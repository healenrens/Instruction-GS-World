"""RGB diagnostics split by observed temporal change and static regions."""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .counterfactuals import predict_shuffled_action, render_state
from .goal_eval_statistics import clustered_paired_comparison
from .train_runtime import move_to_device


PREDICTIONS = ("posterior", "zero_action", "shuffled_action")
REGIONS = ("change", "static", "ambiguous")
ERROR_METRICS = ("charbonnier", "mse", "psnr_db")


@dataclass(frozen=True)
class TemporalRegionConfig:
    blur_kernel: int = 5
    blur_sigma: float = 1.0
    low_floor: float = 0.015
    high_floor: float = 0.04
    low_mad_scale: float = 2.0
    high_mad_scale: float = 4.0
    hysteresis_steps: int = 4
    change_dilation: int = 2

    def validate(self) -> None:
        if self.blur_kernel < 3 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be an odd integer of at least 3")
        if self.blur_sigma <= 0.0:
            raise ValueError("blur_sigma must be positive")
        if not 0.0 <= self.low_floor < self.high_floor < 1.0:
            raise ValueError("change floors must satisfy 0 <= low < high < 1")
        if not 0.0 <= self.low_mad_scale < self.high_mad_scale:
            raise ValueError("MAD scales must satisfy 0 <= low < high")
        if self.hysteresis_steps < 0 or self.change_dilation < 0:
            raise ValueError("morphology step counts must be non-negative")


def _gaussian_kernel(config: TemporalRegionConfig, tensor: torch.Tensor) -> torch.Tensor:
    radius = config.blur_kernel // 2
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / config.blur_sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def _normalized_blur(
    rgb: torch.Tensor,
    valid: torch.Tensor,
    config: TemporalRegionConfig,
) -> torch.Tensor:
    if rgb.ndim != 5 or valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
        raise ValueError("RGB and valid shapes do not align for temporal blur")
    batch, frames, channels, height, width = rgb.shape
    flat_rgb = rgb.float().reshape(batch * frames, channels, height, width)
    flat_valid = valid.float().reshape(batch * frames, 1, height, width)
    kernel = _gaussian_kernel(config, flat_rgb)
    channel_kernel = kernel[None, None].expand(channels, 1, -1, -1)
    mask_kernel = kernel[None, None]
    padding = config.blur_kernel // 2
    numerator = F.conv2d(
        flat_rgb * flat_valid,
        channel_kernel,
        padding=padding,
        groups=channels,
    )
    denominator = F.conv2d(flat_valid, mask_kernel, padding=padding)
    return (numerator / denominator.clamp_min(1e-6)).reshape_as(rgb)


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask[None, None].float(), 3, 1, 1)[0, 0].bool()


def temporal_region_masks(
    history_rgb: torch.Tensor,
    future_rgb: torch.Tensor,
    history_valid: torch.Tensor,
    future_valid: torch.Tensor,
    config: TemporalRegionConfig,
) -> dict[str, torch.Tensor]:
    """Build GT-only diagnostic regions; these masks are never model inputs."""
    config.validate()
    target = future_rgb.float() / 255.0
    current = history_rgb[:, -1:].float() / 255.0
    current = current.expand_as(target)
    current_valid = history_valid[:, -1:].expand_as(future_valid)
    valid = current_valid & future_valid
    blurred_target = _normalized_blur(target, valid, config)
    blurred_current = _normalized_blur(current, valid, config)
    score = (blurred_target - blurred_current).abs().mean(dim=2)
    change = torch.zeros_like(valid)
    static = torch.zeros_like(valid)
    low_threshold = score.new_zeros(score.shape[:2])
    high_threshold = score.new_zeros(score.shape[:2])
    for batch_index in range(score.shape[0]):
        for frame_index in range(score.shape[1]):
            frame_valid = valid[batch_index, frame_index]
            observed = score[batch_index, frame_index][frame_valid]
            if observed.numel() == 0:
                raise ValueError("temporal region evaluation received an empty RGB frame")
            median = observed.median()
            mad = (observed - median).abs().median()
            robust_sigma = 1.4826 * mad
            low = torch.maximum(
                score.new_tensor(config.low_floor),
                median + config.low_mad_scale * robust_sigma,
            )
            high = torch.maximum(
                score.new_tensor(config.high_floor),
                median + config.high_mad_scale * robust_sigma,
            )
            high = torch.maximum(high, low + score.new_tensor(1e-6))
            support = (score[batch_index, frame_index] >= low) & frame_valid
            connected = (score[batch_index, frame_index] >= high) & frame_valid
            for _ in range(config.hysteresis_steps):
                connected = _dilate(connected) & support
            for _ in range(config.change_dilation):
                connected = _dilate(connected) & frame_valid
            change[batch_index, frame_index] = connected
            static[batch_index, frame_index] = (
                frame_valid
                & (score[batch_index, frame_index] < low)
                & ~connected
            )
            low_threshold[batch_index, frame_index] = low
            high_threshold[batch_index, frame_index] = high
    ambiguous = valid & ~change & ~static
    return {
        "valid": valid,
        "change": change,
        "static": static,
        "ambiguous": ambiguous,
        "score": score,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }


def regional_rgb_error_frames(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("regional RGB prediction and target shapes differ")
    if mask.shape != (prediction.shape[0], prediction.shape[1], *prediction.shape[-2:]):
        raise ValueError("regional RGB mask shape mismatch")
    error = prediction.float() - target.float() / 255.0
    weight = mask.to(error.dtype)
    denominator = weight.flatten(2).sum(dim=-1)
    charbonnier_map = torch.sqrt(error.square() + 1e-6).mean(dim=2)
    mse_map = error.square().mean(dim=2)
    charbonnier = (charbonnier_map * weight).flatten(2).sum(dim=-1) / denominator.clamp_min(1.0)
    mse = (mse_map * weight).flatten(2).sum(dim=-1) / denominator.clamp_min(1.0)
    return {
        "charbonnier": charbonnier,
        "mse": mse,
        "psnr_db": -10.0 * torch.log10(mse.clamp_min(1e-8)),
        "nonempty": denominator > 0,
    }


def localization_frames(
    prediction: torch.Tensor,
    current: torch.Tensor,
    regions: dict[str, torch.Tensor],
    config: TemporalRegionConfig,
) -> dict[str, torch.Tensor]:
    valid = regions["valid"]
    prediction_blurred = _normalized_blur(prediction.float(), valid, config)
    current_blurred = _normalized_blur(current.float(), valid, config)
    magnitude = (prediction_blurred - current_blurred).abs().mean(dim=2)
    change = regions["change"]
    static = regions["static"]

    def region_mean(mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(magnitude.dtype)
        return (magnitude * weight).flatten(2).sum(dim=-1) / weight.flatten(2).sum(
            dim=-1
        ).clamp_min(1.0)

    change_mean = region_mean(change)
    static_mean = region_mean(static)
    valid_mass = (magnitude * valid).flatten(2).sum(dim=-1)
    change_mass = (magnitude * change).flatten(2).sum(dim=-1)
    mass_fraction = change_mass / valid_mass.clamp_min(1e-8)
    area_fraction = change.flatten(2).sum(dim=-1).float() / valid.flatten(2).sum(
        dim=-1
    ).clamp_min(1).float()
    topk_iou = magnitude.new_zeros(magnitude.shape[:2])
    nonempty = change.flatten(2).any(dim=-1)
    for batch_index in range(magnitude.shape[0]):
        for frame_index in range(magnitude.shape[1]):
            if not bool(nonempty[batch_index, frame_index]):
                continue
            frame_valid = valid[batch_index, frame_index]
            target_change = change[batch_index, frame_index][frame_valid]
            count = int(target_change.sum())
            selected = torch.topk(
                magnitude[batch_index, frame_index][frame_valid],
                count,
            ).indices
            overlap = target_change[selected].float().sum()
            topk_iou[batch_index, frame_index] = overlap / (2 * count - overlap).clamp_min(1.0)
    return {
        "change_magnitude": change_mean,
        "static_magnitude": static_mean,
        "change_static_ratio": change_mean / static_mean.clamp_min(1e-6),
        "change_mass_fraction": mass_fraction,
        "change_mass_lift": mass_fraction / area_fraction.clamp_min(1e-6),
        "topk_iou_at_gt_area": topk_iou,
        "nonempty": nonempty,
    }


def _amp_context(device: torch.device, amp: str):
    if amp == "fp32":
        return contextlib.nullcontext()
    if amp != "bf16" or device.type != "cuda":
        raise ValueError("bf16 temporal-region evaluation requires CUDA")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _available_comparison(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    clusters: torch.Tensor,
) -> dict:
    unique_clusters = torch.unique(clusters)
    if len(prediction) == 0 or len(unique_clusters) < 2:
        return {"available": False, "frames": len(prediction), "clusters": len(unique_clusters)}
    return {
        "available": True,
        **clustered_paired_comparison(prediction, reference, clusters),
    }


def _summarize_error_slice(
    values: dict[str, dict[str, torch.Tensor]],
    clusters: torch.Tensor,
    selected: torch.Tensor,
) -> dict:
    means = {
        metric: {
            name: float(prediction[selected].mean())
            for name, prediction in predictions.items()
        }
        for metric, predictions in values.items()
    }
    comparisons = {}
    selected_clusters = clusters[selected]
    for metric in ("charbonnier", "mse"):
        metric_values = values[metric]
        comparisons[metric] = {
            "posterior_vs_zero": _available_comparison(
                metric_values["posterior"][selected],
                metric_values["zero_action"][selected],
                selected_clusters,
            ),
            "posterior_vs_shuffled": _available_comparison(
                metric_values["posterior"][selected],
                metric_values["shuffled_action"][selected],
                selected_clusters,
            ),
            "shuffled_vs_zero": _available_comparison(
                metric_values["shuffled_action"][selected],
                metric_values["zero_action"][selected],
                selected_clusters,
            ),
        }
    return {
        "frames": int(selected.sum()),
        "clusters": len(torch.unique(selected_clusters)),
        "mean": means,
        "comparison": comparisons,
    }


@torch.no_grad()
def evaluate_temporal_regions(model, loader, device: torch.device, amp: str, config: TemporalRegionConfig) -> dict:
    config.validate()
    action_bank = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        history_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with _amp_context(device, amp):
            output = model(batch, history_mask=history_mask)
        action_bank.append(output["posterior_actions"].float().cpu())
    actions = torch.cat(action_bank)
    if len(actions) < 2:
        raise ValueError("temporal-region evaluation requires at least two samples")
    shuffle_index = torch.arange(len(actions)).roll(len(actions) // 2)

    error_chunks = {
        region: {metric: {name: [] for name in PREDICTIONS} for metric in ERROR_METRICS}
        for region in REGIONS
    }
    region_clusters = {region: [] for region in REGIONS}
    region_horizons = {region: [] for region in REGIONS}
    region_times = {region: [] for region in REGIONS}
    coverage_chunks = {name: [] for name in REGIONS}
    threshold_chunks = {"low_threshold": [], "high_threshold": []}
    localization_chunks = {
        metric: {name: [] for name in PREDICTIONS}
        for metric in (
            "change_magnitude",
            "static_magnitude",
            "change_static_ratio",
            "change_mass_fraction",
            "change_mass_lift",
            "topk_iou_at_gt_area",
        )
    }
    localization_clusters = []
    localization_horizons = []
    sample_offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        shuffled_actions = actions[
            shuffle_index[sample_offset : sample_offset + batch_size]
        ].to(device, non_blocking=True)
        history_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with _amp_context(device, amp):
            output = model(batch, history_mask=history_mask)
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
            shuffled_slots, shuffled_centers = predict_shuffled_action(
                model,
                batch,
                output,
                shuffled_actions,
            )
            shuffled_feature, shuffled_rgb = render_state(
                model,
                batch,
                output,
                shuffled_slots,
                shuffled_centers,
            )
        del zero_feature, shuffled_feature, shuffled_slots, shuffled_centers
        if output["rendered_future_rgb"] is None or zero_rgb is None or shuffled_rgb is None:
            raise ValueError("temporal-region evaluation requires RGB predictions")
        predictions = {
            "posterior": output["rendered_future_rgb"].float(),
            "zero_action": zero_rgb.float(),
            "shuffled_action": shuffled_rgb.float(),
        }
        regions = temporal_region_masks(
            batch["history_rgb"],
            batch["future_rgb"],
            batch["history_rgb_valid"],
            batch["future_rgb_valid"],
            config,
        )
        frames = batch["future_rgb"].shape[1]
        clusters = batch["sequence_index"][:, None].expand(-1, frames)
        horizons = torch.arange(frames, device=device)[None].expand(batch_size, -1)
        times = batch["future_times"].float()
        valid_pixels = regions["valid"].flatten(2).sum(dim=-1).clamp_min(1)
        for region in REGIONS:
            mask = regions[region]
            pixel_count = mask.flatten(2).sum(dim=-1)
            nonempty = pixel_count > 0
            coverage_chunks[region].append((pixel_count / valid_pixels).float().cpu())
            region_clusters[region].append(clusters[nonempty].cpu())
            region_horizons[region].append(horizons[nonempty].cpu())
            region_times[region].append(times[nonempty].cpu())
            for name, prediction in predictions.items():
                errors = regional_rgb_error_frames(prediction, batch["future_rgb"], mask)
                for metric in ERROR_METRICS:
                    error_chunks[region][metric][name].append(errors[metric][nonempty].cpu())
        for name in threshold_chunks:
            threshold_chunks[name].append(regions[name].cpu())
        current = batch["history_rgb"][:, -1:].float() / 255.0
        current = current.expand_as(predictions["posterior"])
        changed = regions["change"].flatten(2).any(dim=-1)
        localization_clusters.append(clusters[changed].cpu())
        localization_horizons.append(horizons[changed].cpu())
        for name, prediction in predictions.items():
            metrics = localization_frames(prediction, current, regions, config)
            for metric in localization_chunks:
                localization_chunks[metric][name].append(metrics[metric][changed].cpu())
        sample_offset += batch_size
    if sample_offset != len(actions):
        raise RuntimeError("action bank and temporal-region loader differ")

    region_reports = {}
    for region in REGIONS:
        values = {
            metric: {name: torch.cat(chunks) for name, chunks in predictions.items()}
            for metric, predictions in error_chunks[region].items()
        }
        clusters = torch.cat(region_clusters[region])
        horizons = torch.cat(region_horizons[region])
        times = torch.cat(region_times[region])
        all_frames = torch.ones(len(clusters), dtype=torch.bool)
        report = _summarize_error_slice(values, clusters, all_frames)
        report["mean_time_seconds"] = float(times.mean())
        report["by_horizon"] = {}
        for horizon in torch.unique(horizons, sorted=True).tolist():
            selected = horizons == horizon
            horizon_report = _summarize_error_slice(values, clusters, selected)
            horizon_report["mean_time_seconds"] = float(times[selected].mean())
            report["by_horizon"][str(horizon)] = horizon_report
        region_reports[region] = report

    localization = {
        metric: {name: torch.cat(chunks) for name, chunks in predictions.items()}
        for metric, predictions in localization_chunks.items()
    }
    localization_report = {
        "frames": len(torch.cat(localization_clusters)),
        "clusters": len(torch.unique(torch.cat(localization_clusters))),
        "mean": {
            metric: {name: float(value.mean()) for name, value in predictions.items()}
            for metric, predictions in localization.items()
        },
    }
    coverage = {name: torch.cat(chunks) for name, chunks in coverage_chunks.items()}
    thresholds = {name: torch.cat(chunks) for name, chunks in threshold_chunks.items()}
    headline = {
        "change_posterior_vs_zero_relative": region_reports["change"]["comparison"]["charbonnier"]["posterior_vs_zero"].get("relative_improvement"),
        "change_posterior_vs_shuffled_relative": region_reports["change"]["comparison"]["charbonnier"]["posterior_vs_shuffled"].get("relative_improvement"),
        "static_posterior_vs_zero_relative": region_reports["static"]["comparison"]["charbonnier"]["posterior_vs_zero"].get("relative_improvement"),
        "posterior_change_topk_iou": localization_report["mean"]["topk_iou_at_gt_area"]["posterior"],
        "zero_change_topk_iou": localization_report["mean"]["topk_iou_at_gt_area"]["zero_action"],
    }
    return {
        "samples": len(actions),
        "action_source": "future_conditioned_posterior_oracle",
        "deployable_prediction": False,
        "mask_source": "ground_truth_current_and_future_rgb_for_evaluation_only",
        "region_config": asdict(config),
        "coverage": {
            "mean_fraction": {name: float(value.mean()) for name, value in coverage.items()},
            "nonempty_frame_fraction": {
                name: float((value > 0).float().mean()) for name, value in coverage.items()
            },
            "mean_low_threshold": float(thresholds["low_threshold"].mean()),
            "mean_high_threshold": float(thresholds["high_threshold"].mean()),
        },
        "regions": region_reports,
        "localization": localization_report,
        "headline": headline,
    }
