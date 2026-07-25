"""Evaluate current-frame Object-JEPA and RGB reconstruction without future paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.goal_eval_statistics import (  # noqa: E402
    clustered_paired_comparison,
)
from igsw.adaptive_gaussian_wm.representation import (  # noqa: E402
    reconstruct_current,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    masked_rgb_mean,
    rgb_reconstruction_loss,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


def weighted_per_sample(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return (value * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def feature_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    coverage: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid_weight = valid.to(prediction.dtype)
    covered_weight = valid_weight * (coverage > 1e-4)
    mse = (prediction - target).square().mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    target_mean = (
        (target * valid_weight[..., None]).sum(dim=2, keepdim=True)
        / valid_weight.sum(dim=2, keepdim=True)[..., None].clamp_min(1.0)
    )
    baseline_mse = (target_mean - target).square().mean(dim=-1)
    return {
        "mse": weighted_per_sample(mse, valid_weight),
        "objective": weighted_per_sample(mse + 0.1 * cosine, covered_weight),
        "global_feature_mse": weighted_per_sample(baseline_mse, valid_weight),
        "coverage_fraction": weighted_per_sample(
            (coverage > 1e-4).to(prediction.dtype),
            valid_weight,
        ),
        "coverage_mean": weighted_per_sample(coverage, valid_weight),
    }


def rgb_distance_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    return torch.stack(
        [
            rgb_reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                ssim_weight,
            )[0]
            for index in range(len(prediction))
        ]
    )


def effective_rank(features: torch.Tensor) -> float:
    flat = features.reshape(-1, features.shape[-1]).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def within_sample_slot_cosine(slots: torch.Tensor) -> float:
    normalized = F.normalize(slots.float(), dim=-1)
    count = normalized.shape[1]
    if count < 2:
        return 1.0
    summed = normalized.sum(dim=1)
    off_diagonal = summed.square().sum(dim=-1) - count
    return float((off_diagonal / (count * (count - 1))).mean())


def source_cluster_ids(paths: list[str]) -> torch.Tensor:
    stems = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    names = [stem.rsplit("_t", 1)[0] for stem in stems]
    if any(name == stem for name, stem in zip(names, stems)):
        raise ValueError("pair filename lacks a source/time boundary")
    identities = {name: index for index, name in enumerate(sorted(set(names)))}
    return torch.tensor([identities[name] for name in names], dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    clusters: torch.Tensor,
) -> dict:
    collected: dict[str, list[torch.Tensor]] = {
        "feature_mse": [],
        "feature_objective": [],
        "global_feature_mse": [],
        "feature_coverage_fraction": [],
        "feature_coverage_mean": [],
        "feature_mse_shuffled_slots": [],
        "effective_tokens": [],
        "assignment_entropy": [],
        "effective_slots": [],
    }
    if model.config.rgb_supervision:
        collected.update(
            {
                "rgb_distance": [],
                "rgb_distance_shuffled_slots": [],
                "global_color_rgb_distance": [],
                "rgb_coverage_mean": [],
                "object_rgb_mse": [],
            }
        )
    slot_latents = []
    decoded_slot_features = []
    reconstructed_features = []
    target_features = []
    covariance_eigenvalues = []
    readout_opacity = []
    readout_activation = []
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    slot_bank = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        with amp_context():
            history = model.encode_history(batch)
        slot_bank.append(history["slot_states"][-1].slots.cpu())
    slot_bank = torch.cat(slot_bank)
    if slot_bank.shape[0] < 2:
        raise ValueError("slot counterfactual requires at least two samples")
    shuffle_index = torch.arange(slot_bank.shape[0]).roll(
        slot_bank.shape[0] // 2
    )
    sample_offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        with amp_context():
            output = reconstruct_current(model, batch)
            batch_size = batch["history_features"].shape[0]
            shuffled_slots = slot_bank[
                shuffle_index[sample_offset : sample_offset + batch_size]
            ].to(device, non_blocking=True)
            shuffled_output = reconstruct_current(
                model,
                batch,
                output["history"],
                shuffled_slots,
            )
            sample_offset += batch_size
        current_target = batch["history_features"][:, -1:]
        current_valid = batch["history_valid"][:, -1:]
        values = feature_metrics(
            output["feature"].float(),
            current_target.float(),
            current_valid,
            output["feature_coverage"].float(),
        )
        shuffled_values = feature_metrics(
            shuffled_output["feature"].float(),
            current_target.float(),
            current_valid,
            shuffled_output["feature_coverage"].float(),
        )
        collected["feature_mse_shuffled_slots"].append(
            shuffled_values["mse"].cpu()
        )
        for name, value in values.items():
            collected[f"feature_{name}" if name != "global_feature_mse" else name].append(
                value.cpu()
            )
        tokens = output["tokens"]
        slots = output["slots"]
        slot_latents.append(slots.slots.float().cpu())
        decoded_slot_features.append(slots.decoded_feature.float().cpu())
        reconstructed_features.append(output["feature"].float().cpu())
        target_features.append(current_target.float().cpu())
        covariance_eigenvalues.append(
            torch.linalg.eigvalsh(output["readout"].covariance.float()).cpu()
        )
        readout_opacity.append(output["readout"].opacity.float().cpu())
        readout_activation.append(output["readout"].activation.float().cpu())
        collected["effective_tokens"].append(
            tokens.activation.sum(dim=1).squeeze(-1).float().cpu()
        )
        entropy = -(
            slots.assignment.float().clamp_min(1e-8).log()
            * slots.assignment.float()
        ).sum(dim=-1).mean(dim=-1)
        collected["assignment_entropy"].append(entropy.cpu())
        slot_mass = (
            slots.assignment.float()
            * tokens.activation.float()
        ).sum(dim=1)
        slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        collected["effective_slots"].append(
            slot_mass.square().sum(dim=-1).reciprocal().cpu()
        )
        if model.config.rgb_supervision:
            target_rgb = batch["history_rgb"][:, -1:]
            valid_rgb = batch["history_rgb_valid"][:, -1:]
            background = masked_rgb_mean(target_rgb, valid_rgb)
            background = background[..., None, None].expand_as(target_rgb)
            collected["rgb_distance"].append(
                rgb_distance_per_sample(
                    output["rgb"].float(),
                    target_rgb,
                    valid_rgb,
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
            collected["rgb_distance_shuffled_slots"].append(
                rgb_distance_per_sample(
                    shuffled_output["rgb"].float(),
                    target_rgb,
                    valid_rgb,
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
            collected["global_color_rgb_distance"].append(
                rgb_distance_per_sample(
                    background,
                    target_rgb,
                    valid_rgb,
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
            collected["rgb_coverage_mean"].append(
                weighted_per_sample(
                    output["rgb_coverage"].float(),
                    valid_rgb.to(torch.float32),
                ).cpu()
            )
            with amp_context():
                predicted_object_rgb = torch.sigmoid(
                    model.object_aggregator.decode_rgb_logits(slots.slots)
                )
            target_object_rgb = output[
                "readout_context"
            ].current_object_rgb
            collected["object_rgb_mse"].append(
                (predicted_object_rgb - target_object_rgb)
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
    metric_tensors = {
        name: torch.cat(values)
        for name, values in collected.items()
    }
    if len(clusters) != len(next(iter(metric_tensors.values()))):
        raise RuntimeError("source clusters do not align with representation samples")
    metrics = {
        name: float(value.mean())
        for name, value in metric_tensors.items()
    }
    latent = torch.cat(slot_latents)
    decoded = torch.cat(decoded_slot_features)
    reconstructed = torch.cat(reconstructed_features)
    target = torch.cat(target_features)
    eigenvalues = torch.cat(covariance_eigenvalues).flatten(0, -2)
    condition = eigenvalues[:, 1] / eigenvalues[:, 0].clamp_min(1e-12)
    centered_latent = latent - latent.mean(dim=0, keepdim=True)
    metrics.update(
        {
            "slot_latent_sample_variance": float(centered_latent.square().mean()),
            "slot_latent_sample_variation_fraction": float(
                centered_latent.square().mean()
                / latent.square().mean().clamp_min(1e-12)
            ),
            "within_sample_slot_cosine": within_sample_slot_cosine(latent),
            "decoded_slot_feature_variance": float(
                decoded.var(dim=(0, 1), unbiased=False).mean()
            ),
            "decoded_slot_feature_effective_rank": effective_rank(decoded),
            "reconstructed_feature_effective_rank": effective_rank(reconstructed),
            "target_feature_effective_rank": effective_rank(target),
            "readout_covariance_min_eigenvalue": float(eigenvalues[:, 0].min()),
            "readout_covariance_p01_eigenvalue": float(
                torch.quantile(eigenvalues[:, 0], 0.01)
            ),
            "readout_covariance_condition_median": float(
                condition.median()
            ),
            "readout_covariance_condition_p99": float(
                torch.quantile(condition, 0.99)
            ),
            "readout_covariance_condition_max": float(condition.max()),
            "readout_opacity_mean": float(torch.cat(readout_opacity).mean()),
            "readout_activation_mean": float(
                torch.cat(readout_activation).mean()
            ),
            "reconstructed_spatial_variance": float(
                reconstructed.var(dim=2, unbiased=False).mean()
            ),
            "target_spatial_variance": float(
                target.var(dim=2, unbiased=False).mean()
            ),
        }
    )
    metrics["feature_improvement_vs_global"] = (
        metrics["global_feature_mse"] - metrics["feature_mse"]
    ) / max(metrics["global_feature_mse"], 1e-8)
    metrics["feature_vs_global_paired"] = clustered_paired_comparison(
        metric_tensors["feature_mse"],
        metric_tensors["global_feature_mse"],
        clusters,
    )
    metrics["feature_slot_conditioning_paired"] = clustered_paired_comparison(
        metric_tensors["feature_mse"],
        metric_tensors["feature_mse_shuffled_slots"],
        clusters,
    )
    if model.config.rgb_supervision:
        metrics["rgb_improvement_vs_global_color"] = (
            metrics["global_color_rgb_distance"] - metrics["rgb_distance"]
        ) / max(metrics["global_color_rgb_distance"], 1e-8)
        metrics["rgb_vs_global_color_paired"] = clustered_paired_comparison(
            metric_tensors["rgb_distance"],
            metric_tensors["global_color_rgb_distance"],
            clusters,
        )
        metrics["rgb_slot_conditioning_paired"] = clustered_paired_comparison(
            metric_tensors["rgb_distance"],
            metric_tensors["rgb_distance_shuffled_slots"],
            clusters,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", default="")
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_items <= 0 or args.batch <= 0:
        raise ValueError("max_items and batch must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    condition_cache = args.condition_cache if config.condition_dim > 0 else ""
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        condition_cache=condition_cache,
        load_rgb=config.rgb_supervision,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(
        config,
        dataset,
        config.condition_dim > 0,
        config.rgb_supervision,
    )
    expected_cache = checkpoint.get("args", {}).get("condition_feature_sha256", "")
    if expected_cache and expected_cache != dataset.condition_store.feature_sha256:
        raise ValueError("evaluation condition cache differs from checkpoint")
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    clusters = source_cluster_ids(dataset.paths)
    metrics = evaluate(model, loader, device, args.amp, clusters)
    path_digest = hashlib.sha256(
        "\n".join(os.path.abspath(path) for path in dataset.paths).encode()
    ).hexdigest()
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "dino": os.path.abspath(args.dino),
        "split": args.split,
        "samples": len(dataset),
        "source_clusters": len(torch.unique(clusters)),
        "path_digest": path_digest,
        "first_pair": os.path.abspath(dataset.paths[0]),
        "last_pair": os.path.abspath(dataset.paths[-1]),
        "config": {
            "covariance_floor": config.covariance_floor,
            "rgb_loss_weight": config.rgb_loss_weight,
        },
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
