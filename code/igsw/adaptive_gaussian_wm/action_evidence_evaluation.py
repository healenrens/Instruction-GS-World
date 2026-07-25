"""Causal, component, and object-local latent-action evaluation."""
from __future__ import annotations

import torch

from .action_interventions import (
    component_actions,
    current_object_support,
    replace_history,
    spatial_locality,
    state_locality,
    support_to_rgb,
    swap_one_object_action,
)
from .counterfactuals import (
    predict_shuffled_action,
    render_state,
)
from .goal_eval_statistics import (
    clustered_paired_comparison,
)
from .matched_flat_evaluation import (
    feature_mse_by_query,
    rgb_distance_by_query,
    rgb_region_errors_by_query,
)
from .task_group_evidence import (
    remap_task_layout,
    selected_task_group_layout,
    task_group_comparison,
)
from .temporal_region_evaluation import (
    TemporalRegionConfig,
)
from .train_runtime import move_to_device


VARIANTS = (
    "posterior",
    "shuffled_action",
    "zero_action",
    "canonical_only",
    "residual_only",
    "history_swapped_fixed_action",
)
COMPARISONS = (
    ("posterior_over_shuffled", "posterior", "shuffled_action"),
    ("posterior_over_zero", "posterior", "zero_action"),
    ("posterior_over_canonical", "posterior", "canonical_only"),
    ("posterior_over_residual", "posterior", "residual_only"),
    (
        "posterior_over_history_swapped",
        "posterior",
        "history_swapped_fixed_action",
    ),
)
METRICS = (
    "feature_mse",
    "rgb_distance",
    "object_feature_mse",
    "object_center_mse",
    "object_rgb_mse",
)
REGIONS = ("change", "static")


def _amp_context(device: torch.device, amp: str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and amp == "bf16"
        else torch.no_grad()
    )


def cross_episode_donors(clusters: torch.Tensor) -> torch.Tensor:
    if clusters.ndim != 1 or len(clusters) < 2:
        raise ValueError("action evidence requires at least two samples")
    donors = torch.arange(len(clusters)).roll(len(clusters) // 2)
    for index in range(len(donors)):
        while clusters[donors[index]] == clusters[index]:
            donors[index] = (donors[index] + 1) % len(donors)
            if donors[index] == index:
                raise ValueError("every sample needs a donor episode")
    return donors


def _weighted_object_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or activity.shape != prediction.shape[:-1]:
        raise ValueError("object prediction, target, and activity do not align")
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    weight = activity.float()
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)


def _state(model, slots, centers, feature, rgb) -> dict[str, torch.Tensor]:
    if rgb is None:
        raise ValueError("action evidence requires RGB predictions")
    return {
        "slots": slots,
        "centers": centers,
        "object_features": model.object_aggregator.decode_feature(slots),
        "object_rgb": torch.sigmoid(
            model.object_aggregator.decode_rgb_logits(slots).float()
        ),
        "feature": feature,
        "rgb": rgb,
    }


def _comparisons(
    variants: dict[str, torch.Tensor],
    clusters: torch.Tensor,
) -> dict[str, dict]:
    return {
        name: clustered_paired_comparison(
            variants[prediction], variants[reference], clusters
        )
        for name, prediction, reference in COMPARISONS
    }


def _task_comparisons(
    variants: dict[str, torch.Tensor],
    clusters: torch.Tensor,
    layout,
) -> dict[str, dict]:
    return {
        name: task_group_comparison(
            variants[prediction], variants[reference], clusters, layout
        )
        for name, prediction, reference in COMPARISONS
    }


def _task_scalar(values: torch.Tensor, clusters: torch.Tensor, layout) -> dict:
    episode_values = []
    episode_tasks = []
    for episode in torch.unique(clusters, sorted=True):
        mask = clusters == episode
        tasks = torch.unique(layout.ids[mask])
        if len(tasks) != 1:
            raise ValueError("one episode maps to multiple tasks")
        episode_values.append(values[mask].mean())
        episode_tasks.append(tasks[0])
    episode_values = torch.stack(episode_values)
    episode_tasks = torch.stack(episode_tasks)
    per_task = {}
    for task_id, name in enumerate(layout.names):
        selected = episode_tasks == task_id
        if not bool(selected.any()):
            raise ValueError(f"locality has no evidence for task {name}")
        per_task[name] = {
            "episodes": int(selected.sum()),
            "mean": float(episode_values[selected].mean()),
        }
    task_means = torch.tensor([entry["mean"] for entry in per_task.values()])
    return {
        "task_count": len(per_task),
        "mean_over_tasks": float(task_means.mean()),
        "worst_task": float(task_means.min()),
        "per_task": per_task,
    }


@torch.no_grad()
def evaluate(model, loader, donor_loader, device, amp, layout) -> dict:
    action_chunks = []
    cluster_chunks = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with _amp_context(device, amp):
            output = model(batch, history_mask=mask)
        action_chunks.append(output["posterior_actions"].float().cpu())
        cluster_chunks.append(batch["sequence_index"].long().cpu())
    action_bank = torch.cat(action_chunks)
    clusters = torch.cat(cluster_chunks)
    donors = cross_episode_donors(clusters)

    metric_chunks = {
        metric: {variant: [] for variant in VARIANTS} for metric in METRICS
    }
    region_values = {
        region: {variant: [] for variant in VARIANTS} for region in REGIONS
    }
    region_clusters = {region: [] for region in REGIONS}
    region_horizons = {region: [] for region in REGIONS}
    locality_chunks: dict[str, list[torch.Tensor]] = {}
    locality_per_object: dict[str, list[list[torch.Tensor]]] = {}
    history_state_difference = {"feature": [], "center": [], "rgb": []}
    query_times = []
    offset = 0
    for cpu_batch, donor_cpu in zip(loader, donor_loader, strict=True):
        batch = move_to_device(cpu_batch, device)
        donor_batch = move_to_device(donor_cpu, device)
        batch_size = batch["history_features"].shape[0]
        donor_actions = action_bank[
            donors[offset : offset + batch_size]
        ].to(device, non_blocking=True)
        mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with _amp_context(device, amp):
            output = model(batch, history_mask=mask)
            posterior_actions = output["posterior_actions"]
            states = {
                "posterior": _state(
                    model,
                    output["predicted_future_slots"],
                    output["predicted_future_centers"],
                    output["rendered_future_features"],
                    output["rendered_future_rgb"],
                )
            }
            action_variants = component_actions(
                posterior_actions, model.config.canonical_action_dim
            )
            action_variants["shuffled_action"] = donor_actions
            for name, actions in action_variants.items():
                slots, centers = predict_shuffled_action(
                    model, batch, output, actions
                )
                feature, rgb = render_state(
                    model, batch, output, slots, centers
                )
                states[name] = _state(model, slots, centers, feature, rgb)
            swapped_batch = replace_history(batch, donor_batch)
            swapped = model(
                swapped_batch,
                history_mask=mask,
                actions_override=posterior_actions,
            )
            states["history_swapped_fixed_action"] = _state(
                model,
                swapped["predicted_future_slots"],
                swapped["predicted_future_centers"],
                swapped["rendered_future_features"],
                swapped["rendered_future_rgb"],
            )
        activity = output["target_future_activity"].float()
        targets = {
            "object_features": output["target_future_object_features"],
            "centers": output["target_future_centers"],
            "object_rgb": output["target_future_object_rgb"],
        }
        rgb_predictions = {name: state["rgb"] for name, state in states.items()}
        regions = rgb_region_errors_by_query(
            rgb_predictions,
            batch,
            TemporalRegionConfig(),
        )
        frame_clusters = batch["sequence_index"][:, None].expand_as(
            batch["future_times"]
        )
        frame_horizons = torch.arange(
            batch["future_times"].shape[1], device=device
        )[None].expand_as(batch["future_times"])
        for region in REGIONS:
            nonempty = regions[region]["nonempty"]
            region_clusters[region].append(frame_clusters[nonempty].cpu())
            region_horizons[region].append(frame_horizons[nonempty].cpu())
            for name in VARIANTS:
                region_values[region][name].append(
                    regions[region][name][nonempty].cpu()
                )
        for name, state in states.items():
            metric_chunks["feature_mse"][name].append(
                feature_mse_by_query(
                    state["feature"],
                    batch["future_features"],
                    batch["future_valid"],
                ).cpu()
            )
            metric_chunks["rgb_distance"][name].append(
                rgb_distance_by_query(
                    state["rgb"],
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
            metric_chunks["object_feature_mse"][name].append(
                _weighted_object_error(
                    state["object_features"], targets["object_features"], activity
                ).cpu()
            )
            metric_chunks["object_center_mse"][name].append(
                _weighted_object_error(
                    state["centers"], targets["centers"], activity
                ).cpu()
            )
            metric_chunks["object_rgb_mse"][name].append(
                _weighted_object_error(
                    state["object_rgb"], targets["object_rgb"], activity
                ).cpu()
            )
        base = states["posterior"]
        history_state_difference["feature"].append(
            (base["object_features"] - states["history_swapped_fixed_action"]["object_features"])
            .float().square().mean(dim=(-1, -2)).sqrt().cpu()
        )
        history_state_difference["center"].append(
            (base["centers"] - states["history_swapped_fixed_action"]["centers"])
            .float().square().mean(dim=(-1, -2)).sqrt().cpu()
        )
        history_state_difference["rgb"].append(
            (base["object_rgb"] - states["history_swapped_fixed_action"]["object_rgb"])
            .float().square().mean(dim=(-1, -2)).sqrt().cpu()
        )
        support = current_object_support(output)
        rgb_support = support_to_rgb(
            support, batch["feature_grid_hw"], batch["future_rgb_valid"]
        )
        object_metrics: dict[str, list[torch.Tensor]] = {}
        for object_index in range(model.config.object_slots):
            actions = swap_one_object_action(
                posterior_actions, donor_actions, object_index
            )
            with _amp_context(device, amp):
                slots, centers = predict_shuffled_action(
                    model, batch, output, actions
                )
                feature, rgb = render_state(
                    model, batch, output, slots, centers
                )
            slot_rms, slot_own = state_locality(
                base["slots"], slots, object_index
            )
            center_rms, center_own = state_locality(
                base["centers"], centers, object_index
            )
            feature_local = spatial_locality(
                base["feature"], feature, support, batch["future_valid"], object_index
            )
            rgb_local = spatial_locality(
                base["rgb"], rgb, rgb_support, batch["future_rgb_valid"], object_index
            )
            entries = {
                "action_swap_rms": (
                    posterior_actions[:, :, object_index]
                    - donor_actions[:, :, object_index]
                ).float().square().mean(dim=-1).sqrt(),
                "slot_effect_rms": slot_rms,
                "slot_own_fraction": slot_own,
                "slot_own_lift": slot_own * model.config.object_slots,
                "center_effect_rms": center_rms,
                "center_own_fraction": center_own,
                "center_own_lift": center_own * model.config.object_slots,
                **{f"feature_{key}": value for key, value in feature_local.items()},
                **{f"rgb_{key}": value for key, value in rgb_local.items()},
            }
            for name, value in entries.items():
                object_metrics.setdefault(name, []).append(value.cpu())
        weight = (
            output["history_slot_states"][-1].activity[:, None].float()
            * activity
        ).cpu()
        for name, values in object_metrics.items():
            stacked = torch.stack(values, dim=-1)
            locality_chunks.setdefault(name, []).append(
                (stacked * weight).sum(dim=-1)
                / weight.sum(dim=-1).clamp_min(1e-6)
            )
            if name not in locality_per_object:
                locality_per_object[name] = [list() for _ in values]
            for index, value in enumerate(values):
                locality_per_object[name][index].append(value)
        query_times.append(batch["future_times"].float().cpu())
        offset += batch_size
    if offset != len(action_bank):
        raise RuntimeError("action evidence loaders did not cover the action bank")

    tensors = {
        metric: {name: torch.cat(chunks) for name, chunks in variants.items()}
        for metric, variants in metric_chunks.items()
    }
    query_times = torch.cat(query_times)
    overall = {
        metric: _comparisons(
            {name: value.mean(dim=1) for name, value in variants.items()}, clusters
        )
        for metric, variants in tensors.items()
    }
    task_evidence = {
        metric: _task_comparisons(
            {name: value.mean(dim=1) for name, value in variants.items()},
            clusters,
            layout,
        )
        for metric, variants in tensors.items()
    }
    region_report = {}
    for region in REGIONS:
        values = {name: torch.cat(chunks) for name, chunks in region_values[region].items()}
        frame_ids = torch.cat(region_clusters[region])
        horizons = torch.cat(region_horizons[region])
        frame_layout = remap_task_layout(clusters, layout, frame_ids)
        region_report[region] = {
            "frames": len(frame_ids),
            "mean": {name: float(value.mean()) for name, value in values.items()},
            "comparison": _comparisons(values, frame_ids),
            "task_group_evidence": _task_comparisons(
                values, frame_ids, frame_layout
            ),
            "by_horizon": {
                str(index): _comparisons(
                    {name: value[horizons == index] for name, value in values.items()},
                    frame_ids[horizons == index],
                )
                for index in torch.unique(horizons, sorted=True).tolist()
            },
        }
    locality = {name: torch.cat(chunks) for name, chunks in locality_chunks.items()}
    return {
        "samples": len(clusters),
        "clusters": len(torch.unique(clusters)),
        "action_source": "future_conditioned_continuous_object_posterior",
        "deployable_prediction": False,
        "history_swap_contract": "future_targets_fixed_all_causal_history_fields_replaced",
        "mean": {
            metric: {name: float(value.mean()) for name, value in variants.items()}
            for metric, variants in tensors.items()
        },
        "comparison": overall,
        "task_group_evidence": task_evidence,
        "by_future_query": {
            str(index): {
                "mean_time_seconds": float(query_times[:, index].mean()),
                "comparison": {
                    metric: _comparisons(
                        {name: value[:, index] for name, value in variants.items()},
                        clusters,
                    )
                    for metric, variants in tensors.items()
                },
            }
            for index in range(query_times.shape[1])
        },
        "rgb_regions": region_report,
        "fixed_action_history_sensitivity": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in history_state_difference.items()
        },
        "object_action_locality": {
            "support_source": "current_gpstoken_assignment_times_object_assignment",
            "spatial_domains": {
                "feature": "DINO_feature_grid",
                "rgb": "valid_RGB_content_grid",
            },
            "activity_weighted_mean": {
                name: float(value.mean()) for name, value in locality.items()
            },
            "by_future_query": {
                str(index): {
                    name: float(value[:, index].mean())
                    for name, value in locality.items()
                }
                for index in range(query_times.shape[1])
            },
            "by_task": {
                name: _task_scalar(value.mean(dim=1), clusters, layout)
                for name, value in locality.items()
            },
            "per_object_mean": {
                name: [
                    float(torch.cat(chunks).mean()) for chunks in object_chunks
                ]
                for name, object_chunks in locality_per_object.items()
            },
        },
    }
