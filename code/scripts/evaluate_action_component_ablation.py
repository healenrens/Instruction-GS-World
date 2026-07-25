"""Ablate canonical and residual posterior-action components."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from action_transfer_evaluation import (  # noqa: E402
    TRANSFER_COMPARISONS,
    TRANSFER_VARIANTS,
    component_action_variants,
    nearest_effect_donors,
    transfer_action_variants,
)
from evaluate_posterior_dynamics_gate import (  # noqa: E402
    masked_feature_mse,
    rgb_distance,
    weighted_latent_mse,
)
from evaluate_visual_sequence_temporal_regions import _available_samples  # noqa: E402
from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.counterfactuals import (  # noqa: E402
    predict_shuffled_action,
    render_state,
)
from igsw.adaptive_gaussian_wm.goal_eval_statistics import (  # noqa: E402
    clustered_paired_comparison,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    selected_task_group_layout, task_group_metric_evidence,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)

VARIANTS = (
    "posterior",
    "shuffled_action",
    "canonical_only",
    "residual_only",
    "zero_action",
    "copy",
    *TRANSFER_VARIANTS,
)
METRICS = ("feature_mse", "latent_mse", "rgb_distance")
STRATIFIED_METRICS = (
    "change_weighted_feature_mse",
    "change_weighted_rgb_charbonnier",
)
COMPONENT_COMPARISONS = (
    ("posterior_over_zero", "posterior", "zero_action"),
    ("posterior_over_shuffled", "posterior", "shuffled_action"),
    ("posterior_over_copy", "posterior", "copy"),
    ("posterior_over_canonical", "posterior", "canonical_only"),
    ("posterior_over_residual", "posterior", "residual_only"),
    ("canonical_over_zero", "canonical_only", "zero_action"),
    ("residual_over_zero", "residual_only", "zero_action"),
    *TRANSFER_COMPARISONS,
)

def saved_or_override(saved: dict, name: str, override):
    return override if override not in (0, "") else saved[name]


def change_weighted_rgb_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    target = target.float() / 255.0
    error = torch.sqrt(
        (prediction.float() - target).square() + 1e-6
    ).mean(dim=2)
    weight = valid.to(error.dtype) * (target - current).abs().mean(dim=2)
    return (error * weight).flatten(2).sum(dim=-1) / weight.flatten(2).sum(
        dim=-1
    ).clamp_min(1e-6)

def change_weighted_feature_mse(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_features"].float()
    current = batch["history_features"][:, -1:].float()
    error = (prediction.float() - target).square().mean(dim=-1)
    change = (target - current).square().mean(dim=-1).sqrt()
    weight = change * batch["future_valid"].to(change.dtype)
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)

def clustered_comparisons(
    variants: dict[str, torch.Tensor],
    clusters: torch.Tensor,
) -> dict[str, dict]:
    return {
        name: clustered_paired_comparison(
            variants[prediction],
            variants[reference],
            clusters,
        )
        for name, prediction, reference in COMPONENT_COMPARISONS
    }

@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    task_layout,
) -> dict:
    values = {
        metric: {variant: [] for variant in VARIANTS}
        for metric in METRICS
    }
    stratified = {
        metric: {variant: [] for variant in VARIANTS}
        for metric in STRATIFIED_METRICS
    }
    query_times = []
    sequence_clusters = []
    component_slot_rms = {
        variant: [] for variant in ("canonical_only", "residual_only")
    }
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    action_bank = []
    bank_clusters = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        object_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=object_mask)
        action_bank.append(output["posterior_actions"].float().cpu())
        bank_clusters.append(batch["sequence_index"].long().cpu())
    action_bank = torch.cat(action_bank)
    bank_clusters = torch.cat(bank_clusters)
    if len(action_bank) < 2:
        raise ValueError("component ablation requires two actions to shuffle")
    shuffle_index = torch.arange(len(action_bank)).roll(len(action_bank) // 2)
    matched_index, transfer_diagnostics = nearest_effect_donors(
        action_bank,
        bank_clusters,
        shuffle_index,
        model.config.canonical_action_dim,
    )
    sample_offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        shuffled_actions = action_bank[
            shuffle_index[sample_offset : sample_offset + batch_size]
        ].to(device, non_blocking=True)
        matched_actions = action_bank[
            matched_index[sample_offset : sample_offset + batch_size]
        ].to(device, non_blocking=True)
        object_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=object_mask)
            predictions = {
                "posterior": (
                    output["predicted_future_slots"],
                    output["rendered_future_features"],
                    output["rendered_future_rgb"],
                )
            }
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
            predictions["shuffled_action"] = (
                shuffled_slots,
                shuffled_feature,
                shuffled_rgb,
            )
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
            predictions["zero_action"] = (
                output["zero_action_future_slots"],
                zero_feature,
                zero_rgb,
            )
            future_count = batch["future_features"].shape[1]
            predictions["copy"] = (
                output["online_history_slots"][:, -1:].expand(
                    -1, future_count, -1, -1
                ),
                batch["history_features"][:, -1:].expand(
                    -1, future_count, -1, -1
                ),
                batch["history_rgb"][:, -1:].expand(
                    -1, future_count, -1, -1, -1
                ).float()
                / 255.0,
            )
            action_variants = component_action_variants(
                output["posterior_actions"],
                model.config.canonical_action_dim,
            )
            action_variants.update(
                transfer_action_variants(
                    output["posterior_actions"],
                    matched_actions,
                    shuffled_actions,
                    model.config.canonical_action_dim,
                )
            )
            for name, actions in action_variants.items():
                slots, centers = predict_shuffled_action(
                    model,
                    batch,
                    output,
                    actions,
                )
                feature, rgb = render_state(
                    model,
                    batch,
                    output,
                    slots,
                    centers,
                )
                predictions[name] = (slots, feature, rgb)
                if name in component_slot_rms:
                    component_slot_rms[name].append(
                        (
                            output["predicted_future_slots"].float()
                            - slots.float()
                        )
                        .square()
                        .mean(dim=(1, 2, 3))
                        .sqrt()
                        .cpu()
                    )
        for name, (slots, feature, rgb) in predictions.items():
            if rgb is None:
                raise ValueError("action component ablation requires RGB")
            values["feature_mse"][name].append(
                masked_feature_mse(
                    feature.float(),
                    batch["future_features"].float(),
                    batch["future_valid"],
                ).cpu()
            )
            values["latent_mse"][name].append(
                weighted_latent_mse(
                    slots.float(),
                    output["target_future_slots"].float(),
                    output["target_future_activity"],
                ).cpu()
            )
            values["rgb_distance"][name].append(
                rgb_distance(
                    rgb.float(),
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
            current_rgb = batch["history_rgb"][:, -1:].float() / 255.0
            stratified["change_weighted_feature_mse"][name].append(
                change_weighted_feature_mse(feature, batch).cpu()
            )
            stratified["change_weighted_rgb_charbonnier"][name].append(
                change_weighted_rgb_charbonnier(
                    rgb,
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    current_rgb,
                ).cpu()
            )
        query_times.append(batch["future_times"].float().cpu())
        sequence_clusters.append(batch["sequence_index"].long().cpu())
        sample_offset += batch_size
    if sample_offset != len(action_bank):
        raise RuntimeError("action bank and component loader differ")
    tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in variants.items()
        }
        for metric, variants in values.items()
    }
    stratified_tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in variants.items()
        }
        for metric, variants in stratified.items()
    }
    query_times = torch.cat(query_times)
    sequence_clusters = torch.cat(sequence_clusters)
    if not torch.equal(sequence_clusters, bank_clusters):
        raise RuntimeError("action bank and evaluation episode order differ")
    comparisons = {
        metric: clustered_comparisons(variants, sequence_clusters)
        for metric, variants in tensors.items()
    }
    change_comparisons = {
        metric: clustered_comparisons(
            {name: value.mean(dim=1) for name, value in variants.items()},
            sequence_clusters,
        )
        for metric, variants in stratified_tensors.items()
        if metric.startswith("change_weighted_")
    }
    return {
        "samples": len(next(iter(tensors["feature_mse"].values()))),
        "clusters": len(torch.unique(sequence_clusters)),
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in variants.items()
            }
            for metric, variants in tensors.items()
        },
        "comparison": comparisons,
        "change_weighted_comparison": change_comparisons,
        "task_group_evidence": task_group_metric_evidence(
            stratified_tensors, sequence_clusters, task_layout,
            COMPONENT_COMPARISONS),
        "posterior_component_slot_rms": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in component_slot_rms.items()
        },
        "transfer_diagnostics": transfer_diagnostics,
        "by_future_query": {
            str(index): {
                "time_seconds": {
                    "min": float(query_times[:, index].min()),
                    "mean": float(query_times[:, index].mean()),
                    "max": float(query_times[:, index].max()),
                },
                "mean": {
                    metric: {
                        name: float(value[:, index].mean())
                        for name, value in variants.items()
                    }
                    for metric, variants in stratified_tensors.items()
                },
                "comparison": {
                    metric: clustered_comparisons(
                        {
                            name: value[:, index]
                            for name, value in variants.items()
                        },
                        sequence_clusters,
                    )
                    for metric, variants in stratified_tensors.items()
                    if metric.startswith("change_weighted_")
                },
            }
            for index in range(query_times.shape[1])
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--history_frames", type=int, default=0)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--sequence_anchors", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items < args.batch:
        raise ValueError("component ablation requires at least two samples")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_step = int(checkpoint.get("global_step", -1))
    if checkpoint_step < 1 or checkpoint.get("phase") != "joint":
        raise ValueError("component ablation requires a joint checkpoint")
    saved = checkpoint.get("args", {})
    if saved.get("data_format") != "sequence":
        raise ValueError("checkpoint was not trained with visual sequences")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("component ablation requires RGB and no language")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved_or_override(
            saved,
            "history_frames",
            args.history_frames,
        ),
        future_frames=saved_or_override(
            saved,
            "future_frames",
            args.future_frames,
        ),
        anchors=saved_or_override(
            saved,
            "sequence_anchors",
            args.sequence_anchors,
        ),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, True)
    task_layout = selected_task_group_layout(dataset, args.data, args.split)
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
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": checkpoint_step,
        "checkpoint_phase": checkpoint["phase"],
        "data": os.path.abspath(args.data),
        "data_sha256": dataset.data_sha256,
        "split": args.split,
        "amp": args.amp,
        "requested_max_items": args.max_items,
        "available_samples": _available_samples(dataset),
        "anchors": list(dataset.anchors),
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "action_dim": config.action_dim,
        "canonical_action_dim": config.canonical_action_dim,
        "action_residual_dim": config.action_residual_dim,
        "evaluation": evaluate(model, loader, device, args.amp, task_layout),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
