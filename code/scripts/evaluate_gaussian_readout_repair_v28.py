#!/usr/bin/env python3
"""Held A/B gate for the causal Gaussian feature-readout repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.carrier_contracts import (  # noqa: E402
    READOUT_GATE_CONTRACT,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.decoder import GaussianReadoutState  # noqa: E402
from igsw.adaptive_gaussian_wm.gaussian_math import (  # noqa: E402
    mahalanobis_squared_from_precision,
    precision_2d,
)
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_diagnostics import (  # noqa: E402
    teacher_future_readout,
)
from igsw.adaptive_gaussian_wm.readout_repair import (  # noqa: E402
    direct_current_state,
    first_query,
)
from igsw.adaptive_gaussian_wm.readout_runtime import (  # noqa: E402
    decode_gaussian_readout,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v28_training import ARCHITECTURE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation_mode", choices=("isolated", "joint"), required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_model(path: str, device: torch.device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.architecture != ARCHITECTURE:
        raise ValueError(f"checkpoint is not {ARCHITECTURE}: {path}")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model


def verify_lineage(saved: dict, mode: str, baseline: str) -> dict:
    initialization = saved.get("init_from", "")
    if not os.path.isfile(initialization):
        raise ValueError("candidate initialization checkpoint is unavailable")
    if mode == "isolated":
        parent = initialization
    else:
        isolated = torch.load(
            initialization, map_location="cpu", weights_only=False, mmap=True
        )
        isolated_args = isolated.get("args", {})
        if (
            isolated.get("phase") != "readout"
            or isolated_args.get("training_stage") != "readout"
            or isolated_args.get("readout_scope") != "isolated"
            or int(isolated.get("phase_step", -1))
            != int(isolated_args.get("representation_steps", -2))
        ):
            raise ValueError("joint candidate does not descend from isolation")
        parent = isolated_args.get("init_from", "")
        if not os.path.isfile(parent):
            raise ValueError("isolated parent checkpoint is unavailable")
    if file_sha256(parent) != file_sha256(baseline):
        raise ValueError("candidate and baseline do not share a parent")
    return {"initialization_checkpoint": os.path.abspath(initialization)}


def per_sample_feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float() * (coverage.float() > 1e-4)
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def per_sample_object_error(output: dict) -> torch.Tensor:
    prediction = F.normalize(
        output["predicted_future_object_features"].float(), dim=-1
    )
    target = F.normalize(
        output["target_future_object_features"].float(), dim=-1
    )
    error = (prediction - target).square().mean(dim=-1)
    weight = output["target_future_activity"].float()
    if weight.ndim == error.ndim + 1:
        weight = weight.squeeze(-1)
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def current_intervention_state(
    model, batch: dict, output: dict, mode: str
) -> GaussianReadoutState:
    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    if mode == "shuffle":
        order = torch.arange(slots.slots.shape[1] - 1, -1, -1, device=slots.slots.device)
        predicted_slots = slots.slots[:, order]
        predicted_centers = slots.center[:, order]
        predicted_scale = slots.relative_scale[:, order]
        predicted_disparity = slots.relative_disparity[:, order]
    elif mode == "zero":
        predicted_slots = torch.zeros_like(slots.slots)
        predicted_centers = slots.center
        predicted_scale = slots.relative_scale
        predicted_disparity = slots.relative_disparity
    else:
        raise ValueError(f"unknown readout intervention: {mode}")
    state, _ = decode_gaussian_readout(
        model,
        batch,
        tokens,
        slots,
        predicted_slots[:, None],
        predicted_centers[:, None],
        predicted_relative_scale=predicted_scale[:, None],
        predicted_relative_disparity=predicted_disparity[:, None],
    )
    return state
def mixture_health(
    state: GaussianReadoutState,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = first_query(state)
    covariance = state.covariance.float()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-8)
    difference = coordinates[:, :, None].float() - state.center[..., None, :].float()
    distance = mahalanobis_squared_from_precision(
        precision_2d(covariance), difference
    )
    weight = torch.exp(-0.5 * distance)
    weight = weight * state.opacity.squeeze(-1)[..., None].float()
    weight = weight * state.activation.squeeze(-1)[..., None].float()
    order = torch.softmax(state.depth_order.squeeze(-1).float(), dim=2)
    weight = weight * order[..., None] * state.center.shape[2]
    coverage = weight.sum(dim=2)
    mixture = weight / coverage[:, :, None].clamp_min(1e-6)
    effective = mixture.square().sum(dim=2).clamp_min(1e-8).reciprocal()
    valid_float = valid.float()
    denominator = valid_float.flatten(1).sum(dim=1).clamp_min(1.0)
    opacity_saturated = (
        (state.opacity < 0.01) | (state.opacity > 0.99)
    ).float().flatten(1).mean(dim=1)
    return {
        "coverage": ((coverage > 1e-4) & valid).float().flatten(1).sum(dim=1)
        / denominator,
        "effective_components": (effective * valid_float).flatten(1).sum(dim=1)
        / denominator,
        "covariance_min_eigenvalue": eigenvalues[..., 0].flatten(1).mean(dim=1),
        "covariance_condition": (
            eigenvalues[..., 1] / eigenvalues[..., 0]
        ).flatten(1).mean(dim=1),
        "opacity_saturation_fraction": opacity_saturated,
        "active_token_fraction": (state.activation > 0.5).float().flatten(1).mean(dim=1),
    }


def evaluate_batch(model, batch: dict) -> dict[str, torch.Tensor]:
    history_mask = torch.zeros(
        batch["history_features"].shape[0],
        batch["history_features"].shape[1],
        model.config.object_slots,
        dtype=torch.bool,
        device=batch["history_features"].device,
    )
    weights = AdaptiveGaussianLossWeights(
        flow=0.0,
        action=0.0,
        action_specificity=0.0,
        rgb=0.0,
    )
    output = model(
        batch,
        history_mask=history_mask,
        phase="object_memory_representation_loss",
        loss_weights=weights,
    )
    current_target = batch["history_features"][:, -1:]
    current_valid = batch["history_valid"][:, -1:]
    current_coordinates = batch["history_coordinates"][:, -1:]
    conditioned = first_query(output["current_gaussian_readout"])
    current, current_coverage = model.gaussian_readout.splat_features(
        conditioned, current_coordinates
    )
    direct, direct_coverage = model.gaussian_readout.splat_features(
        direct_current_state(output["history_token_states"][-1]),
        current_coordinates,
    )
    tokens = output["history_token_states"][-1].reconstructed_features[:, None]
    valid_coverage = torch.ones_like(current_valid, dtype=current.dtype)
    scene = current_target.mean(dim=2, keepdim=True).expand_as(current_target)
    shuffled_state = current_intervention_state(model, batch, output, "shuffle")
    zero_state = current_intervention_state(model, batch, output, "zero")
    shuffled, shuffled_coverage = model.gaussian_readout.splat_features(
        shuffled_state, current_coordinates
    )
    zero, zero_coverage = model.gaussian_readout.splat_features(
        zero_state, current_coordinates
    )
    teacher, teacher_coverage = teacher_future_readout(model, batch, output)
    persistence = current_target.expand_as(batch["future_features"])
    result = {
        "current": per_sample_feature_error(
            current, current_target, current_valid, current_coverage
        ),
        "direct": per_sample_feature_error(
            direct, current_target, current_valid, direct_coverage
        ),
        "token": per_sample_feature_error(
            tokens, current_target, current_valid, valid_coverage
        ),
        "scene": per_sample_feature_error(
            scene, current_target, current_valid, valid_coverage
        ),
        "shuffled": per_sample_feature_error(
            shuffled, current_target, current_valid, shuffled_coverage
        ),
        "zero": per_sample_feature_error(
            zero, current_target, current_valid, zero_coverage
        ),
        "future": per_sample_feature_error(
            output["rendered_future_features"],
            batch["future_features"],
            batch["future_valid"],
            output["render_coverage"],
        ),
        "teacher_future": per_sample_feature_error(
            teacher,
            batch["future_features"],
            batch["future_valid"],
            teacher_coverage,
        ),
        "persistence": per_sample_feature_error(
            persistence,
            batch["future_features"],
            batch["future_valid"],
            torch.ones_like(batch["future_valid"], dtype=persistence.dtype),
        ),
        "object": per_sample_object_error(output),
    }
    for horizon in range(batch["future_features"].shape[1]):
        result[f"future_h{horizon}"] = per_sample_feature_error(
            output["rendered_future_features"][:, horizon : horizon + 1],
            batch["future_features"][:, horizon : horizon + 1],
            batch["future_valid"][:, horizon : horizon + 1],
            output["render_coverage"][:, horizon : horizon + 1],
        )
    result.update(mixture_health(conditioned, current_coordinates, current_valid))
    return result


def collect_metrics(model, loader, device: torch.device) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {"sequence_index": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            values = evaluate_batch(model, batch)
            rows["sequence_index"].append(batch["sequence_index"].detach().cpu())
            for name, value in values.items():
                rows.setdefault(name, []).append(value.detach().float().cpu())
    return {name: torch.cat(parts) for name, parts in rows.items()}


def ratio_with_cluster_ci(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    unique = torch.unique(clusters, sorted=True)
    numerator_sum = torch.stack([numerator[clusters == value].sum() for value in unique])
    denominator_sum = torch.stack(
        [denominator[clusters == value].sum() for value in unique]
    )
    estimate = float(numerator_sum.sum() / denominator_sum.sum().clamp_min(1e-12))
    draws = torch.randint(len(unique), (samples, len(unique)), generator=generator)
    values = numerator_sum[draws].sum(dim=1) / denominator_sum[draws].sum(
        dim=1
    ).clamp_min(1e-12)
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return {"estimate": estimate, "ci95_low": float(bounds[0]), "ci95_high": float(bounds[1])}


def compare(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    mode: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict[str, bool]]:
    clusters = candidate["sequence_index"]
    if not torch.equal(clusters, baseline["sequence_index"]):
        raise ValueError("baseline and candidate sample orders differ")
    generator = torch.Generator().manual_seed(seed)

    def ratio(numerator, denominator):
        return ratio_with_cluster_ci(
            numerator, denominator, clusters, bootstrap_samples, generator
        )

    metrics = {
        "current_improvement": ratio(baseline["current"] - candidate["current"], baseline["current"]),
        "gap_reduction": ratio(
            (baseline["current"] - baseline["token"])
            - (candidate["current"] - candidate["token"]),
            (baseline["current"] - baseline["token"]).abs(),
        ),
        "scene_mean_gain": ratio(candidate["scene"] - candidate["current"], candidate["scene"]),
        "shuffle_margin": ratio(candidate["shuffled"] - candidate["current"], candidate["current"]),
        "zero_margin": ratio(candidate["zero"] - candidate["current"], candidate["current"]),
        "future_improvement": ratio(baseline["future"] - candidate["future"], baseline["future"]),
        "future_gain_over_persistence": ratio(candidate["persistence"] - candidate["future"], candidate["persistence"]),
        "object_degradation": ratio(candidate["object"] - baseline["object"], baseline["object"]),
        "teacher_advantage": ratio(candidate["future"] - candidate["teacher_future"], candidate["future"]),
    }
    horizon_names = sorted(name for name in candidate if name.startswith("future_h"))
    metrics["horizon_improvements"] = {
        name: ratio(baseline[name] - candidate[name], baseline[name])
        for name in horizon_names
    }
    metrics["candidate_means"] = {
        name: float(candidate[name].mean())
        for name in (
            "current",
            "direct",
            "token",
            "scene",
            "future",
            "teacher_future",
            "persistence",
            "object",
            "coverage",
            "effective_components",
            "covariance_min_eigenvalue",
            "covariance_condition",
            "opacity_saturation_fraction",
            "active_token_fraction",
        )
    }
    checks = {
        "current_loss_reduced_10pct": metrics["current_improvement"]["estimate"] >= 0.10
        and metrics["current_improvement"]["ci95_low"] > 0.0,
        "token_gap_reduced_20pct": metrics["gap_reduction"]["estimate"] >= 0.20,
        "scene_mean_gain_20pct": metrics["scene_mean_gain"]["estimate"] >= 0.20
        and metrics["scene_mean_gain"]["ci95_low"] > 0.0,
        "coverage_99pct": metrics["candidate_means"]["coverage"] >= 0.99,
        "shuffle_worsens_2pct": metrics["shuffle_margin"]["estimate"] >= 0.02,
        "zero_worsens_2pct": metrics["zero_margin"]["estimate"] >= 0.02,
        "object_degradation_at_most_2pct": metrics["object_degradation"]["ci95_high"] <= 0.02,
        "covariance_is_finite": metrics["candidate_means"]["covariance_min_eigenvalue"] >= 1e-5
        and metrics["candidate_means"]["covariance_condition"] <= 1e4,
        "opacity_not_saturated": metrics["candidate_means"]["opacity_saturation_fraction"] <= 0.25,
        "active_fraction_is_valid": 0.25 <= metrics["candidate_means"]["active_token_fraction"] <= 0.90,
        "multiple_components_contribute": metrics["candidate_means"]["effective_components"] >= 2.0,
    }
    if mode == "isolated":
        checks["future_degradation_at_most_2pct"] = metrics["future_improvement"]["ci95_low"] >= -0.02
    else:
        checks["future_improves_2pct"] = metrics["future_improvement"]["estimate"] >= 0.02
        checks["future_beats_persistence_2pct"] = metrics["future_gain_over_persistence"]["estimate"] >= 0.02
        checks["teacher_not_worse_than_model"] = metrics["teacher_advantage"]["estimate"] >= -0.02
        checks["no_horizon_degrades_2pct"] = all(
            value["ci95_low"] >= -0.02
            for value in metrics["horizon_improvements"].values()
        )
    return metrics, checks


def main() -> None:
    args = parse_args()
    if min(args.max_items, args.batch, args.bootstrap_samples) <= 0:
        raise ValueError("evaluation sizes must be positive")
    if not torch.cuda.is_available():
        raise ValueError("held readout evaluation requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    if status.strip():
        raise ValueError("held readout evaluation requires a clean worktree")
    device = torch.device("cuda:0")
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    baseline_checkpoint, baseline_model = load_checkpoint_model(args.baseline, device)
    baseline_phase_step = int(baseline_checkpoint.get("phase_step", -1))
    baseline = collect_metrics(baseline_model, loader, device)
    del baseline_model, baseline_checkpoint
    torch.cuda.empty_cache()
    candidate_checkpoint, candidate_model = load_checkpoint_model(args.candidate, device)
    saved = candidate_checkpoint.get("args", {})
    if saved.get("training_stage") != "readout" or saved.get("readout_scope") != args.evaluation_mode:
        raise ValueError("candidate checkpoint does not match evaluation mode")
    if candidate_checkpoint.get("phase") != "readout" or int(
        candidate_checkpoint.get("phase_step", -1)
    ) != int(saved.get("representation_steps", -2)):
        raise ValueError("candidate readout stage is incomplete")
    lineage = verify_lineage(saved, args.evaluation_mode, args.baseline)
    candidate = collect_metrics(candidate_model, loader, device)
    metrics, checks = compare(
        baseline,
        candidate,
        args.evaluation_mode,
        args.bootstrap_samples,
        args.seed,
    )
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": READOUT_GATE_CONTRACT,
        "evaluation_mode": args.evaluation_mode,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "data_manifest_sha256": dataset.data_sha256,
        "split": args.split,
        "samples": len(dataset),
        "episode_clusters": int(torch.unique(candidate["sequence_index"]).numel()),
        "bootstrap_samples": args.bootstrap_samples,
        "baseline_checkpoint": os.path.abspath(args.baseline),
        "baseline_checkpoint_sha256": file_sha256(args.baseline),
        "candidate_checkpoint": os.path.abspath(args.candidate),
        "candidate_checkpoint_sha256": file_sha256(args.candidate),
        "baseline_phase_step": baseline_phase_step,
        "candidate_phase_step": int(candidate_checkpoint.get("phase_step", -1)),
        "lineage": lineage,
        "checks": checks,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
