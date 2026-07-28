#!/usr/bin/env python3
"""Held A/B promotion gate for the v29 hierarchical Gaussian carrier."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.gaussian_math import (  # noqa: E402
    mahalanobis_squared_from_precision,
    precision_2d,
)
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_repair import first_query  # noqa: E402
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
    parser.add_argument("--seed", type=int, default=2901)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.architecture != ARCHITECTURE:
        raise ValueError(f"checkpoint is not {ARCHITECTURE}: {path}")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model.eval()


def verify_candidate(
    checkpoint: dict, args: argparse.Namespace, current_commit: str
) -> dict:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("candidate is not a v29 checkpoint")
    if checkpoint.get("git_commit") != current_commit:
        raise ValueError("candidate checkpoint git commit differs")
    saved = checkpoint.get("args", {})
    expected = {
        "training_stage": "readout",
        "readout_scope": args.evaluation_mode,
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate training stage differs from evaluation mode")
    if checkpoint.get("phase") != "readout" or int(
        checkpoint.get("phase_step", -1)
    ) != int(saved.get("representation_steps", -2)):
        raise ValueError("candidate readout stage is incomplete")
    initialization = saved.get("init_from", "")
    if not os.path.isfile(initialization):
        raise ValueError("candidate initialization checkpoint is unavailable")
    if file_sha256(initialization) != file_sha256(args.baseline):
        raise ValueError("candidate does not descend directly from baseline")
    return {"initialization_checkpoint": os.path.abspath(initialization)}


def sample_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float()
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def sample_object_error(output: dict) -> torch.Tensor:
    prediction = F.normalize(
        output["predicted_future_object_features"].float(), dim=-1
    )
    target = F.normalize(output["target_future_object_features"].float(), dim=-1)
    error = (prediction - target).square().mean(dim=-1)
    weight = output["target_future_activity"].float()
    if weight.ndim == error.ndim + 1:
        weight = weight.squeeze(-1)
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def intervention_state(model, batch: dict, output: dict, mode: str):
    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    if mode == "shuffle":
        order = torch.arange(
            slots.slots.shape[1] - 1, -1, -1, device=slots.slots.device
        )
        predicted_slots = slots.slots[:, order]
    elif mode == "zero":
        predicted_slots = torch.zeros_like(slots.slots)
    else:
        raise ValueError(f"unknown intervention: {mode}")
    state, _ = decode_gaussian_readout(
        model,
        batch,
        tokens,
        slots,
        predicted_slots[:, None],
        slots.center[:, None],
        predicted_relative_scale=slots.relative_scale[:, None],
        predicted_relative_disparity=slots.relative_disparity[:, None],
    )
    return state


def mixture_health(state, coordinates: torch.Tensor, valid: torch.Tensor) -> dict:
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
    return {
        "coverage_fraction": ((coverage > 1e-4) & valid).float().flatten(1).sum(dim=1)
        / denominator,
        "effective_components": (effective * valid_float).flatten(1).sum(dim=1)
        / denominator,
        "covariance_min_eigenvalue": eigenvalues[..., 0].flatten(1).mean(dim=1),
        "covariance_condition": (
            eigenvalues[..., 1] / eigenvalues[..., 0]
        ).flatten(1).mean(dim=1),
        "opacity_saturation_fraction": (
            (state.opacity < 0.01) | (state.opacity > 0.99)
        ).float().flatten(1).mean(dim=1),
        "component_count": state.center.new_full(
            (state.center.shape[0],), float(state.center.shape[2])
        ),
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
        flow=0.0, action=0.0, action_specificity=0.0, rgb=0.0
    )
    output = model(
        batch,
        history_mask=history_mask,
        phase="object_memory_representation_loss",
        loss_weights=weights,
    )
    current_target = batch["history_features"][:, -1:]
    current_valid = batch["history_valid"][:, -1:]
    coordinates = batch["history_coordinates"][:, -1:]
    conditioned = first_query(output["current_gaussian_readout"])
    current, _ = model.gaussian_readout.splat_features(conditioned, coordinates)
    tokens = output["history_token_states"][-1].reconstructed_features[:, None]
    valid_float = batch["history_valid"][:, -1].float()
    current_frame = batch["history_features"][:, -1].float()
    scene_mean = (current_frame * valid_float[..., None]).sum(dim=1)
    scene_mean = scene_mean / valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    scene = scene_mean[:, None, None].expand_as(current_target)
    shuffled_state = intervention_state(model, batch, output, "shuffle")
    zero_state = intervention_state(model, batch, output, "zero")
    shuffled, _ = model.gaussian_readout.splat_features(shuffled_state, coordinates)
    zero, _ = model.gaussian_readout.splat_features(zero_state, coordinates)
    persistence = current_target.expand_as(batch["future_features"])
    result = {
        "current": sample_error(current, current_target, current_valid),
        "token": sample_error(tokens, current_target, current_valid),
        "scene": sample_error(scene, current_target, current_valid),
        "shuffled": sample_error(shuffled, current_target, current_valid),
        "zero": sample_error(zero, current_target, current_valid),
        "future": sample_error(
            output["rendered_future_features"],
            batch["future_features"],
            batch["future_valid"],
        ),
        "persistence": sample_error(
            persistence, batch["future_features"], batch["future_valid"]
        ),
        "object": sample_object_error(output),
    }
    for horizon in range(batch["future_features"].shape[1]):
        result[f"future_h{horizon}"] = sample_error(
            output["rendered_future_features"][:, horizon : horizon + 1],
            batch["future_features"][:, horizon : horizon + 1],
            batch["future_valid"][:, horizon : horizon + 1],
        )
    result.update(mixture_health(conditioned, coordinates, current_valid))
    return result


def collect(model, loader, device: torch.device) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {"sequence_index": []}
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with torch.no_grad(), amp:
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            rows["sequence_index"].append(batch["sequence_index"].cpu())
            for name, value in evaluate_batch(model, batch).items():
                rows.setdefault(name, []).append(value.detach().float().cpu())
    return {name: torch.cat(parts) for name, parts in rows.items()}


def cluster_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    clusters: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> dict[str, float]:
    unique = torch.unique(clusters, sorted=True)
    numerator_sum = torch.stack([numerator[clusters == key].sum() for key in unique])
    denominator_sum = torch.stack(
        [denominator[clusters == key].sum() for key in unique]
    )
    estimate = float(numerator_sum.sum() / denominator_sum.sum().clamp_min(1e-12))
    draws = torch.randint(len(unique), (samples, len(unique)), generator=generator)
    values = numerator_sum[draws].sum(dim=1) / denominator_sum[draws].sum(
        dim=1
    ).clamp_min(1e-12)
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return {
        "estimate": estimate,
        "ci95_low": float(bounds[0]),
        "ci95_high": float(bounds[1]),
    }


def compare(baseline: dict, candidate: dict, args: argparse.Namespace):
    clusters = candidate["sequence_index"]
    if not torch.equal(clusters, baseline["sequence_index"]):
        raise ValueError("baseline and candidate held rows differ")
    generator = torch.Generator().manual_seed(args.seed)

    def ratio(numerator, denominator):
        return cluster_ratio(
            numerator,
            denominator,
            clusters,
            args.bootstrap_samples,
            generator,
        )

    metrics = {
        "current_improvement": ratio(
            baseline["current"] - candidate["current"], baseline["current"]
        ),
        "token_gap_recovery": ratio(
            (baseline["current"] - baseline["token"])
            - (candidate["current"] - candidate["token"]),
            (baseline["current"] - baseline["token"]).abs(),
        ),
        "scene_mean_gain": ratio(
            candidate["scene"] - candidate["current"], candidate["scene"]
        ),
        "current_degradation": ratio(
            candidate["current"] - baseline["current"], baseline["current"]
        ),
        "future_improvement": ratio(
            baseline["future"] - candidate["future"], baseline["future"]
        ),
        "future_gain_over_persistence": ratio(
            candidate["persistence"] - candidate["future"],
            candidate["persistence"],
        ),
        "shuffle_margin": ratio(
            candidate["shuffled"] - candidate["current"], candidate["current"]
        ),
        "zero_margin": ratio(
            candidate["zero"] - candidate["current"], candidate["current"]
        ),
        "object_degradation": ratio(
            candidate["object"] - baseline["object"], baseline["object"]
        ),
    }
    horizons = sorted(name for name in candidate if name.startswith("future_h"))
    metrics["horizon_improvements"] = {
        name: ratio(baseline[name] - candidate[name], baseline[name])
        for name in horizons
    }
    metrics["candidate_means"] = {
        name: float(candidate[name].mean())
        for name in (
            "current",
            "token",
            "scene",
            "future",
            "persistence",
            "object",
            "coverage_fraction",
            "effective_components",
            "covariance_min_eigenvalue",
            "covariance_condition",
            "opacity_saturation_fraction",
            "component_count",
        )
    }
    checks = {
        "covariance_is_finite": (
            metrics["candidate_means"]["covariance_min_eigenvalue"] >= 1e-5
            and metrics["candidate_means"]["covariance_condition"] <= 1e4
        ),
        "opacity_not_saturated": (
            metrics["candidate_means"]["opacity_saturation_fraction"] <= 0.25
        ),
        "multiple_components_contribute": (
            metrics["candidate_means"]["effective_components"] >= 2.0
        ),
        "object_degradation_at_most_2pct": (
            metrics["object_degradation"]["ci95_high"] <= 0.02
        ),
    }
    if args.evaluation_mode == "isolated":
        checks.update(
            current_loss_reduced_10pct=(
                metrics["current_improvement"]["estimate"] >= 0.10
                and metrics["current_improvement"]["ci95_low"] > 0.0
            ),
            token_gap_recovered_70pct=(
                metrics["token_gap_recovery"]["estimate"] >= 0.70
            ),
            scene_mean_gain_20pct=(
                metrics["scene_mean_gain"]["estimate"] >= 0.20
                and metrics["scene_mean_gain"]["ci95_low"] > 0.0
            ),
        )
    else:
        checks.update(
            current_degradation_at_most_2pct=(
                metrics["current_degradation"]["ci95_high"] <= 0.02
            ),
            future_improves_2pct=(
                metrics["future_improvement"]["estimate"] >= 0.02
                and metrics["future_improvement"]["ci95_low"] > 0.0
            ),
            future_beats_persistence_2pct=(
                metrics["future_gain_over_persistence"]["estimate"] >= 0.02
            ),
            shuffle_worsens_2pct=metrics["shuffle_margin"]["estimate"] >= 0.02,
            zero_worsens_2pct=metrics["zero_margin"]["estimate"] >= 0.02,
            no_horizon_degrades_2pct=all(
                value["ci95_low"] >= -0.02
                for value in metrics["horizon_improvements"].values()
            ),
        )
    return metrics, checks


def main() -> None:
    args = parse_args()
    if min(args.max_items, args.batch, args.bootstrap_samples) <= 0:
        raise ValueError("held evaluation sizes must be positive")
    if not torch.cuda.is_available():
        raise ValueError("held carrier evaluation requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    if status.strip():
        raise ValueError("held carrier evaluation requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
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
    device = torch.device("cuda:0")
    _, baseline_model = load_model(args.baseline, device)
    baseline = collect(baseline_model, loader, device)
    del baseline_model
    torch.cuda.empty_cache()
    candidate_checkpoint, candidate_model = load_model(args.candidate, device)
    lineage = verify_candidate(candidate_checkpoint, args, commit)
    candidate = collect(candidate_model, loader, device)
    metrics, checks = compare(baseline, candidate, args)
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": READOUT_GATE_CONTRACT,
        "evaluation_mode": args.evaluation_mode,
        "git_commit": commit,
        "data_manifest_sha256": dataset.data_sha256,
        "held_split": args.split,
        "held_items": len(dataset),
        "baseline_checkpoint": os.path.abspath(args.baseline),
        "baseline_checkpoint_sha256": file_sha256(args.baseline),
        "candidate_checkpoint": os.path.abspath(args.candidate),
        "candidate_checkpoint_sha256": file_sha256(args.candidate),
        "metrics": metrics,
        "checks": checks,
        **lineage,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise RuntimeError("hierarchical Gaussian carrier held gate failed")


if __name__ == "__main__":
    main()
