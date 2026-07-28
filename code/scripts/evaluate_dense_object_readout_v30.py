#!/usr/bin/env python3
"""Held-seed promotion gates for isolated and joint v30 dense readout stages."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.dense_readout_contracts import (  # noqa: E402
    DENSE_HELD_CONTRACT,
    DENSE_PREFLIGHT_CONTRACT,
)
from igsw.adaptive_gaussian_wm.feature_readout_runtime import (  # noqa: E402
    decode_current_dense_readout,
)
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_repair import (  # noqa: E402
    direct_current_state,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evaluation_mode", choices=("isolated", "joint"), required=True
    )
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3001)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_feature_error(
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
    prediction = F.normalize(output["predicted_future_object_features"].float(), dim=-1)
    target = F.normalize(output["target_future_object_features"].float(), dim=-1)
    error = (prediction - target).square().mean(dim=-1)
    weight = output["target_future_activity"].float()
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)


def current_metrics(model, batch: dict, history: dict) -> dict[str, torch.Tensor]:
    tokens = history["token_states"][-1]
    dense = decode_current_dense_readout(model, batch, history)
    target = batch["history_features"][:, -1]
    valid = batch["history_valid"][:, -1]
    gaussian, _ = model.gaussian_readout.splat_features(
        direct_current_state(tokens), batch["history_coordinates"][:, -1:]
    )
    scene = (target.float() * valid[..., None].float()).sum(dim=1)
    scene = scene / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    scene = scene[:, None].expand_as(target)
    effective = dense.assignment[:, 0].square().sum(dim=1).clamp_min(1e-8).reciprocal()
    valid_weight = valid.float()
    return {
        "dense": sample_feature_error(dense.feature[:, 0], target, valid),
        "token": sample_feature_error(tokens.reconstructed_features, target, valid),
        "gaussian": sample_feature_error(gaussian[:, 0], target, valid),
        "scene": sample_feature_error(scene, target, valid),
        "effective": (effective * valid_weight).sum(dim=1)
        / valid_weight.sum(dim=1).clamp_min(1.0),
        "coverage": ((dense.coverage[:, 0] > 1e-4) & valid).float().sum(dim=1)
        / valid_weight.sum(dim=1).clamp_min(1.0),
    }


def action_free_output(model, batch: dict) -> dict:
    history_mask = torch.zeros(
        *batch["history_features"].shape[:2],
        model.config.object_slots,
        dtype=torch.bool,
        device=batch["history_features"].device,
    )
    weights = AdaptiveGaussianLossWeights(
        flow=0.0,
        action=0.0,
        action_specificity=0.0,
        rgb=0.0,
        current_readout=0.0,
        readout_regularization=0.0,
    )
    return model(
        batch,
        history_mask=history_mask,
        phase="object_memory_representation_loss",
        loss_weights=weights,
    )


def collect(model, loader, device: torch.device, include_future: bool) -> dict:
    rows: dict[str, list[torch.Tensor]] = {"sequence_index": []}
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with torch.no_grad(), amp:
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            if include_future:
                output = action_free_output(model, batch)
                history = {
                    "token_states": output["history_token_states"],
                    "slot_states": output["history_slot_states"],
                }
            else:
                output = None
                history = model.encode_history(batch)
            values = current_metrics(model, batch, history)
            if output is not None:
                future = batch["future_features"]
                valid = batch["future_valid"]
                persistence = batch["history_features"][:, -1:].expand_as(future)
                values.update(
                    future=sample_feature_error(
                        output["rendered_future_features"], future, valid
                    ),
                    persistence=sample_feature_error(persistence, future, valid),
                    object=sample_object_error(output),
                )
                for horizon in range(future.shape[1]):
                    values[f"future_h{horizon}"] = sample_feature_error(
                        output["rendered_future_features"][:, horizon : horizon + 1],
                        future[:, horizon : horizon + 1],
                        valid[:, horizon : horizon + 1],
                    )
            rows["sequence_index"].append(batch["sequence_index"].detach().cpu())
            for name, value in values.items():
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
    numerator_sum = torch.stack(
        [numerator[clusters == value].sum() for value in unique]
    )
    denominator_sum = torch.stack(
        [denominator[clusters == value].sum() for value in unique]
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


def compare_isolated(candidate: dict, args) -> tuple[dict, dict[str, bool]]:
    generator = torch.Generator().manual_seed(args.seed)
    clusters = candidate["sequence_index"]

    def ratio(numerator, denominator):
        return cluster_ratio(
            numerator, denominator, clusters, args.bootstrap_samples, generator
        )

    metrics = {
        "dense_to_token": ratio(candidate["dense"], candidate["token"]),
        "gaussian_gain": ratio(
            candidate["gaussian"] - candidate["dense"], candidate["gaussian"]
        ),
        "scene_gain": ratio(
            candidate["scene"] - candidate["dense"], candidate["scene"]
        ),
        "candidate_means": {
            name: float(candidate[name].mean())
            for name in ("dense", "token", "gaussian", "scene", "effective", "coverage")
        },
    }
    checks = {
        "all_metrics_finite": all(
            math.isfinite(value) for value in metrics["candidate_means"].values()
        ),
        "within_token_parity": metrics["dense_to_token"]["estimate"] <= 1.10,
        "gaussian_gain_lower_ci_at_least_30pct": (
            metrics["gaussian_gain"]["ci95_low"] >= 0.30
        ),
        "scene_gain_lower_ci_positive": metrics["scene_gain"]["ci95_low"] > 0.0,
        "assignment_effective_tokens": metrics["candidate_means"]["effective"] >= 2.0,
        "coverage_fraction": metrics["candidate_means"]["coverage"] >= 0.90,
    }
    return metrics, checks


def compare_joint(
    baseline: dict, candidate: dict, args
) -> tuple[dict, dict[str, bool]]:
    require(
        torch.equal(baseline["sequence_index"], candidate["sequence_index"]),
        "baseline and candidate held rows differ",
    )
    generator = torch.Generator().manual_seed(args.seed)
    clusters = candidate["sequence_index"]

    def ratio(numerator, denominator):
        return cluster_ratio(
            numerator, denominator, clusters, args.bootstrap_samples, generator
        )

    metrics = {
        "current_degradation": ratio(
            candidate["dense"] - baseline["dense"], baseline["dense"]
        ),
        "future_improvement": ratio(
            baseline["future"] - candidate["future"], baseline["future"]
        ),
        "future_gain_over_persistence": ratio(
            candidate["persistence"] - candidate["future"],
            candidate["persistence"],
        ),
        "object_degradation": ratio(
            candidate["object"] - baseline["object"], baseline["object"]
        ),
        "dense_to_token": ratio(candidate["dense"], candidate["token"]),
        "candidate_means": {
            name: float(candidate[name].mean())
            for name in (
                "dense",
                "token",
                "future",
                "persistence",
                "object",
                "effective",
                "coverage",
            )
        },
    }
    horizons = sorted(name for name in candidate if name.startswith("future_h"))
    metrics["horizon_improvements"] = {
        name: ratio(baseline[name] - candidate[name], baseline[name])
        for name in horizons
    }
    checks = {
        "all_metrics_finite": all(
            math.isfinite(value) for value in metrics["candidate_means"].values()
        ),
        "current_degradation_at_most_5pct": (
            metrics["current_degradation"]["ci95_high"] <= 0.05
        ),
        "future_improves_2pct": (
            metrics["future_improvement"]["estimate"] >= 0.02
            and metrics["future_improvement"]["ci95_low"] > 0.0
        ),
        "future_beats_persistence_2pct": (
            metrics["future_gain_over_persistence"]["estimate"] >= 0.02
            and metrics["future_gain_over_persistence"]["ci95_low"] > 0.0
        ),
        "object_degradation_at_most_2pct": (
            metrics["object_degradation"]["ci95_high"] <= 0.02
        ),
        "no_horizon_degrades_2pct": all(
            value["ci95_low"] >= -0.02
            for value in metrics["horizon_improvements"].values()
        ),
        "within_token_parity": metrics["dense_to_token"]["estimate"] <= 1.10,
        "assignment_effective_tokens": metrics["candidate_means"]["effective"] >= 2.0,
        "coverage_fraction": metrics["candidate_means"]["coverage"] >= 0.90,
    }
    return metrics, checks


def load_model(path: str, device: torch.device) -> tuple[dict, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    require(config.dense_object_readout, "checkpoint has no dense readout")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model.eval()


def verify_complete(checkpoint: dict, commit: str, mode: str) -> dict:
    require(
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
        "candidate is not a v30 checkpoint",
    )
    require(checkpoint.get("git_commit") == commit, "candidate commit differs")
    require(checkpoint.get("phase") == "readout", "candidate phase differs")
    saved = checkpoint.get("args", {})
    require(saved.get("training_stage") == "readout", "candidate stage differs")
    require(saved.get("readout_scope") == mode, "candidate scope differs")
    require(
        int(checkpoint.get("phase_step", -1))
        == int(saved.get("representation_steps", -2)),
        "candidate readout stage is incomplete",
    )
    return saved


def verify_preflight(path: str, commit: str, data_sha256: str) -> dict:
    require(os.path.isfile(path), "dense preflight report is unavailable")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "contract": DENSE_PREFLIGHT_CONTRACT,
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
    }
    require(
        all(report.get(name) == value for name, value in expected.items()),
        "dense preflight report differs",
    )
    return report


def main() -> None:
    args = parse_args()
    for name in ("data", "baseline", "candidate", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(min(args.max_items, args.batch, args.bootstrap_samples) > 0, "bad sizes")
    require(torch.cuda.is_available(), "dense held gate requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "dense held gate requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    baseline_sha256 = file_sha256(args.baseline)
    candidate_sha256 = file_sha256(args.candidate)
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

    candidate_checkpoint, candidate_model = load_model(args.candidate, device)
    candidate_saved = verify_complete(
        candidate_checkpoint, commit, args.evaluation_mode
    )
    initialization = candidate_saved.get("init_from", "")
    require(os.path.isfile(initialization), "candidate initialization is unavailable")
    require(
        file_sha256(initialization) == baseline_sha256,
        "candidate does not descend directly from baseline",
    )
    candidate_preflight_path = candidate_saved.get("dense_preflight_report", "")
    candidate = collect(
        candidate_model, loader, device, args.evaluation_mode == "joint"
    )
    candidate_phase_step = int(candidate_checkpoint["phase_step"])
    del candidate_model, candidate_checkpoint
    torch.cuda.empty_cache()

    baseline = None
    preflight_path = candidate_preflight_path
    if args.evaluation_mode == "joint":
        baseline_checkpoint, baseline_model = load_model(args.baseline, device)
        baseline_saved = verify_complete(baseline_checkpoint, commit, "isolated")
        preflight_path = baseline_saved.get("dense_preflight_report", "")
        baseline = collect(baseline_model, loader, device, True)
        del baseline_model, baseline_checkpoint
        torch.cuda.empty_cache()
    preflight = verify_preflight(preflight_path, commit, dataset.data_sha256)
    expected_preflight_sha256 = (
        candidate_saved.get("dense_preflight_report_sha256", "")
        if args.evaluation_mode == "isolated"
        else baseline_saved.get("dense_preflight_report_sha256", "")
    )
    require(
        file_sha256(preflight_path) == expected_preflight_sha256,
        "dense preflight report changed after launch",
    )
    if args.evaluation_mode == "isolated":
        require(
            preflight.get("source_checkpoint_sha256") == baseline_sha256,
            "isolated preflight used another baseline",
        )
        metrics, checks = compare_isolated(candidate, args)
    else:
        metrics, checks = compare_joint(baseline, candidate, args)

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": DENSE_HELD_CONTRACT,
        "evaluation_mode": args.evaluation_mode,
        "git_commit": commit,
        "checkpoint_version": CHECKPOINT_VERSION,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": dataset.data_sha256,
        "held_split": args.split,
        "held_items": len(dataset),
        "episode_clusters": int(torch.unique(candidate["sequence_index"]).numel()),
        "bootstrap_samples": args.bootstrap_samples,
        "baseline_checkpoint": os.path.abspath(args.baseline),
        "baseline_checkpoint_sha256": baseline_sha256,
        "candidate_checkpoint": os.path.abspath(args.candidate),
        "candidate_checkpoint_sha256": candidate_sha256,
        "candidate_phase_step": candidate_phase_step,
        "dense_preflight_report": os.path.abspath(preflight_path),
        "dense_preflight_report_sha256": file_sha256(preflight_path),
        "checks": checks,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise RuntimeError("dense object readout held gate failed")


if __name__ == "__main__":
    main()
