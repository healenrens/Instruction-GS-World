#!/usr/bin/env python3
"""Held promotion gates for v39 representation, posterior, and history Prior."""

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
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.dynamic_dual_horizon_dataset import (  # noqa: E402
    DynamicDualHorizonEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.jit_dino_runtime import (  # noqa: E402
    JitDinoFeatureRuntime,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    file_sha256,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage", choices=("representation", "posterior", "prior"), required=True
    )
    parser.add_argument("--split", choices=("heldseed", "heldtask"), default="heldseed")
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--max_items", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--jit_dino_batch", type=int, default=16)
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--goal_stability_threshold", type=float, default=0.05)
    parser.add_argument("--prior_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum_shuffle_degradation", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("data", "checkpoint", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(
        not args.teacher_sidecar or os.path.isabs(args.teacher_sidecar),
        "--teacher_sidecar must be absolute",
    )
    require(args.max_items >= args.batch >= 2, "evaluation needs at least one batch")
    require(args.prior_samples > 0, "prior sample count must be positive")
    return args


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Metrics:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add(
        self,
        name: str,
        values: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        values = values.detach().float().reshape(-1)
        weight = torch.ones_like(values) if valid is None else valid.float().reshape(-1)
        self.sums[name] = self.sums.get(name, 0.0) + float((values * weight).sum())
        self.counts[name] = self.counts.get(name, 0.0) + float(weight.sum())

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / max(self.counts[name], 1.0)
            for name in sorted(self.sums)
        }


def feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float()
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def representation_weights() -> AdaptiveGaussianLossWeights:
    return AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=1.0,
        allocator=0.2,
        slot=0.2,
        action=0.0,
        action_specificity=0.0,
        geometry=0.25,
        rgb=0.0,
    )


def load_model(args, dataset, device: torch.device):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    require(checkpoint.get("checkpoint_version") == 39, "checkpoint is not v39")
    require(checkpoint.get("phase") == args.stage, "checkpoint stage differs")
    saved = checkpoint.get("args", {})
    expected_contract = {
        "data": os.path.realpath(args.data),
        "sequence_data_sha256": dataset.data_sha256,
        "temporal_contract": "dynamic_dual_horizon_v1",
        "history_frames_min": 1,
        "history_frames_max": 4,
        "history_span_frames": args.history_span_frames,
        "future_frames": 2,
        "short_horizon_frames": 30,
        "goal_query_seconds": args.goal_query_seconds,
        "goal_tail_guard_frames": args.goal_tail_guard_frames,
        "goal_probe_frames": args.goal_probe_frames,
        "goal_stability_threshold": args.goal_stability_threshold,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
    }
    mismatch = {
        name: {"checkpoint": saved.get(name), "evaluation": value}
        for name, value in expected_contract.items()
        if (os.path.realpath(saved.get(name, "")) if name == "data" else saved.get(name))
        != value
    }
    require(not mismatch, f"held temporal contract differs: {mismatch}")
    expected_steps = (
        saved.get("representation_steps")
        if args.stage == "representation"
        else saved.get("joint_steps")
    )
    require(
        expected_steps is not None
        and int(checkpoint.get("phase_step", -1)) == int(expected_steps),
        "promotion requires a completed stage checkpoint",
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    require(
        config.feature_dim == dataset.feature_dim,
        "checkpoint feature dimension differs",
    )
    require(config.dual_horizon_dynamics, "checkpoint is not dual-horizon")
    model = AdaptiveGaussianObjectWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def build_dataset(args):
    return DynamicDualHorizonEpisodeDataset(
        args.data,
        args.split,
        history_frames_min=1,
        history_frames_max=4,
        history_span_frames=args.history_span_frames,
        short_horizon_frames=30,
        goal_query_seconds=args.goal_query_seconds,
        goal_tail_guard_frames=args.goal_tail_guard_frames,
        goal_probe_frames=args.goal_probe_frames,
        max_items=args.max_items,
        teacher_sidecar=args.teacher_sidecar,
        feature_source="jit",
    )


def batch_indices(length: int, batch: int):
    usable = length // batch * batch
    rows = torch.arange(usable).reshape(batch, usable // batch).transpose(0, 1)
    for row in rows:
        yield row.tolist()


def cross_episode_shuffle(actions: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    for shift in range(1, actions.shape[0]):
        donor = sequence.roll(shift, dims=0)
        if bool((donor != sequence).all()):
            return actions.roll(shift, dims=0)
    raise RuntimeError("posterior shuffle batch has no cross-episode derangement")


def add_horizon_metrics(
    metrics: Metrics,
    prefix: str,
    history_length: int,
    result: dict,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch["future_features"]
    valid = batch["future_valid"]
    current = batch["history_features"][:, -1:].expand_as(target)
    direct = feature_error(result["rendered_future_features"], target, valid)
    persistence = feature_error(current, target, valid)
    metrics.add(f"{prefix}/h{history_length}/short_direct", direct[:, 0])
    metrics.add(f"{prefix}/h{history_length}/short_persistence", persistence[:, 0])
    metrics.add(f"{prefix}/all/short_direct", direct[:, 0])
    metrics.add(f"{prefix}/all/short_persistence", persistence[:, 0])
    goal_valid = result["future_horizon_valid"][:, 1]
    if "rollout_goal_rendered_features" in result:
        rollout = feature_error(
            result["rollout_goal_rendered_features"],
            target[:, 1:2],
            valid[:, 1:2],
        )[:, 0]
        path = feature_error(
            result["rollout_goal_rendered_features"],
            result["rendered_future_features"][:, 1:2],
            valid[:, 1:2],
        )[:, 0]
        for scope in (f"h{history_length}", "all"):
            metrics.add(f"{prefix}/{scope}/goal_direct", direct[:, 1], goal_valid)
            metrics.add(f"{prefix}/{scope}/goal_rollout", rollout, goal_valid)
            metrics.add(
                f"{prefix}/{scope}/goal_persistence", persistence[:, 1], goal_valid
            )
            metrics.add(f"{prefix}/{scope}/goal_path", path, goal_valid)
        return direct[:, 1], rollout, goal_valid
    return direct[:, 1], direct[:, 1], goal_valid


def evaluate(args, model, dataset, runtime, device, amp_context) -> dict:
    metrics = Metrics()
    for history_length in range(1, 5):
        for indices in batch_indices(len(dataset), args.batch):
            raw = default_collate([dataset[(index, history_length)] for index in indices])
            raw = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in raw.items()
            }
            batch = runtime(raw)
            history_mask = torch.zeros(
                batch["history_times"].shape[0],
                batch["history_times"].shape[1],
                model.config.object_slots,
                device=device,
                dtype=torch.bool,
            )
            with torch.no_grad(), amp_context():
                if args.stage == "representation":
                    result = model(
                        batch,
                        history_mask=history_mask,
                        phase="object_memory_representation_loss",
                        loss_weights=representation_weights(),
                    )
                elif args.stage == "posterior":
                    result = model(batch, history_mask=history_mask, phase="joint")
                else:
                    prior = model.predict_prior_features(
                        batch, args.prior_samples, stochastic=True
                    )
                    target = batch["future_features"][None].expand_as(prior)
                    valid = batch["future_valid"][None].expand(prior.shape[:-1])
                    sampled_error = feature_error(
                        prior.flatten(0, 1),
                        target.flatten(0, 1),
                        valid.flatten(0, 1),
                    ).reshape(args.prior_samples, -1, 2)
                    persistence = feature_error(
                        batch["history_features"][:, -1:].expand_as(
                            batch["future_features"]
                        ),
                        batch["future_features"],
                        batch["future_valid"],
                    )
                    goal_valid = batch["future_horizon_valid"][:, 1]
                    metrics.add("goal_valid", goal_valid.float())
                    for scope in (f"h{history_length}", "all"):
                        metrics.add(
                            f"prior/{scope}/short_single", sampled_error[0, :, 0]
                        )
                        metrics.add(
                            f"prior/{scope}/short_best_of_n",
                            sampled_error[:, :, 0].min(dim=0).values,
                        )
                        metrics.add(
                            f"prior/{scope}/short_persistence", persistence[:, 0]
                        )
                        metrics.add(
                            f"prior/{scope}/goal_single",
                            sampled_error[0, :, 1],
                            goal_valid,
                        )
                        metrics.add(
                            f"prior/{scope}/goal_best_of_n",
                            sampled_error[:, :, 1].min(dim=0).values,
                            goal_valid,
                        )
                        metrics.add(
                            f"prior/{scope}/goal_persistence",
                            persistence[:, 1],
                            goal_valid,
                        )
                    continue
            _, _, goal_valid = add_horizon_metrics(
                metrics, args.stage, history_length, result, batch
            )
            metrics.add("goal_valid", goal_valid.float())
            if args.stage == "posterior":
                shuffled_actions = cross_episode_shuffle(
                    result["posterior_actions"], batch["sequence_index"]
                )
                with torch.no_grad(), amp_context():
                    shuffled = model(
                        batch,
                        history_mask=history_mask,
                        phase="joint",
                        actions_override=shuffled_actions,
                    )
                shuffled_error = feature_error(
                    shuffled["rollout_goal_rendered_features"],
                    batch["future_features"][:, 1:2],
                    batch["future_valid"][:, 1:2],
                )[:, 0]
                for scope in (f"h{history_length}", "all"):
                    metrics.add(
                        f"posterior/{scope}/goal_shuffled", shuffled_error, goal_valid
                    )
    return metrics.means()


def acceptance(stage: str, means: dict, minimum_shuffle: float) -> dict[str, bool]:
    checks = {
        "all_metrics_finite": all(math.isfinite(value) for value in means.values())
    }
    for history in range(1, 5):
        prefix = f"{stage}/h{history}"
        checks[f"h{history}_short_beats_persistence"] = (
            means[f"{prefix}/short_direct"] < means[f"{prefix}/short_persistence"]
        )
        if stage == "posterior":
            checks[f"h{history}_goal_direct_beats_persistence"] = (
                means[f"{prefix}/goal_direct"]
                < means[f"{prefix}/goal_persistence"]
            )
            checks[f"h{history}_goal_rollout_beats_persistence"] = (
                means[f"{prefix}/goal_rollout"]
                < means[f"{prefix}/goal_persistence"]
            )
    if stage == "posterior":
        checks.update(
            nonempty_stable_goal_subset=means["goal_valid"] > 0.0,
            goal_direct_beats_persistence=(
                means["posterior/all/goal_direct"]
                < means["posterior/all/goal_persistence"]
            ),
            goal_rollout_beats_persistence=(
                means["posterior/all/goal_rollout"]
                < means["posterior/all/goal_persistence"]
            ),
            shuffled_effects_are_worse=(
                means["posterior/all/goal_shuffled"]
                >= means["posterior/all/goal_rollout"] * (1.0 + minimum_shuffle)
            ),
        )
    return checks


def prior_acceptance(means: dict) -> dict[str, bool]:
    return {
        "all_metrics_finite": all(math.isfinite(value) for value in means.values()),
        "nonempty_stable_goal_subset": means["goal_valid"] > 0.0,
        "short_best_of_n_is_coverage": (
            means["prior/all/short_best_of_n"] <= means["prior/all/short_single"]
        ),
        "goal_best_of_n_is_coverage": (
            means["prior/all/goal_best_of_n"] <= means["prior/all/goal_single"]
        ),
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v39 held evaluation requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "held evaluation rejects tracked worktree changes")
    checksum = subprocess.run(
        ["sha256sum", "-c", "--status", EPISODE_VERIFIED_NAME],
        cwd=args.data,
        check=False,
    )
    require(checksum.returncode == 0, "RGB manifest checksum failed")
    dataset = build_dataset(args)
    require(len(dataset) >= args.batch, "held split is smaller than one batch")
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    device = torch.device("cuda:0")
    model, checkpoint = load_model(args, dataset, device)
    require(checkpoint.get("git_commit") == commit, "checkpoint commit differs")
    dino_amp = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    runtime = JitDinoFeatureRuntime(
        device,
        dino_amp,
        args.jit_dino_batch,
        args.goal_stability_threshold,
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    means = evaluate(args, model, dataset, runtime, device, amp_context)
    checks = (
        prior_acceptance(means)
        if args.stage == "prior"
        else acceptance(args.stage, means, args.minimum_shuffle_degradation)
    )
    passed = all(checks.values())
    manifest_path = os.path.join(args.data, EPISODE_MANIFEST_NAME)
    report = {
        "status": "passed" if passed else "failed",
        "contract": f"object_memory_v39_{args.stage}_held_v1",
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": file_sha256(manifest_path),
        "held_split": args.split,
        "held_items": len(dataset),
        "history_lengths": [1, 2, 3, 4],
        "history_span_frames": list(dataset.history_span_frames),
        "short_horizon_frames": 30,
        "goal_query_seconds": args.goal_query_seconds,
        "goal_tail_guard_frames": args.goal_tail_guard_frames,
        "goal_probe_frames": args.goal_probe_frames,
        "goal_stability_threshold": args.goal_stability_threshold,
        "source_checkpoint": os.path.realpath(args.checkpoint),
        "source_checkpoint_sha256": file_digest(args.checkpoint),
        "source_checkpoint_phase_step": checkpoint["phase_step"],
        "prior_samples": args.prior_samples if args.stage == "prior" else 0,
        "evaluation_seed": args.seed,
        "minimum_shuffle_degradation": args.minimum_shuffle_degradation,
        "checks": checks,
        "metrics": means,
        "decision": (
            "coverage_evaluation_only"
            if args.stage == "prior" and passed
            else "promote" if passed else "reject"
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    require(passed, f"v39 {args.stage} held gate failed")


if __name__ == "__main__":
    main()
