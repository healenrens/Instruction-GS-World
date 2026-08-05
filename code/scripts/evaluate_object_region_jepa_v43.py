#!/usr/bin/env python3
"""Held checkpoint evaluator for Object-Region JEPA v43 promotion criteria."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.compact_state_probe import (  # noqa: E402
    CompactStateProbe,
)
from igsw.adaptive_gaussian_wm.dynamic_dual_horizon_dataset import (  # noqa: E402
    DynamicDualHorizonEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    file_sha256,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v43_model_runtime import _encode_sequence  # noqa: E402
from verify_object_memory_jepa_v39 import verify_manifest  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--startup_gate", required=True)
    parser.add_argument("--held_split", default="heldseed")
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--probe_train_items", type=int, default=32)
    parser.add_argument("--probe_eval_items", type=int, default=32)
    parser.add_argument("--probe_steps", type=int, default=200)
    parser.add_argument("--world_eval_items", type=int, default=32)
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()
    for name in ("data", "checkpoint", "output", "startup_gate"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(min(args.probe_train_items, args.probe_eval_items, args.probe_steps)
            > 0, "probe evaluation sizes must be positive")
    require(args.world_eval_items >= 2, "world evaluation needs two held samples")
    return args


def dataset(args, split: str, items: int):
    return DynamicDualHorizonEpisodeDataset(
        args.data,
        split,
        history_frames_min=1,
        history_frames_max=4,
        history_span_frames=args.history_span_frames,
        short_horizon_frames=30,
        goal_query_seconds=args.goal_query_seconds,
        goal_tail_guard_frames=args.goal_tail_guard_frames,
        goal_probe_frames=args.goal_probe_frames,
        max_items=items,
        teacher_sidecar=args.teacher_sidecar,
        feature_source="jit",
    )


def load_model(args, device):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    require(checkpoint.get("checkpoint_version") == 43, "checkpoint is not v43")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    require(config.architecture == "object_region_memory_v1",
            "checkpoint architecture differs")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def cosine_error(prediction, target):
    return 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1
    )


def valid_mean(value, valid):
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def compact_example(model, sample, device, amp_context):
    batch = move_to_device(default_collate([sample]), device)
    with amp_context():
        online_features = model.online_dino(
            batch["history_jit_rgb"], batch["history_jit_valid"]
        )
        online = _encode_sequence(
            model,
            online_features,
            batch["history_times"],
            target=False,
            masked=False,
        )
        target_features = model.target_dino(
            batch["history_jit_rgb"], batch["history_jit_valid"]
        )
    token = online["token_states"][-1]
    return {
        "region": online["regions"]["feature"][:, -1].cpu().float(),
        "root": online["roots"]["slots"][:, -1].cpu().float(),
        "owner": online["regions"]["owner"][:, -1].cpu().float(),
        "center": online["regions"]["center"][:, -1].cpu().float(),
        "covariance": online["regions"]["covariance"][:, -1].cpu().float(),
        "coordinates": target_features.coordinates[:, -1].cpu().float(),
        "presence": online["regions"]["presence"][:, -1].cpu().float(),
        "target": target_features.native[:, -1].cpu().float(),
        "valid": target_features.valid[:, -1].cpu(),
        "token": token.reconstructed_features.cpu().float(),
    }


def cache_probe_examples(model, source, count, device, amp_context):
    return [
        compact_example(model, source[(index, 4)], device, amp_context)
        for index in range(min(count, len(source)))
    ]


def stack_examples(examples, indices, device):
    return {
        name: torch.cat([examples[index][name] for index in indices]).to(device)
        for name in examples[0]
    }


def fit_probe(model, train_examples, steps, device):
    probe = CompactStateProbe(model.config).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-4, weight_decay=1e-4)
    batch_size = min(4, len(train_examples))
    for step in range(steps):
        indices = [
            (step * batch_size + offset) % len(train_examples)
            for offset in range(batch_size)
        ]
        values = stack_examples(train_examples, indices, device)
        prediction = probe(
            values["region"], values["root"], values["owner"],
            values["center"], values["covariance"], values["coordinates"],
            values["presence"], values["valid"],
        )
        loss = valid_mean(
            cosine_error(prediction, values["target"]), values["valid"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        optimizer.step()
    return probe.eval()


@torch.no_grad()
def evaluate_probe(probe, examples, device):
    compact_errors = []
    token_errors = []
    scene_errors = []
    for start in range(0, len(examples), 4):
        indices = list(range(start, min(start + 4, len(examples))))
        values = stack_examples(examples, indices, device)
        compact = probe(
            values["region"], values["root"], values["owner"],
            values["center"], values["covariance"], values["coordinates"],
            values["presence"], values["valid"],
        )
        weight = values["valid"].to(values["target"].dtype)[..., None]
        scene = (values["target"] * weight).sum(dim=1, keepdim=True)
        scene = scene / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        scene = scene.expand_as(values["target"])
        compact_errors.append(
            valid_mean(cosine_error(compact, values["target"]), values["valid"])
        )
        token_errors.append(
            valid_mean(
                cosine_error(values["token"], values["target"]), values["valid"]
            )
        )
        scene_errors.append(
            valid_mean(cosine_error(scene, values["target"]), values["valid"])
        )
    compact = torch.stack(compact_errors).mean()
    token = torch.stack(token_errors).mean()
    scene = torch.stack(scene_errors).mean()
    recovery = (scene - compact) / (scene - token).clamp_min(1e-6)
    return {
        "compact_probe_error": float(compact),
        "gpstoken_error": float(token),
        "scene_baseline_error": float(scene),
        "compact_gap_recovery": float(recovery),
    }


@torch.no_grad()
def evaluate_samples(model, samples, step, device, amp_context, diagnostics=False):
    batch = move_to_device(default_collate(samples), device)
    model.set_curriculum_step(step)
    with amp_context():
        output = model(batch, collect_diagnostics=diagnostics)
    return {
        name: float(value.detach())
        for name, value in output["parts"].items()
        if value.numel() == 1
    }


def evaluate_one(model, sample, step, device, amp_context, diagnostics=False):
    return evaluate_samples(
        model, [sample], step, device, amp_context, diagnostics
    )


def mean(values):
    return sum(values) / max(1, len(values))


def world_metrics(model, held, count, device, amp_context):
    h1_errors = []
    h4_errors = []
    h4_motion_gains = []
    h1_motion_gains = []
    short_gains = []
    temporal_gains = []
    zero_gains = []
    shuffled_gains = []
    direct_gains = []
    rollout_gains = []
    goal_path_errors = []
    goal_target_errors = []
    rank_fractions = []
    budget_boundary = []
    goal_valid_items = 0
    for index in range(min(count, len(held))):
        action_free_h1 = evaluate_one(
            model, held[(index, 1)], 7000, device, amp_context
        )
        action_free_h4 = evaluate_one(
            model, held[(index, 4)], 7000, device, amp_context
        )
        held_count = min(count, len(held))
        partner = (index + max(1, held_count // 2)) % held_count
        posterior = evaluate_samples(
            model,
            [held[(index, 4)], held[(partner, 4)]],
            22000,
            device,
            amp_context,
            diagnostics=True,
        )
        h1 = action_free_h1["loss_short_root"] + action_free_h1["loss_short_region"]
        h4 = action_free_h4["loss_short_root"] + action_free_h4["loss_short_region"]
        h4_persistence = (
            action_free_h4["diagnostic_short_persistence_root"]
            + action_free_h4["diagnostic_short_persistence_region"]
        )
        h1_persistence = (
            action_free_h1["diagnostic_short_persistence_root"]
            + action_free_h1["diagnostic_short_persistence_region"]
        )
        h1_errors.append(h1)
        h4_errors.append(h4)
        short_gains.append(
            action_free_h4["diagnostic_short_relative_gain_over_persistence"]
        )
        if max(h1_persistence, h4_persistence) > 0.05:
            h1_motion_gains.append(
                (h1_persistence - h1) / max(h1_persistence, 1e-6)
            )
            h4_motion_gains.append(
                (h4_persistence - h4) / max(h4_persistence, 1e-6)
            )
        temporal_gains.append(posterior.get("temporal_order_relative_gain", 0.0))
        zero_gains.append(posterior["diagnostic_effect_relative_gain_over_zero"])
        shuffled_gains.append(
            posterior["diagnostic_effect_relative_gain_over_shuffled"]
        )
        if posterior["goal_valid_fraction"] > 0.0:
            goal_valid_items += 1
            direct_gains.append(
                posterior["diagnostic_goal_direct_gain_over_persistence"]
            )
            rollout_gains.append(
                posterior["diagnostic_goal_rollout_gain_over_persistence"]
            )
            goal_path_errors.append(
                posterior["loss_path_root"] + posterior["loss_path_region"]
            )
            goal_target_errors.append(
                0.5
                * (
                    posterior["diagnostic_goal_direct"]
                    + posterior["diagnostic_goal_rollout"]
                )
            )
        rank_fractions.append(posterior["region_effective_rank_fraction"])
        budget_boundary.append(
            posterior["region_min_budget_fraction"]
            + posterior["region_max_budget_fraction"]
        )
    history_gain = mean(h4_motion_gains) - mean(h1_motion_gains)
    return {
        "short_relative_gain_over_persistence": mean(short_gains),
        "h1_short_error": mean(h1_errors),
        "h4_short_error": mean(h4_errors),
        "motion_active_items": len(h1_motion_gains),
        "motion_h1_gain_over_persistence": mean(h1_motion_gains),
        "motion_h4_gain_over_persistence": mean(h4_motion_gains),
        "motion_h4_gain_over_h1": history_gain,
        "temporal_order_relative_gain": mean(temporal_gains),
        "posterior_gain_over_zero": mean(zero_gains),
        "posterior_gain_over_shuffled": mean(shuffled_gains),
        "goal_valid_items": goal_valid_items,
        "goal_direct_gain_over_persistence": mean(direct_gains),
        "goal_rollout_gain_over_persistence": mean(rollout_gains),
        "goal_path_error": mean(goal_path_errors),
        "goal_path_error_ratio_to_target": mean(goal_path_errors)
        / max(mean(goal_target_errors), 1e-6),
        "region_effective_rank_fraction": mean(rank_fractions),
        "region_budget_boundary_fraction": mean(budget_boundary),
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v43 evaluation requires CUDA")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest_sha256, _ = verify_manifest(args.data)
    with open(args.startup_gate, encoding="utf-8") as handle:
        startup_gate = json.load(handle)
    require(startup_gate.get("status") == "passed", "startup gate did not pass")
    require(startup_gate.get("git_commit") == commit, "startup gate commit differs")
    require(startup_gate.get("data_manifest_sha256") == manifest_sha256,
            "startup gate data differs")
    device = torch.device("cuda:0")
    model, checkpoint = load_model(args, device)
    require(checkpoint.get("git_commit") == commit, "checkpoint commit differs")
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    train_data = dataset(args, "train", args.probe_train_items)
    held_data = dataset(
        args,
        args.held_split,
        max(args.probe_eval_items, args.world_eval_items),
    )
    train_examples = cache_probe_examples(
        model, train_data, args.probe_train_items, device, amp_context
    )
    held_examples = cache_probe_examples(
        model, held_data, args.probe_eval_items, device, amp_context
    )
    probe = fit_probe(model, train_examples, args.probe_steps, device)
    capacity = evaluate_probe(probe, held_examples, device)
    world = world_metrics(
        model, held_data, args.world_eval_items, device, amp_context
    )
    checks = {
        "compact_capacity_recovers_80_percent": capacity["compact_gap_recovery"] >= 0.8,
        "short_beats_persistence_10_percent": (
            world["short_relative_gain_over_persistence"] >= 0.1
        ),
        "h4_beats_h1_on_motion_3_percent": (
            world["motion_active_items"] > 0
            and world["motion_h4_gain_over_h1"] >= 0.03
        ),
        "ordered_beats_reversed_5_percent": (
            world["temporal_order_relative_gain"] >= 0.05
        ),
        "posterior_beats_zero_10_percent": world["posterior_gain_over_zero"] >= 0.1,
        "posterior_beats_shuffled_10_percent": (
            world["posterior_gain_over_shuffled"] >= 0.1
        ),
        "goal_direct_beats_persistence": (
            world["goal_valid_items"] > 0
            and world["goal_direct_gain_over_persistence"] > 0.0
        ),
        "goal_rollout_beats_persistence": (
            world["goal_valid_items"] > 0
            and world["goal_rollout_gain_over_persistence"] > 0.0
        ),
        "goal_direct_rollout_are_consistent": (
            world["goal_valid_items"] > 0
            and world["goal_path_error_ratio_to_target"] <= 1.0
        ),
        "region_effective_rank_above_20_percent": (
            world["region_effective_rank_fraction"] >= 0.2
        ),
        "region_budget_not_stuck": world["region_budget_boundary_fraction"] < 0.95,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": "object_region_jepa_v43_held_promotion_v1",
        "git_commit": commit,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "held_split": args.held_split,
        "probe_steps": args.probe_steps,
        "checks": checks,
        "capacity": capacity,
        "world": world,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed" and not args.report_only:
        raise RuntimeError("v43 held promotion criteria did not pass")


if __name__ == "__main__":
    main()
