"""Held-split evaluation for predictive Object Tube JEPA v45."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.predictive_object_tube_model import (  # noqa: E402
    PredictiveObjectTubeWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TemporalObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.v45_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    PredictiveObjectTubeConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("v45 evaluator collected no values")
    return sum(values) / len(values)


def _distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1, eps=1e-6
    )


def _endpoint_identity_top1(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    source = F.normalize(source.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    batch, slots = source.shape[:2]
    candidates = target.reshape(batch * slots, -1)
    logits = torch.einsum("bsd,qd->bsq", source, candidates)
    labels = torch.arange(batch * slots, device=source.device).reshape(batch, slots)
    correct = (logits.argmax(dim=-1) == labels).float()
    return (correct * weight).sum() / weight.sum().clamp_min(1.0)


def _temporal_identity_top1(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    source = F.normalize(source.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    batch, steps, slots = source.shape[:3]
    candidates = target.permute(1, 0, 2, 3).reshape(steps, batch * slots, -1)
    logits = torch.einsum("btsd,tqd->btsq", source, candidates)
    labels = torch.arange(batch * slots, device=source.device).reshape(batch, slots)
    labels = labels[:, None].expand(batch, steps, slots)
    correct = (logits.argmax(dim=-1) == labels).float()
    return (correct * weight).sum() / weight.sum().clamp_min(1.0)


def _tube_metrics(
    model, output
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full = output["full_state"]
    assignment = full["assignment"].float()
    transported = torch.einsum(
        "btsn,btnm->btsm",
        assignment[:, :-1],
        output["correspondence"].forward.float(),
    )
    patches = full["tube_target_patches"][:, 1:].float()
    target_valid = assignment[:, 1:].sum(dim=2).clamp(0.0, 1.0)
    weight = transported * target_valid[:, :, None]
    mass = weight.sum(dim=-1)
    denominator = mass + model.config.observation_mass_tau
    target = torch.einsum("btsn,btnd->btsd", weight, patches) / denominator[..., None]
    support = mass / denominator
    correct = _distance(full["tube_prediction"], target)
    persistence = _distance(full["decoded_slots"][:, :-1], target)
    correct = (correct * support).sum() / support.sum().clamp_min(1.0)
    persistence = (persistence * support).sum() / support.sum().clamp_min(1.0)
    identity = _temporal_identity_top1(
        full["tube_prediction"], target, support
    )
    return correct, persistence, identity


def _long_tube_target(model, output) -> tuple[torch.Tensor, torch.Tensor]:
    full = output["full_state"]
    assignment = full["assignment"].float()
    transported = assignment[:, 0]
    for transition in output["correspondence"].forward.float().unbind(dim=1):
        transported = torch.einsum("bsn,bnm->bsm", transported, transition)
    final_valid = assignment[:, -1].sum(dim=1).clamp(0.0, 1.0)
    weight = transported * final_valid[:, None]
    mass = weight.sum(dim=-1)
    denominator = mass + model.config.observation_mass_tau
    target = torch.einsum(
        "bsn,bnd->bsd", weight, full["tube_target_patches"][:, -1].float()
    ) / denominator[..., None]
    return target, mass / denominator


def _batch_metrics(model, output) -> dict[str, float]:
    full, masked = output["full_state"], output["masked_state"]
    tube, persistence, adjacent_identity = _tube_metrics(model, output)
    long_target, long_weight = _long_tube_target(model, output)
    long_identity = _endpoint_identity_top1(
        long_target, full["decoded_slots"][:, -1], long_weight
    )
    reappearance = _endpoint_identity_top1(
        long_target, masked["decoded_slots"][:, -1], long_weight
    )
    parts = output["parts"]
    return {
        "tube_correct_distance": float(tube),
        "tube_persistence_distance": float(persistence),
        "tube_relative_gain_over_persistence": float(
            (persistence - tube) / persistence.clamp_min(1e-6)
        ),
        "adjacent_tube_identity_top1": float(adjacent_identity),
        "long_horizon_identity_top1": float(long_identity),
        "masked_reappearance_identity_top1": float(reappearance),
        "effect_correct_distance": float(parts["effect_correct_distance"]),
        "effect_zero_distance": float(parts["effect_zero_distance"]),
        "effect_shuffled_distance": float(parts["effect_shuffled_distance"]),
        "effect_relative_gain_over_zero": float(
            parts["effect_relative_gain_over_zero"]
        ),
        "effect_relative_gain_over_shuffled": float(
            parts["effect_relative_gain_over_shuffled"]
        ),
        "image_goal_correct_distance": float(parts["goal_correct_distance"]),
        "image_goal_zero_distance": float(parts["goal_zero_distance"]),
        "image_goal_shuffled_distance": float(parts["goal_shuffled_distance"]),
        "image_goal_relative_gain_over_zero": float(
            parts["goal_relative_gain_over_zero"]
        ),
        "effect_action_norm": float(parts["effect_action_norm"]),
        "effect_action_std": float(parts["effect_action_std"]),
        "object_effective_count": float(parts["object_effective_count"]),
        "object_identity_top1": float(parts["object_identity_top1"]),
        "object_correction_gate": float(parts["object_correction_gate"]),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v45 evaluator requires a visible CUDA device")
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v45 evaluator requires a version-45 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v45 evaluator checkpoint architecture differs")
    config = PredictiveObjectTubeConfig(**checkpoint["config"])
    dataset = TemporalObjectVideoDataset(
        args.data,
        args.split,
        max_items=args.max_items,
        seed=17,
        record_manifest_hash=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    model = PredictiveObjectTubeWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    collected: dict[str, list[float]] = {}
    items = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                name: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for name, value in batch.items()
            }
            features = encoder(batch)
            with amp_context():
                output = model(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                    batch["observation_mask"],
                    config.total_steps,
                )
            metrics = _batch_metrics(model, output)
            for name, value in metrics.items():
                collected.setdefault(name, []).append(value)
            items += len(batch["video_rgb"])
    metrics = {name: _mean(values) for name, values in collected.items()}
    report = {
        "status": "completed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "items": items,
        "metrics": metrics,
        "checks": {
            "tube_beats_persistence": metrics["tube_relative_gain_over_persistence"] > 0,
            "effect_beats_zero": metrics["effect_relative_gain_over_zero"] > 0,
            "effect_beats_shuffled": metrics["effect_relative_gain_over_shuffled"] > 0,
            "image_goal_beats_zero": metrics["image_goal_relative_gain_over_zero"] > 0,
            "effect_not_unit_norm": metrics["effect_action_norm"] < 0.95,
        },
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
