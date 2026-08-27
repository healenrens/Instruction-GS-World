"""Real-GPU verifier for frozen-state posterior carrier Dynamics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_carrier_dynamics_model_v61 import (  # noqa: E402
    ContinuousCarrierDynamicsModelV61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_dynamics_v61 import (  # noqa: E402
    EFFECT_CAPACITIES,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_world_model_v61 import (  # noqa: E402
    ContinuousCarrierObjectWorldModelV61,
)
from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v61_config import (  # noqa: E402
    CHECKPOINT_VERSION,
    DYNAMICS_ARCHITECTURE,
    DYNAMICS_STAGE,
    VARIANTS,
    config_for_variant,
)
from igsw.adaptive_gaussian_wm.v61_dynamics_checkpointing import (  # noqa: E402
    validate_state_checkpoint_v61,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--effect_capacity", choices=EFFECT_CAPACITIES, required=True)
    parser.add_argument("--state_checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_length", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--student_frame_batch", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _batch(sample, device):
    values = {
        name: value[None] if torch.is_tensor(value) else value
        for name, value in sample.items()
    }
    return move_to_device(values, device)


def _gradient_contract(model):
    names, norm = [], 0.0
    for name, parameter in model.named_parameters():
        if name.startswith("state_model."):
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError("v61 Dynamics modified the frozen Object State")
            continue
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"v61 Dynamics parameter has no gradient: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"v61 Dynamics has non-finite gradient: {name}")
        names.append(name)
        norm += float(parameter.grad.float().square().sum())
    prefixes = ("effect_posterior.", "dynamics.")
    if any(not any(name.startswith(prefix) for name in names) for prefix in prefixes):
        raise RuntimeError("v61 Dynamics required gradient group is missing")
    return len(names), norm**0.5


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v61 Dynamics verifier requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    config = config_for_variant(args.variant)
    if not config.uses_dino_alignment:
        raise ValueError("v61 Dynamics verifier requires a DINO-aligned variant")
    state_checkpoint_path = os.path.abspath(args.state_checkpoint)
    state_checkpoint = torch.load(
        state_checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_state_checkpoint_v61(
        state_checkpoint, config, args.source_revision, args.held_group_stride
    )
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        str(args.chunk_length),
        args.temporal_step_ms,
        max_items=32,
        seed=args.seed,
        group_partition="train",
        held_group_stride=args.held_group_stride,
    )
    batch = _batch(dataset[(0, args.chunk_length)], device)
    state_model = ContinuousCarrierObjectWorldModelV61(
        config,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.student_frame_batch,
    ).to(device)
    state_model.load_state_dict(state_checkpoint["model"], strict=True)
    model = ContinuousCarrierDynamicsModelV61(state_model, args.effect_capacity).to(
        device
    )
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    teacher_features = dino(batch)
    evidence = tracker(batch, teacher_features.patches, teacher_features.grid_hw)
    relation = build_trajectory_relation_teacher_v56(
        evidence, config, batch["frame_times"]
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    model.train()
    with amp_context():
        output = model(batch, evidence, relation)
    if not bool(torch.isfinite(output["loss"])):
        raise RuntimeError("v61 Dynamics verifier produced a non-finite loss")
    output["loss"].backward()
    gradient_tensors, gradient_norm = _gradient_contract(model)
    effect = output["effect"]
    factors, dimensions, binding = EFFECT_CAPACITIES[args.effect_capacity]
    expected = (1, factors, dimensions)
    if tuple(effect.value.shape) != expected:
        raise RuntimeError(f"v61 latent effect shape differs: {effect.value.shape}")
    owner_error = float((effect.owner.sum(dim=-1) - 1.0).abs().max())
    if owner_error >= 1e-5:
        raise RuntimeError("v61 effect owner probabilities do not sum to one")
    correct_zero_difference = float(
        (output["correct"].carriers.feature - output["zero"].carriers.feature)
        .abs()
        .max()
    )
    if correct_zero_difference <= 1e-6:
        raise RuntimeError("v61 Dynamics ignores the posterior latent effect")
    correct_shuffled_difference = float(
        (output["correct"].carriers.feature - output["shuffled"].carriers.feature)
        .abs()
        .max()
    )
    if correct_shuffled_difference <= 1e-6:
        raise RuntimeError("v61 shuffled effect is not a real intervention")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": DYNAMICS_ARCHITECTURE,
        "stage": DYNAMICS_STAGE,
        "variant": args.variant,
        "effect_capacity": args.effect_capacity,
        "effect_factors": factors,
        "effect_dimension": dimensions,
        "effect_binding": binding,
        "git_commit": args.source_revision,
        "state_checkpoint": state_checkpoint_path,
        "state_checkpoint_step": state_checkpoint["global_step"],
        "state_frozen": True,
        "held_group_stride": args.held_group_stride,
        "posterior_reads_future": True,
        "history_prior_present": False,
        "explicit_action_used": False,
        "effect_owner_partition_max_error": owner_error,
        "correct_zero_feature_max_difference": correct_zero_difference,
        "correct_shuffled_feature_max_difference": correct_shuffled_difference,
        "loss": float(output["loss"].detach()),
        "gradient_tensor_count": gradient_tensors,
        "gradient_norm": gradient_norm,
        "metrics": {name: float(value) for name, value in output["parts"].items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
