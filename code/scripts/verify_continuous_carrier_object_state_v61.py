"""Real-GPU startup contract for one v61 comparison variant."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.carrier_teacher_v61 import (  # noqa: E402
    FrozenSiglipObjectTeacherV61,
    build_object_components_v61,
)
from igsw.adaptive_gaussian_wm.bounded_probability_v61 import (  # noqa: E402
    binary_probability_values,
    normalize_probability_mass,
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
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STAGE,
    VARIANTS,
    config_for_variant,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_length", type=int, default=4)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--student_frame_batch", type=int, default=16)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--siglip_teacher_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def batch_from_sample(sample, device):
    batch = {
        name: value[None] if torch.is_tensor(value) else value
        for name, value in sample.items()
    }
    return move_to_device(batch, device)


def finite_gradients(model):
    names, norm = [], 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"v61 non-finite gradient: {name}")
        names.append(name)
        norm += float(parameter.grad.float().square().sum())
    prefixes = (
        "student.backbone.",
        "student.projector.",
        "state_encoder.carriers.",
        "state_encoder.roots.",
        "motion_readout.",
    )
    missing = [
        prefix
        for prefix in prefixes
        if not any(name.startswith(prefix) for name in names)
    ]
    if missing:
        raise RuntimeError(f"v61 missing required gradient groups: {missing}")
    return len(names), norm**0.5


def bounded_probability_contract(config, device, amp_context):
    low_mass = torch.full(
        (2, config.object_roots),
        1e-12,
        device=device,
        requires_grad=True,
    )
    boundary = torch.tensor((0.0, 1.0), device=device, requires_grad=True)
    with amp_context():
        normalized = normalize_probability_mass(
            low_mass,
            dim=-1,
            prior_mass=config.assignment_prior_mass,
        )
        boundary_loss = binary_probability_values(
            boundary,
            torch.tensor((1.0, 0.0), device=device),
            config.relation_probability_floor,
        ).mean()
        loss = normalized.square().mean() + boundary_loss
    loss.backward()
    values = (loss, normalized, low_mass.grad, boundary.grad)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("v61 bounded probability contract is non-finite")
    normalization_error = float(
        (normalized.sum(dim=-1) - 1.0).abs().max().detach()
    )
    maximum_gradient = max(
        float(low_mass.grad.abs().max()),
        float(boundary.grad.abs().max()),
    )
    if normalization_error >= 1e-6 or not 1.0 < maximum_gradient < 100.0:
        raise RuntimeError("v61 bounded probability contract differs")
    return normalization_error, maximum_gradient


@torch.no_grad()
def causal_contract(model, batch, amp_context):
    frames = batch["video_rgb"].shape[1]
    prefix = max(1, frames // 2)
    changed = dict(batch)
    changed_rgb = batch["video_rgb"].clone()
    changed_rgb[:, prefix:] = changed_rgb[:, prefix:].flip(-1)
    changed["video_rgb"] = changed_rgb
    with amp_context():
        _, original = model.encode_student(
            batch["video_rgb"], batch["video_pixel_valid"]
        )
        _, alternate = model.encode_student(
            changed["video_rgb"], changed["video_pixel_valid"]
        )
    differences = (
        (original.carriers.feature[:, :prefix] - alternate.carriers.feature[:, :prefix])
        .abs()
        .max(),
        (original.roots.feature[:, :prefix] - alternate.roots.feature[:, :prefix])
        .abs()
        .max(),
    )
    maximum = max(float(value) for value in differences)
    if maximum >= 1e-6:
        raise RuntimeError("v61 future suffix changed the causal Student prefix")
    suffix = float(
        (original.carriers.feature[:, prefix:] - alternate.carriers.feature[:, prefix:])
        .abs()
        .max()
    )
    if suffix <= 0.0:
        raise RuntimeError("v61 suffix perturbation did not change Student state")
    return maximum, suffix


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("v61 verifier requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    config = config_for_variant(args.variant)
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
    held_dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        str(args.chunk_length),
        args.temporal_step_ms,
        max_items=32,
        seed=args.seed,
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    training_groups = set(
        zip(dataset._group_source_indices, dataset.sampling_group_names)
    )
    held_groups = set(
        zip(held_dataset._group_source_indices, held_dataset.sampling_group_names)
    )
    if training_groups & held_groups:
        raise RuntimeError("v61 train and held task groups overlap")
    batch = batch_from_sample(dataset[(0, args.chunk_length)], device)
    model = ContinuousCarrierObjectWorldModelV61(
        config,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.student_frame_batch,
    ).to(device)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(
        config, device, args.tracker_checkpoint, sequence_batch=1
    )
    semantic_teacher = (
        FrozenSiglipObjectTeacherV61(
            args.siglip_checkpoint, device, args.siglip_teacher_batch
        )
        if config.uses_object_semantics
        else None
    )
    teacher_features = dino(batch)
    evidence = tracker(batch, teacher_features.patches, teacher_features.grid_hw)
    relation = build_trajectory_relation_teacher_v56(
        evidence, config, batch["frame_times"]
    )
    components = build_object_components_v61(
        batch, evidence, relation, config.object_roots, semantic_teacher
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    probability_error, probability_gradient = bounded_probability_contract(
        config, device, amp_context
    )
    model.train()
    with amp_context():
        output = model(batch, evidence, relation, components)
    if not bool(torch.isfinite(output["loss"])):
        raise RuntimeError("v61 verifier produced a non-finite loss")
    output["loss"].backward()
    gradient_tensors, gradient_norm = finite_gradients(model)
    field, state = output["field"], output["state"]
    expected = {
        "carrier_feature": (
            1,
            args.chunk_length,
            config.carrier_count,
            config.student_dim,
        ),
        "carrier_center": (1, args.chunk_length, config.carrier_count, 2),
        "root_feature": (1, args.chunk_length, config.object_roots, config.student_dim),
        "root_owner": (
            1,
            args.chunk_length,
            config.carrier_count,
            config.total_owners,
        ),
    }
    actual = {
        "carrier_feature": tuple(state.carriers.feature.shape),
        "carrier_center": tuple(state.carriers.center.shape),
        "root_feature": tuple(state.roots.feature.shape),
        "root_owner": tuple(state.roots.owner.shape),
    }
    if actual != expected:
        raise RuntimeError(f"v61 state shapes differ: {actual}")
    owner_error = float((state.roots.owner.sum(dim=-1) - 1.0).abs().max())
    if owner_error >= 1e-5:
        raise RuntimeError("v61 owner probabilities do not partition carriers")
    model.eval()
    prefix_difference, suffix_difference = causal_contract(model, batch, amp_context)
    blocks = (
        model.student.backbone.blocks
        if config.student_encoder == "dino"
        else model.student.backbone.encoder.layers
    )
    frozen_lower = all(
        not parameter.requires_grad
        for block in blocks[: -config.student_trainable_blocks]
        for parameter in block.parameters()
    )
    trainable_upper = all(
        parameter.requires_grad
        for block in blocks[-config.student_trainable_blocks :]
        for parameter in block.parameters()
    )
    if not frozen_lower or not trainable_upper:
        raise RuntimeError("v61 Student trainable-block boundary differs")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STAGE,
        "variant": args.variant,
        "git_commit": args.source_revision,
        "student_encoder": config.student_encoder,
        "single_student_encoder": True,
        "dynamics_present": False,
        "point_tracker_in_deployable_model": False,
        "training_only_dino_teacher": True,
        "training_only_cotracker_teacher": True,
        "training_only_siglip_crop_teacher": config.uses_object_semantics,
        "group_partition": "train",
        "held_group_stride": args.held_group_stride,
        "training_group_count": len(training_groups),
        "held_group_count": len(held_groups),
        "train_held_group_overlap": 0,
        "historical_checkpoint_used": False,
        "student_token_count": field.features.shape[2],
        "carrier_count": config.carrier_count,
        "object_root_count": config.object_roots,
        "owner_partition_max_error": owner_error,
        "causal_prefix_max_difference": prefix_difference,
        "suffix_perturbation_max_difference": suffix_difference,
        "loss": float(output["loss"].detach()),
        "gradient_tensor_count": gradient_tensors,
        "gradient_norm": gradient_norm,
        "bounded_probability_normalization_error": probability_error,
        "bounded_probability_max_gradient": probability_gradient,
        "frozen_lower_blocks": frozen_lower,
        "trainable_upper_blocks": trainable_upper,
        "metrics": {name: float(value) for name, value in output["parts"].items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
