#!/usr/bin/env python3
"""Four-GPU real-data admission for the v67 continuous object field."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_predictive_object_field_v67 import (  # noqa: E402
    ContinuousPredictiveObjectFieldV67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_teacher_v67 import (  # noqa: E402
    ContinuousPredictiveTeacherRuntimeV67,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v67_checkpointing import (  # noqa: E402
    load_predictive_state_checkpoint_v67,
)
from igsw.adaptive_gaussian_wm.v67_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    DYNAMICS_STAGE,
    STAGES,
    ContinuousPredictiveObjectFieldConfigV67,
)
from igsw.distributed import init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--state_checkpoint", default="")
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--expected_world_size", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--siglip_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_batch(args, dataset, context):
    selected = []
    for source_index in range(len(dataset.source_names)):
        candidates = dataset.balanced_source_evaluation_indices(
            source_index, context.world_size
        )
        selected.append(candidates[context.rank])
    selected = selected[: args.batch]
    samples = [dataset[(index, 8)] for index in selected]
    return move_to_device(
        collate_native_video_batch_v65(samples), torch.device(context.device)
    )


def swap_future(batch, target, source_frame):
    swapped_batch = dict(batch)
    swapped_rgb = batch["video_rgb"].clone()
    swapped_rgb[:, source_frame + 1 :] = swapped_rgb[
        :, source_frame + 1 :
    ].roll(1, dims=0)
    swapped_batch["video_rgb"] = swapped_rgb

    def changed(value):
        result = value.clone()
        result[:, source_frame + 1 :] = result[:, source_frame + 1 :].roll(
            1, dims=0
        )
        return result

    swapped_target = replace(
        target,
        track_coordinates=changed(target.track_coordinates),
        dino=changed(target.dino),
        siglip=changed(target.siglip),
        visibility=changed(target.visibility),
    )
    return swapped_batch, swapped_target


def gradient_contract(model):
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not bool(parameter.grad.isfinite().all())
    ]
    squared_norm = sum(
        float(parameter.grad.float().square().sum())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    return missing, nonfinite, squared_norm**0.5


@torch.no_grad()
def causal_contract(model, batch, target, config, stage):
    model.eval()
    swapped_batch, swapped_target = swap_future(
        batch, target, config.source_frame
    )
    _, _, _, source = model.encode_source(batch, target, False)
    _, _, _, swapped_source = model.encode_source(
        swapped_batch, swapped_target, False
    )
    _, goal = model.encode_target(batch, target, config.target_frame)
    _, swapped_goal = model.encode_target(
        swapped_batch, swapped_target, config.target_frame
    )
    source_difference = float((source.mean - swapped_source.mean).abs().max())
    target_difference = float((goal.mean - swapped_goal.mean).abs().max())
    effect_difference = 0.0
    if stage == DYNAMICS_STAGE:
        delta = (
            batch["frame_times"][:, config.target_frame]
            - batch["frame_times"][:, config.source_frame]
        )
        effect = model.effect_posterior(source, goal, delta, sample=False)
        changed_effect = model.effect_posterior(
            swapped_source, swapped_goal, delta, sample=False
        )
        effect_difference = float(
            (effect.mean - changed_effect.mean).abs().max()
        )
    return source_difference, target_difference, effect_difference


def main() -> None:
    args = parse_args()
    context = init_torchrun()
    require(
        context.world_size == args.expected_world_size,
        "v67 admission world size differs from the explicit contract",
    )
    config = ContinuousPredictiveObjectFieldConfigV67()
    config.validate()
    if args.stage == DYNAMICS_STAGE:
        require(
            os.path.isfile(args.state_checkpoint),
            "v67 Dynamics admission requires the E0 state checkpoint",
        )
    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        "8",
        "100,200,400",
        0,
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    batch = build_batch(args, dataset, context)
    device = torch.device(context.device)
    teacher = ContinuousPredictiveTeacherRuntimeV67(
        config,
        device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    target = teacher(batch)
    model = ContinuousPredictiveObjectFieldV67(config, args.stage)
    if args.stage == DYNAMICS_STAGE:
        checkpoint = load_predictive_state_checkpoint_v67(
            args.state_checkpoint, config
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.configure_stage(DYNAMICS_STAGE)
    model = model.to(device).train()
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if context.distributed
        else model
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    with amp_context():
        output = wrapped(batch, target)
    require(bool(output["loss"].isfinite()), "v67 admission loss is non-finite")
    output["loss"].backward()
    missing, nonfinite, gradient_norm = gradient_contract(model)
    require(not missing, f"v67 trainable parameters lack gradients: {missing}")
    require(not nonfinite, f"v67 parameters have non-finite gradients: {nonfinite}")
    source_difference, target_difference, effect_difference = causal_contract(
        model, batch, target, config, args.stage
    )
    require(source_difference < 1e-6, "v67 source path reads future observations")
    require(target_difference > 1e-6, "v67 target path ignores future observations")
    if args.stage == DYNAMICS_STAGE:
        require(effect_difference > 1e-6, "v67 posterior ignores the future target")
    local = {
        "rank": context.rank,
        "loss": float(output["loss"].detach()),
        "gradient_norm": gradient_norm,
        "source_future_swap_max_difference": source_difference,
        "target_future_swap_max_difference": target_difference,
        "posterior_future_swap_max_difference": effect_difference,
        "source_names": [dataset.source_names[index] for index in range(args.batch)],
        "native_height": int(batch["native_image_hw"][:, 0].max()),
        "native_width": int(batch["native_image_hw"][:, 1].max()),
    }
    reports = [local]
    if context.distributed:
        reports = [None] * context.world_size
        dist.all_gather_object(reports, local)
    if context.is_main:
        report = {
            "status": "passed",
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "stage": args.stage,
            "git_commit": args.source_revision,
            "world_size": context.world_size,
            "fixed_object_count": False,
            "patch_grid_is_object_state": False,
            "rgb_reconstruction": False,
            "tracker_role": "training_only_correspondence_visibility_reliability",
            "tracker_is_dynamic_target": False,
            "training_only_teachers": ["DINOv2-L", "SigLIP", "CoTracker"],
            "per_rank": reports,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
    if context.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
