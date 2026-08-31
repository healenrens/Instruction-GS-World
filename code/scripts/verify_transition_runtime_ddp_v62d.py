"""Multi-rank runtime probe for the v62 E1 deterministic transition oracle."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields, replace
import json
import math
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_effect_posterior_v62 import (  # noqa: E402
    zero_object_effect_v62,
)
from igsw.adaptive_gaussian_wm.object_transition_teacher_runtime_v62 import (  # noqa: E402
    ObjectTransitionTeacherRuntimeV62,
)
from igsw.adaptive_gaussian_wm.teacher_transition_oracle_v62 import (  # noqa: E402
    TeacherTransitionOracleV62,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    cuda_memory_metrics,
    move_to_device,
    reduce_metrics,
)
from igsw.adaptive_gaussian_wm.v62_checkpointing import (  # noqa: E402
    load_e0_codec_checkpoint_v62,
)
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    ObjectTransitionConfigV62,
)
from igsw.distributed import init_torchrun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--siglip_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_group", default="object-transition-v62d-ddp-runtime")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def balanced_indices(dataset, count):
    per_source = math.ceil(count / len(dataset.source_names))
    pools = [
        dataset.balanced_source_evaluation_indices(index, per_source)
        for index in range(len(dataset.source_names))
    ]
    ordered = []
    for position in range(per_source):
        for pool in pools:
            ordered.append(pool[position])
    return ordered[:count]


def roll_state(state):
    return replace(
        state,
        **{
            field.name: torch.roll(getattr(state, field.name), shifts=1, dims=0)
            for field in fields(state)
        },
    )


def state_max_difference(first, second):
    return torch.stack(
        [
            (getattr(first, field.name).float() - getattr(second, field.name).float())
            .abs()
            .amax()
            for field in fields(first)
        ]
    ).amax()


@torch.no_grad()
def causal_intervention_metrics(model, output, frame_times):
    source = output["source"]
    target = output["target"]
    delta_seconds = frame_times[:, 1] - frame_times[:, 0]
    swapped_effect = model.posterior(source, roll_state(target), delta_seconds)
    zero = zero_object_effect_v62(output["effect"])
    zero_swapped = zero_object_effect_v62(swapped_effect)
    zero_prediction, _ = model.dynamics(source, zero, delta_seconds)
    zero_swapped_prediction, _ = model.dynamics(source, zero_swapped, delta_seconds)
    posterior_difference = (
        (output["effect"].value.float() - swapped_effect.value.float())
        .square()
        .mean()
        .sqrt()
    )
    identity_difference = (
        (output["correct"].identity.float() - source.identity.float()).abs().amax()
    )
    return {
        "posterior_future_swap_rms": posterior_difference,
        "zero_future_swap_max_difference": state_max_difference(
            zero_prediction, zero_swapped_prediction
        ),
        "identity_copy_max_difference": identity_difference,
    }


def parameter_sync_difference(model, context, device):
    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in model.parameters()]
    ).sum()
    minimum = checksum.clone()
    maximum = checksum.clone()
    if context.distributed:
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return (maximum - minimum).abs().to(device)


def write_wandb(args, report):
    if args.wandb_mode == "disabled":
        return
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "contract": report["contract"],
            "source_revision": args.source_revision,
            "world_size": report["world_size"],
            "steps": args.steps,
            "batch_per_rank": args.batch,
        },
    )
    run.log({f"runtime/{name}": value for name, value in report["metrics"].items()})
    run.summary.update(report)
    run.finish()


def main():
    args = parse_args()
    context = init_torchrun()
    device = torch.device(context.device)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    torch.set_float32_matmul_precision("high")
    config = ObjectTransitionConfigV62()
    codec_checkpoint = load_e0_codec_checkpoint_v62(args.codec_checkpoint, config)
    model = TeacherTransitionOracleV62(config).to(device).train()
    model.load_codec_state(codec_checkpoint["model"])
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
    teacher = ObjectTransitionTeacherRuntimeV62(
        config,
        device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        "3",
        "100",
        0,
        args.seed,
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    total_items = args.steps * args.batch * context.world_size
    indices = balanced_indices(dataset, total_items)
    local_indices = indices[context.rank :: context.world_size]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    metric_sums = {}
    unused_parameter_count = torch.zeros((), device=device)
    for step in range(args.steps):
        selected = local_indices[step * args.batch : (step + 1) * args.batch]
        cpu_batch = default_collate([dataset[(index, 3)] for index in selected])
        batch = move_to_device(cpu_batch, device)
        observation = teacher(batch)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            output = wrapped(observation, batch["frame_times"], batch["source_index"])
        output["loss"].backward()
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        unused_parameter_count = torch.tensor(
            sum(parameter.grad is None for _, parameter in trainable),
            device=device,
            dtype=torch.float32,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            max_norm=5.0,
            error_if_nonfinite=True,
        )
        metrics = {
            **output["parts"],
            **causal_intervention_metrics(model, output, batch["frame_times"]),
            "gradient_norm": grad_norm,
            "unused_parameter_count": unused_parameter_count,
        }
        optimizer.step()
        for name, value in metrics.items():
            tensor = (
                value
                if torch.is_tensor(value)
                else torch.tensor(value, device=device, dtype=torch.float32)
            )
            metric_sums[name] = (
                metric_sums.get(name, tensor.detach() * 0.0) + tensor.detach()
            )
    averaged = {name: value / args.steps for name, value in metric_sums.items()}
    averaged["parameter_sync_max_difference"] = parameter_sync_difference(
        model, context, device
    )
    reduced = reduce_metrics(
        averaged,
        context.world_size,
        max_names=frozenset(
            (
                "zero_future_swap_max_difference",
                "identity_copy_max_difference",
                "unused_parameter_count",
                "parameter_sync_max_difference",
            )
        ),
    )
    reduced.update(cuda_memory_metrics(device))
    if context.is_main:
        report = {
            "status": "completed",
            "contract": "transition_oracle_ddp_runtime_v62d",
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "source_revision": args.source_revision,
            "codec_checkpoint": os.path.abspath(args.codec_checkpoint),
            "codec_checkpoint_step": int(codec_checkpoint["global_step"]),
            "data": os.path.abspath(args.data_index),
            "world_size": context.world_size,
            "batch_per_rank": args.batch,
            "steps_completed": args.steps,
            "writes_training_checkpoint": False,
            "metrics": reduced,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        write_wandb(args, report)
        print(json.dumps(report, sort_keys=True))
    if context.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
