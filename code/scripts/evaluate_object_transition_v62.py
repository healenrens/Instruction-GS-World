"""Six-source held evaluation for v62 E0 and E1 with W&B export."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_transition_teacher_runtime_v62 import (  # noqa: E402
    ObjectTransitionTeacherRuntimeV62,
)
from igsw.adaptive_gaussian_wm.teacher_object_autoencoder_v62 import (  # noqa: E402
    TeacherObjectAutoencoderV62,
)
from igsw.adaptive_gaussian_wm.teacher_transition_oracle_v62 import (  # noqa: E402
    TeacherTransitionOracleV62,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    E0_STAGE,
    STAGES,
    ObjectTransitionConfigV62,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--items_per_source", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--siglip_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="object-transition-v62-evaluation")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def load_model(args, checkpoint, config, device):
    if args.stage == E0_STAGE:
        model = TeacherObjectAutoencoderV62(config)
    else:
        model = TeacherTransitionOracleV62(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def validate_checkpoint(args, checkpoint, config):
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "config": config.to_dict(),
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if differences:
        raise ValueError(f"v62 evaluation checkpoint differs: {differences}")


@torch.no_grad()
def evaluate_source(args, dataset, source_index, model, teacher, device):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    totals, batches = {}, 0
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        batch = default_collate([dataset[(index, 3)] for index in selected])
        batch = move_to_device(batch, device)
        observation = teacher(batch)
        with amp_context():
            if args.stage == E0_STAGE:
                output = model(observation)
            else:
                output = model(observation, batch["frame_times"], batch["source_index"])
        for name, value in output["parts"].items():
            totals[name] = totals.get(name, 0.0) + float(value)
        totals["teacher_object_valid_fraction"] = totals.get(
            "teacher_object_valid_fraction", 0.0
        ) + float(observation.object_valid.float().mean())
        batches += 1
    return {name: value / batches for name, value in totals.items()}


def main():
    args = parse_args()
    device = torch.device("cuda")
    config = ObjectTransitionConfigV62()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    validate_checkpoint(args, checkpoint, config)
    model = load_model(args, checkpoint, config, device)
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
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    source_metrics = {
        name: evaluate_source(args, dataset, index, model, teacher, device)
        for index, name in enumerate(dataset.source_names)
    }
    metric_names = sorted(next(iter(source_metrics.values())))
    overall = {
        name: sum(metrics[name] for metrics in source_metrics.values())
        / len(source_metrics)
        for name in metric_names
    }
    report = {
        "status": "completed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data_index),
        "held_group_stride": args.held_group_stride,
        "items_per_source": args.items_per_source,
        "source_metrics": source_metrics,
        "overall": overall,
        "patch_grid_is_object_target": False,
        "evaluation_unit": "continuous query-object track coordinates",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.wandb_mode != "disabled":
        import wandb

        os.makedirs(args.wandb_dir, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or None,
            group=args.wandb_group,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config={
                "checkpoint_version": CHECKPOINT_VERSION,
                "architecture": ARCHITECTURE,
                "stage": args.stage,
                "checkpoint": os.path.abspath(args.checkpoint),
                "items_per_source": args.items_per_source,
            },
        )
        payload = {f"eval/overall/{name}": value for name, value in overall.items()}
        for source, metrics in source_metrics.items():
            payload.update(
                {f"eval/{source}/{name}": value for name, value in metrics.items()}
            )
        run.log(payload, step=int(checkpoint["global_step"]))
        run.summary.update(report)
        run.finish()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
