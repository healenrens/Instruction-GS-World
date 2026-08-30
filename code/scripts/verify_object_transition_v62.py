"""Real-teacher startup verifier for v62 E0 and E1."""

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
from igsw.adaptive_gaussian_wm.v62_checkpointing import (  # noqa: E402
    load_e0_codec_checkpoint_v62,
)
from igsw.adaptive_gaussian_wm.v62_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    E0_STAGE,
    E1_STAGE,
    STAGES,
    ObjectTransitionConfigV62,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--codec_checkpoint", default="")
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--siglip_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def build_real_batch(args, device):
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
    per_source = (args.batch + len(dataset.source_names) - 1) // len(
        dataset.source_names
    )
    indices = []
    for source_index in range(len(dataset.source_names)):
        indices.extend(
            dataset.balanced_source_evaluation_indices(source_index, per_source)
        )
    indices = indices[: args.batch]
    cpu_batch = default_collate([dataset[(index, 3)] for index in indices])
    return move_to_device(cpu_batch, device), dataset


def build_model(args, config, device):
    if args.stage == E0_STAGE:
        return TeacherObjectAutoencoderV62(config).to(device)
    checkpoint = load_e0_codec_checkpoint_v62(args.codec_checkpoint, config)
    model = TeacherTransitionOracleV62(config).to(device)
    model.load_codec_state(checkpoint["model"])
    return model


def main():
    args = parse_args()
    device = torch.device("cuda")
    config = ObjectTransitionConfigV62()
    config.validate()
    batch, dataset = build_real_batch(args, device)
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
    observation = teacher(batch)
    if not bool(observation.object_valid.any()):
        raise RuntimeError("v62 verifier sample contains no relation-supported object")
    model = build_model(args, config, device).train()
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.amp == "bf16"
        else nullcontext()
    )
    with amp_context:
        if args.stage == E0_STAGE:
            output = model(observation)
        else:
            output = model(observation, batch["frame_times"], batch["source_index"])
    output["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise RuntimeError("v62 verifier found missing or non-finite gradients")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "git_commit": args.source_revision,
        "data": os.path.abspath(args.data_index),
        "data_contract": dataset.contract_label,
        "historical_checkpoint_used": False,
        "patch_grid_is_object_target": False,
        "rgb_reconstruction_used": False,
        "explicit_action_used": False,
        "language_used": False,
        "teacher_roles": ["DINOv2-L", "SigLIP", "CoTracker relation"],
        "object_valid_fraction": float(observation.object_valid.float().mean()),
        "source_coverage": float(batch["source_index"].unique().numel()),
        "delta_seconds_mean": float(
            (batch["frame_times"][:, 1] - batch["frame_times"][:, 0]).mean()
        ),
        "point_count": float(observation.coordinates.shape[2]),
        "carrier_count": float(config.carrier_count),
        "effect_shape": [config.effect_factors, config.effect_dim],
        "loss": float(output["loss"].detach()),
        "trainable_gradient_tensors": float(len(gradients)),
        "gradient_norm": float(
            torch.stack([value.float().square().sum() for value in gradients])
            .sum()
            .sqrt()
        ),
    }
    if args.stage == E1_STAGE:
        report.update(
            {
                "correct_gain_over_zero": float(
                    output["parts"]["correct_gain_over_zero"]
                ),
                "correct_gain_over_shuffled": float(
                    output["parts"]["correct_gain_over_shuffled"]
                ),
                "matched_shuffle_fraction": float(
                    output["parts"]["matched_shuffle_fraction"]
                ),
                "effect_rms": float(output["parts"]["effect_rms"]),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
