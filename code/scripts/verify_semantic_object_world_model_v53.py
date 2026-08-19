"""One-batch CUDA contract verification for v53 and its selected stage."""

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

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.semantic_object_world_model_v53 import (  # noqa: E402
    SemanticObjectLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v53_checkpointing import validate_init_from  # noqa: E402
from igsw.adaptive_gaussian_wm.v53_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STAGES,
    SemanticObjectWorldModelConfig,
)
from igsw.adaptive_gaussian_wm.v53_training_loop import select_stage_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init_from", default="")
    parser.add_argument("--chunk_length", type=int, default=3)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--dino_frame_batch", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    for name in ("data_index", "dino_checkpoint", "output", "init_from"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    require(torch.cuda.is_available(), "v53 verifier requires one visible CUDA device")
    require(os.path.isfile(args.data_index), "v53 multisource index is missing")
    require(
        os.path.isfile(args.dino_checkpoint), "v53 frozen DINO checkpoint is missing"
    )
    if args.stage == "dynamics":
        require(
            bool(args.init_from), "v53 dynamics verifier requires tokenizer init_from"
        )
        require(os.path.isfile(args.init_from), "v53 tokenizer init_from is missing")
    elif args.init_from:
        raise ValueError("v53 tokenizer verifier does not accept init_from")
    config = SemanticObjectWorldModelConfig()
    config.validate()
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        str(args.chunk_length),
        args.temporal_step_ms,
        max_items=8,
        seed=173,
    )
    batch = default_collate(
        [dataset[(0, args.chunk_length)], dataset[(1, args.chunk_length)]]
    )
    device = torch.device("cuda:0")
    batch = select_stage_frames(move_to_device(batch, device), args.stage)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    features = dino(batch)
    require(
        not features.patches.requires_grad,
        "frozen DINO features unexpectedly require grad",
    )
    model = SemanticObjectLatentWorldModel(config).to(device)
    if args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", weights_only=False, mmap=True
        )
        validate_init_from(checkpoint, config)
        model.load_state_dict(checkpoint["model"], strict=True)
    model.configure_stage(args.stage)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
        )
    require(bool(torch.isfinite(output["loss"])), "v53 verifier loss is not finite")
    output["loss"].backward()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradients = [
        parameter.grad for _, parameter in trainable if parameter.grad is not None
    ]
    require(bool(gradients), "v53 selected stage produced no gradients")
    require(
        all(bool(torch.isfinite(value).all()) for value in gradients),
        "v53 has non-finite gradients",
    )
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    require(
        not frozen_with_grad,
        f"v53 frozen modules received gradients: {frozen_with_grad}",
    )
    with torch.no_grad():
        encoding = model.tokenizer(
            features.patches, features.coordinates, features.valid
        )
        swapped = features.patches.clone()
        swapped[:, -1] = swapped.flip(0)[:, -1]
        swapped_encoding = model.tokenizer(
            swapped, features.coordinates, features.valid
        )
        source_difference = (
            (encoding.slots[:, 0] - swapped_encoding.slots[:, 0]).abs().max()
        )
        target_difference = (
            (encoding.slots[:, -1] - swapped_encoding.slots[:, -1]).abs().max()
        )
    require(float(source_difference) < 1e-6, "v53 source state reads future content")
    require(
        float(target_difference) > 1e-5,
        "v53 target state ignores changed future content",
    )
    parts = {
        name: float(value.detach().float()) for name, value in output["parts"].items()
    }
    if args.stage == "tokenizer":
        require(
            parts["object_effective_slot_count"] > 1.0,
            "v53 tokenizer collapsed in verifier",
        )
    else:
        require(
            "dynamics_correct_gain_over_zero" in parts,
            "v53 dynamics diagnostics missing",
        )
        require(
            "dynamics_correct_gain_over_shuffled" in parts,
            "v53 effect diagnostics missing",
        )
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": args.stage,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "source_count": len(dataset.source_names),
        "source_names": list(dataset.source_names),
        "object_slots": config.object_slots,
        "scene_slots": config.scene_slots,
        "action_dim": config.action_dim,
        "frozen_dino": True,
        "point_tracker_used": False,
        "rgb_reconstruction_used": False,
        "instance_segmentation_used": False,
        "language_used": False,
        "explicit_action_used": False,
        "historical_checkpoint_used": False,
        "source_future_swap_max_difference": float(source_difference),
        "target_future_swap_max_difference": float(target_difference),
        "trainable_parameter_tensors": len(trainable),
        "gradient_parameter_tensors": len(gradients),
        "loss": float(output["loss"].detach().float()),
        "metrics": parts,
    }
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
