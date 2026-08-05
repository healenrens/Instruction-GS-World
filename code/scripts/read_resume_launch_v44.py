#!/usr/bin/env python3
"""Emit strict v44 launch settings as NUL-delimited values."""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("checkpoint_version") != 44:
        raise ValueError("resume launch source is not v44")
    saved = checkpoint["args"]
    if saved.get("architecture") != "object_region_dual_encoder_v1":
        raise ValueError("resume launch source architecture differs")
    names = (
        "data", "gate_report", "teacher_sidecar", "seed", "batch",
        "grad_accum", "target_global_batch", "workers", "jit_dino_batch",
        "max_train_items", "history_span_frames", "short_horizon_frames",
        "goal_query_seconds", "goal_tail_guard_frames", "goal_probe_frames",
        "goal_stability_threshold", "goal_rollout_weight",
        "path_consistency_weight", "save_every", "recovery_every", "log_every",
        "wandb_mode", "wandb_project", "wandb_entity", "wandb_name",
        "wandb_group", "wandb_tags", "wandb_dir", "video_vae_model",
        "video_vae_contract", "video_vae_pythonpath", "video_vae_short_side",
        "video_vae_clip_frames", "video_vae_batch",
    )
    values = [str(saved.get(name, "")) for name in names]
    values.append(str(checkpoint["world_size"]))
    sys.stdout.buffer.write("\0".join(values).encode() + b"\0")


if __name__ == "__main__":
    main()
