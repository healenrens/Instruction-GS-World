#!/usr/bin/env python3
"""Emit null-delimited launcher settings from a trusted v42 checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if not os.path.isabs(args.checkpoint) or not os.path.isfile(args.checkpoint):
        raise ValueError("--checkpoint must be an existing absolute path")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint.get("checkpoint_version") != 42:
        raise ValueError("resume launch settings require a v42 checkpoint")
    saved = checkpoint["args"]
    stage = saved["training_stage"]
    if stage not in ("representation", "posterior", "prior"):
        raise ValueError(f"invalid checkpoint training stage: {stage}")
    steps = (
        saved["representation_steps"]
        if stage == "representation"
        else saved["joint_steps"]
    )
    values = (
        stage,
        saved["data"],
        saved["gate_report"],
        saved.get("teacher_sidecar", ""),
        saved.get("representation_gate_report", ""),
        saved.get("posterior_gate_report", ""),
        saved["seed"],
        saved["batch"],
        saved["grad_accum"],
        saved["target_global_batch"],
        saved["workers"],
        saved["jit_dino_batch"],
        steps,
        saved["core_lr"],
        saved["action_lr"],
        saved["save_every"],
        saved["recovery_every"],
        saved["log_every"],
        checkpoint["world_size"],
        saved["wandb_mode"],
        saved["wandb_project"],
        saved["wandb_entity"],
        saved["wandb_name"],
        saved["wandb_group"],
        saved["wandb_tags"],
        saved["wandb_dir"],
        saved["history_span_frames"],
        saved["short_horizon_frames"],
        saved["goal_query_seconds"],
        saved["goal_tail_guard_frames"],
        saved["goal_probe_frames"],
        saved["goal_stability_threshold"],
        saved["goal_rollout_weight"],
        saved["path_consistency_weight"],
        saved["max_train_items"],
    )
    for value in values:
        sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")


if __name__ == "__main__":
    main()
