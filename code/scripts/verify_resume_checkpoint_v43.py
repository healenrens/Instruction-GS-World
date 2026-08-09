#!/usr/bin/env python3
"""Validate a v43 full-state checkpoint before strict single-node resume."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v43_resume_compatibility import (  # noqa: E402
    validate_v43_resume_commit,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--resume_compatible_git_commit", default="")
    parser.add_argument("--world_size", type=int, required=True)
    args = parser.parse_args()
    for name in ("out", "checkpoint"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    manifest_path = os.path.join(args.out, "checkpoint_manifest.json")
    latest_path = os.path.join(args.out, "latest.pt")
    run_id_path = os.path.join(args.out, "wandb_run_id.txt")
    require(os.path.isfile(manifest_path), "checkpoint manifest is missing")
    require(os.path.lexists(latest_path), "latest.pt is missing")
    require(os.path.isfile(run_id_path), "W&B run id is missing")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    checkpoint_path = os.path.realpath(args.checkpoint)
    require(os.path.realpath(latest_path) == checkpoint_path, "latest.pt differs")
    require(
        os.path.realpath(manifest["checkpoint_path"]) == checkpoint_path,
        "manifest checkpoint differs",
    )
    require(os.path.dirname(checkpoint_path) == os.path.realpath(args.out),
            "checkpoint is outside OUT")
    require(os.path.getsize(checkpoint_path) == int(manifest["size_bytes"]),
            "checkpoint size differs")
    require(int(manifest["checkpoint_version"]) == 43, "manifest is not v43")
    require(int(manifest["world_size"]) == args.world_size, "world size differs")
    require(manifest["phase"] == "representation", "checkpoint phase differs")
    with open(run_id_path, encoding="utf-8") as handle:
        run_id = handle.read().strip()
    require(bool(run_id), "W&B run id is empty")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_v43_resume_commit(checkpoint, args)
    for name in (
        "checkpoint_version",
        "git_commit",
        "world_size",
        "phase",
        "phase_step",
        "global_step",
    ):
        require(checkpoint.get(name) == manifest.get(name),
                f"checkpoint {name} differs from manifest")
    require(checkpoint.get("parallelism") == "ddp_full_state_dict",
            "checkpoint parallelism differs")
    require(len(checkpoint.get("rng_states", ())) == args.world_size,
            "checkpoint RNG state count differs")
    require(checkpoint["config"].get("architecture") == "object_region_memory_v1",
            "checkpoint architecture is not v43")
    require(checkpoint["config"].get("object_region_memory") is True,
            "checkpoint has no Region Memory")
    require(int(checkpoint["model"]["curriculum_step"]) == int(checkpoint["global_step"]),
            "checkpoint curriculum and global steps differ")
    for section in ("model", "optimizer", "scheduler", "config", "args"):
        require(section in checkpoint, f"checkpoint section is missing: {section}")
    print(json.dumps({
        "status": "passed",
        "contract": "strict_resume_v43",
        "checkpoint": checkpoint_path,
        "phase_step": int(manifest["phase_step"]),
        "global_step": int(manifest["global_step"]),
        "world_size": int(manifest["world_size"]),
        "git_commit": manifest["git_commit"],
        "current_git_commit": args.git_commit,
        "resume_compatibility_reason": args.resume_compatibility_reason,
        "wandb_run_id": run_id,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
