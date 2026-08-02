#!/usr/bin/env python3
"""Validate a v41 full-state checkpoint before strict single-node resume."""

from __future__ import annotations

import argparse
import json
import os

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument(
        "--stage", choices=("representation", "posterior", "prior"), required=True
    )
    args = parser.parse_args()
    require(os.path.isabs(args.out), "--out must be absolute")
    require(os.path.isabs(args.checkpoint), "--checkpoint must be absolute")
    require(args.world_size > 0, "--world_size must be positive")
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
    require(
        os.path.dirname(checkpoint_path) == os.path.realpath(args.out),
        "checkpoint is outside OUT",
    )
    require(os.path.isfile(checkpoint_path), "checkpoint target is missing")
    require(
        os.path.getsize(checkpoint_path) == int(manifest["size_bytes"]),
        "checkpoint size differs",
    )
    require(int(manifest["checkpoint_version"]) == 41, "manifest is not v41")
    require(manifest["git_commit"] == args.git_commit, "checkpoint commit differs")
    require(int(manifest["world_size"]) == args.world_size, "world size differs")
    require(manifest["phase"] == args.stage, "checkpoint stage differs")
    with open(run_id_path, encoding="utf-8") as handle:
        run_id = handle.read().strip()
    require(bool(run_id), "W&B run id is empty")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    for name in (
        "checkpoint_version",
        "git_commit",
        "world_size",
        "phase",
        "phase_step",
        "global_step",
    ):
        require(
            checkpoint.get(name) == manifest.get(name),
            f"checkpoint {name} differs from manifest",
        )
    require(
        checkpoint.get("parallelism") == "ddp_full_state_dict",
        "checkpoint parallelism differs",
    )
    require(
        len(checkpoint.get("rng_states", ())) == args.world_size,
        "checkpoint RNG state count differs",
    )
    for section in ("model", "optimizer", "scheduler", "config", "args"):
        require(section in checkpoint, f"checkpoint section is missing: {section}")
    saved = checkpoint["args"]
    require(saved.get("feature_source") == "jit", "checkpoint is not JIT DINO")
    require(
        saved.get("temporal_contract") == "dynamic_dual_horizon_v1",
        "checkpoint temporal contract differs",
    )
    require(
        saved.get("architecture") == "object_memory_v3",
        "checkpoint architecture is not object_memory_v3",
    )
    report = {
        "status": "passed",
        "contract": "strict_resume_v41",
        "checkpoint": checkpoint_path,
        "checkpoint_kind": manifest["checkpoint_kind"],
        "phase": manifest["phase"],
        "phase_step": int(manifest["phase_step"]),
        "global_step": int(manifest["global_step"]),
        "world_size": int(manifest["world_size"]),
        "git_commit": manifest["git_commit"],
        "wandb_run_id": run_id,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
