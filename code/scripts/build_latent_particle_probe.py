"""Build the strict-causal latent particle probe cache on the remote data root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.probe_data import (  # noqa: E402
    ProbeRoots,
    build_probe_cache,
    fixed_grid_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pfs/public/xuhaoming/instruct_gs_world")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_clips", type=int, default=1200)
    parser.add_argument("--heldseed_clips", type=int, default=300)
    parser.add_argument("--heldtask_clips", type=int, default=300)
    parser.add_argument("--particle_side", type=int, default=16)
    parser.add_argument("--active_count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    data = os.path.join(args.root, "data")
    roots = ProbeRoots(
        plan=os.path.join(data, "rt2_win", "window_plan.json"),
        causal=os.path.join(data, "rt2_causal_v1"),
        source=os.path.join(data, "rt2_joint_src"),
        tracked=os.path.join(data, "rt2_joint"),
    )
    limits = {
        "train": args.train_clips,
        "heldseed": args.heldseed_clips,
        "heldtask": args.heldtask_clips,
    }
    cache = build_probe_cache(
        roots,
        args.out,
        limits,
        particle_side=args.particle_side,
        active_count=args.active_count,
        seed=args.seed,
    )
    clip_splits = cache["clips"]["split"]
    split_names = {0: "train", 1: "heldseed", 2: "heldtask"}
    split_counts = Counter(split_names[int(value)] for value in clip_splits)
    record_clip = cache["records"]["clip_index"]
    record_splits = Counter(split_names[int(clip_splits[int(index)])] for index in record_clip)
    valid = cache["records"]["valid"]
    motion_valid = cache["records"]["motion_valid"]
    target = cache["records"]["target"]
    mover = (target[..., :2].norm(dim=-1) > 0.01) & motion_valid
    active = cache["clips"]["active"][record_clip]
    active_recall = float((active & mover).sum() / mover.sum().clamp_min(1))
    fixed = fixed_grid_indices(48, args.particle_side).numpy().tobytes()
    summary = {
        "status": "ok",
        "cache": os.path.abspath(args.out),
        "version": cache["version"],
        "clip_counts": dict(split_counts),
        "record_counts": dict(record_splits),
        "task_counts": dict(Counter(cache["clips"]["task"])),
        "particles": int(cache["clips"]["state"].shape[1]),
        "state_dim": int(cache["clips"]["state"].shape[2]),
        "valid_fraction": float(valid.float().mean()),
        "motion_valid_fraction": float(motion_valid.float().mean()),
        "mover_fraction_of_motion_valid": float(mover.sum() / motion_valid.sum().clamp_min(1)),
        "current_only_active_mover_recall": active_recall,
        "fixed_indices_sha256": hashlib.sha256(fixed).hexdigest(),
        "future_conditioned_input_fields": [],
        "target_only_fields": [
            "target",
            "valid",
            "visible",
            "motion_valid",
            "target_xyz",
        ],
    }
    summary_path = os.path.splitext(args.out)[0] + ".summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
