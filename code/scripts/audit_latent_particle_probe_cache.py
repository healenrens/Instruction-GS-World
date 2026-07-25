"""Produce split/horizon quality statistics for the strict-causal probe cache."""
from __future__ import annotations

import argparse
import json
import os

import torch


def summarize(cache: dict, keep: torch.Tensor) -> dict:
    records = cache["records"]
    clip_index = records["clip_index"][keep]
    valid = records["valid"][keep]
    visible = records["visible"][keep]
    motion_valid = records["motion_valid"][keep]
    target = records["target"][keep]
    active = cache["clips"]["active"][clip_index]
    mover = (target[..., :2].norm(dim=-1) > 0.01) & motion_valid
    active_valid = active & motion_valid
    active_mover = active & mover
    plausible_denominator = (visible & valid).sum().clamp_min(1)
    flow = target[..., :2].norm(dim=-1)[motion_valid]
    return {
        "records": int(keep.sum()),
        "geometry_valid_fraction": float(valid.float().mean()),
        "visible_fraction_of_geometry_valid": float(visible.sum() / valid.sum().clamp_min(1)),
        "motion_valid_fraction_of_geometry_valid": float(
            motion_valid.sum() / valid.sum().clamp_min(1)
        ),
        "tracker_jump_rejection_fraction": float(
            ((visible & valid) & ~motion_valid).sum() / plausible_denominator
        ),
        "mover_fraction_of_motion_valid": float(
            mover.sum() / motion_valid.sum().clamp_min(1)
        ),
        "active_fraction": float(active.float().mean()),
        "active_mover_recall": float(active_mover.sum() / mover.sum().clamp_min(1)),
        "active_mover_precision": float(
            active_mover.sum() / active_valid.sum().clamp_min(1)
        ),
        "flow_norm_quantiles": {
            "p50": float(flow.quantile(0.50)),
            "p90": float(flow.quantile(0.90)),
            "p99": float(flow.quantile(0.99)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    record_clip = cache["records"]["clip_index"]
    clip_split = torch.tensor(cache["clips"]["split"], dtype=torch.long)
    record_split = clip_split[record_clip]
    result = {
        "status": "ok",
        "cache": os.path.abspath(args.cache),
        "version": cache["version"],
        "causal_contract": {
            "input": "single-frame VGGT fixed grid plus current RGB descriptors",
            "active_mask": "current-frame GPSToken entropy only",
            "posterior_targets": "future RGB plus full-video SpaTracker pseudo-labels",
            "future_conditioned_input_fields": [],
        },
        "splits": {},
    }
    for split_id, split in enumerate(("train", "heldseed", "heldtask")):
        split_keep = record_split == split_id
        split_result = {"all": summarize(cache, split_keep), "horizons": {}}
        for horizon in cache["records"]["horizon"].unique().tolist():
            keep = split_keep & (cache["records"]["horizon"] == horizon)
            split_result["horizons"][str(int(horizon))] = summarize(cache, keep)
        result["splits"][split] = split_result
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
