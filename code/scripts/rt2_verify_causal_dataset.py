"""Verify the strict-causal RoboTwin VLA dataset before cache prep or training."""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.causal_geometry import CAUSAL_GEOMETRY_VERSION, project_xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rt2_causal_v1")
    ap.add_argument("--source", default="data/rt2_joint_src")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    data_files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    source_names = {os.path.basename(p) for p in glob.glob(os.path.join(a.source, "*.pt"))}
    data_names = {os.path.basename(p) for p in data_files}
    if data_names != source_names:
        raise ValueError(f"dataset/source names differ: missing={len(source_names - data_names)} "
                         f"extra={len(data_names - source_names)}")

    splits = Counter()
    targets = Counter()
    valid_counts = []
    max_reprojection = 0.0
    max_traj0_error = 0.0
    for index, path in enumerate(data_files, 1):
        clip = torch.load(path, map_location="cpu", weights_only=False)
        if clip.get("causal_geometry_version") != CAUSAL_GEOMETRY_VERSION:
            raise ValueError(f"geometry contract mismatch: {path}")
        if clip.get("geometry_input_source") != "current_head_rgb_only":
            raise ValueError(f"noncausal input source: {path}")
        expected = {
            "means": (2304, 3), "uv": (2304, 2), "traj": (13, 2304, 3),
            "geom_valid": (2304,), "gt_rgb": (1, 392, 518, 3),
            "left_rgb": (240, 320, 3), "right_rgb": (240, 320, 3),
            "dq": (50, 14), "anchor": (14,),
        }
        for key, shape in expected.items():
            if tuple(clip[key].shape) != shape:
                raise ValueError(f"{path}: {key} shape {tuple(clip[key].shape)} != {shape}")
        if not bool(torch.isfinite(clip["means"]).all()) or not bool(torch.isfinite(clip["traj"]).all()):
            raise ValueError(f"nonfinite geometry: {path}")
        target = clip["geometry_target_source"]
        valid = int(clip["geom_valid"].sum())
        if target == "none" and valid:
            raise ValueError(f"action-only clip has geometry labels: {path}")
        reproj = (project_xyz(clip["means"], clip["K_intr"]) - clip["uv"]).norm(dim=-1).max()
        traj0 = (clip["traj"][0] - clip["means"]).abs().max()
        max_reprojection = max(max_reprojection, float(reproj))
        max_traj0_error = max(max_traj0_error, float(traj0))
        splits[clip["split"]] += 1
        targets[target] += 1
        valid_counts.append(valid)
        if index % 2000 == 0:
            print(f"[causal-verify] {index}/{len(data_files)}", flush=True)

    report = {
        "status": "ok", "data": os.path.abspath(a.data), "source": os.path.abspath(a.source),
        "causal_geometry_version": CAUSAL_GEOMETRY_VERSION, "count": len(data_files),
        "splits": dict(splits), "target_sources": dict(targets),
        "geometry_valid_min": min(valid_counts), "geometry_valid_max": max(valid_counts),
        "geometry_valid_mean": sum(valid_counts) / len(valid_counts),
        "max_reprojection_px": max_reprojection, "max_traj0_abs_error": max_traj0_error,
    }
    if max_reprojection > 1e-3 or max_traj0_error != 0.0:
        raise ValueError(f"geometry invariant failed: {report}")
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        with open(a.report, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
