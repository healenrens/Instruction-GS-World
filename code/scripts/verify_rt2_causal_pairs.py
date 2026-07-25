"""Verify arbitrary-frame pair data and future-swap input invariance."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.causal_geometry import project_xyz  # noqa: E402
from igsw.latent_particle_wm.pair_targets import CAUSAL_PAIR_VERSION  # noqa: E402


def tensor_digest(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source")
    parser.add_argument("--legacy_causal")
    parser.add_argument("--spatrack_root", default="/mnt/pfs/public/xuhaoming/SpaTrackerV2")
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    if not files:
        raise ValueError(f"no pair files in {args.data}")
    preprocess_image = None
    if args.source:
        sys.path.insert(0, args.spatrack_root)
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    split_counts = Counter()
    horizon_counts = Counter()
    start_counts = Counter()
    input_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    target_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    future_ends: dict[tuple[str, int], set[int]] = defaultdict(set)
    group_counts: dict[tuple[str, int], int] = defaultdict(int)
    valid_counts = []
    max_reprojection = 0.0
    max_traj0_error = 0.0
    source_rgb_checks: set[tuple[str, int]] = set()
    legacy_checks: set[str] = set()
    current_source_name = None
    current_source = None
    for index, path in enumerate(files, 1):
        pair = torch.load(path, map_location="cpu", weights_only=False)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError(f"pair contract mismatch: {path}")
        start, end, horizon = int(pair["start"]), int(pair["end"]), int(pair["horizon"])
        if not (0 <= start < end <= 12 and horizon == end - start):
            raise ValueError(f"invalid pair metadata: {path}")
        expected = {
            "means": (2304, 3),
            "uv": (2304, 2),
            "K_intr": (3, 3),
            "traj": (horizon + 1, 2304, 3),
            "vis": (horizon + 1, 2304),
            "geom_valid": (2304,),
            "rgb_path": (horizon + 1, int(pair["H"]), int(pair["W"]), 3),
        }
        for key, shape in expected.items():
            if tuple(pair[key].shape) != shape:
                raise ValueError(f"{path}: {key}={tuple(pair[key].shape)} expected={shape}")
        if pair["geometry_input_source"] != f"source_rgb_frame_{start}_only":
            raise ValueError(f"noncausal geometry source: {path}")
        if pair["geometry_target_source"] != "spatrack_full_video_relative_motion":
            raise ValueError(f"unexpected target source: {path}")
        if pair["input_fields"] != ["means", "uv", "K_intr", "rgb_path[0]", "horizon"]:
            raise ValueError(f"unexpected input contract: {path}")
        if pair["target_fields"] != ["traj", "vis", "geom_valid", "rgb_path[1:]"]:
            raise ValueError(f"unexpected target contract: {path}")
        if not torch.isfinite(pair["means"]).all() or not torch.isfinite(pair["traj"]).all():
            raise ValueError(f"nonfinite geometry: {path}")
        valid_count = int(pair["geom_valid"].sum())
        if valid_count != int(pair["geometry_valid_count"]):
            raise ValueError(f"geometry valid count mismatch: {path}")
        reprojection = (project_xyz(pair["means"], pair["K_intr"]) - pair["uv"]).norm(dim=-1).max()
        traj0_error = (pair["traj"][0] - pair["means"]).abs().max()
        max_reprojection = max(max_reprojection, float(reprojection))
        max_traj0_error = max(max_traj0_error, float(traj0_error))
        key = (pair["source_name"], start)
        input_hashes[key].add(
            tensor_digest([pair["means"], pair["uv"], pair["K_intr"], pair["rgb_path"][0]])
        )
        target_hashes[key].add(
            tensor_digest([pair["traj"][-1], pair["vis"][-1], pair["rgb_path"][-1]])
        )
        future_ends[key].add(end)
        group_counts[key] += 1
        split_counts[pair["split"]] += 1
        horizon_counts[horizon] += 1
        start_counts[start] += 1
        valid_counts.append(valid_count)
        if args.source and key not in source_rgb_checks:
            if pair["source_name"] != current_source_name:
                source_path = os.path.join(args.source, pair["source_name"])
                current_source = torch.load(source_path, map_location="cpu", weights_only=False)
                current_source_name = pair["source_name"]
            raw = current_source["gt_rgb"][start].permute(2, 0, 1).float()
            expected_rgb = (
                preprocess_image(raw)
                .permute(1, 2, 0)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            if not torch.equal(pair["rgb_path"][0], expected_rgb):
                raise ValueError(f"current RGB does not match standalone source frame: {path}")
            source_rgb_checks.add(key)
        if args.legacy_causal and start == 0 and pair["source_name"] not in legacy_checks:
            legacy_path = os.path.join(args.legacy_causal, pair["source_name"])
            legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
            for field, legacy_value in {
                "means": legacy["means"],
                "uv": legacy["uv"],
                "K_intr": legacy["K_intr"],
                "rgb_path[0]": legacy["gt_rgb"][0],
            }.items():
                value = pair["rgb_path"][0] if field == "rgb_path[0]" else pair[field]
                if not torch.equal(value, legacy_value):
                    raise ValueError(f"t=0 legacy mismatch for {field}: {path}")
            legacy_checks.add(pair["source_name"])
        if index % 500 == 0:
            print(f"[verify-pairs] {index}/{len(files)}", flush=True)

    inconsistent_inputs = [key for key, hashes in input_hashes.items() if len(hashes) != 1]
    if inconsistent_inputs:
        raise ValueError(f"future-conditioned inputs for {inconsistent_inputs[:5]}")
    comparable_groups = sum(count > 1 for count in group_counts.values())
    multiple_future_groups = sum(
        group_counts[key] > 1 and len(future_ends[key]) > 1
        for key in future_ends
    )
    multi_future_groups = sum(
        group_counts[key] > 1 and len(target_hashes[key]) > 1
        for key in target_hashes
    )
    if comparable_groups == 0:
        raise ValueError("dataset has no same-current, multiple-future comparison groups")
    if multiple_future_groups != comparable_groups:
        raise ValueError("comparable groups do not contain distinct future indices")
    report = {
        "status": "ok",
        "data": os.path.abspath(args.data),
        "pair_version": CAUSAL_PAIR_VERSION,
        "count": len(files),
        "split_counts": dict(split_counts),
        "horizon_counts": dict(sorted(horizon_counts.items())),
        "start_counts": dict(sorted(start_counts.items())),
        "input_identity_groups": len(input_hashes),
        "future_invariant_observation_groups": len(input_hashes) - len(inconsistent_inputs),
        "groups_with_multiple_future_indices": multiple_future_groups,
        "groups_with_distinct_future_targets": multi_future_groups,
        "comparable_groups": comparable_groups,
        "geometry_valid_min": min(valid_counts),
        "geometry_valid_mean": sum(valid_counts) / len(valid_counts),
        "geometry_valid_max": max(valid_counts),
        "max_reprojection_px": max_reprojection,
        "max_traj0_abs_error": max_traj0_error,
        "standalone_source_rgb_checks": len(source_rgb_checks),
        "exact_t0_legacy_checks": len(legacy_checks),
    }
    if max_reprojection > 1e-3 or max_traj0_error != 0.0:
        raise ValueError(f"geometry invariant failed: {report}")
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
