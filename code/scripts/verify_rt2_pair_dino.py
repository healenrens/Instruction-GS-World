"""Verify DINO sidecars against strict-causal pair metadata."""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    pair_paths = sorted(glob.glob(os.path.join(args.pairs, "*.pt")))
    sidecar_paths = sorted(glob.glob(os.path.join(args.dino, "*.pt")))
    pair_names = {os.path.basename(path) for path in pair_paths}
    sidecar_names = {os.path.basename(path) for path in sidecar_paths}
    if pair_names != sidecar_names:
        raise ValueError(
            f"pair/sidecar names differ: missing={sorted(pair_names - sidecar_names)[:5]} "
            f"extra={sorted(sidecar_names - pair_names)[:5]}"
        )

    projection_hashes = set()
    models = set()
    feature_dims = set()
    norm_min = float("inf")
    norm_max = 0.0
    for index, pair_path in enumerate(pair_paths, 1):
        name = os.path.basename(pair_path)
        pair = torch.load(pair_path, map_location="cpu", weights_only=False)
        sidecar = torch.load(
            os.path.join(args.dino, name),
            map_location="cpu",
            weights_only=False,
        )
        expected_metadata = (
            pair["source_name"],
            int(pair["start"]),
            int(pair["end"]),
            int(pair["horizon"]),
        )
        actual_metadata = (
            sidecar["source_name"],
            int(sidecar["start"]),
            int(sidecar["end"]),
            int(sidecar["horizon"]),
        )
        if actual_metadata != expected_metadata:
            raise ValueError(f"metadata mismatch: {name}")
        feature_dim = int(sidecar["feature_dim"])
        expected_shape = (
            int(sidecar["image_size"]) // 14,
            int(sidecar["image_size"]) // 14,
            feature_dim,
        )
        for key in ("dino0", "dino1"):
            feature = sidecar[key].float()
            if tuple(feature.shape) != expected_shape or not torch.isfinite(feature).all():
                raise ValueError(f"invalid {key}: {name}")
            norms = feature.norm(dim=-1)
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
        projection_hashes.add(sidecar["projection_sha256"])
        models.add(sidecar["model"])
        feature_dims.add(feature_dim)
        if index % 500 == 0:
            print(f"[verify-pair-dino] {index}/{len(pair_paths)}", flush=True)
    if len(projection_hashes) != 1 or len(models) != 1 or len(feature_dims) != 1:
        raise ValueError("DINO cache contract is not uniform")
    if norm_min < 0.98 or norm_max > 1.02:
        raise ValueError("projected DINO features are not unit normalized")

    report = {
        "status": "ok",
        "pairs": os.path.abspath(args.pairs),
        "dino": os.path.abspath(args.dino),
        "count": len(pair_paths),
        "model": next(iter(models)),
        "feature_dim": next(iter(feature_dims)),
        "projection_sha256": next(iter(projection_hashes)),
        "feature_norm_min": norm_min,
        "feature_norm_max": norm_max,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    temporary = f"{args.report}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
