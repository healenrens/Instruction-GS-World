"""Audit a completed visual-sequence cache against its source file inventory."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
import sys

import torch
from torchvision.io import ImageReadMode, decode_jpeg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    CONTROL_HZ,
    EXPECTED_FRAME_COUNT,
    SEQUENCE_CACHE_VERSION,
)


def split_from_name(name: str) -> str:
    for split in ("heldseed", "heldtask", "train"):
        if name.endswith(f"_{split}.pt"):
            return split
    raise ValueError(f"cannot infer split from cache name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--finite_stride", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.finite_stride < 1:
        raise ValueError("finite stride must be positive")
    source_paths = sorted(glob.glob(os.path.join(args.source, "*.pt")))
    cache_paths = sorted(glob.glob(os.path.join(args.cache, "*.pt")))
    source_by_name = {
        os.path.basename(path): path
        for path in source_paths
    }
    source_names = {os.path.basename(path) for path in source_paths}
    cache_names = {os.path.basename(path) for path in cache_paths}
    missing = sorted(source_names - cache_names)
    extra = sorted(cache_names - source_names)
    if missing or extra:
        raise ValueError(
            f"cache inventory mismatch: missing={len(missing)} extra={len(extra)}"
        )

    splits = Counter()
    projection_hashes = set()
    tensor_shapes = set()
    rgb_shapes = set()
    total_bytes = 0
    finite_checked = 0
    jpeg_checked = 0
    source_metadata_checked = 0
    forbidden = {
        "instruction",
        "condition_feature",
        "condition_tokens",
        "task",
        "task_index",
        "language",
    }
    for index, path in enumerate(cache_paths):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        present = sorted(forbidden.intersection(cache))
        if present:
            raise ValueError(f"semantic fields in {path}: {present}")
        if cache.get("sequence_version") != SEQUENCE_CACHE_VERSION:
            raise ValueError(f"sequence version mismatch: {path}")
        if cache.get("source_name") != os.path.basename(path):
            raise ValueError(f"source identity mismatch: {path}")
        if cache.get("split") != split_from_name(os.path.basename(path)):
            raise ValueError(f"split metadata mismatch: {path}")
        if abs(float(cache.get("control_hz", 0.0)) - CONTROL_HZ) > 1e-9:
            raise ValueError(f"control frequency mismatch: {path}")
        controls = cache.get("frame_control_indices")
        dino = cache.get("dino")
        rgb = cache.get("rgb")
        if (
            not torch.is_tensor(controls)
            or len(controls) != EXPECTED_FRAME_COUNT
            or not bool((controls[1:] > controls[:-1]).all())
        ):
            raise ValueError(f"invalid control times: {path}")
        if (
            not torch.is_tensor(dino)
            or dino.ndim != 4
            or len(dino) != EXPECTED_FRAME_COUNT
            or dino.shape[-1] != int(cache.get("feature_dim", -1))
        ):
            raise ValueError(f"invalid DINO tensor: {path}")
        if (
            not isinstance(rgb, dict)
            or len(rgb.get("jpeg_frames", ())) != EXPECTED_FRAME_COUNT
        ):
            raise ValueError(f"invalid RGB cache: {path}")
        if index % args.finite_stride == 0:
            if not bool(torch.isfinite(dino.float()).all()):
                raise ValueError(f"non-finite DINO values: {path}")
            decoded = decode_jpeg(
                rgb["jpeg_frames"][0],
                mode=ImageReadMode.RGB,
            )
            expected_rgb = (
                3,
                int(rgb["content_height"]),
                int(rgb["content_width"]),
            )
            if tuple(decoded.shape) != expected_rgb:
                raise ValueError(f"decoded RGB shape mismatch: {path}")
            source = torch.load(
                source_by_name[os.path.basename(path)],
                map_location="cpu",
                weights_only=False,
            )
            source_rgb = source.get("gt_rgb")
            if (
                int(source["s"]) != int(cache["source_start"])
                or int(source["win"]) != int(cache["source_window"])
                or source["split"] != cache["split"]
                or not torch.is_tensor(source_rgb)
                or source_rgb.dtype != torch.uint8
                or len(source_rgb) != EXPECTED_FRAME_COUNT
            ):
                raise ValueError(f"source metadata mismatch: {path}")
            finite_checked += 1
            jpeg_checked += 1
            source_metadata_checked += 1
        splits[cache["split"]] += 1
        projection_hashes.add(cache["projection_sha256"])
        tensor_shapes.add(tuple(dino.shape))
        rgb_shapes.add(
            (
                int(rgb["content_height"]),
                int(rgb["content_width"]),
                int(rgb["padded_height"]),
                int(rgb["padded_width"]),
            )
        )
        total_bytes += os.path.getsize(path)
    if len(projection_hashes) != 1:
        raise ValueError("sequence cache contains multiple DINO projections")
    report = {
        "status": "passed",
        "source": os.path.abspath(args.source),
        "cache": os.path.abspath(args.cache),
        "files": len(cache_paths),
        "split_counts": dict(sorted(splits.items())),
        "total_bytes": total_bytes,
        "dino_shapes": [list(shape) for shape in sorted(tensor_shapes)],
        "rgb_shapes": [list(shape) for shape in sorted(rgb_shapes)],
        "projection_sha256": next(iter(projection_hashes)),
        "semantic_fields_present": False,
        "finite_files_checked": finite_checked,
        "jpeg_files_checked": jpeg_checked,
        "source_metadata_files_checked": source_metadata_checked,
        "missing": 0,
        "extra": 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
