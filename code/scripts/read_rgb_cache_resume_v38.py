#!/usr/bin/env python3
"""Emit null-delimited RGB cache contract values for a safe resume."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", required=True)
    args = parser.parse_args()
    if not os.path.isabs(args.pending) or not os.path.isfile(args.pending):
        raise ValueError("--pending must be an existing absolute path")
    with open(args.pending, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not False:
        raise ValueError("resume settings require an incomplete pending manifest")
    source = manifest["source"]
    sampling = manifest["sampling"]
    rgb = manifest["rgb_cache"]
    values = (
        source["path"],
        ",".join(source["variants"]),
        source["expected_source_fps"],
        ",".join(str(value) for value in sampling["window_lengths"]),
        sampling["sample_stride"],
        rgb["jpeg_quality"],
    )
    for value in values:
        sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")


if __name__ == "__main__":
    main()
