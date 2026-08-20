#!/usr/bin/env python3
"""Exercise the exact distributed v53 decode frontier before GPU training."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.video_file_decoder import (  # noqa: E402
    VIDEO_DECODER_CONTRACT,
)


DECODE_FRONTIER_CONTRACT = "multisource_v53_distributed_decode_frontier_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--workers_per_rank", type=int, required=True)
    parser.add_argument("--prefetch_factor", type=int, required=True)
    parser.add_argument("--chunk_lengths", default="3,4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def rank_report_path(output: str, rank: int) -> str:
    return f"{output}.rank{rank}.json"


def verify_rank(rank: int, args: argparse.Namespace) -> None:
    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        0,
        args.seed,
    )
    sampler = build_training_sampler(
        dataset,
        args.world_size,
        rank,
        args.seed,
        args.batch_size,
        1,
    )
    sampler.set_epoch(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers_per_rank,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor,
    )
    batches = args.workers_per_rank * args.prefetch_factor + 2
    source_counts = {name: 0 for name in dataset.source_names}
    observed_batches = 0
    for batch in loader:
        require(
            batch["video_rgb"].shape[0] == args.batch_size,
            f"rank {rank} decode batch has the wrong size",
        )
        require(
            batch["video_rgb"].dtype == torch.uint8,
            f"rank {rank} decode batch changed RGB dtype",
        )
        for source_index in batch["source_index"].tolist():
            source_counts[dataset.source_names[int(source_index)]] += 1
        observed_batches += 1
        if observed_batches == batches:
            break
    require(observed_batches == batches, f"rank {rank} decode frontier was truncated")
    write_json(
        rank_report_path(args.output, rank),
        {
            "rank": rank,
            "batches": observed_batches,
            "samples": observed_batches * args.batch_size,
            "source_counts": source_counts,
        },
    )


def main() -> None:
    args = parse_args()
    args.data_index = os.path.abspath(args.data_index)
    args.output = os.path.abspath(args.output)
    dimensions = (
        args.world_size,
        args.batch_size,
        args.workers_per_rank,
        args.prefetch_factor,
    )
    require(min(dimensions) > 0, "decode frontier dimensions must be positive")
    mp.spawn(verify_rank, args=(args,), nprocs=args.world_size, join=True)
    rank_reports = []
    for rank in range(args.world_size):
        with open(rank_report_path(args.output, rank), encoding="utf-8") as handle:
            rank_reports.append(json.load(handle))
    samples = sum(report["samples"] for report in rank_reports)
    report = {
        "status": "passed",
        "contract": DECODE_FRONTIER_CONTRACT,
        "decoder_contract": VIDEO_DECODER_CONTRACT,
        "data_index": args.data_index,
        "world_size": args.world_size,
        "batch_size": args.batch_size,
        "workers_per_rank": args.workers_per_rank,
        "prefetch_factor": args.prefetch_factor,
        "sampler_epoch": args.seed,
        "batches_per_rank": args.workers_per_rank * args.prefetch_factor + 2,
        "decoded_samples": samples,
        "rank_reports": rank_reports,
    }
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
