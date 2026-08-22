"""Admission contract for the multisource video decode frontier."""

from __future__ import annotations

import json
import os

from .video_file_decoder import VIDEO_DECODER_CONTRACT

DECODE_FRONTIER_CONTRACT = "multisource_v53_distributed_decode_frontier_v1"


def audit_decode_frontier(
    path: str,
    data_index: str,
    seed: int,
    source_names: tuple[str, ...],
) -> dict:
    path = os.path.abspath(path)
    data_index = os.path.abspath(data_index)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    world_size = int(payload["world_size"])
    batch_size = int(payload["batch_size"])
    batches_per_rank = int(payload["batches_per_rank"])
    rank_reports = payload["rank_reports"]
    expected_ranks = list(range(world_size))
    observed_ranks = sorted(int(report["rank"]) for report in rank_reports)
    source_counts = {name: 0 for name in source_names}
    rank_samples_consistent = True
    rank_batches_consistent = True
    report_sources_exact = True
    for report in rank_reports:
        batches = int(report["batches"])
        samples = int(report["samples"])
        counts = report["source_counts"]
        rank_batches_consistent &= batches == batches_per_rank
        rank_samples_consistent &= samples == batches * batch_size
        report_sources_exact &= set(counts) == set(source_names)
        for source in source_names:
            source_counts[source] += int(counts.get(source, 0))

    decoded_samples = int(payload["decoded_samples"])
    checks = {
        "status_passed": payload["status"] == "passed",
        "contract_matches": payload["contract"] == DECODE_FRONTIER_CONTRACT,
        "decoder_contract_matches": (
            payload["decoder_contract"] == VIDEO_DECODER_CONTRACT
        ),
        "data_index_matches": os.path.abspath(payload["data_index"]) == data_index,
        "sampler_seed_matches": int(payload["sampler_epoch"]) == int(seed),
        "positive_runtime_dimensions": min(
            world_size,
            batch_size,
            int(payload["workers_per_rank"]),
            int(payload["prefetch_factor"]),
            batches_per_rank,
            decoded_samples,
        )
        > 0,
        "rank_set_is_complete": observed_ranks == expected_ranks,
        "rank_batches_are_complete": rank_batches_consistent,
        "rank_sample_counts_are_consistent": rank_samples_consistent,
        "rank_source_tables_match_index": report_sources_exact,
        "decoded_sample_total_is_consistent": (
            decoded_samples == sum(int(report["samples"]) for report in rank_reports)
        ),
        "every_training_source_was_decoded": all(
            count > 0 for count in source_counts.values()
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "contract": payload["contract"],
        "decoder_contract": payload["decoder_contract"],
        "data_index": data_index,
        "sampler_epoch": int(payload["sampler_epoch"]),
        "world_size": world_size,
        "batch_size": batch_size,
        "workers_per_rank": int(payload["workers_per_rank"]),
        "prefetch_factor": int(payload["prefetch_factor"]),
        "batches_per_rank": batches_per_rank,
        "decoded_samples": decoded_samples,
        "source_counts": source_counts,
    }
