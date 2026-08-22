#!/usr/bin/env python3
"""CPU counterfactual test for the v56 target."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v56_config import (  # noqa: E402
    VerifiedRelationObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v56_data_contract import (  # noqa: E402
    DECODE_FRONTIER_CONTRACT,
    audit_decode_frontier,
)
from igsw.adaptive_gaussian_wm.v56_independent_gates import (  # noqa: E402
    run_v56_independent_gates,
)
from igsw.adaptive_gaussian_wm.video_file_decoder import (  # noqa: E402
    VIDEO_DECODER_CONTRACT,
)


def decode_contract_test() -> dict:
    sources = ("robotwin", "agibot", "droid", "robomind", "bridge", "hy_embodied")
    with tempfile.TemporaryDirectory() as directory:
        index_path = os.path.join(directory, "index.json")
        report_path = os.path.join(directory, "decode.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        payload = {
            "status": "passed",
            "contract": DECODE_FRONTIER_CONTRACT,
            "decoder_contract": VIDEO_DECODER_CONTRACT,
            "data_index": index_path,
            "world_size": 2,
            "batch_size": 8,
            "workers_per_rank": 2,
            "prefetch_factor": 2,
            "sampler_epoch": 17,
            "batches_per_rank": 6,
            "decoded_samples": 96,
            "rank_reports": [
                {
                    "rank": rank,
                    "batches": 6,
                    "samples": 48,
                    "source_counts": {source: 8 for source in sources},
                }
                for rank in range(2)
            ],
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        passed = audit_decode_frontier(report_path, index_path, 17, sources)
        payload["rank_reports"][0]["source_counts"]["hy_embodied"] = 0
        payload["rank_reports"][1]["source_counts"]["hy_embodied"] = 0
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        missing_source = audit_decode_frontier(report_path, index_path, 17, sources)
    if passed["status"] != "passed":
        raise RuntimeError(f"valid v56 decode contract failed: {passed}")
    if missing_source["status"] != "failed":
        raise RuntimeError("v56 decode contract accepted a missing training source")
    return {
        "valid_decode_frontier": passed["status"],
        "missing_source_decode_frontier": missing_source["status"],
    }


def main() -> None:
    config = VerifiedRelationObjectStateConfig()
    config.validate()
    report = run_v56_independent_gates(config, torch.device("cpu"))
    if report["status"] != "passed":
        raise RuntimeError(f"v56 objective gates failed: {report}")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bf16 = run_v56_independent_gates(config, torch.device("cpu"))
    if bf16["status"] != "passed":
        raise RuntimeError(f"v56 BF16 objective gates failed: {bf16}")
    decode = decode_contract_test()
    print(
        json.dumps({**report, "bf16_status": bf16["status"], **decode}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
