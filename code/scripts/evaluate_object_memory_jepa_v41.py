#!/usr/bin/env python3
"""Held promotion gates for Object Memory JEPA v41."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/pfs/public/xuhaoming/.cache")

sys.path.insert(0, os.path.dirname(__file__))

import evaluate_dynamic_dual_horizon_v39 as evaluator  # noqa: E402

from igsw.adaptive_gaussian_wm.v41_held_metrics import (  # noqa: E402
    add_v41_held_metrics,
    v41_acceptance,
)


evaluator.EXPECTED_CHECKPOINT_VERSION = 41
evaluator.EXPECTED_ARCHITECTURE = "object_memory_v3"
evaluator.HELD_CONTRACT_PREFIX = "object_memory_v41"
evaluator.add_v40_held_metrics = add_v41_held_metrics
evaluator.v40_acceptance = v41_acceptance


if __name__ == "__main__":
    evaluator.main()
