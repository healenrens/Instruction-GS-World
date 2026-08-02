#!/usr/bin/env python3
"""Held promotion gates for Object Memory JEPA v40."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import evaluate_dynamic_dual_horizon_v39 as evaluator  # noqa: E402


evaluator.EXPECTED_CHECKPOINT_VERSION = 40
evaluator.EXPECTED_ARCHITECTURE = "object_memory_v2"
evaluator.HELD_CONTRACT_PREFIX = "object_memory_v40"


if __name__ == "__main__":
    evaluator.main()
