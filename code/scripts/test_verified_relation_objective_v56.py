#!/usr/bin/env python3
"""CPU counterfactual test for the v56 target."""

from __future__ import annotations

import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v56_config import (  # noqa: E402
    VerifiedRelationObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v56_independent_gates import (  # noqa: E402
    run_v56_independent_gates,
)


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
    print(json.dumps({**report, "bf16_status": bf16["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
