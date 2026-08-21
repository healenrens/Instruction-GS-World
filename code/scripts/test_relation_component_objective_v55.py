#!/usr/bin/env python3
"""CPU contract test for the v55 relation-component objective."""

from __future__ import annotations

import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v55_config import (  # noqa: E402
    RelationComponentObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v55_independent_gates import (  # noqa: E402
    run_v55_independent_gates,
)


def main() -> None:
    config = RelationComponentObjectStateConfig()
    config.validate()
    report = run_v55_independent_gates(config, torch.device("cpu"))
    if report["status"] != "passed":
        raise RuntimeError(f"v55 independent gates failed: {report}")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bf16_report = run_v55_independent_gates(config, torch.device("cpu"))
    if bf16_report["status"] != "passed":
        raise RuntimeError(f"v55 BF16 objective gate failed: {bf16_report}")
    print(
        json.dumps(
            {
                **report,
                "bf16_autocast_objective": bf16_report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
