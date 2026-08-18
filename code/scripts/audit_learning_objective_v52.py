"""Run the v52 objective counterexample audit without model or dataset access."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v52_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    LearningObjectiveObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v52_falsification import run_objective_falsification  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    args = parser.parse_args()
    config = LearningObjectiveObjectStateConfig()
    config.validate()
    audit = run_objective_falsification(config, torch.device("cpu"))
    report = {
        "status": audit["status"],
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "objective_falsification": audit,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if audit["status"] != "passed":
        raise RuntimeError("v52 learning objective does not reject every counterexample")


if __name__ == "__main__":
    main()
