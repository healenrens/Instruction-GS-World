#!/usr/bin/env python3
"""Upload an already-computed v66 G0 JSON report to W&B."""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from audit_object_motion_field_g0_v66 import flatten_wandb  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_group", default="object-motion-field-g0-v66")
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)

    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        dir=args.wandb_dir,
        config={
            "contract": report["contract"],
            "source_revision": report["source_revision"],
            "items_per_source": report["items_per_source"],
            "chunk_length": report["chunk_length"],
            "local_motion_modes": report["local_motion_modes"],
            "bootstrap_samples": report["bootstrap_samples"],
            "report_only_upload": True,
        },
    )
    run.log(flatten_wandb(report))
    run.summary.update(report)
    run.finish()
    print(f"uploaded_report={os.path.abspath(args.report)}")
    print(f"wandb_run_path={run.entity}/{run.project}/{run.id}")


if __name__ == "__main__":
    main()
