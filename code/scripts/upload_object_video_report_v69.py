#!/usr/bin/env python3
"""Upload existing V69 evidence after a W&B failure without rerunning any computation."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="instruct-gs-world")
    p.add_argument("--entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    args = p.parse_args()
    import wandb
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    run = wandb.init(project=args.project, entity=args.entity or None, name=args.name,
                     group="object-video-sequence-v69", job_type="evidence-upload", config=vars(args))
    directory = Path(args.directory)
    artifact = wandb.Artifact(args.name, type="object-video-evidence")
    allowed = {".json", ".jsonl", ".html", ".png", ".mp4", ".csv"}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in allowed:
            artifact.add_file(str(path), name=path.relative_to(directory).as_posix())
    for name in ("report.json", "test_report.json"):
        path = directory / name
        if path.is_file():
            report = json.loads(path.read_text())
            for key in ("status", "independent_status", "checkpoint_step", "cases_by_source", "parameter_inventory", "probes"):
                if key in report:
                    run.summary[key] = report[key]
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
