#!/usr/bin/env python3
"""Publish existing V69 evidence to native SwanLab without rerunning computation."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.adaptive_gaussian_wm.swanlab_tracking_v69 import (
    add_swanlab_arguments, start_swanlab_v69, log_video_v69, log_evidence_v69, set_results_v69,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    add_swanlab_arguments(p, default_name="object_video_v69_evidence_upload")
    p.add_argument("--name", dest="swanlab_name", default=argparse.SUPPRESS, help="Alias for --swanlab_name.")
    p.add_argument("--project", dest="swanlab_project", default=argparse.SUPPRESS, help="Alias for --swanlab_project.")
    p.add_argument("--entity", dest="legacy_upload_entity", default=argparse.SUPPRESS,
                   help="Ignored legacy W&B entity; use --swanlab_workspace for a SwanLab workspace.")
    args = p.parse_args()
    directory = Path(args.directory)
    args.out = str(directory / "evidence_upload" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run = start_swanlab_v69(args, group="object-video-sequence-v69", job_type="evidence-upload", config=vars(args))
    if run is not None:
        files = []
        for root, directories, names in os.walk(directory):
            directories[:] = [name for name in directories if name not in ("swanlog", "evidence_upload")]
            files.extend(Path(root) / name for name in names)
        files.sort()
        log_evidence_v69(run, files, base_path=directory)
        for path in files:
            if path.suffix == ".mp4":
                log_video_v69(run, "evidence/" + path.relative_to(directory).as_posix(), path)
        for name in ("report.json", "test_report.json"):
            path = directory / name
            if path.is_file():
                report = json.loads(path.read_text())
                set_results_v69(run, {key: report[key]
                                     for key in ("status", "independent_status", "checkpoint_step", "cases_by_source", "parameter_inventory", "probes")
                                     if key in report})
        run.finish()


if __name__ == "__main__":
    main()
