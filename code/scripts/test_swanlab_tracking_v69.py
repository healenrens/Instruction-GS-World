#!/usr/bin/env python3
"""Exercise training scalar logging and resume offline, without running a model."""

import argparse
import json
import os
from pathlib import Path
import runpy

def main():
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "igsw/adaptive_gaussian_wm/swanlab_tracking_v69.py"))
    parser = api["add_swanlab_arguments"](argparse.ArgumentParser(description=__doc__), default_name="v69_swanlab_tracking_test")
    parser.set_defaults(swanlab_mode="offline")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.swanlab_mode = "offline"
    out = Path(args.out)
    audit_args = argparse.Namespace(**vars(args))
    audit_args.swanlab_mode = "online"
    assert api["start_swanlab_v69"](audit_args, "object-video-sequence-v69", "held-evaluation") is None
    os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    os.environ["SWANLAB_WORKSPACE"] = args.swanlab_workspace
    os.environ["SWANLAB_RUN_ID"] = ""
    os.environ["SWANLAB_RESUME"] = ""
    run = api["start_swanlab_v69"](args, "object-video-sequence-v69", "state")
    assert all(name not in os.environ for name in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_RUN_ID", "SWANLAB_RESUME"))
    api["log_values_v69"](run, {"loss": 1.5, "gradient_norm": 2.0, "lr": 0.0002}, step=1)
    tracking = json.loads((out / "tracking.json").read_text())
    assert tracking["project"] == args.swanlab_project
    assert tracking["workspace"] == args.swanlab_workspace
    run.finish()
    checkpoint = {"step": 1, "tracking": tracking}
    os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    os.environ["SWANLAB_WORKSPACE"] = args.swanlab_workspace
    os.environ["SWANLAB_RUN_ID"] = ""
    os.environ["SWANLAB_RESUME"] = ""
    resumed = api["start_swanlab_v69"](args, "object-video-sequence-v69", "state", checkpoint=checkpoint)
    assert all(name not in os.environ for name in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_RUN_ID", "SWANLAB_RESUME"))
    api["log_values_v69"](resumed, {"loss": 1.0, "gradient_norm": 1.5, "lr": 0.0002}, step=2)
    assert resumed.id == tracking["id"]
    resumed.finish()
    (out / "test_report.json").write_text(json.dumps({"status": "passed_sdk_logging", "run_id": tracking["id"],
                                                    "mode": args.swanlab_mode, "model_executed": False,
                                                    "uploaded": False, "tracking_scope": "training_scalars_only",
                                                    "project": tracking["project"],
                                                    "workspace": tracking["workspace"],
                                                    "legacy_project_environment_compatible": True,
                                                    "workspace_environment_compatible": True}, indent=2), encoding="utf-8")
    print(f"[swanlab-v69-test] passed mode={args.swanlab_mode} report={out / 'test_report.json'}", flush=True)


if __name__ == "__main__":
    main()
