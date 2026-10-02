#!/usr/bin/env python3
"""Exercise V69 tracking with the real SDK, without importing or running the model."""

import argparse
import json
from pathlib import Path
import runpy

import av
from PIL import Image


def main():
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "igsw/adaptive_gaussian_wm/swanlab_tracking_v69.py"))
    parser = api["add_swanlab_arguments"](argparse.ArgumentParser(description=__doc__), default_name="v69_swanlab_tracking_test")
    parser.set_defaults(swanlab_mode="offline")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    run = api["start_swanlab_v69"](args, "object-video-sequence-v69", "tracking-integration")
    api["log_values_v69"](run, {"loss": 1.5, "gradient_norm": 2.0, "status": "started"}, step=1)
    api["log_table_v69"](run, "cases", ["case", "error", "points", "details"],
                         [{"case": "case_a", "error": None, "points": 0, "details": {"observed": False}},
                          {"case": "case_b", "error": 3.2, "points": 4, "details": [1, 2]}], step=1)
    evidence = out / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    payload = {"status": "passed_sdk_logging", "unmeasured": None, "content": "object-state-evidence-" * 7000}
    report_path = evidence / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    (evidence / "index.html").write_text("<html><body><p>V69 SwanLab tracking integration</p></body></html>", encoding="utf-8")
    video = evidence / "clip.mp4"
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for color in ("red", "green", "blue"):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    api["log_video_v69"](run, "review/clip", video, step=1)
    api["set_results_v69"](run, {"status": "passed_sdk_logging", "nullable_cells": True})
    api["log_evidence_v69"](run, [evidence], out)
    tracking = json.loads((out / "tracking.json").read_text())
    run.finish()
    checkpoint = {"step": 1, "tracking": tracking}
    resumed = api["start_swanlab_v69"](args, "object-video-sequence-v69", "tracking-integration", checkpoint=checkpoint)
    api["log_values_v69"](resumed, {"loss": 1.0, "gradient_norm": 1.5, "status": "resumed"}, step=2)
    assert resumed.id == tracking["id"]
    resumed.finish()
    (out / "test_report.json").write_text(json.dumps({"status": "passed_sdk_logging", "run_id": tracking["id"],
                                                    "mode": args.swanlab_mode, "model_executed": False}, indent=2), encoding="utf-8")
    print(f"[swanlab-v69-test] passed mode={args.swanlab_mode} report={out / 'test_report.json'}", flush=True)


if __name__ == "__main__":
    main()
