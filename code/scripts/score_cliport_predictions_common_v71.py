#!/usr/bin/env python3
"""Re-score existing lossless prediction files without running any model."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.common_cliport_scoring import score_prediction_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "predictions", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-edge", type=int, default=128)
    parser.add_argument("--model-image-size", type=int, default=None, help="Record a known size for older exports lacking this metadata.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    report = score_prediction_directory(args.manifest, args.predictions, args.out, args.max_edge, args.model_image_size)
    print(json.dumps({"report": str(Path(args.out)/"report.json"), "cases": report["cases"],
                      "common_max_edge": report["common_max_edge"], "unscored_conditions": report["unscored_conditions"]}), flush=True)
