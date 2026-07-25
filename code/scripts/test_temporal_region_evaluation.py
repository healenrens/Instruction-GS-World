"""Remote CPU contracts for temporal change/static RGB diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.temporal_region_evaluation import (  # noqa: E402
    TemporalRegionConfig,
    localization_frames,
    regional_rgb_error_frames,
    temporal_region_masks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = TemporalRegionConfig(
        low_floor=0.01,
        high_floor=0.03,
        hysteresis_steps=2,
        change_dilation=1,
    )
    history = torch.zeros(1, 1, 3, 32, 32, dtype=torch.uint8)
    future = torch.zeros(1, 1, 3, 32, 32, dtype=torch.uint8)
    future[:, :, :, 10:18, 12:20] = 204
    valid_history = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    valid_future = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    regions = temporal_region_masks(
        history,
        future,
        valid_history,
        valid_future,
        config,
    )
    if bool((regions["change"] & regions["static"]).any()):
        raise AssertionError("change and static regions overlap")
    union = regions["change"] | regions["static"] | regions["ambiguous"]
    if not torch.equal(union, regions["valid"]):
        raise AssertionError("temporal regions do not partition valid pixels")
    target_square = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    target_square[:, :, 10:18, 12:20] = True
    recall = (regions["change"] & target_square).sum().float() / target_square.sum()
    if float(recall) < 0.9:
        raise AssertionError("synthetic changed square was not recovered")

    posterior = future.float() / 255.0
    zero = history.float() / 255.0
    shuffled = torch.zeros_like(posterior)
    shuffled[:, :, :, 2:10, 2:10] = 0.8
    posterior_error = regional_rgb_error_frames(posterior, future, regions["change"])
    zero_error = regional_rgb_error_frames(zero, future, regions["change"])
    if not bool((posterior_error["charbonnier"] < zero_error["charbonnier"]).all()):
        raise AssertionError("correct prediction did not beat zero in change region")
    current = history.float() / 255.0
    posterior_location = localization_frames(posterior, current, regions, config)
    shuffled_location = localization_frames(shuffled, current, regions, config)
    if not bool(
        (
            posterior_location["topk_iou_at_gt_area"]
            > shuffled_location["topk_iou_at_gt_area"]
        ).all()
    ):
        raise AssertionError("change localization did not distinguish correct and wrong motion")
    report = {
        "status": "ok",
        "change_fraction": float(regions["change"].float().mean()),
        "static_fraction": float(regions["static"].float().mean()),
        "square_recall": float(recall),
        "posterior_change_charbonnier": float(posterior_error["charbonnier"].mean()),
        "zero_change_charbonnier": float(zero_error["charbonnier"].mean()),
        "posterior_topk_iou": float(posterior_location["topk_iou_at_gt_area"].mean()),
        "shuffled_topk_iou": float(shuffled_location["topk_iou_at_gt_area"].mean()),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
