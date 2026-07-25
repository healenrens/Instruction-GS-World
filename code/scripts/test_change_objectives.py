"""Server-side contracts for collapse-resistant world-model objectives."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.change_objectives import (  # noqa: E402
    change_balanced_rgb_delta_loss,
    scale_invariant_object_change_loss,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _feature_scale_contract() -> dict[str, float]:
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(2, 3, 4, 12, generator=generator)
    target = torch.randn(2, 3, 4, 12, generator=generator)
    current_prediction = torch.randn(2, 4, 12, generator=generator)
    current_target = torch.randn(2, 4, 12, generator=generator)
    activity = torch.rand(2, 3, 4, generator=generator)
    reference = scale_invariant_object_change_loss(
        prediction,
        target,
        current_prediction,
        current_target,
        activity,
    )
    rescaled = scale_invariant_object_change_loss(
        prediction * 13.0,
        target * 0.25,
        current_prediction * 13.0,
        current_target * 0.25,
        activity,
    )
    difference = float((reference - rescaled).abs())
    _require(difference < 1e-7, "feature-change loss depends on feature scale")
    return {
        "reference": float(reference),
        "rescaled": float(rescaled),
        "absolute_difference": difference,
    }


def _rgb_change_contract() -> dict[str, float]:
    current = torch.full((1, 1, 3, 12, 12), 96, dtype=torch.uint8)
    target = current.clone()
    target[:, :, 0, 3:8, 4:9] = 176
    target[:, :, 1, 3:8, 4:9] = 136
    future_valid = torch.ones(1, 1, 12, 12, dtype=torch.bool)
    current_valid = future_valid.clone()
    copy_prediction = current.float() / 255.0
    perfect_prediction = target.float() / 255.0

    copy_loss, copy_parts = change_balanced_rgb_delta_loss(
        copy_prediction,
        target,
        current,
        future_valid,
        current_valid,
        0.04,
    )
    perfect_loss, perfect_parts = change_balanced_rgb_delta_loss(
        perfect_prediction,
        target,
        current,
        future_valid,
        current_valid,
        0.04,
    )
    polluted_prediction = perfect_prediction.clone()
    polluted_prediction[:, :, :, :2, :] += 0.1
    polluted_loss, polluted_parts = change_balanced_rgb_delta_loss(
        polluted_prediction,
        target,
        current,
        future_valid,
        current_valid,
        0.04,
    )
    _require(float(perfect_loss) < float(copy_loss) * 0.1, "perfect future does not beat current-copy")
    _require(
        float(polluted_parts["static_charbonnier"])
        > float(perfect_parts["static_charbonnier"]),
        "static-region pollution is not penalized",
    )
    _require(float(polluted_loss) > float(perfect_loss), "RGB objective ignores static pollution")
    _require(
        0.0 < float(copy_parts["change_weight_fraction"]) < 1.0,
        "change weighting did not separate the moving patch",
    )

    gradient_prediction = copy_prediction.clone().requires_grad_(True)
    gradient_loss, _ = change_balanced_rgb_delta_loss(
        gradient_prediction,
        target,
        current,
        future_valid,
        current_valid,
        0.04,
    )
    gradient_loss.backward()
    changed_gradient = gradient_prediction.grad[:, :, :, 3:8, 4:9].abs().sum()
    _require(float(changed_gradient) > 0.0, "change-region loss has no prediction gradient")
    return {
        "copy_loss": float(copy_loss),
        "perfect_loss": float(perfect_loss),
        "polluted_loss": float(polluted_loss),
        "change_weight_fraction": float(copy_parts["change_weight_fraction"]),
        "copy_change_charbonnier": float(copy_parts["copy_change_charbonnier"]),
        "changed_gradient_l1": float(changed_gradient),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "status": "ok",
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_scale": _feature_scale_contract(),
        "rgb_change": _rgb_change_contract(),
    }
    _require(CHECKPOINT_VERSION == 27, "checkpoint version was not upgraded")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
