"""Remote CPU contract for activity-gated canonical actions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.action_posterior import (  # noqa: E402
    ObjectDeltaActionPosterior,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(101)
    base = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=6,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
    )
    plain = ObjectDeltaActionPosterior(base).eval()
    gated = ObjectDeltaActionPosterior(
        replace(base, canonical_activity_gate=True)
    ).eval()
    gated.load_state_dict(plain.state_dict(), strict=True)
    history = torch.randn(2, 1, 4, base.object_dim)
    future = history[:, -1, None] + 0.05 * torch.randn(
        2, 1, 4, base.object_dim
    )
    history_activity = torch.tensor(
        [[[1.0, 0.25, 0.0, 0.81]], [[0.49, 1.0, 0.36, 0.0]]]
    )
    future_activity = torch.tensor(
        [[[1.0, 1.0, 1.0, 0.25]], [[1.0, 0.64, 0.25, 1.0]]]
    )
    history_centers = torch.randn(2, 1, 4, 2)
    future_centers = history_centers[:, -1, None] + 0.05 * torch.randn(
        2, 1, 4, 2
    )
    inputs = (
        history,
        history_activity,
        future,
        future_activity,
        torch.ones(2, 1),
        history_centers,
        future_centers,
    )
    reference = plain(*inputs)
    actual = gated(*inputs)
    confidence = (
        history_activity[:, -1, None] * future_activity
    ).pow(base.canonical_activity_power)[..., None]
    expected = reference * confidence
    error = float((actual - expected).abs().max())
    inactive = confidence.squeeze(-1) == 0.0
    inactive_max = float(actual[inactive].abs().max())
    if error != 0.0:
        raise AssertionError("canonical action did not follow activity confidence")
    if inactive_max != 0.0:
        raise AssertionError("inactive object retained a canonical action")
    report = {
        "status": "ok",
        "max_error": error,
        "inactive_action_max": inactive_max,
        "confidence": confidence.squeeze(-1).tolist(),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
