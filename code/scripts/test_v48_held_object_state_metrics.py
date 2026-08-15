#!/usr/bin/env python3
"""CPU contracts for the v48 held-video metric implementations."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v48_cross_episode_evaluation import (  # noqa: E402
    cross_episode_consistency,
)
from igsw.adaptive_gaussian_wm.v48_held_metrics import (  # noqa: E402
    slot_deletion_metrics,
)


class ToyDecoderModel:
    def __init__(self):
        self.config = SimpleNamespace(object_slots=2, active_slot_fraction=0.1)

    def decode_frame(self, slots, coordinates, slot_valid=None):
        preferred = coordinates[..., 0].ge(0.0).long()
        logits = coordinates.new_full((*coordinates.shape[:2], 2), -12.0)
        logits.scatter_(2, preferred[..., None], 12.0)
        if slot_valid is not None:
            logits = logits.masked_fill(~slot_valid[:, None], -1e4)
        assignment = logits.softmax(dim=-1)
        decoded = torch.einsum("bnk,bkd->bnd", assignment, slots)
        return F.normalize(decoded, dim=-1), assignment


def test_deletion_locality() -> dict:
    model = ToyDecoderModel()
    slots = F.normalize(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), dim=-1)
    coordinates = torch.tensor([[[[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]]]])
    full, assignment = model.decode_frame(slots, coordinates[:, -1])
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    activity = assignment.sum(dim=1) / 4.0
    state = {
        "slots": slots[:, None],
        "reconstruction": full[:, None],
        "assignment": assignment[:, None],
        "activity": activity[:, None],
    }
    metrics = slot_deletion_metrics(
        model,
        full[:, None],
        coordinates,
        valid,
        state,
    )
    if float(metrics["deletion_error_increase"]) <= 0.0:
        raise AssertionError("deleting a useful toy slot did not increase error")
    if float(metrics["deletion_change_locality_precision"]) < 0.99:
        raise AssertionError("toy slot deletion was not localized")
    if float(metrics["deletion_locality_enrichment"]) < 1.9:
        raise AssertionError("toy slot deletion did not exceed its area baseline")
    return {name: float(value) for name, value in metrics.items()}


def test_cross_episode_proxy() -> dict:
    slots = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.98, 0.02, 0.0], [0.02, 0.98, 0.0]],
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        ]
    )
    active = torch.ones(3, 2, dtype=torch.bool)
    report = cross_episode_consistency(
        slots,
        active,
        episode_ids=torch.tensor([0, 1, 2]),
        group_ids=torch.tensor([0, 0, 1]),
        maximum_pairs_per_category=16,
    )
    margin = report["same_group_margin_over_different"]
    if margin is None or margin <= 0.0:
        raise AssertionError("same-group toy slot sets did not exceed different groups")
    if report["semantic_object_correspondence_verified"]:
        raise AssertionError("annotation-free proxy claimed semantic verification")
    return report


def main() -> None:
    report = {
        "status": "passed",
        "deletion": test_deletion_locality(),
        "cross_episode": test_cross_episode_proxy(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
