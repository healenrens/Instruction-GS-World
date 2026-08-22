#!/usr/bin/env python3
"""Tensor contracts used by the v56 evaluation metrics."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.v56_evaluation_metrics import (  # noqa: E402
    _reappearance,
    _track_correspondence,
    visible_mask,
)


def main() -> None:
    visibility = torch.tensor(
        [[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    teacher = SimpleNamespace(
        visibility=visibility,
        object_confidence=torch.ones(1, 3),
    )
    owners = F.one_hot(torch.arange(3), num_classes=3).float()
    assignment = owners[None, None].expand(1, 4, 3, 3).clone()
    mask = visible_mask(teacher)
    if mask.dtype != torch.bool or mask.shape != visibility.shape:
        raise RuntimeError("v56 float visibility did not produce a boolean mask")
    correct, shuffled = _track_correspondence(assignment, teacher)
    if correct[0] <= shuffled[0]:
        raise RuntimeError("v56 correspondence contract lost track identity")
    reappearance, wrong_reappearance, events = _reappearance(assignment, teacher)
    if events < 1.0 or reappearance[0] <= wrong_reappearance[0]:
        raise RuntimeError("v56 reappearance contract lost track identity")
    print(
        {
            "status": "passed",
            "visibility_input_dtype": str(visibility.dtype),
            "visibility_mask_dtype": str(mask.dtype),
            "correspondence_margin": correct[0] - shuffled[0],
            "reappearance_events": events,
        }
    )


if __name__ == "__main__":
    main()
