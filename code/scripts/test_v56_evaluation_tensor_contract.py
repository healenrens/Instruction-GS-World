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
from igsw.adaptive_gaussian_wm.object_state_target_v52 import (  # noqa: E402
    component_motion_targets,
)
from igsw.adaptive_gaussian_wm.v56_state_probe_metrics import (  # noqa: E402
    held_group_binary_metrics,
    held_group_vector_metrics,
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

    relation = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    motion = torch.tensor(
        [[[[[1.0, 0.0]], [[3.0, 0.0]], [[9.0, 0.0]]]]]
    )
    component_teacher = SimpleNamespace(
        same_confidence=relation,
        motion=motion,
        motion_valid=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        object_confidence=torch.ones(1, 3),
    )
    component_motion_target, component_motion_weight, _ = component_motion_targets(
        component_teacher
    )
    expected = torch.tensor([2.0, 2.0, 0.0])
    if not torch.allclose(component_motion_target[0, 0, :, 0, 0], expected):
        raise RuntimeError("v56 component motion target differs from relation pooling")
    if not torch.equal(
        component_motion_weight[0, 0, :, 0] > 0.0,
        torch.tensor([True, True, False]),
    ):
        raise RuntimeError("v56 component motion support differs from training")

    groups = torch.arange(4).repeat_interleave(8)
    vector_target = torch.stack(
        (torch.linspace(-1.0, 1.0, 32), torch.linspace(1.0, -1.0, 32)),
        dim=-1,
    )
    vector_metrics = held_group_vector_metrics(
        vector_target, vector_target, torch.ones(32), groups
    )
    if vector_metrics["zero_relative_gain"] < 0.999:
        raise RuntimeError("v56 aligned motion readout metric rejected exact prediction")

    binary_target = (torch.arange(32) % 2).float()
    binary_metrics = held_group_binary_metrics(
        binary_target, binary_target, torch.ones(32), groups
    )
    if binary_metrics["balanced_accuracy"] != 1.0:
        raise RuntimeError("v56 explicit visibility metric rejected exact prediction")
    print(
        {
            "status": "passed",
            "visibility_input_dtype": str(visibility.dtype),
            "visibility_mask_dtype": str(mask.dtype),
            "correspondence_margin": correct[0] - shuffled[0],
            "reappearance_events": events,
            "component_motion_target": component_motion_target[0, 0, :, 0, 0].tolist(),
            "component_motion_readout_gain": vector_metrics["zero_relative_gain"],
            "explicit_visibility_balanced_accuracy": binary_metrics[
                "balanced_accuracy"
            ],
        }
    )


if __name__ == "__main__":
    main()
