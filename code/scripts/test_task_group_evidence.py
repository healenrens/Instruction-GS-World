"""Deterministic regression tests for episode-balanced task statistics."""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    TaskGroupLayout,
    remap_task_layout,
    task_group_comparison,
)


def main() -> None:
    episodes = torch.tensor([0, 0, 1, 2, 2])
    layout = TaskGroupLayout(
        ids=torch.tensor([0, 0, 0, 1, 1]),
        hashes=("hash-a", "hash-b"),
        names=("task-a", "task-b"),
        source_index_sha256="a" * 64,
    )
    prediction = torch.tensor([0.0, 0.0, 1.0, 0.8, 0.8])
    reference = torch.ones(5)
    result = task_group_comparison(
        prediction,
        reference,
        episodes,
        layout,
    )
    task_a = result["per_task"]["task-a"]
    task_b = result["per_task"]["task-b"]
    if result["samples"] != 5 or result["episodes"] != 3:
        raise AssertionError("task evidence count contract differs")
    if abs(task_a["prediction_mean"] - 0.5) > 1e-7:
        raise AssertionError("task-a is not episode-balanced")
    if abs(task_b["prediction_mean"] - 0.8) > 1e-7:
        raise AssertionError("task-b mean differs")
    if abs(result["positive_task_fraction"] - 1.0) > 1e-7:
        raise AssertionError("positive task fraction differs")
    region_layout = remap_task_layout(
        episodes,
        layout,
        torch.tensor([0, 2, 2]),
    )
    if region_layout.ids.tolist() != [0, 1, 1]:
        raise AssertionError("region episode-to-task remap differs")
    print(
        {
            "status": "ok",
            "task_a_episode_balanced_prediction": task_a["prediction_mean"],
            "task_b_prediction": task_b["prediction_mean"],
            "region_task_ids": region_layout.ids.tolist(),
        }
    )


if __name__ == "__main__":
    main()
