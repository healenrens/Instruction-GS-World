"""Passing synthetic task evidence shared by remote contract tests."""
from __future__ import annotations

from igsw.adaptive_gaussian_wm.matched_flat_evaluation import COMPARISONS
from igsw.adaptive_gaussian_wm.task_group_evidence import (
    TASK_AGGREGATION,
    TASK_CONTRACT,
)


def object_flat_task_evidence(split: str, samples: int, episodes: int) -> dict:
    task_count = TASK_CONTRACT[split]["task_count"]
    summary = {
        "aggregation": TASK_AGGREGATION,
        "samples": samples,
        "episodes": episodes,
        "task_count": task_count,
        "positive_tasks": task_count,
        "positive_task_fraction": 1.0,
        "mean_relative_improvement": 0.10,
        "median_relative_improvement": 0.10,
        "worst_relative_improvement": 0.08,
    }
    comparison_names = [name for name, _, _ in COMPARISONS]

    def comparisons() -> dict:
        return {name: dict(summary) for name in comparison_names}

    return {
        "aggregation": TASK_AGGREGATION,
        "samples": samples,
        "episodes": episodes,
        "task_count": task_count,
        "task_hashes": [f"hash-{index}" for index in range(task_count)],
        "task_names": [f"task-{index}" for index in range(task_count)],
        "source_index_sha256": "c" * 64,
        "comparison": {
            "change_weighted_feature_mse": comparisons(),
            "change_weighted_rgb_distance": comparisons(),
        },
        "rgb_regions": {
            "change": comparisons(),
            "static": comparisons(),
        },
    }
