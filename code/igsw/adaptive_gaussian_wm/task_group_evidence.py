"""Episode-balanced task evidence for held-split world-model evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from collections.abc import Mapping, Sequence

import torch

from .goal_eval_statistics import clustered_paired_comparison


EPISODE_SOURCE_INDEX = "episode_source_index.json"
TASK_AGGREGATION = "mean_samples_per_episode_then_mean_episodes_per_task"
MINIMUM_TASK_MEDIAN_RELATIVE = 0.03
TASK_CONTRACT = {
    "heldseed": {"task_count": 47, "minimum_positive_fraction": 0.75},
    "heldtask": {"task_count": 3, "minimum_positive_fraction": 1.0},
}


@dataclass(frozen=True)
class TaskGroupLayout:
    ids: torch.Tensor
    hashes: tuple[str, ...]
    names: tuple[str, ...]
    source_index_sha256: str


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def task_source_index_sha256(cache_root: str) -> str:
    path = os.path.join(cache_root, EPISODE_SOURCE_INDEX)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_task_group_layout(
    dataset,
    cache_root: str,
    split: str,
) -> TaskGroupLayout:
    """Recover selected-sample task ids from the episode dataset contract."""
    if split not in TASK_CONTRACT:
        raise ValueError(f"task evidence does not support split: {split}")
    hashes = tuple(getattr(dataset, "sampling_group_names", ()))
    spans = tuple(getattr(dataset, "sampling_group_spans", ()))
    if not hashes or len(hashes) != len(spans) or len(set(hashes)) != len(hashes):
        raise ValueError("dataset has no unique task-group partition")
    manifest_path = os.path.join(cache_root, "episode_manifest.json")
    source_path = os.path.join(cache_root, EPISODE_SOURCE_INDEX)
    manifest = _load_json(manifest_path)
    source = _load_json(source_path)
    source_tasks = {}
    for entry in source:
        if entry.get("split") != split:
            continue
        filename = str(entry["filename"])
        if filename in source_tasks:
            raise ValueError(f"duplicate source-index filename: {filename}")
        source_tasks[filename] = str(entry["task"])
    grouped_tasks: dict[str, set[str]] = {}
    for entry in manifest.get("episodes", ()):
        if entry.get("split") != split:
            continue
        filename = str(entry["filename"])
        if filename not in source_tasks:
            raise ValueError(f"source index is missing episode: {filename}")
        task = source_tasks[filename]
        group_hash = str(entry["sampling_group"])
        expected = hashlib.sha256(task.encode()).hexdigest()[:16]
        if group_hash != expected:
            raise ValueError(f"task hash differs for episode: {filename}")
        grouped_tasks.setdefault(group_hash, set()).add(task)
    names = []
    for group_hash in hashes:
        tasks = grouped_tasks.get(group_hash, set())
        if len(tasks) != 1:
            raise ValueError(f"task group does not map one-to-one: {group_hash}")
        names.append(next(iter(tasks)))
    ids = torch.full((len(dataset),), -1, dtype=torch.long)
    cursor = 0
    for task_id, (start, end) in enumerate(spans):
        if start != cursor or not start < end or end > len(dataset):
            raise ValueError("task-group spans do not partition selected samples")
        ids[start:end] = task_id
        cursor = end
    if cursor != len(dataset) or bool((ids < 0).any()):
        raise ValueError("task-group ids do not cover selected samples")
    return TaskGroupLayout(
        ids=ids,
        hashes=hashes,
        names=tuple(names),
        source_index_sha256=task_source_index_sha256(cache_root),
    )


def remap_task_layout(
    source_episode_ids: torch.Tensor,
    source_layout: TaskGroupLayout,
    target_episode_ids: torch.Tensor,
) -> TaskGroupLayout:
    """Map frame-level region rows back to task ids through episode identity."""
    if source_episode_ids.shape != source_layout.ids.shape:
        raise ValueError("source episodes and task ids must align")
    episode_to_task = {}
    for episode in torch.unique(source_episode_ids, sorted=True):
        task_ids = torch.unique(source_layout.ids[source_episode_ids == episode])
        if len(task_ids) != 1:
            raise ValueError("one episode maps to multiple task groups")
        episode_to_task[int(episode)] = int(task_ids[0])
    missing = sorted(
        {int(episode) for episode in torch.unique(target_episode_ids)}
        - set(episode_to_task)
    )
    if missing:
        raise ValueError(f"region rows reference unknown episodes: {missing}")
    ids = torch.tensor(
        [episode_to_task[int(episode)] for episode in target_episode_ids],
        dtype=torch.long,
    )
    return TaskGroupLayout(
        ids=ids,
        hashes=source_layout.hashes,
        names=source_layout.names,
        source_index_sha256=source_layout.source_index_sha256,
    )


def _sample_vector(values: torch.Tensor, samples: int) -> torch.Tensor:
    if values.shape[0] != samples:
        raise ValueError("metric values do not align with task samples")
    values = values.float()
    if values.ndim > 1:
        values = values.mean(dim=tuple(range(1, values.ndim)))
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("task evidence requires finite per-sample values")
    return values


def task_group_comparison(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    episode_ids: torch.Tensor,
    layout: TaskGroupLayout,
) -> dict:
    samples = len(episode_ids)
    prediction = _sample_vector(prediction, samples)
    reference = _sample_vector(reference, samples)
    if episode_ids.ndim != 1 or layout.ids.shape != episode_ids.shape:
        raise ValueError("episode and task ids must be aligned vectors")
    episode_prediction = []
    episode_reference = []
    episode_tasks = []
    episode_samples = []
    for episode in torch.unique(episode_ids, sorted=True):
        mask = episode_ids == episode
        task_ids = torch.unique(layout.ids[mask])
        if len(task_ids) != 1:
            raise ValueError("one episode contributes to multiple tasks")
        episode_prediction.append(prediction[mask].mean())
        episode_reference.append(reference[mask].mean())
        episode_tasks.append(task_ids[0])
        episode_samples.append(int(mask.sum()))
    episode_prediction = torch.stack(episode_prediction)
    episode_reference = torch.stack(episode_reference)
    episode_tasks = torch.stack(episode_tasks)
    task_prediction = []
    task_reference = []
    per_task = {}
    for task_id, (group_hash, name) in enumerate(zip(layout.hashes, layout.names)):
        mask = episode_tasks == task_id
        if not bool(mask.any()):
            raise ValueError(f"selected task has no episode evidence: {name}")
        task_prediction.append(episode_prediction[mask].mean())
        task_reference.append(episode_reference[mask].mean())
        absolute = task_reference[-1] - task_prediction[-1]
        relative = absolute / task_reference[-1].clamp_min(1e-8)
        sample_count = sum(
            count for count, selected in zip(episode_samples, mask) if bool(selected)
        )
        per_task[name] = {
            "group_hash": group_hash,
            "samples": sample_count,
            "episodes": int(mask.sum()),
            "prediction_mean": float(task_prediction[-1]),
            "reference_mean": float(task_reference[-1]),
            "absolute_improvement": float(absolute),
            "relative_improvement": float(relative),
            "positive": bool(absolute > 0.0),
        }
    task_prediction = torch.stack(task_prediction)
    task_reference = torch.stack(task_reference)
    relative = (task_reference - task_prediction) / task_reference.clamp_min(1e-8)
    positive = task_reference > task_prediction
    return {
        "aggregation": TASK_AGGREGATION,
        "samples": samples,
        "episodes": len(episode_prediction),
        "task_count": len(task_prediction),
        "positive_tasks": int(positive.sum()),
        "positive_task_fraction": float(positive.float().mean()),
        "mean_relative_improvement": float(relative.mean()),
        "median_relative_improvement": float(relative.median()),
        "worst_relative_improvement": float(relative.min()),
        "task_mean_comparison": clustered_paired_comparison(
            task_prediction,
            task_reference,
            torch.arange(len(task_prediction)),
        ),
        "per_task": per_task,
    }


def task_group_metric_evidence(
    metrics: Mapping[str, Mapping[str, torch.Tensor]],
    episode_ids: torch.Tensor,
    layout: TaskGroupLayout,
    comparisons: Sequence[tuple[str, str, str]],
) -> dict:
    return {
        "aggregation": TASK_AGGREGATION,
        "samples": len(episode_ids),
        "episodes": len(torch.unique(episode_ids)),
        "task_count": len(layout.names),
        "task_hashes": list(layout.hashes),
        "task_names": list(layout.names),
        "source_index_sha256": layout.source_index_sha256,
        "comparison": {
            metric: {
                name: task_group_comparison(
                    variants[prediction],
                    variants[reference],
                    episode_ids,
                    layout,
                )
                for name, prediction, reference in comparisons
            }
            for metric, variants in metrics.items()
        },
    }


def object_flat_task_evidence(
    metrics: Mapping[str, Mapping[str, torch.Tensor]],
    episode_ids: torch.Tensor,
    layout: TaskGroupLayout,
    region_metrics: Mapping[str, Mapping[str, torch.Tensor]],
    region_episode_ids: Mapping[str, torch.Tensor],
    comparisons: Sequence[tuple[str, str, str]],
) -> dict:
    changed = {
        name: variants
        for name, variants in metrics.items()
        if name.startswith("change_weighted_")
    }
    result = task_group_metric_evidence(
        changed,
        episode_ids,
        layout,
        comparisons,
    )
    result["rgb_regions"] = {}
    for region, variants in region_metrics.items():
        region_layout = remap_task_layout(
            episode_ids,
            layout,
            region_episode_ids[region],
        )
        evidence = task_group_metric_evidence(
            {"rgb_distance": variants},
            region_episode_ids[region],
            region_layout,
            comparisons,
        )
        result["rgb_regions"][region] = evidence["comparison"]["rgb_distance"]
    return result


def task_comparison_passes(
    comparison: dict,
    split: str,
    minimum_median_relative: float,
) -> bool:
    contract = TASK_CONTRACT[split]
    return (
        comparison.get("aggregation") == TASK_AGGREGATION
        and int(comparison.get("task_count", 0)) == contract["task_count"]
        and float(comparison.get("positive_task_fraction", 0.0))
        >= contract["minimum_positive_fraction"]
        and float(comparison.get("median_relative_improvement", float("-inf")))
        >= minimum_median_relative
    )


def task_evidence_passes(
    evidence: dict,
    split: str,
    expected_samples: int,
    expected_episodes: int,
    expected_source_index_sha256: str,
    metrics: Sequence[str],
    comparisons: Sequence[str],
    minimum_median_relative: float,
) -> bool:
    contract = TASK_CONTRACT[split]
    task_count = contract["task_count"]
    return (
        evidence.get("aggregation") == TASK_AGGREGATION
        and int(evidence.get("samples", 0)) == expected_samples
        and int(evidence.get("episodes", 0)) == expected_episodes
        and evidence.get("source_index_sha256") == expected_source_index_sha256
        and int(evidence.get("task_count", 0)) == task_count
        and len(set(evidence.get("task_hashes", ()))) == task_count
        and len(set(evidence.get("task_names", ()))) == task_count
        and all(
            task_comparison_passes(
                evidence.get("comparison", {}).get(metric, {}).get(name, {}),
                split,
                minimum_median_relative,
            )
            for metric in metrics
            for name in comparisons
        )
    )


def task_noninferiority_passes(
    comparison: dict,
    split: str,
    relative_margin: float,
) -> bool:
    return (
        comparison.get("aggregation") == TASK_AGGREGATION
        and int(comparison.get("task_count", 0))
        == TASK_CONTRACT[split]["task_count"]
        and float(comparison.get("worst_relative_improvement", float("-inf")))
        >= -relative_margin
    )


def object_flat_task_gate_entries(
    evidence: dict,
    split: str,
    expected_samples: int,
    expected_episodes: int,
    expected_source_index_sha256: str,
    minimum_relative: float,
) -> list[tuple[str, bool, dict, dict]]:
    contract = TASK_CONTRACT[split]
    contract_observed = {
        "aggregation": evidence.get("aggregation"),
        "samples": evidence.get("samples"),
        "episodes": evidence.get("episodes"),
        "source_index_sha256": evidence.get("source_index_sha256"),
        "task_count": evidence.get("task_count"),
        "unique_task_hashes": len(set(evidence.get("task_hashes", ()))),
        "unique_task_names": len(set(evidence.get("task_names", ()))),
    }
    contract_required = {
        "aggregation": TASK_AGGREGATION,
        "samples": expected_samples,
        "episodes": expected_episodes,
        "source_index_sha256": expected_source_index_sha256,
        "task_count": contract["task_count"],
    }
    entries = [
        (
            "task_group_contract",
            contract_observed
            == {**contract_required, "unique_task_hashes": contract["task_count"],
                "unique_task_names": contract["task_count"]},
            contract_observed,
            contract_required,
        )
    ]
    comparisons = evidence.get("comparison", {})
    for label, metric in (
        ("feature", "change_weighted_feature_mse"),
        ("rgb", "change_weighted_rgb_distance"),
    ):
        observed = comparisons.get(metric, {}).get(
            "object_over_flat_posterior", {}
        )
        entries.append(
            (
                f"object_beats_flat_by_task_on_change_{label}",
                task_comparison_passes(
                    observed, split, minimum_relative
                ),
                observed,
                {
                    "minimum_median_relative": minimum_relative,
                    **contract,
                },
            )
        )
    region_comparisons = evidence.get("rgb_regions", {})
    observed_change = region_comparisons.get("change", {}).get(
        "object_over_flat_posterior", {}
    )
    entries.append(
        (
            "object_beats_flat_by_task_on_observed_change_rgb",
            task_comparison_passes(
                observed_change, split, minimum_relative
            ),
            observed_change,
            {"minimum_median_relative": minimum_relative, **contract},
        )
    )
    static = region_comparisons.get("static", {}).get(
        "object_over_flat_posterior", {}
    )
    entries.append(
        (
            "object_static_rgb_task_noninferior",
            task_noninferiority_passes(static, split, 0.05),
            static,
            {"worst_relative_improvement": ">=-0.05", **contract},
        )
    )
    return entries
