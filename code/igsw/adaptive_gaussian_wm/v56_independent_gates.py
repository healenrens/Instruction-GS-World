"""Counterfactual gates for the v56 relation objective."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .v52_falsification import build_synthetic_objective_contract
from .verified_relation_objective_v56 import (
    verified_relation_object_state_terms,
)


def _one_hot(index: torch.Tensor, owners: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=owners).float()


def _replace_assignment(prediction, assignment):
    return replace(
        prediction,
        assignment=assignment,
        decoder_assignment=assignment.clone(),
    )


def _semantic_identity(teacher):
    frames = teacher.visibility.shape[1]
    return teacher.track_identity[:, None].expand(-1, frames, -1, -1).clone()


def _corruptions(prediction, teacher, config):
    index = prediction.assignment.argmax(dim=-1)
    object_track = teacher.object_confidence[:, None] > 0.5

    merged = torch.where(object_track, torch.zeros_like(index), index)
    merge_all = _replace_assignment(prediction, _one_hot(merged, config.owner_count))

    split = index.clone()
    split[:, split.shape[1] // 2 :, 0:2] = 2
    split_by_time = _replace_assignment(prediction, _one_hot(split, config.owner_count))

    scene = torch.where(
        object_track,
        torch.full_like(index, config.object_slots),
        index,
    )
    all_scene = _replace_assignment(prediction, _one_hot(scene, config.owner_count))

    uniform = prediction.assignment.clone()
    uniform[..., : config.object_slots] = 1.0 / config.object_slots
    uniform[..., config.object_slots :] = 0.0
    uniform_roots = _replace_assignment(prediction, uniform)

    per_track = index.clone()
    per_track[:, :, 0:4] = torch.arange(4, device=index.device)[None, None]
    track_fragmentation = _replace_assignment(
        prediction, _one_hot(per_track, config.owner_count)
    )

    identity_swap = prediction.identity.clone()
    semantic_swap = _semantic_identity(teacher)
    midpoint = identity_swap.shape[1] // 2
    identity_swap[:, midpoint:, 0:2] = prediction.identity[:, midpoint:, 2:4]
    identity_swap[:, midpoint:, 2:4] = prediction.identity[:, midpoint:, 0:2]
    semantic_swap[:, midpoint:, 0:2] = semantic_swap[:, midpoint:, 2:4].clone()
    semantic_swap[:, midpoint:, 2:4] = semantic_swap[:, midpoint:, 0:2].clone()

    return {
        "merge_all": (merge_all, _semantic_identity(teacher)),
        "split_by_time": (split_by_time, _semantic_identity(teacher)),
        "all_scene": (all_scene, _semantic_identity(teacher)),
        "uniform_roots": (uniform_roots, _semantic_identity(teacher)),
        "track_fragmentation": (
            track_fragmentation,
            _semantic_identity(teacher),
        ),
        "identity_swap": (
            replace(prediction, identity=identity_swap),
            semantic_swap,
        ),
        "dynamic_corruption": (
            replace(prediction, motion=torch.zeros_like(prediction.motion)),
            _semantic_identity(teacher),
        ),
        "lifecycle_corruption": (
            replace(
                prediction,
                visibility=1.0 - prediction.visibility,
                presence=torch.full_like(prediction.presence, 0.02),
            ),
            _semantic_identity(teacher),
        ),
    }


def run_v56_independent_gates(config, device: torch.device) -> dict:
    teacher, evidence, reasonable = build_synthetic_objective_contract(config, device)
    teacher = replace(
        teacher,
        scene_confidence=torch.zeros_like(teacher.scene_confidence),
        transient_confidence=torch.zeros_like(teacher.transient_confidence),
    )
    semantic = _semantic_identity(teacher)
    reference = verified_relation_object_state_terms(
        reasonable, semantic, teacher, evidence, config
    )
    corruptions = _corruptions(reasonable, teacher, config)
    results = {
        name: verified_relation_object_state_terms(
            prediction, corrupted_semantic, teacher, evidence, config
        )
        for name, (prediction, corrupted_semantic) in corruptions.items()
    }
    expected = {
        "merge_all": "relation_partition",
        "split_by_time": "contrastive_cycle",
        "all_scene": "object_support",
        "uniform_roots": "relation_same_partition",
        "track_fragmentation": "relation_same_partition",
        "identity_swap": "identity",
        "dynamic_corruption": "motion",
        "lifecycle_corruption": "lifecycle",
    }
    scores = {"reasonable": float(reference["target_total"])}
    margins, attribution = {}, {}
    for name, terms in results.items():
        scores[name] = float(terms["target_total"])
        margins[name] = scores[name] - scores["reasonable"]
        attribution[name] = float(terms[expected[name]] - reference[expected[name]])
    ranking = all(
        margin >= config.objective_falsification_margin for margin in margins.values()
    )
    attributed = all(value > 0.0 for value in attribution.values())
    checks = {
        "all_counterfactuals_ranked": ranking,
        "counterfactual_attribution": attributed,
        "reasonable_state_finite": bool(torch.isfinite(reference["target_total"])),
        "reasonable_uses_multiple_roots": (
            float(reference["verified_effective_roots"]) > 1.5
        ),
        "collapse_baseline_is_worse": (
            float(reference["relation_collapse_margin"])
            >= config.minimum_real_target_collapse_margin
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reasonable_score": scores["reasonable"],
        "scores": scores,
        "margins": margins,
        "attribution": attribution,
        "expected_attribution": expected,
        "reasonable_effective_roots": float(reference["verified_effective_roots"]),
        "reasonable_collapse_margin": float(reference["relation_collapse_margin"]),
    }
