"""Independent counterfactual gates for v55 Object State semantics."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .relation_component_objective_v55 import (
    relation_component_object_state_terms,
)
from .v52_falsification import build_synthetic_objective_contract


def _semantic_identity(teacher) -> torch.Tensor:
    frames = teacher.visibility.shape[1]
    return teacher.track_identity[:, None].expand(-1, frames, -1, -1).clone()


def _merge_moving_components(prediction, teacher, config):
    index = prediction.assignment.argmax(dim=-1)
    moving_object = teacher.object_confidence[:, None] > 0.5
    merged = torch.where(moving_object, torch.zeros_like(index), index)
    assignment = F.one_hot(merged, num_classes=config.owner_count).float()
    return replace(
        prediction,
        assignment=assignment,
        decoder_assignment=assignment.clone(),
    )


def _swap_cross_frame_identity(prediction, semantic_identity):
    midpoint = prediction.identity.shape[1] // 2
    identity = prediction.identity.clone()
    semantic = semantic_identity.clone()
    identity[:, midpoint:, 0:2] = prediction.identity[:, midpoint:, 2:4]
    identity[:, midpoint:, 2:4] = prediction.identity[:, midpoint:, 0:2]
    semantic[:, midpoint:, 0:2] = semantic_identity[:, midpoint:, 2:4]
    semantic[:, midpoint:, 2:4] = semantic_identity[:, midpoint:, 0:2]
    return replace(prediction, identity=identity), semantic


def external_root_deletion_locality(prediction, teacher, config):
    """Measure deletion using externally related tracks, not predicted masks."""

    baseline = prediction.decoder_assignment.float()
    deleted = baseline.clone()
    removed = deleted[..., 0].clone()
    deleted[..., 0] = 0.0
    deleted[..., config.object_slots] += removed
    change = (deleted - baseline).abs().sum(dim=-1).mean(dim=1)

    seed = 0
    inside = teacher.same_confidence[:, seed].clone()
    inside[:, seed] = 1.0
    outside = teacher.different_confidence[:, seed]
    inside_change = (change * inside).sum() / inside.sum().clamp_min(1.0)
    outside_change = (change * outside).sum() / outside.sum().clamp_min(1.0)
    ratio = inside_change / outside_change.clamp_min(1e-6)
    return inside_change, outside_change, ratio


def run_v55_independent_gates(config, device: torch.device) -> dict:
    teacher, evidence, reasonable = build_synthetic_objective_contract(config, device)
    semantic = _semantic_identity(teacher)
    reference = relation_component_object_state_terms(
        reasonable, semantic, teacher, evidence, config
    )

    merged = _merge_moving_components(reasonable, teacher, config)
    merged_terms = relation_component_object_state_terms(
        merged, semantic, teacher, evidence, config
    )
    motion_separation_margin = float(
        merged_terms["component_utilization"]
        - reference["component_utilization"]
    )

    swapped, swapped_semantic = _swap_cross_frame_identity(reasonable, semantic)
    swapped_terms = relation_component_object_state_terms(
        swapped, swapped_semantic, teacher, evidence, config
    )
    identity_margin = float(swapped_terms["identity"] - reference["identity"])
    semantic_margin = float(
        swapped_terms["semantic_alignment"] - reference["semantic_alignment"]
    )

    inside, outside, locality = external_root_deletion_locality(
        reasonable, teacher, config
    )
    checks = {
        "root_deletion_locality": bool(
            float(inside) > 0.5 and float(locality) >= 10.0
        ),
        "cross_frame_identity": identity_margin
        >= config.objective_falsification_margin,
        "cross_frame_semantic_identity": semantic_margin
        >= config.objective_falsification_margin,
        "different_motion_component_separation": motion_separation_margin
        >= config.objective_falsification_margin,
        "reasonable_state_is_finite": bool(torch.isfinite(reference["target_total"])),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "root_deletion_inside_change": float(inside),
        "root_deletion_outside_change": float(outside),
        "root_deletion_locality_ratio": float(locality),
        "cross_frame_identity_margin": identity_margin,
        "cross_frame_semantic_margin": semantic_margin,
        "different_motion_component_margin": motion_separation_margin,
        "reasonable_component_loss": float(reference["component_utilization"]),
        "merged_component_loss": float(merged_terms["component_utilization"]),
    }
