"""Counterexamples that the v52 Object State objective must reject."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .object_state_target_v52 import ObjectStatePredictions, object_state_target_terms
from .point_track_teacher import PointTrackEvidence
from .trajectory_lifecycle import LIFECYCLE_OCCLUDED, LIFECYCLE_VISIBLE
from .trajectory_relation_teacher import TrajectoryRelationTeacher


def _one_hot_owner(index: torch.Tensor, owners: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=owners).float()


def build_synthetic_objective_contract(config, device: torch.device):
    batch, frames, points = 1, 6, 8
    owners = config.owner_count
    coordinates = torch.zeros(batch, frames, points, 2, device=device)
    base = torch.tensor(
        [
            [-0.65, -0.20], [-0.50, -0.20], [0.45, 0.15], [0.60, 0.15],
            [-0.75, 0.70], [0.75, 0.70], [-0.30, -0.75], [0.30, -0.75],
        ],
        device=device,
    )
    coordinates[:] = base
    time = torch.arange(frames, device=device).float()
    coordinates[0, :, 0:2, 0] += 0.04 * time[:, None]
    coordinates[0, :, 2:4, 0] -= 0.04 * time[:, None]
    visibility = torch.ones(batch, frames, points, device=device, dtype=torch.bool)
    visibility[:, 2:4, 1] = False
    track_identity = torch.zeros(batch, points, config.patch_dim, device=device)
    track_identity[:, 0:2, 0] = 1.0
    track_identity[:, 2:4, 1] = 1.0
    track_identity[:, 4:6, 2] = 1.0
    track_identity[:, 6:8, 3] = 1.0
    track_identity = F.normalize(track_identity, dim=-1)
    same = torch.zeros(batch, points, points, device=device)
    same[:, 0, 1] = same[:, 1, 0] = 1.0
    same[:, 2, 3] = same[:, 3, 2] = 1.0
    different = torch.zeros_like(same)
    different[:, 0:2, 2:4] = 1.0
    different[:, 2:4, 0:2] = 1.0
    object_confidence = torch.zeros(batch, points, device=device)
    object_confidence[:, :4] = 1.0
    scene_confidence = torch.zeros_like(object_confidence)
    scene_confidence[:, 4:6] = 1.0
    transient_confidence = torch.zeros_like(object_confidence)
    transient_confidence[:, 6:8] = 1.0
    lifecycle_known = torch.ones_like(visibility)
    lifecycle_state = torch.full_like(visibility, LIFECYCLE_VISIBLE, dtype=torch.long)
    lifecycle_state[:, 2:4, 1] = LIFECYCLE_OCCLUDED
    presence = torch.ones_like(visibility, dtype=torch.float32)
    motion = torch.zeros(
        batch, frames, points, len(config.dynamic_horizons), 2, device=device
    )
    motion[:, :, 0:2, :, 0] = 0.50
    motion[:, :, 2:4, :, 0] = -0.50
    motion_valid = visibility[..., None].expand_as(motion[..., 0]).clone()
    teacher = TrajectoryRelationTeacher(
        track_identity=track_identity,
        persistence=visibility.float().mean(dim=1),
        object_confidence=object_confidence,
        scene_confidence=scene_confidence,
        transient_confidence=transient_confidence,
        same_confidence=same,
        different_confidence=different,
        visibility=visibility.float(),
        presence=presence,
        lifecycle_known=lifecycle_known,
        lifecycle_state=lifecycle_state,
        motion=motion,
        motion_valid=motion_valid,
        relation_score=same,
    )
    flow = coordinates[:, 1:] - coordinates[:, :-1]
    evidence = PointTrackEvidence(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=flow,
        motion_salience=flow.norm(dim=-1),
        query_times=torch.zeros(points, device=device, dtype=torch.long),
        sampled_features=track_identity[:, None].expand(-1, frames, -1, -1),
    )
    owner_index = torch.tensor(
        [0, 0, 1, 1, config.object_slots, config.object_slots,
         config.object_slots + 1, config.object_slots + 1],
        device=device,
    )[None, None].expand(batch, frames, -1).clone()
    assignment = _one_hot_owner(owner_index, owners)
    identity = torch.zeros(
        batch, frames, points, config.identity_dim, device=device
    )
    identity[:, :, 0:2, 0] = 1.0
    identity[:, :, 2:4, 1] = 1.0
    identity[:, :, 4:6, 2] = 1.0
    identity[:, :, 6:8, 3] = 1.0
    center = coordinates.clone()
    center[:, :, 0:2] = coordinates[:, :, 0:2].mean(dim=2, keepdim=True)
    center[:, :, 2:4] = coordinates[:, :, 2:4].mean(dim=2, keepdim=True)
    visible_prediction = visibility.float() * 0.98 + (~visibility).float() * 0.02
    prediction = ObjectStatePredictions(
        assignment=assignment,
        identity=identity,
        motion=motion.clone(),
        center=center,
        visibility=visible_prediction,
        presence=torch.full_like(visible_prediction, 0.98),
        decoder_assignment=assignment.clone(),
    )
    return teacher, evidence, prediction


def _replace_assignment(prediction, assignment):
    return replace(
        prediction,
        assignment=assignment,
        decoder_assignment=assignment.clone(),
    )


def _corruptions(prediction, teacher, config):
    assignments = prediction.assignment
    owner_index = assignments.argmax(dim=-1)
    object_track = teacher.object_confidence[:, None] > 0.5

    merged_index = torch.where(object_track, torch.zeros_like(owner_index), owner_index)
    merge_all = _replace_assignment(
        prediction, _one_hot_owner(merged_index, config.owner_count)
    )

    split_index = owner_index.clone()
    split_index[:, 3:, 0:2] = 2
    split_by_time = _replace_assignment(
        prediction, _one_hot_owner(split_index, config.owner_count)
    )

    swap_index = owner_index.clone()
    swap_index[:, 3:, 0:2] = 1
    swap_index[:, 3:, 2:4] = 0
    identity_swap = _replace_assignment(
        prediction, _one_hot_owner(swap_index, config.owner_count)
    )
    swapped_identity = identity_swap.identity.clone()
    swapped_identity[:, 3:, 0:2] = prediction.identity[:, :1, 2:3]
    swapped_identity[:, 3:, 2:4] = prediction.identity[:, :1, 0:1]
    identity_swap = replace(identity_swap, identity=swapped_identity)

    scene_index = torch.where(
        object_track,
        torch.full_like(owner_index, config.object_slots),
        owner_index,
    )
    all_scene = _replace_assignment(
        prediction, _one_hot_owner(scene_index, config.owner_count)
    )

    background_index = owner_index.clone()
    background_index[:, :, 4:6] = 0
    background_lock = _replace_assignment(
        prediction, _one_hot_owner(background_index, config.owner_count)
    )

    fragmented_index = owner_index.clone()
    fragmented_index[:, :, 0] = 0
    fragmented_index[:, :, 1] = 2
    fragmented_index[:, :, 2] = 1
    fragmented_index[:, :, 3] = 3
    track_per_slot = _replace_assignment(
        prediction, _one_hot_owner(fragmented_index, config.owner_count)
    )

    dynamic_corruption = replace(prediction, motion=torch.zeros_like(prediction.motion))
    lifecycle_corruption = replace(
        prediction,
        visibility=1.0 - prediction.visibility,
        presence=torch.full_like(prediction.presence, 0.02),
    )
    return {
        "merge_all": merge_all,
        "split_by_time": split_by_time,
        "identity_swap": identity_swap,
        "all_scene": all_scene,
        "background_lock": background_lock,
        "track_per_slot": track_per_slot,
        "dynamic_corruption": dynamic_corruption,
        "lifecycle_corruption": lifecycle_corruption,
    }


def run_objective_falsification(config, device: torch.device) -> dict:
    teacher, evidence, reasonable = build_synthetic_objective_contract(config, device)
    reference = object_state_target_terms(reasonable, teacher, evidence, config)
    corruptions = _corruptions(reasonable, teacher, config)
    results = {
        name: object_state_target_terms(value, teacher, evidence, config)
        for name, value in corruptions.items()
    }
    expected = {
        "merge_all": "different_relation",
        "split_by_time": "track_cycle",
        "identity_swap": "identity",
        "all_scene": "owner_evidence",
        "background_lock": "owner_evidence",
        "track_per_slot": "same_relation",
        "dynamic_corruption": "motion",
        "lifecycle_corruption": "lifecycle",
    }
    scores = {"reasonable": float(reference["target_total"])}
    margins, attribution = {}, {}
    for name, terms in results.items():
        scores[name] = float(terms["target_total"])
        margins[name] = scores[name] - scores["reasonable"]
        term = expected[name]
        attribution[name] = float(terms[term] - reference[term])
    ranking_passed = all(
        value >= config.objective_falsification_margin for value in margins.values()
    )
    attribution_passed = all(value > 0.0 for value in attribution.values())
    return {
        "status": "passed" if ranking_passed and attribution_passed else "failed",
        "reasonable_score": scores["reasonable"],
        "scores": scores,
        "margins": margins,
        "attribution": attribution,
        "expected_attribution": expected,
        "ranking_passed": ranking_passed,
        "attribution_passed": attribution_passed,
    }
