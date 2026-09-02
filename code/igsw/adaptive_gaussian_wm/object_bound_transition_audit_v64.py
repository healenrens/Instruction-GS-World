"""Held-time falsification metrics for the object-bound transition teacher."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_bound_transition_teacher_v64 import (
    build_object_bound_membership_v64,
    build_seed_membership_v64,
    fit_shared_transition_v64,
    transition_horizon_tensors_v64,
)


def _weighted_mean(value, weight, dimensions):
    numerator = (value.float() * weight.float()).sum(dim=dimensions)
    denominator = weight.float().sum(dim=dimensions).clamp_min(1e-6)
    return numerator / denominator


def _track_features(features, visibility):
    weight = visibility.float()
    pooled = (features.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / weight.sum(dim=1).clamp_min(1.0)[..., None]
    return F.normalize(pooled, dim=-1, eps=1e-6)


def _feature_dispersion(features, visibility, membership):
    tracks = _track_features(features, visibility)
    persistence = visibility.float().mean(dim=1)
    weight = membership.float() * persistence
    centroid = (tracks * weight[..., None]).sum(dim=1)
    centroid = F.normalize(centroid, dim=-1, eps=1e-6)
    error = 1.0 - F.cosine_similarity(tracks, centroid[:, None], dim=-1)
    return _weighted_mean(error, weight, (1,))


def _motion_dispersion(evidence, membership, start, stop):
    pair_visible = evidence.visibility[:, start + 1 : stop]
    pair_visible = pair_visible & evidence.visibility[:, start : stop - 1]
    weight = membership[:, None].float() * pair_visible.float()
    flow = evidence.residual_flow[:, start : stop - 1].float()
    mean = (flow * weight[..., None]).sum(dim=(1, 2))
    mean = mean / weight.sum(dim=(1, 2)).clamp_min(1e-6)[..., None]
    error = (flow - mean[:, None, None]).square().sum(dim=-1)
    return _weighted_mean(error, weight, (1, 2)).sqrt()


def _geometry_instability(evidence, membership, start, stop):
    coordinates = evidence.coordinates[:, start:stop].float()
    visibility = evidence.visibility[:, start:stop].float()
    distance = (coordinates[:, :, :, None] - coordinates[:, :, None]).norm(dim=-1)
    pair_membership = membership[:, :, None] * membership[:, None]
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    weight = joint * pair_membership[:, None]
    diagonal = torch.eye(
        membership.shape[-1], device=membership.device, dtype=torch.bool
    )[None, None]
    weight = weight.masked_fill(diagonal, 0.0)
    temporal_count = weight.sum(dim=1)
    mean = (distance * weight).sum(dim=1) / temporal_count.clamp_min(1e-6)
    variance = ((distance - mean[:, None]).square() * weight).sum(dim=1)
    variance = variance / temporal_count.clamp_min(1e-6)
    pair_valid = (temporal_count > 1e-5).float()
    return _weighted_mean(variance.sqrt(), pair_valid, (1, 2))


def _persistence_transition_error(evidence, membership, start, stop, config):
    errors = []
    for horizon in config.transition_horizons:
        if stop - start <= horizon:
            continue
        _, flow, visible = transition_horizon_tensors_v64(
            evidence, start, stop, horizon
        )
        weight = membership[:, None].float() * visible
        error = _weighted_mean(flow.square().sum(dim=-1), weight, (1, 2)).sqrt()
        scale = torch.quantile(
            flow.norm(dim=-1).reshape(len(flow), -1), 0.75, dim=1
        ).clamp_min(0.005)
        errors.append(error / scale)
    return torch.stack(errors, dim=1).mean(dim=1)


def _membership_metrics(bundle, membership, start, stop, config):
    visibility = bundle.evidence.visibility[:, start:stop]
    _, _, transition, transition_valid = fit_shared_transition_v64(
        bundle.evidence, membership, start, stop, config
    )
    persistence = _persistence_transition_error(
        bundle.evidence, membership, start, stop, config
    )
    return {
        "dino_group_dispersion": _feature_dispersion(
            bundle.observation.dino[:, start:stop], visibility, membership
        ),
        "siglip_group_dispersion": _feature_dispersion(
            bundle.observation.siglip[:, start:stop], visibility, membership
        ),
        "motion_dispersion": _motion_dispersion(
            bundle.evidence, membership, start, stop
        ),
        "relative_geometry_instability": _geometry_instability(
            bundle.evidence, membership, start, stop
        ),
        "shared_transition_residual": transition,
        "shared_transition_valid": transition_valid.float(),
        "persistence_transition_error": persistence,
        "shared_transition_gain_over_persistence": persistence - transition,
    }


def _old_seed_membership(bundle, seeds, valid):
    batch = torch.arange(len(seeds), device=seeds.device)
    points = bundle.relation.same_confidence.shape[-1]
    membership = bundle.relation.same_confidence.float()[batch, seeds]
    membership = torch.maximum(membership, F.one_hot(seeds, points).float())
    return membership * valid[:, None].float()


def _alternate_membership(candidate):
    scores = candidate.effective_track_count.clone()
    batch = torch.arange(len(scores), device=scores.device)
    scores[batch, candidate.selected_component] = -1.0
    alternate_index = scores.argmax(dim=1)
    alternate = candidate.components[batch, alternate_index]
    alternate_valid = candidate.valid[batch, alternate_index]
    return alternate * alternate_valid[:, None].float(), alternate_valid


def object_bound_transition_audit_v64(bundle, sequence_index, config):
    frames = bundle.evidence.visibility.shape[1]
    split = frames // 2
    candidate = build_object_bound_membership_v64(
        bundle.observation,
        bundle.evidence,
        sequence_index,
        config,
        0,
        split,
    )
    membership = candidate.selected.float()
    rolled = torch.roll(membership, shifts=membership.shape[-1] // 2, dims=-1)
    swapped = torch.roll(membership, shifts=1, dims=0)
    alternate, alternate_valid = _alternate_membership(candidate)
    merged = torch.maximum(membership, alternate)
    old = _old_seed_membership(
        bundle, candidate.selected_seed, candidate.selected_valid
    )
    suffix_membership, suffix_valid, _, _ = build_seed_membership_v64(
        bundle.observation,
        bundle.evidence,
        candidate.selected_seed,
        config,
        split,
        frames,
    )
    candidate_metrics = _membership_metrics(bundle, membership, split, frames, config)
    corruptions = {
        "rolled": _membership_metrics(bundle, rolled, split, frames, config),
        "swapped": _membership_metrics(bundle, swapped, split, frames, config),
        "merged": _membership_metrics(bundle, merged, split, frames, config),
    }
    old_metrics = _membership_metrics(bundle, old, split, frames, config)
    suffix_visibility = bundle.evidence.visibility[:, split:].float().mean(dim=1)
    suffix_visible_mass = (membership * suffix_visibility).sum(dim=-1)
    suffix_visible_tracks = ((membership > 0.0) & (suffix_visibility > 0.0)).sum(dim=-1)
    audit_valid = candidate.selected_valid & (
        suffix_visible_tracks >= config.minimum_audit_visible_tracks
    )
    audit_valid = audit_valid & candidate_metrics["shared_transition_valid"].bool()
    audit_valid = audit_valid & suffix_valid
    result = {
        "candidate_valid": candidate.selected_valid.float(),
        "audit_valid": audit_valid.float(),
        "candidate_effective_track_count": (candidate.selected_effective_track_count),
        "candidate_membership_fraction": membership.mean(dim=-1),
        "candidate_suffix_visible_mass": suffix_visible_mass,
        "candidate_suffix_visible_tracks": suffix_visible_tracks.float(),
        "candidate_prefix_transition_residual": (
            candidate.selected_prefix_transition_residual
        ),
        "candidate_scene_fraction": candidate.scene_membership.mean(dim=-1),
        "candidate_unknown_fraction": candidate.unknown_membership.mean(dim=-1),
        "alternate_valid": alternate_valid.float(),
        "suffix_seed_valid": suffix_valid.float(),
        "prefix_suffix_membership_cosine_error": 1.0
        - F.cosine_similarity(membership, suffix_membership.float(), dim=-1),
    }
    for name, value in candidate_metrics.items():
        result[f"candidate_{name}"] = value
        result[f"old_seed_{name}"] = old_metrics[name]
        if name != "shared_transition_valid":
            result[f"improvement_over_old_{name}"] = old_metrics[name] - value
        for corruption, metrics in corruptions.items():
            result[f"{corruption}_{name}"] = metrics[name]
            if name != "shared_transition_valid":
                result[f"margin_{corruption}_{name}"] = metrics[name] - value
    return result
