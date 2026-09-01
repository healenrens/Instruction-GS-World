"""Temporally held-out evidence for the consensus object-membership teacher."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .consensus_object_membership_v63 import (
    build_consensus_object_membership_v63,
    consensus_affinity_v63,
    diffuse_seed_membership_v63,
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
    distance = (
        coordinates[:, :, :, None] - coordinates[:, :, None]
    ).norm(dim=-1)
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


def _membership_metrics(bundle, membership, start, stop):
    visibility = bundle.evidence.visibility[:, start:stop]
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
    }


def _old_seed_membership(bundle, seeds, valid):
    batch = torch.arange(len(seeds), device=seeds.device)
    points = bundle.relation.same_confidence.shape[-1]
    membership = bundle.relation.same_confidence.float()[batch, seeds]
    membership = torch.maximum(membership, F.one_hot(seeds, points).float())
    return membership * valid[:, None].float()


def consensus_teacher_audit_v63(bundle, sequence_index, config):
    frames = bundle.evidence.visibility.shape[1]
    split = frames // 2
    candidate = build_consensus_object_membership_v63(
        bundle.observation,
        bundle.evidence,
        sequence_index,
        config,
        0,
        split,
    )
    membership = candidate.selected.float()
    rolled = torch.roll(membership, shifts=membership.shape[-1] // 2, dims=-1)
    old = _old_seed_membership(
        bundle, candidate.selected_seed, candidate.selected_valid
    )
    suffix_graph = consensus_affinity_v63(
        bundle.observation, bundle.evidence, split, frames, config
    )
    suffix_membership = diffuse_seed_membership_v63(
        suffix_graph.affinity,
        suffix_graph.persistence,
        candidate.selected_seed,
    )
    candidate_metrics = _membership_metrics(bundle, membership, split, frames)
    rolled_metrics = _membership_metrics(bundle, rolled, split, frames)
    old_metrics = _membership_metrics(bundle, old, split, frames)
    suffix_visibility = bundle.evidence.visibility[:, split:].float().mean(dim=1)
    suffix_visible_mass = (membership * suffix_visibility).sum(dim=-1)
    audit_valid = candidate.selected_valid & (suffix_visible_mass >= 2.0)
    result = {
        "candidate_valid": candidate.selected_valid.float(),
        "audit_valid": audit_valid.float(),
        "candidate_effective_track_count": candidate.effective_track_count,
        "candidate_membership_fraction": membership.mean(dim=-1),
        "candidate_suffix_visible_mass": suffix_visible_mass,
        "prefix_suffix_membership_cosine_error": 1.0
        - F.cosine_similarity(membership, suffix_membership, dim=-1),
    }
    for name, value in candidate_metrics.items():
        result[f"candidate_{name}"] = value
        result[f"rolled_{name}"] = rolled_metrics[name]
        result[f"old_seed_{name}"] = old_metrics[name]
        result[f"margin_roll_{name}"] = rolled_metrics[name] - value
        result[f"improvement_over_old_{name}"] = old_metrics[name] - value
    return result
