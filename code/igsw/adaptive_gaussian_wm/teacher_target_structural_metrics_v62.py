"""Independent coherence measurements for the v62 teacher object target."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_sample_mean(value, weight, dimensions):
    numerator = (value.float() * weight.float()).sum(dim=dimensions)
    denominator = weight.float().sum(dim=dimensions).clamp_min(1e-6)
    return numerator / denominator


def _track_features(features, visibility):
    weight = visibility.float()
    pooled = (features.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / weight.sum(dim=1, keepdim=False).clamp_min(1.0)[..., None]
    return F.normalize(pooled, dim=-1, eps=1e-6)


def _feature_dispersion(features, visibility, membership):
    tracks = _track_features(features, visibility)
    persistence = visibility.float().mean(dim=1)
    weight = membership.float() * persistence
    centroid = (tracks * weight[..., None]).sum(dim=1)
    centroid = F.normalize(centroid, dim=-1, eps=1e-6)
    error = 1.0 - F.cosine_similarity(tracks, centroid[:, None], dim=-1)
    return _weighted_sample_mean(error, weight, (1,))


def _motion_dispersion(evidence, membership):
    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    weight = membership[:, None].float() * pair_visible.float()
    flow = evidence.residual_flow.float()
    mean = (flow * weight[..., None]).sum(dim=(1, 2))
    mean = mean / weight.sum(dim=(1, 2)).clamp_min(1e-6)[..., None]
    error = (flow - mean[:, None, None]).square().sum(dim=-1)
    return _weighted_sample_mean(error, weight, (1, 2)).sqrt()


def _rigidity_error(evidence, membership):
    coordinates = evidence.coordinates.float()
    visibility = evidence.visibility.float()
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    pair_membership = membership[:, :, None] * membership[:, None]
    diagonal = torch.eye(
        membership.shape[-1], device=membership.device, dtype=torch.bool
    )[None, None]
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    weight = joint * pair_membership[:, None]
    weight = weight.masked_fill(diagonal, 0.0)
    temporal_count = weight.sum(dim=1)
    temporal_weight = temporal_count.clamp_min(1e-6)
    mean = (distance * weight).sum(dim=1) / temporal_weight
    variance = ((distance - mean[:, None]).square() * weight).sum(dim=1)
    variance = variance / temporal_weight
    pair_valid = (temporal_count > 1e-5).float()
    return _weighted_sample_mean(variance.sqrt(), pair_valid, (1, 2))


def _relation_mean(relation, membership, name):
    values = getattr(relation, name).float()
    weight = membership[:, :, None] * membership[:, None]
    diagonal = torch.eye(
        membership.shape[-1], device=membership.device, dtype=torch.bool
    )[None]
    weight = weight.masked_fill(diagonal, 0.0)
    return _weighted_sample_mean(values, weight, (1, 2))


def _effective_tracks(membership):
    membership = membership.float()
    return membership.sum(dim=-1).square() / membership.square().sum(dim=-1).clamp_min(
        1e-6
    )


def _membership_metrics(bundle, membership):
    observation = bundle.observation
    evidence = bundle.evidence
    relation = bundle.relation
    persistence = evidence.visibility.float().mean(dim=1)
    motion = evidence.motion_salience.float().amax(dim=1)
    return {
        "dino_group_dispersion": _feature_dispersion(
            observation.dino, evidence.visibility, membership
        ),
        "siglip_group_dispersion": _feature_dispersion(
            observation.siglip, evidence.visibility, membership
        ),
        "motion_dispersion": _motion_dispersion(evidence, membership),
        "relative_geometry_instability": _rigidity_error(evidence, membership),
        "same_relation_evidence": _relation_mean(
            relation, membership, "same_confidence"
        ),
        "different_relation_evidence": _relation_mean(
            relation, membership, "different_confidence"
        ),
        "track_persistence": _weighted_sample_mean(persistence, membership, (1,)),
        "motion_salience": _weighted_sample_mean(motion, membership, (1,)),
        "effective_track_count": _effective_tracks(membership),
    }


def teacher_target_structural_metrics_v62(bundle):
    membership = bundle.observation.membership.float()
    shift = max(membership.shape[-1] // 2, 1)
    shuffled = torch.roll(membership, shifts=shift, dims=-1)
    selected_metrics = _membership_metrics(bundle, membership)
    shuffled_metrics = _membership_metrics(bundle, shuffled)
    result = {f"selected_{name}": value for name, value in selected_metrics.items()}
    result.update(
        {f"shuffled_{name}": value for name, value in shuffled_metrics.items()}
    )
    lower_is_better = (
        "dino_group_dispersion",
        "siglip_group_dispersion",
        "motion_dispersion",
        "relative_geometry_instability",
        "different_relation_evidence",
    )
    higher_is_better = (
        "same_relation_evidence",
        "track_persistence",
        "motion_salience",
    )
    for name in lower_is_better:
        result[f"margin_{name}"] = shuffled_metrics[name] - selected_metrics[name]
    for name in higher_is_better:
        result[f"margin_{name}"] = selected_metrics[name] - shuffled_metrics[name]
    lifecycle_known = bundle.relation.lifecycle_known.float()
    result.update(
        {
            "object_valid": bundle.observation.object_valid.float(),
            "selected_support_fraction": bundle.observation.support.float().mean(
                dim=(1, 2)
            ),
            "selected_lifecycle_known_fraction": _weighted_sample_mean(
                lifecycle_known,
                membership[:, None].expand_as(lifecycle_known),
                (1, 2),
            ),
        }
    )
    return result
