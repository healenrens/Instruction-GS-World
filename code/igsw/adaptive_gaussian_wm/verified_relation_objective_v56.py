"""Non-collapsing relation objective for pure-video Object State learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_state_target_v52 import (
    _identity_term,
    _lifecycle_term,
    _motion_geometry_terms,
    visible_track_mean,
)
from .relation_semantic_objective_v54 import semantic_alignment_terms


def _batch_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions)
    usable = (denominator > 0.0).float()
    per_sample = numerator / denominator.clamp_min(1e-6)
    return (per_sample * usable).sum() / usable.sum().clamp_min(1.0)


def _graph_bce(similarity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = similarity.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float().clamp(0.0, 1.0)
    return -(target * probability.log() + (1.0 - target) * torch.log1p(-probability))


def _conditional_object_assignment(assignment, object_slots):
    objects = assignment[..., :object_slots].float()
    object_probability = objects.sum(dim=-1, keepdim=True)
    conditional = objects / object_probability.clamp_min(1e-6)
    return conditional, object_probability.squeeze(-1)


def balanced_relation_weights(teacher):
    support = teacher.object_confidence.float()
    support_pair = support[:, :, None] * support[:, None]
    same = teacher.same_confidence.float() * support_pair
    different = teacher.different_confidence.float() * support_pair
    dimensions = tuple(range(1, same.ndim))
    same_mass = same.sum(dim=dimensions, keepdim=True)
    different_mass = different.sum(dim=dimensions, keepdim=True)
    same_present = same_mass > 0.0
    different_present = different_mass > 0.0
    both = same_present & different_present
    same_scale = torch.where(both, 0.5, 1.0)
    different_scale = torch.where(both, 0.5, 1.0)
    same_weight = same / same_mass.clamp_min(1e-6) * same_scale
    different_weight = different / different_mass.clamp_min(1e-6) * different_scale
    return same_weight, different_weight


def _relation_partition_terms(conditional, teacher):
    track_assignment = visible_track_mean(conditional, teacher.visibility)
    track_assignment = track_assignment / track_assignment.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    similarity = torch.einsum(
        "bpk,bqk->bpq", track_assignment.float(), track_assignment.float()
    )
    same_weight, different_weight = balanced_relation_weights(teacher)
    weight = same_weight + different_weight
    target = same_weight / weight.clamp_min(1e-6)
    partition = _batch_weighted_mean(_graph_bce(similarity, target), weight)
    same_loss = _batch_weighted_mean(1.0 - similarity, same_weight)
    different_loss = _batch_weighted_mean(similarity, different_weight)
    collapsed = _batch_weighted_mean(
        _graph_bce(torch.ones_like(similarity), target), weight
    )
    return {
        "track_assignment": track_assignment,
        "relation_partition": partition,
        "relation_same_partition": same_loss,
        "relation_different_partition": different_loss,
        "relation_collapse_baseline": collapsed,
        "relation_collapse_margin": collapsed - partition,
    }


def _contrastive_cycle_terms(conditional, track_assignment, teacher, config):
    positive = torch.einsum(
        "btpk,bpk->btp", conditional.float(), track_assignment.float()
    )
    negative = torch.einsum(
        "btpk,bqk->btpq", conditional.float(), track_assignment.float()
    )
    different = teacher.different_confidence.float()
    support_pair = (
        teacher.object_confidence[:, :, None] * teacher.object_confidence[:, None]
    )
    weight = (
        teacher.visibility.float()[..., None]
        * different[:, None]
        * support_pair[:, None]
    )
    ranking = (
        F.softplus(
            (negative - positive[..., None] + config.cycle_margin)
            / config.cycle_temperature
        )
        * config.cycle_temperature
    )
    loss = _batch_weighted_mean(ranking, weight)
    positive_metric = _batch_weighted_mean(
        positive[..., None].expand_as(negative), weight
    )
    negative_metric = _batch_weighted_mean(negative, weight)
    return {
        "contrastive_cycle": loss,
        "cycle_positive_similarity": positive_metric,
        "cycle_negative_similarity": negative_metric,
        "cycle_similarity_margin": positive_metric - negative_metric,
    }


def _object_support_term(object_probability, teacher):
    weight = teacher.visibility.float() * teacher.object_confidence[:, None]
    loss = _batch_weighted_mean(-object_probability.clamp_min(1e-6).log(), weight)
    probability = _batch_weighted_mean(object_probability, weight)
    return loss, probability


def _decoder_support_term(prediction, teacher):
    target = prediction.assignment.detach().float()
    decoder = prediction.decoder_assignment.float().clamp_min(1e-6)
    cross_entropy = -(target * decoder.log()).sum(dim=-1)
    weight = teacher.visibility.float() * teacher.object_confidence[:, None]
    return _batch_weighted_mean(cross_entropy, weight)


def _root_metrics(track_assignment, teacher):
    relation_degree = (teacher.same_confidence + teacher.different_confidence).amax(
        dim=-1
    )
    track_weight = teacher.object_confidence * relation_degree
    root_mass = (track_assignment * track_weight[..., None]).sum(dim=1)
    share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -(share * share.clamp_min(1e-6).log()).sum(dim=-1)
    effective = entropy.exp()
    maximum = share.amax(dim=-1)
    negative_supported = teacher.different_confidence.amax(dim=(-2, -1)) > 0.0
    usable = negative_supported.float()
    return {
        "verified_effective_roots": effective.mean(),
        "verified_maximum_root_share": maximum.mean(),
        "verified_supported_roots": (root_mass > 0.05).float().sum(dim=-1).mean(),
        "verified_negative_supported_effective_roots": (
            (effective * usable).sum() / usable.sum().clamp_min(1.0)
        ),
        "verified_negative_supported_maximum_root_share": (
            (maximum * usable).sum() / usable.sum().clamp_min(1.0)
        ),
        "verified_negative_supported_sample_fraction": usable.mean(),
    }


def verified_relation_object_state_terms(
    prediction,
    semantic_identity: torch.Tensor,
    teacher,
    evidence,
    config,
) -> dict[str, torch.Tensor]:
    conditional, object_probability = _conditional_object_assignment(
        prediction.assignment, config.object_slots
    )
    relation = _relation_partition_terms(conditional, teacher)
    cycle = _contrastive_cycle_terms(
        conditional, relation["track_assignment"], teacher, config
    )
    object_support, supported_probability = _object_support_term(
        object_probability, teacher
    )
    identity, identity_temporal, identity_same, identity_negative = _identity_term(
        prediction, teacher, config
    )
    motion, geometry = _motion_geometry_terms(prediction, teacher, evidence)
    lifecycle, visibility, presence = _lifecycle_term(prediction, teacher)
    decoder_support = _decoder_support_term(prediction, teacher)
    semantic = semantic_alignment_terms(semantic_identity, teacher)
    semantic_total = (
        semantic["semantic_alignment"]
        + 0.25 * semantic["semantic_same_relation"]
        + 0.25 * semantic["semantic_different_relation"]
    )
    contributions = {
        "contribution_relation_partition": (
            config.relation_partition_weight * relation["relation_partition"]
        ),
        "contribution_contrastive_cycle": (
            config.contrastive_cycle_weight * cycle["contrastive_cycle"]
        ),
        "contribution_object_support": (config.object_support_weight * object_support),
        "contribution_identity": config.identity_weight * identity,
        "contribution_semantic": config.semantic_alignment_weight * semantic_total,
        "contribution_motion": config.motion_weight * motion,
        "contribution_lifecycle": config.lifecycle_weight * lifecycle,
        "contribution_geometry": config.geometry_weight * geometry,
        "contribution_decoder_support": (
            config.decoder_support_weight * decoder_support
        ),
    }
    total = sum(contributions.values())
    roots = _root_metrics(relation["track_assignment"], teacher)
    terms = {
        "target_total": total,
        "relation_partition": relation["relation_partition"],
        "relation_same_partition": relation["relation_same_partition"],
        "relation_different_partition": relation["relation_different_partition"],
        "relation_collapse_baseline": relation["relation_collapse_baseline"],
        "relation_collapse_margin": relation["relation_collapse_margin"],
        **cycle,
        "object_support": object_support,
        "supported_object_probability": supported_probability,
        "identity": identity,
        "identity_temporal": identity_temporal,
        "identity_same": identity_same,
        "identity_negative": identity_negative,
        "motion": motion,
        "geometry": geometry,
        "lifecycle": lifecycle,
        "visibility": visibility,
        "presence": presence,
        "decoder_support": decoder_support,
        **semantic,
        "semantic_total": semantic_total,
        **roots,
        **contributions,
    }
    nonfinite = [
        name for name, value in terms.items() if not bool(torch.isfinite(value))
    ]
    if nonfinite:
        raise RuntimeError(
            "v56 verified relation objective contains non-finite terms: "
            + ", ".join(nonfinite)
        )
    return terms
