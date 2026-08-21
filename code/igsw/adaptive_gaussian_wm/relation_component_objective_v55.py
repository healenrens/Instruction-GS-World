"""Relation-graph factorization objective for adaptive object components."""

from __future__ import annotations

import torch

from .object_state_target_v52 import (
    object_state_target_terms,
    visible_track_mean,
    weighted_mean,
)
from .relation_semantic_objective_v54 import semantic_alignment_terms


def relation_graph_component_terms(assignment, teacher, config):
    """Factor an external soft relation graph without a fixed component count."""

    object_assignment = assignment[..., : config.object_slots].float()
    object_probability = object_assignment.sum(dim=-1, keepdim=True)
    conditional = object_assignment / object_probability.clamp_min(1e-6)
    track_assignment = visible_track_mean(conditional, teacher.visibility)
    track_assignment = track_assignment / track_assignment.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)

    similarity = torch.einsum("bpk,bqk->bpq", track_assignment, track_assignment)
    similarity = similarity.clamp(1e-6, 1.0 - 1e-6)
    same = teacher.same_confidence.float()
    different = teacher.different_confidence.float()
    relation_evidence = same + different
    target = same / relation_evidence.clamp_min(1e-6)
    object_pair = (
        teacher.object_confidence[:, :, None]
        * teacher.object_confidence[:, None]
    )
    graph_weight = relation_evidence * object_pair
    graph_bce = -(
        target * similarity.log() + (1.0 - target) * (1.0 - similarity).log()
    )
    utilization = weighted_mean(graph_bce, graph_weight)
    same_partition = weighted_mean(1.0 - similarity, same * object_pair)
    different_partition = weighted_mean(similarity, different * object_pair)

    relation_degree = relation_evidence.amax(dim=-1)
    track_weight = teacher.object_confidence * relation_degree
    root_mass = torch.einsum("bpk,bp->bk", track_assignment, track_weight)
    root_share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    root_entropy = -(root_share * root_share.clamp_min(1e-6).log()).sum(dim=-1)
    effective_components = root_entropy.exp().mean()
    maximum_component_share = root_share.amax(dim=-1).mean()
    supported_roots = (root_mass > 0.05).float().sum(dim=-1).mean()
    return {
        "component_utilization": utilization,
        "component_same_partition": same_partition,
        "component_different_partition": different_partition,
        "component_effective_roots": effective_components,
        "component_maximum_root_share": maximum_component_share,
        "component_supported_roots": supported_roots,
        "component_relation_evidence": weighted_mean(
            relation_evidence, object_pair
        ),
    }


def relation_component_object_state_terms(
    prediction,
    semantic_identity: torch.Tensor,
    teacher,
    evidence,
    config,
) -> dict[str, torch.Tensor]:
    base = object_state_target_terms(prediction, teacher, evidence, config)
    component = relation_graph_component_terms(
        prediction.assignment, teacher, config
    )
    semantic = semantic_alignment_terms(semantic_identity, teacher)
    semantic_total = (
        semantic["semantic_alignment"]
        + 0.25 * semantic["semantic_same_relation"]
        + 0.25 * semantic["semantic_different_relation"]
    )
    target_without_semantic = (
        base["target_total"]
        + config.component_utilization_weight
        * component["component_utilization"]
    )
    target_total = (
        target_without_semantic
        + config.semantic_alignment_weight * semantic_total
    )
    terms = {
        **base,
        **component,
        **semantic,
        "semantic_total": semantic_total,
        "target_total_without_semantic": target_without_semantic,
        "target_total": target_total,
    }
    if not all(bool(torch.isfinite(value)) for value in terms.values()):
        raise RuntimeError("v55 relation-component objective contains non-finite terms")
    return terms
