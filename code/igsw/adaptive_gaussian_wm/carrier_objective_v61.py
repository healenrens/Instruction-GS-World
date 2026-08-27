"""External-teacher objective for single-encoder continuous Object State."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .fixed_teacher_projection_v61 import fixed_group_projection_v61
from .point_track_teacher import sample_patch_field


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _binary_probability_loss(prediction, target, weight):
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    value = (
        -target.float() * prediction.log()
        - (1.0 - target.float()) * (1.0 - prediction).log()
    )
    return _weighted_mean(value, weight.float())


def track_assignments_v61(state, field, evidence, config):
    support_field = state.carriers.support.permute(0, 1, 3, 2)
    assignment = sample_patch_field(
        support_field.float(), evidence.coordinates.float(), field.grid_hw
    )
    assignment = assignment * state.carriers.presence[:, :, None]
    assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    root = torch.einsum(
        "btpq,btqm->btpm",
        assignment,
        state.roots.owner[..., : config.object_roots],
    )
    root = root / root.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return assignment, root


def _relation_terms(root_assignment, relation, evidence):
    visible = evidence.visibility.float()
    average = (root_assignment * visible[..., None]).sum(dim=1)
    average = average / visible.sum(dim=1)[..., None].clamp_min(1.0)
    coassignment = torch.einsum("bpm,bqm->bpq", average, average).clamp(
        1e-6, 1.0 - 1e-6
    )
    same = relation.same_confidence.float()
    different = relation.different_confidence.float()
    same_loss = _weighted_mean(-coassignment.log(), same)
    different_loss = _weighted_mean(-(1.0 - coassignment).log(), different)
    return same_loss, different_loss, average, coassignment


def _temporal_track_terms(state, carrier_assignment, root_assignment, evidence):
    pair_visible = evidence.visibility[:, 1:] & evidence.visibility[:, :-1]
    pair_weight = pair_visible.float()
    carrier_cycle = 1.0 - F.cosine_similarity(
        carrier_assignment[:, 1:].float(), carrier_assignment[:, :-1].float(), dim=-1
    )
    root_cycle = 1.0 - F.cosine_similarity(
        root_assignment[:, 1:].float(), root_assignment[:, :-1].float(), dim=-1
    )
    identity = torch.einsum(
        "btpq,btqd->btpd", carrier_assignment, state.carriers.identity.float()
    )
    identity = F.normalize(identity, dim=-1, eps=1e-6)
    identity_cycle = 1.0 - F.cosine_similarity(
        identity[:, 1:], identity[:, :-1], dim=-1
    )
    return (
        _weighted_mean(carrier_cycle, pair_weight),
        _weighted_mean(root_cycle, pair_weight),
        _weighted_mean(identity_cycle, pair_weight),
        identity,
    )


def _geometry_lifecycle_terms(
    model, state, assignment, root_assignment, evidence, relation
):
    coordinate = torch.einsum(
        "btpq,btqd->btpd", assignment, state.carriers.center.float()
    )
    coordinate_error = (coordinate - evidence.coordinates.float()).norm(dim=-1)
    visible = evidence.visibility.float()
    geometry = _weighted_mean(coordinate_error, visible)
    visibility = torch.einsum(
        "btpm,btm->btp", root_assignment, state.roots.visibility.float()
    )
    presence = torch.einsum(
        "btpm,btm->btp", root_assignment, state.roots.presence.float()
    )
    lifecycle_weight = relation.lifecycle_known.float()
    visibility_loss = _binary_probability_loss(
        visibility, relation.visibility, lifecycle_weight
    )
    presence_loss = _binary_probability_loss(
        presence, relation.presence, lifecycle_weight
    )
    track_dynamic = torch.einsum(
        "btpq,btqd->btpd", assignment, state.carriers.dynamic.float()
    )
    predicted_motion = model.motion_readout(track_dynamic).reshape(
        *track_dynamic.shape[:3], len(model.config.dynamic_horizons), 2
    )
    motion_error = (predicted_motion - relation.motion.float()).square().sum(dim=-1)
    motion = _weighted_mean(motion_error, relation.motion_valid.float())
    return (
        geometry,
        visibility_loss,
        presence_loss,
        motion,
        coordinate_error,
        visibility,
        presence,
    )


def _dino_alignment(model, state, assignment, evidence):
    identity = torch.einsum(
        "btpq,btqd->btpd", assignment, state.carriers.identity.float()
    )
    prediction = F.normalize(model.identity_to_dino(identity), dim=-1, eps=1e-6)
    target = fixed_group_projection_v61(
        evidence.sampled_features, model.config.teacher_projection_dim
    )
    error = 1.0 - (prediction * target).sum(dim=-1)
    return _weighted_mean(error, evidence.visibility.float()), error


def _object_semantic_alignment(model, state, root_average, components):
    membership = components.membership.float()
    component_root = torch.einsum("bmp,bpr->bmr", membership, root_average)
    component_root = component_root / membership.sum(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    component_root = component_root / component_root.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    root_identity = state.roots.identity.index_select(1, components.frame_indices)
    root_semantic = F.normalize(model.root_to_siglip(root_identity), dim=-1, eps=1e-6)
    prediction = torch.einsum("bmr,bard->bamd", component_root, root_semantic)
    prediction = F.normalize(prediction, dim=-1, eps=1e-6)
    target = fixed_group_projection_v61(
        components.semantic, model.config.teacher_projection_dim
    )
    error = 1.0 - (prediction * target).sum(dim=-1)
    valid = components.semantic_valid & components.valid[:, None]
    alignment = _weighted_mean(error, valid.float())
    final_valid = valid[:, -1]
    predicted_vectors = prediction[:, -1][final_valid]
    target_vectors = target[:, -1][final_valid]
    if len(predicted_vectors) > 1:
        logits = predicted_vectors @ target_vectors.T / 0.07
        labels = torch.arange(len(predicted_vectors), device=logits.device)
        retrieval = F.cross_entropy(logits, labels)
        retrieval_accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    else:
        retrieval = alignment * 0.0
        retrieval_accuracy = alignment.detach() * 0.0
    return alignment, retrieval, retrieval_accuracy, error


def _state_regularizers(state):
    center = state.carriers.center.float()
    distance = (center[:, :, :, None] - center[:, :, None]).norm(dim=-1)
    diagonal = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)
    repulsion = (
        torch.exp(-distance / 0.10).masked_fill(diagonal[None, None], 0.0).mean()
    )
    owner = state.roots.owner.float().mean(dim=(0, 1, 2))
    owner_entropy = -(owner * owner.clamp_min(1e-6).log()).sum()
    maximum_entropy = torch.log(torch.tensor(float(len(owner)), device=owner.device))
    root_balance = maximum_entropy - owner_entropy
    carrier_presence = state.carriers.presence.float().mean()
    effective_carriers = state.carriers.presence.float().sum(dim=-1).mean()
    effective_roots = (
        state.roots.presence.float().sum(dim=-1).square()
        / state.roots.presence.float().square().sum(dim=-1).clamp_min(1e-6)
    ).mean()
    weight = state.roots.owner[..., : state.roots.identity.shape[2]]
    weight = weight / weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
    carrier_identity = torch.einsum(
        "btqm,btqd->btmd", weight, state.carriers.identity.float()
    )
    carrier_dynamic = torch.einsum(
        "btqm,btqd->btmd", weight, state.carriers.dynamic.float()
    )
    identity_alignment = (
        1.0
        - F.cosine_similarity(
            F.normalize(carrier_identity, dim=-1, eps=1e-6),
            state.roots.identity.float(),
            dim=-1,
        ).mean()
    )
    dynamic_alignment = F.smooth_l1_loss(carrier_dynamic, state.roots.dynamic.float())
    return (
        repulsion,
        root_balance,
        identity_alignment,
        dynamic_alignment,
        carrier_presence,
        effective_carriers,
        effective_roots,
    )


def _track_appearance_retrieval(identity, visibility):
    source = identity[:, 0, 1::2]
    target = identity[:, -1, 1::2]
    valid = visibility[:, 0, 1::2] & visibility[:, -1, 1::2]
    temporal_error = _weighted_mean(
        1.0 - F.cosine_similarity(source, target, dim=-1), valid.float()
    )
    source_vectors = source[valid]
    target_vectors = target[valid]
    if len(source_vectors) > 1:
        logits = source_vectors @ target_vectors.T / 0.07
        labels = torch.arange(len(source_vectors), device=logits.device)
        retrieval_accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    else:
        retrieval_accuracy = temporal_error.detach() * 0.0
    return temporal_error, retrieval_accuracy


def _reappearance_identity_error(identity, evidence, relation):
    visibility = evidence.visibility.bool()
    lifecycle_known = relation.lifecycle_known.bool()
    presence = relation.presence.float() >= 0.5
    reference = identity[:, 0]
    seen = visibility[:, 0] & lifecycle_known[:, 0]
    occluded = torch.zeros_like(seen)
    errors, weights = [], []
    for frame in range(1, identity.shape[1]):
        current_visible = visibility[:, frame] & lifecycle_known[:, frame]
        reappeared = current_visible & occluded & seen
        errors.append(1.0 - F.cosine_similarity(identity[:, frame], reference, dim=-1))
        weights.append(reappeared.float())
        hidden = (
            lifecycle_known[:, frame]
            & presence[:, frame]
            & ~visibility[:, frame]
            & seen
        )
        occluded = (occluded | hidden) & ~current_visible
        reference = torch.where(
            current_visible[..., None], identity[:, frame], reference
        )
        seen = seen | current_visible
    error = torch.stack(errors, dim=1)
    weight = torch.stack(weights, dim=1)
    return _weighted_mean(error, weight), weight.sum()


def continuous_carrier_objective_v61(
    model, field, state, evidence, relation, components
):
    config = model.config
    carrier_assignment, root_assignment = track_assignments_v61(
        state, field, evidence, config
    )
    same, different, root_average, _ = _relation_terms(
        root_assignment, relation, evidence
    )
    carrier_cycle, root_cycle, identity_cycle, identity = _temporal_track_terms(
        state, carrier_assignment, root_assignment, evidence
    )
    values = _geometry_lifecycle_terms(
        model,
        state,
        carrier_assignment,
        root_assignment,
        evidence,
        relation,
    )
    (
        geometry,
        visibility,
        presence,
        motion,
        coordinate_error,
        visibility_prediction,
        _,
    ) = values
    dino = geometry * 0.0
    dino_error = coordinate_error * 0.0
    if config.uses_dino_alignment:
        dino, dino_error = _dino_alignment(model, state, carrier_assignment, evidence)
    object_semantic = geometry * 0.0
    object_retrieval = geometry * 0.0
    object_retrieval_accuracy = geometry.detach() * 0.0
    object_semantic_error = coordinate_error[:, :1, : config.object_roots] * 0.0
    if config.uses_object_semantics:
        values = _object_semantic_alignment(model, state, root_average, components)
        (
            object_semantic,
            object_retrieval,
            object_retrieval_accuracy,
            object_semantic_error,
        ) = values
    regularizers = _state_regularizers(state)
    (
        repulsion,
        root_balance,
        root_identity_alignment,
        root_dynamic_alignment,
        carrier_presence,
        effective_carriers,
        effective_roots,
    ) = regularizers
    track_appearance_error, track_appearance_retrieval = _track_appearance_retrieval(
        identity, evidence.visibility
    )
    reappearance_error, reappearance_count = _reappearance_identity_error(
        identity, evidence, relation
    )
    relation_loss = same + different
    assignment_loss = carrier_cycle + root_cycle
    lifecycle_loss = visibility + presence
    loss = (
        config.track_assignment_weight * assignment_loss
        + config.relation_weight * relation_loss
        + config.geometry_weight * geometry
        + config.geometry_weight * motion
        + config.lifecycle_weight * lifecycle_loss
        + config.identity_weight * identity_cycle
        + config.root_carrier_alignment_weight * root_identity_alignment
        + config.root_carrier_alignment_weight * root_dynamic_alignment
        + config.dino_alignment_weight * dino
        + config.object_semantic_weight * (object_semantic + object_retrieval)
        + config.carrier_diversity_weight * repulsion
        + config.root_balance_weight * root_balance
    )
    parts = {
        "object_state_loss": loss.detach(),
        "carrier_track_cycle_error": carrier_cycle.detach(),
        "object_root_track_cycle_error": root_cycle.detach(),
        "identity_temporal_error": identity_cycle.detach(),
        "relation_same_error": same.detach(),
        "relation_different_error": different.detach(),
        "track_coordinate_error": geometry.detach(),
        "track_motion_error": motion.detach(),
        "visibility_error": visibility.detach(),
        "presence_error": presence.detach(),
        "carrier_repulsion": repulsion.detach(),
        "root_balance_penalty": root_balance.detach(),
        "root_carrier_identity_alignment_error": root_identity_alignment.detach(),
        "root_carrier_dynamic_alignment_error": root_dynamic_alignment.detach(),
        "carrier_presence_mean": carrier_presence.detach(),
        "effective_carriers": effective_carriers.detach(),
        "effective_object_roots": effective_roots.detach(),
        "scene_owner_fraction": state.roots.owner[..., -1].mean().detach(),
        "track_visibility_prediction_mean": visibility_prediction.mean().detach(),
        "track_reappearance_identity_error": reappearance_error.detach(),
        "track_reappearance_count": reappearance_count.detach(),
        "heldout_track_appearance_temporal_error": track_appearance_error.detach(),
        "heldout_track_appearance_retrieval_accuracy": (
            track_appearance_retrieval.detach()
        ),
        "continuous_track_count": evidence.visibility.float()
        .sum(dim=-1)
        .mean()
        .detach(),
    }
    if config.uses_dino_alignment:
        parts["dino_local_alignment_error"] = dino.detach()
        parts["dino_local_error_unweighted"] = dino_error.mean().detach()
        parts["heldout_track_dino_error"] = _weighted_mean(
            dino_error[..., 1::2], evidence.visibility[..., 1::2].float()
        ).detach()
    if config.uses_object_semantics:
        parts["object_semantic_alignment_error"] = object_semantic.detach()
        parts["object_semantic_retrieval_loss"] = object_retrieval.detach()
        parts["object_semantic_retrieval_accuracy"] = object_retrieval_accuracy.detach()
        parts["object_semantic_error_unweighted"] = (
            object_semantic_error.mean().detach()
        )
    return loss, parts, carrier_assignment, root_assignment
