"""External-observation objective for posterior carrier Dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .fixed_teacher_projection_v61 import fixed_group_projection_v61


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _probability_loss(prediction, target, weight):
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    value = -target.float() * prediction.log()
    value = value - (1.0 - target.float()) * (1.0 - prediction).log()
    return _weighted_mean(value, weight.float())


def _decode_tracks(model, state, carrier_assignment, root_assignment):
    carrier = state.carriers
    center = torch.einsum(
        "bpq,bqd->bpd", carrier_assignment, carrier.center[:, 0].float()
    )
    identity = torch.einsum(
        "bpq,bqd->bpd", carrier_assignment, carrier.identity[:, 0].float()
    )
    appearance = F.normalize(
        model.state_model.identity_to_dino(identity), dim=-1, eps=1e-6
    )
    visibility = torch.einsum(
        "bpm,bm->bp", root_assignment, state.roots.visibility[:, 0].float()
    )
    presence = torch.einsum(
        "bpm,bm->bp", root_assignment, state.roots.presence[:, 0].float()
    )
    return center, appearance, visibility, presence


def _route_errors(model, state, assignment, root_assignment, evidence, relation):
    center, appearance, visibility, presence = _decode_tracks(
        model, state, assignment, root_assignment
    )
    target_visible = evidence.visibility[:, -1].float()
    coordinate = _weighted_mean(
        (center - evidence.coordinates[:, -1].float()).norm(dim=-1), target_visible
    )
    target_appearance = fixed_group_projection_v61(
        evidence.sampled_features[:, -1], model.config.teacher_projection_dim
    )
    appearance_error = 1.0 - (appearance * target_appearance).sum(dim=-1)
    appearance = _weighted_mean(appearance_error, target_visible)
    lifecycle_weight = relation.lifecycle_known[:, -1].float()
    visibility_error = _probability_loss(
        visibility, relation.visibility[:, -1], lifecycle_weight
    )
    presence_error = _probability_loss(
        presence, relation.presence[:, -1], lifecycle_weight
    )
    total = coordinate + appearance + 0.25 * (visibility_error + presence_error)
    return {
        "total": total,
        "coordinate": coordinate,
        "appearance": appearance,
        "visibility": visibility_error,
        "presence": presence_error,
        "appearance_field": appearance_error,
    }


def _state_alignment(prediction, target):
    carrier_feature = (
        1.0
        - F.cosine_similarity(
            prediction.carriers.feature.float(), target.carriers.feature.float(), dim=-1
        ).mean()
    )
    carrier_identity = (
        1.0
        - F.cosine_similarity(
            prediction.carriers.identity.float(),
            target.carriers.identity.float(),
            dim=-1,
        ).mean()
    )
    carrier_dynamic = F.smooth_l1_loss(
        prediction.carriers.dynamic.float(), target.carriers.dynamic.float()
    )
    carrier_center = F.smooth_l1_loss(
        prediction.carriers.center.float(), target.carriers.center.float()
    )
    carrier_covariance = F.smooth_l1_loss(
        prediction.carriers.covariance.float(), target.carriers.covariance.float()
    )
    carrier_lifecycle = F.smooth_l1_loss(
        prediction.carriers.presence.float(), target.carriers.presence.float()
    ) + F.smooth_l1_loss(
        prediction.carriers.visibility.float(), target.carriers.visibility.float()
    )
    root_feature = (
        1.0
        - F.cosine_similarity(
            prediction.roots.feature.float(), target.roots.feature.float(), dim=-1
        ).mean()
    )
    root_identity = (
        1.0
        - F.cosine_similarity(
            prediction.roots.identity.float(), target.roots.identity.float(), dim=-1
        ).mean()
    )
    root_dynamic = F.smooth_l1_loss(
        prediction.roots.dynamic.float(), target.roots.dynamic.float()
    )
    root_center = F.smooth_l1_loss(
        prediction.roots.center.float(), target.roots.center.float()
    )
    root_scale = F.smooth_l1_loss(
        prediction.roots.relative_scale.float(), target.roots.relative_scale.float()
    )
    root_lifecycle = F.smooth_l1_loss(
        prediction.roots.presence.float(), target.roots.presence.float()
    ) + F.smooth_l1_loss(
        prediction.roots.visibility.float(), target.roots.visibility.float()
    )
    root_owner = (
        F.kl_div(
            prediction.roots.owner.float().clamp_min(1e-6).log(),
            target.roots.owner.float(),
            reduction="batchmean",
        )
        / prediction.roots.owner.shape[2]
    )
    return (
        carrier_feature
        + carrier_identity
        + carrier_dynamic
        + carrier_center
        + carrier_covariance
        + carrier_lifecycle
        + root_feature
        + root_identity
        + root_dynamic
        + root_center
        + root_scale
        + root_lifecycle
        + root_owner
    ) / 13.0


def carrier_dynamics_objective_v61(
    model,
    source,
    target,
    correct,
    zero,
    shuffled,
    effect,
    carrier_assignment,
    root_assignment,
    evidence,
    relation,
):
    source_carrier = carrier_assignment[:, 0]
    source_root = root_assignment[:, 0]
    correct_errors = _route_errors(
        model, correct, source_carrier, source_root, evidence, relation
    )
    zero_errors = _route_errors(
        model, zero, source_carrier, source_root, evidence, relation
    )
    shuffled_errors = _route_errors(
        model, shuffled, source_carrier, source_root, evidence, relation
    )
    persistence_errors = _route_errors(
        model, source, source_carrier, source_root, evidence, relation
    )
    state_alignment = _state_alignment(correct, target)
    intervention = F.relu(
        0.05 + correct_errors["total"] - zero_errors["total"].detach()
    )
    intervention = intervention + F.relu(
        0.05 + correct_errors["total"] - shuffled_errors["total"].detach()
    )
    active_effect = effect.value.float() * effect.activation[..., None].float()
    statistical_effect = gather_batch_with_grad(active_effect)
    effect_std = statistical_effect.std(dim=(0, 1), unbiased=False).mean()
    effect_variance = F.relu(0.20 - effect_std)
    owner_entropy = (
        -(effect.owner.float() * effect.owner.float().clamp_min(1e-6).log())
        .sum(dim=-1)
        .mean()
    )
    owner_overlap = torch.einsum(
        "bko,blo->bkl", effect.owner.float(), effect.owner.float()
    )
    diagonal = torch.eye(
        owner_overlap.shape[-1], device=owner_overlap.device, dtype=torch.bool
    )
    owner_diversity = owner_overlap.masked_fill(diagonal[None], 0.0).mean()
    activation_sparsity = effect.activation.float().mean()
    loss = correct_errors["total"] + 0.25 * state_alignment + intervention
    loss = loss + 0.05 * effect_variance
    loss = loss + 0.01 * owner_diversity + 0.01 * activation_sparsity
    persistence_gain = persistence_errors["total"] - correct_errors["total"]
    zero_gain = zero_errors["total"] - correct_errors["total"]
    shuffled_gain = shuffled_errors["total"] - correct_errors["total"]
    parts = {
        "dynamics_loss": loss.detach(),
        "future_track_error": correct_errors["total"].detach(),
        "future_track_coordinate_error": correct_errors["coordinate"].detach(),
        "future_track_appearance_error": correct_errors["appearance"].detach(),
        "future_visibility_error": correct_errors["visibility"].detach(),
        "future_presence_error": correct_errors["presence"].detach(),
        "future_state_alignment_error": state_alignment.detach(),
        "persistence_future_track_error": persistence_errors["total"].detach(),
        "zero_effect_future_track_error": zero_errors["total"].detach(),
        "shuffled_effect_future_track_error": shuffled_errors["total"].detach(),
        "persistence_relative_gain": (
            persistence_gain / persistence_errors["total"].clamp_min(1e-6)
        ).detach(),
        "zero_effect_relative_gain": (
            zero_gain / zero_errors["total"].clamp_min(1e-6)
        ).detach(),
        "shuffled_effect_relative_gain": (
            shuffled_gain / shuffled_errors["total"].clamp_min(1e-6)
        ).detach(),
        "effect_intervention_margin_loss": intervention.detach(),
        "effect_standard_deviation": effect_std.detach(),
        "effect_variance_penalty": effect_variance.detach(),
        "effect_absolute_mean": active_effect.abs().mean().detach(),
        "effect_active_token_count": effect.activation.float()
        .sum(dim=-1)
        .mean()
        .detach(),
        "effect_owner_entropy": owner_entropy.detach(),
        "effect_owner_overlap": owner_diversity.detach(),
        "effect_activation_mean": activation_sparsity.detach(),
        "heldout_future_track_appearance_error": _weighted_mean(
            correct_errors["appearance_field"][:, 1::2],
            evidence.visibility[:, -1, 1::2].float(),
        ).detach(),
    }
    return loss, parts
