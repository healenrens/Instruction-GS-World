"""Held-coordinate objectives for the continuous predictive object field v67."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .continuous_predictive_teacher_v67 import teacher_relation_evidence_v67


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = value.float()
    weight = weight.float()
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def _cosine_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)


def _held_mask(target) -> torch.Tensor:
    return ~target.context_mask


def _relation_loss(logits, target_relation, weight, coordinate_mask=None):
    if coordinate_mask is not None:
        weight = weight * coordinate_mask[:, None].float()
    error = F.binary_cross_entropy_with_logits(
        logits.float(), target_relation.float(), reduction="none"
    )
    return _weighted_mean(error, weight)


def _semantic_errors(field, target, frame, relation, weight, coordinate_mask):
    support_weight = weight * coordinate_mask[:, None].float()
    semantic_weight = support_weight * relation * target.visibility[:, frame, None].float()
    dino = _cosine_error(field.dino, target.dino[:, frame, None])
    siglip = _cosine_error(field.siglip, target.siglip[:, frame, None])
    dino_error = _weighted_mean(dino, semantic_weight)
    siglip_error = _weighted_mean(siglip, semantic_weight)
    support = _relation_loss(
        field.support_logits, relation, weight, coordinate_mask
    )
    visibility_target = target.visibility[:, frame, None].float().expand_as(
        field.visibility_logits
    )
    visibility_error = F.binary_cross_entropy_with_logits(
        field.visibility_logits.float(), visibility_target, reduction="none"
    )
    visibility_error = _weighted_mean(visibility_error, support_weight * relation)
    semantic_residual = 0.5 * (dino + siglip)
    log_variance = field.log_uncertainty.float()
    uncertainty_nll = semantic_residual * torch.exp(-log_variance) + log_variance
    uncertainty_nll = _weighted_mean(uncertainty_nll, semantic_weight)
    calibration = _weighted_mean(
        (semantic_residual.detach() / torch.exp(log_variance).clamp_min(1e-4) - 1.0).abs(),
        semantic_weight,
    )
    return {
        "dino_error": dino_error,
        "siglip_error": siglip_error,
        "support_bce": support,
        "visibility_bce": visibility_error,
        "uncertainty_nll": uncertainty_nll,
        "uncertainty_calibration_error": calibration,
        "combined": dino_error + siglip_error + support + 0.25 * visibility_error,
    }


def _relation_structure(relation, query_indices):
    matrix = relation.probability.index_select(2, query_indices)
    symmetry = (matrix - matrix.transpose(1, 2)).abs().mean()
    composition = (
        matrix[:, :, :, None] * matrix[:, None, :, :]
    ).amax(dim=2)
    transitivity = F.relu(composition - matrix).mean()
    diagonal = matrix.diagonal(dim1=1, dim2=2)
    reflexive = -diagonal.float().clamp_min(1e-6).log().mean()
    return symmetry, transitivity, reflexive


def _variance_penalty(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = value.float().flatten(0, -2)
    standard_deviation = flat.std(dim=0, unbiased=False)
    return F.relu(1.0 - standard_deviation).mean(), standard_deviation.mean()


def predictive_state_objective_v67(model, target, output, config):
    relation_target, relation_weight = teacher_relation_evidence_v67(
        target, config.source_frame, config
    )
    held = _held_mask(target)
    relation = _relation_loss(
        output["source_relation"].logits,
        relation_target,
        relation_weight,
    )
    relation_visibility_target = target.visibility[:, config.source_frame, None].float()
    relation_visibility_target = relation_visibility_target.expand_as(
        output["source_relation"].visibility_logits
    )
    relation_visibility = F.binary_cross_entropy_with_logits(
        output["source_relation"].visibility_logits.float(),
        relation_visibility_target,
        reduction="none",
    )
    relation_visibility = _weighted_mean(relation_visibility, relation_weight)
    relation_residual = F.binary_cross_entropy_with_logits(
        output["source_relation"].logits.float(),
        relation_target.float(),
        reduction="none",
    )
    relation_log_variance = output["source_relation"].log_uncertainty.float()
    relation_uncertainty = relation_residual * torch.exp(-relation_log_variance)
    relation_uncertainty = relation_uncertainty + relation_log_variance
    relation_uncertainty = _weighted_mean(relation_uncertainty, relation_weight)
    field = _semantic_errors(
        output["source_decoded"],
        target,
        config.source_frame,
        relation_target,
        relation_weight,
        held,
    )
    point_weight = target.visibility[:, config.source_frame].float()
    point_weight = point_weight * target.reliability
    point_dino = _weighted_mean(
        _cosine_error(output["point_dino"], target.dino[:, config.source_frame]),
        point_weight,
    )
    point_siglip = _weighted_mean(
        _cosine_error(output["point_siglip"], target.siglip[:, config.source_frame]),
        point_weight,
    )
    identity = 1.0 - F.cosine_similarity(
        output["source_code"].identity.float(),
        output["target_code"].identity.float(),
        dim=-1,
    )
    query_valid = target.visibility[:, config.source_frame].index_select(
        1, target.query_indices
    ).float()
    query_valid = query_valid * target.visibility[:, config.target_frame].index_select(
        1, target.query_indices
    ).float()
    identity = _weighted_mean(identity, query_valid)
    symmetry, transitivity, reflexive = _relation_structure(
        output["source_relation"], target.query_indices
    )
    code_variance, code_std = _variance_penalty(output["source_code"].mean)
    response_variance, response_std = _variance_penalty(
        output["source_relation"].response
    )
    response_reconstruction = _weighted_mean(
        (
            output["source_decoded"].response.float()
            - output["source_relation"].response.detach().float()
        )
        .square()
        .mean(dim=-1),
        relation_weight * relation_target * held[:, None].float(),
    )
    continuity = output["continuity_error"]
    object_rate = output["source_code"].rate.mean()
    point_rate = _weighted_mean(output["source_code"].point_rate, point_weight)

    semantic_weight = relation_target * relation_weight * held[:, None].float()
    shared_distortion = 0.5 * (
        _cosine_error(output["source_decoded"].dino, target.dino[:, config.source_frame, None])
        + _cosine_error(output["source_decoded"].siglip, target.siglip[:, config.source_frame, None])
    )
    shared_distortion = (
        shared_distortion * semantic_weight
    ).sum(dim=-1) / semantic_weight.sum(dim=-1).clamp_min(1e-6)
    point_distortion = 0.5 * (
        _cosine_error(output["point_dino"], target.dino[:, config.source_frame])
        + _cosine_error(output["point_siglip"], target.siglip[:, config.source_frame])
    )
    separate_distortion = torch.einsum(
        "bqp,bp->bq", semantic_weight, point_distortion
    ) / semantic_weight.sum(dim=-1).clamp_min(1e-6)
    separate_rate = torch.einsum(
        "bqp,bp->bq", semantic_weight, output["source_code"].point_rate.float()
    )
    separate_rate = separate_rate / semantic_weight.amax(dim=-1).clamp_min(1e-6)
    shared_cost = shared_distortion + config.rate_weight * output["source_code"].rate
    separate_cost = separate_distortion + config.rate_weight * separate_rate
    rate_saving = separate_cost - shared_cost

    total = (
        config.semantic_weight * (field["dino_error"] + field["siglip_error"])
        + config.relation_weight * (relation + field["support_bce"])
        + config.visibility_weight * field["visibility_bce"]
        + config.visibility_weight * relation_visibility
        + config.identity_weight * identity
        + config.point_reconstruction_weight * (point_dino + point_siglip)
        + config.rate_weight * object_rate
        + 0.25 * config.rate_weight * point_rate
        + config.uncertainty_weight * field["uncertainty_nll"]
        + config.uncertainty_weight * relation_uncertainty
        + 0.25 * response_reconstruction
        + config.symmetry_weight * (symmetry + reflexive)
        + config.transitivity_weight * transitivity
        + config.continuity_weight * continuity
        + config.variance_weight * (code_variance + response_variance)
    )
    parts = {
        "loss": total.detach(),
        "relation_bce": relation.detach(),
        "heldout_dino_error": field["dino_error"].detach(),
        "heldout_siglip_error": field["siglip_error"].detach(),
        "heldout_support_bce": field["support_bce"].detach(),
        "heldout_visibility_bce": field["visibility_bce"].detach(),
        "relation_visibility_bce": relation_visibility.detach(),
        "relation_uncertainty_nll": relation_uncertainty.detach(),
        "heldout_response_reconstruction": response_reconstruction.detach(),
        "identity_future_cosine_error": identity.detach(),
        "point_dino_error": point_dino.detach(),
        "point_siglip_error": point_siglip.detach(),
        "predictive_object_rate": object_rate.detach(),
        "predictive_point_rate": point_rate.detach(),
        "predictive_rate_saving": rate_saving.mean().detach(),
        "predictive_rate_saving_positive_fraction": (rate_saving > 0).float().mean().detach(),
        "uncertainty_nll": field["uncertainty_nll"].detach(),
        "uncertainty_calibration_error": field["uncertainty_calibration_error"].detach(),
        "relation_symmetry_error": symmetry.detach(),
        "relation_transitivity_error": transitivity.detach(),
        "relation_reflexive_error": reflexive.detach(),
        "coordinate_scale_continuity_error": continuity.detach(),
        "object_code_std": code_std.detach(),
        "object_response_std": response_std.detach(),
        "teacher_relation_mean": _weighted_mean(relation_target, relation_weight).detach(),
        "teacher_reliable_fraction": (target.reliability > 0).float().mean().detach(),
    }
    return total, parts


def _code_error(prediction, target):
    identity = 1.0 - F.cosine_similarity(
        prediction.identity.float(), target.identity.float(), dim=-1
    )
    dynamic = (prediction.dynamic.float() - target.dynamic.float()).square().mean(dim=-1)
    return (identity + dynamic).mean(), identity.mean(), dynamic.mean()


def posterior_dynamics_objective_v67(model, target, output, config):
    short_relation, short_weight = teacher_relation_evidence_v67(
        target, config.midpoint_frame, config
    )
    goal_relation, goal_weight = teacher_relation_evidence_v67(
        target, config.target_frame, config
    )
    held = _held_mask(target)
    short = _semantic_errors(
        output["short_correct"].field,
        target,
        config.midpoint_frame,
        short_relation,
        short_weight,
        held,
    )
    correct = _semantic_errors(
        output["goal_correct"].field,
        target,
        config.target_frame,
        goal_relation,
        goal_weight,
        held,
    )
    zero = _semantic_errors(
        output["goal_zero"].field,
        target,
        config.target_frame,
        goal_relation,
        goal_weight,
        held,
    )
    shuffled = _semantic_errors(
        output["goal_shuffled"].field,
        target,
        config.target_frame,
        goal_relation,
        goal_weight,
        held,
    )
    persistence = _semantic_errors(
        output["goal_persistence"],
        target,
        config.target_frame,
        goal_relation,
        goal_weight,
        held,
    )
    code, identity, dynamic = _code_error(
        output["goal_correct"].code, output["goal_target_code"]
    )
    short_code, short_identity, short_dynamic = _code_error(
        output["short_correct"].code, output["midpoint_target_code"]
    )
    direct_code, _, _ = _code_error(
        output["goal_direct"].code, output["goal_target_code"]
    )
    rollout_code, _, _ = _code_error(
        output["goal_rollout"].code, output["goal_target_code"]
    )
    path_code = (
        output["goal_direct"].code.mean.float()
        - output["goal_rollout"].code.mean.float()
    ).square().mean()
    path_field = 0.5 * (
        _cosine_error(output["goal_direct"].field.dino, output["goal_rollout"].field.dino).mean()
        + _cosine_error(output["goal_direct"].field.siglip, output["goal_rollout"].field.siglip).mean()
    )
    response_error = _weighted_mean(
        (
            output["goal_correct"].field.response.float()
            - output["goal_target_field"].response.float()
        )
        .square()
        .mean(dim=-1),
        goal_relation * goal_weight * held[:, None].float(),
    )
    intervention_zero = F.relu(
        config.intervention_margin + correct["combined"] - zero["combined"]
    )
    intervention_shuffled = F.relu(
        config.intervention_margin + correct["combined"] - shuffled["combined"]
    )
    intervention_persistence = F.relu(
        config.intervention_margin + correct["combined"] - persistence["combined"]
    )
    effect_rate = 0.5 * (
        output["short_effect"].rate.mean() + output["tail_effect"].rate.mean()
    )
    effect_variance, effect_std = _variance_penalty(
        torch.cat((output["short_effect"].mean, output["tail_effect"].mean), dim=1)
    )
    predicted_state_rate = output["goal_correct"].code.rate.mean()
    total = (
        config.short_prediction_weight
        * (
            short["dino_error"]
            + short["siglip_error"]
            + short["support_bce"]
            + config.visibility_weight * short["visibility_bce"]
            + config.uncertainty_weight * short["uncertainty_nll"]
            + config.state_prediction_weight * short_code
        )
        + config.field_prediction_weight
        * (
            correct["dino_error"]
            + correct["siglip_error"]
            + correct["support_bce"]
            + config.visibility_weight * correct["visibility_bce"]
            + config.uncertainty_weight * correct["uncertainty_nll"]
            + 0.25 * response_error
        )
        + config.state_prediction_weight * (code + direct_code + rollout_code)
        + config.path_consistency_weight * (path_code + path_field)
        + config.intervention_weight
        * (intervention_zero + intervention_shuffled + intervention_persistence)
        + config.effect_rate_weight * effect_rate
        + 0.1 * config.effect_rate_weight * predicted_state_rate
        + config.effect_variance_weight * effect_variance
    )
    parts = {
        "loss": total.detach(),
        "short_dino_absolute_error": short["dino_error"].detach(),
        "short_siglip_absolute_error": short["siglip_error"].detach(),
        "short_support_bce": short["support_bce"].detach(),
        "short_visibility_bce": short["visibility_bce"].detach(),
        "short_state_code_error": short_code.detach(),
        "short_identity_error": short_identity.detach(),
        "short_dynamic_error": short_dynamic.detach(),
        "future_dino_absolute_error": correct["dino_error"].detach(),
        "future_siglip_absolute_error": correct["siglip_error"].detach(),
        "future_support_bce": correct["support_bce"].detach(),
        "future_visibility_bce": correct["visibility_bce"].detach(),
        "future_state_code_error": code.detach(),
        "future_identity_error": identity.detach(),
        "future_dynamic_error": dynamic.detach(),
        "future_response_error": response_error.detach(),
        "persistence_field_error": persistence["combined"].detach(),
        "zero_effect_field_error": zero["combined"].detach(),
        "shuffled_effect_field_error": shuffled["combined"].detach(),
        "effect_correct_field_error": correct["combined"].detach(),
        "effect_gain_over_persistence": (persistence["combined"] - correct["combined"]).detach(),
        "effect_gain_over_zero": (zero["combined"] - correct["combined"]).detach(),
        "effect_gain_over_shuffled": (shuffled["combined"] - correct["combined"]).detach(),
        "effect_intervention_zero_hinge": intervention_zero.detach(),
        "effect_intervention_shuffled_hinge": intervention_shuffled.detach(),
        "effect_intervention_persistence_hinge": intervention_persistence.detach(),
        "effect_rate": effect_rate.detach(),
        "future_predicted_state_rate": predicted_state_rate.detach(),
        "effect_std": effect_std.detach(),
        "goal_direct_code_error": direct_code.detach(),
        "goal_rollout_code_error": rollout_code.detach(),
        "goal_path_code_error": path_code.detach(),
        "goal_path_field_error": path_field.detach(),
        "uncertainty_calibration_error": correct["uncertainty_calibration_error"].detach(),
    }
    return total, parts
