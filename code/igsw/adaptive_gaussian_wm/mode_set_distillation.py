"""Frozen-teacher distillation for a history-only unordered action set."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .mode_set_prior import ModeSetActionPrior
from .scale import signed_gap_scale
from .set_matching import exact_target_component_assignment


def _pairwise_weighted_mean(
    value: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity[:, :, None].to(value.dtype)
    return (
        (value * weight).sum(dim=(-1, -2))
        / weight.sum(dim=(-1, -2)).clamp_min(1.0)
    )


def _matched_mean(
    cost: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(
        cost,
        dim=2,
        index=assignment[..., None],
    ).mean()


def _cardinality_matched_mean(
    cost: torch.Tensor,
    assignment: torch.Tensor,
    ambiguity: torch.Tensor,
    deterministic_component: torch.Tensor,
) -> torch.Tensor:
    matched = torch.gather(
        cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    ambiguous_weight = ambiguity[:, None].to(cost.dtype)
    deterministic_weight = (~ambiguity).to(cost.dtype)
    deterministic = torch.gather(
        cost[:, 0],
        dim=1,
        index=deterministic_component[:, None],
    ).squeeze(-1)
    count = (
        ambiguous_weight.sum() * cost.shape[1]
        + deterministic_weight.sum()
    ).clamp_min(1.0)
    return (
        (matched * ambiguous_weight).sum()
        + (deterministic * deterministic_weight).sum()
    ) / count


def _group_layout(
    group_id: torch.Tensor | None,
    components: int,
) -> tuple[int, torch.Tensor]:
    if group_id is None:
        raise ValueError("mode-set distillation requires grouped futures")
    ids, counts = torch.unique_consecutive(
        group_id.flatten(),
        return_counts=True,
    )
    if ids.numel() != torch.unique(ids).numel() or not bool(
        (counts == components).all()
    ):
        raise ValueError("mode-set distillation requires complete groups")
    representatives = torch.arange(
        0,
        ids.numel() * components,
        components,
        device=group_id.device,
    )
    return ids.numel(), representatives


def frozen_teacher_mode_set_loss(
    model,
    batch: dict[str, torch.Tensor],
    teacher: dict,
    latent_weight: float,
    canonical_weight: float,
    effect_weight: float,
    unique_mode_cardinality: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill posterior actions and their effects without moving the teacher."""
    if min(latent_weight, canonical_weight, effect_weight) < 0.0:
        raise ValueError("distillation weights must be non-negative")
    if latent_weight == 0.0 and canonical_weight == 0.0 and effect_weight == 0.0:
        raise ValueError("at least one distillation weight must be positive")
    prior = model.latent_actions.prior
    if not isinstance(prior, ModeSetActionPrior):
        raise ValueError("mode-set distillation requires ModeSetActionPrior")
    groups, representatives = _group_layout(
        batch.get("group_id"),
        prior.components,
    )
    if unique_mode_cardinality and "ambiguity" not in batch:
        raise ValueError(
            "unique mode cardinality requires ambiguity metadata"
        )
    ambiguity = (
        batch["ambiguity"][representatives].bool()
        if unique_mode_cardinality
        else torch.ones(
            groups,
            device=representatives.device,
            dtype=torch.bool,
        )
    )
    history_activity = torch.stack(
        [state.activity for state in teacher["history_slot_states"]],
        dim=1,
    ).detach()
    history_slots = teacher["online_history_slots"].detach()
    history_centers = teacher["online_history_centers"].detach()
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    context = model.latent_actions.prior_context(
        history_slots,
        history_activity,
        future_scale,
        history_centers,
        history_scale,
        teacher.get("language_condition"),
    )
    logits, all_prototypes = prior._distribution(context)
    prototypes = all_prototypes[representatives]
    target_actions = teacher["posterior_actions"].detach().reshape(
        groups,
        prior.components,
        *teacher["posterior_actions"].shape[1:],
    )
    latent_cost = (
        prototypes[:, None] - target_actions[:, :, None]
    ).square().mean(dim=(-1, -2, -3))
    canonical_dimensions = (
        6
        if model.config.canonical_semantic_action
        else 3 if model.config.canonical_center_action else 0
    )
    if canonical_weight > 0.0 and canonical_dimensions == 0:
        raise ValueError(
            "canonical distillation requires canonical posterior dimensions"
        )
    canonical_cost = (
        (
            prototypes[:, None, ..., :canonical_dimensions]
            - target_actions[:, :, None, ..., :canonical_dimensions]
        ).square().mean(dim=(-1, -2, -3))
        if canonical_dimensions > 0
        else latent_cost.new_zeros(latent_cost.shape)
    )

    component_slots = []
    component_centers = []
    component_features = []
    history_mask = torch.zeros(
        history_slots[representatives].shape[:3],
        device=history_slots.device,
        dtype=torch.bool,
    )
    for component in range(prior.components):
        prediction = model.dynamics(
            history_slots[representatives],
            history_activity[representatives],
            history_scale[representatives],
            future_scale[representatives],
            prototypes[:, component],
            history_mask,
            history_centers[representatives],
            (
                teacher["language_condition"][representatives]
                if teacher.get("language_condition") is not None
                else None
            ),
        )
        component_slots.append(prediction.future_slots)
        component_centers.append(
            prediction.future_centers
            if prediction.future_centers is not None
            else model.object_aggregator.decode_center(
                prediction.future_slots
            )
        )
        component_features.append(
            model.object_aggregator.decode_feature(
                prediction.future_slots
            )
        )
    predicted_slots = torch.stack(component_slots, dim=1)
    predicted_centers = torch.stack(component_centers, dim=1)
    predicted_features = torch.stack(component_features, dim=1)
    target_slots = teacher["predicted_future_slots"].detach().reshape(
        groups,
        prior.components,
        *teacher["predicted_future_slots"].shape[1:],
    )
    target_centers = teacher["predicted_future_centers"].detach().reshape(
        groups,
        prior.components,
        *teacher["predicted_future_centers"].shape[1:],
    )
    target_features = teacher[
        "predicted_future_object_features"
    ].detach().reshape(
        groups,
        prior.components,
        *teacher["predicted_future_object_features"].shape[1:],
    )
    activity = teacher["target_future_activity"].detach().reshape(
        groups,
        prior.components,
        *teacher["target_future_activity"].shape[1:],
    )
    slot_cost = _pairwise_weighted_mean(
        (
            F.normalize(predicted_slots[:, None], dim=-1)
            - F.normalize(target_slots[:, :, None], dim=-1)
        ).square().mean(dim=-1),
        activity,
    )
    feature_cost = _pairwise_weighted_mean(
        (
            F.normalize(predicted_features[:, None], dim=-1)
            - F.normalize(target_features[:, :, None], dim=-1)
        ).square().mean(dim=-1),
        activity,
    )
    center_cost = _pairwise_weighted_mean(
        (
            predicted_centers[:, None] - target_centers[:, :, None]
        ).square().mean(dim=-1),
        activity,
    )
    effect_cost = slot_cost + 2.0 * feature_cost + 10.0 * center_cost
    assignment_cost = (
        latent_weight * latent_cost
        + canonical_weight * canonical_cost
        + effect_weight * effect_cost
    )
    assignment = exact_target_component_assignment(
        assignment_cost.detach()
    )
    deterministic_component = assignment_cost[:, 0].detach().argmin(dim=-1)

    def matched_mean(cost: torch.Tensor) -> torch.Tensor:
        if unique_mode_cardinality:
            return _cardinality_matched_mean(
                cost,
                assignment,
                ambiguity,
                deterministic_component,
            )
        return _matched_mean(cost, assignment)

    latent = matched_mean(latent_cost)
    canonical = matched_mean(canonical_cost)
    slot = matched_mean(slot_cost)
    feature = matched_mean(feature_cost)
    center = matched_mean(center_cost)
    effect = slot + 2.0 * feature + 10.0 * center
    log_probability = F.log_softmax(
        logits[representatives],
        dim=-1,
    )
    if unique_mode_cardinality:
        ambiguous_calibration = -log_probability.mean(dim=-1)
        deterministic_calibration = -torch.gather(
            log_probability,
            dim=1,
            index=deterministic_component[:, None],
        ).squeeze(-1)
        calibration = torch.where(
            ambiguity,
            ambiguous_calibration,
            deterministic_calibration,
        ).mean()
    else:
        calibration = -log_probability.mean()
    total = (
        latent_weight * latent
        + canonical_weight * canonical
        + effect_weight * effect
        + 0.1 * calibration
    )
    row_margin = (
        assignment_cost.detach().sort(dim=-1).values[..., 1]
        - assignment_cost.detach().sort(dim=-1).values[..., 0]
    )
    if unique_mode_cardinality:
        ambiguous_weight = ambiguity[:, None].to(row_margin.dtype)
        deterministic_weight = (~ambiguity).to(row_margin.dtype)
        margin_count = (
            ambiguous_weight.sum() * row_margin.shape[1]
            + deterministic_weight.sum()
        ).clamp_min(1.0)
        assignment_margin = (
            (row_margin * ambiguous_weight).sum()
            + (row_margin[:, 0] * deterministic_weight).sum()
        ) / margin_count
    else:
        assignment_margin = row_margin.mean()
    return total, {
        "total": total,
        "latent": latent,
        "canonical": canonical,
        "effect": effect,
        "slot": slot,
        "feature": feature,
        "center": center,
        "calibration": calibration,
        "assignment_margin": assignment_margin,
    }
