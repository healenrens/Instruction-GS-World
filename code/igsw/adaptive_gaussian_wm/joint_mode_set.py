"""Train-time exposure of Dynamics to history-only mode-set actions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .mode_set_prior import (
    ModeSetActionPrior,
    ordered_group_responsibility,
)
from .scale import signed_gap_scale
from .set_matching import (
    exact_target_component_assignment,
    select_components,
)


def _weighted_mean(
    value: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _ordered_mode_set_dynamics_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match hypotheses to futures, then train the actual inference path."""
    prior = model.latent_actions.prior
    if not isinstance(prior, ModeSetActionPrior):
        raise ValueError("joint mode-set loss requires ModeSetActionPrior")
    logits, prototypes = prior._distribution(output["prior_context"])
    responsibility = ordered_group_responsibility(
        output["target_future_centers"],
        output["target_history_centers"][:, -1],
        prior.components,
        batch.get("group_id"),
    ).to(prototypes.dtype)
    actions = (
        responsibility[..., None, None, None] * prototypes
    ).sum(dim=1)
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    prediction = model.dynamics(
        output["online_history_slots"],
        history_activity,
        history_scale,
        future_scale,
        actions,
        output["history_mask"],
        output["online_history_centers"],
        output.get("language_condition"),
    )
    predicted_centers = (
        prediction.future_centers
        if prediction.future_centers is not None
        else model.object_aggregator.decode_center(
            prediction.future_slots
        )
    )
    predicted_features = model.object_aggregator.decode_feature(
        prediction.future_slots
    )
    activity = output["target_future_activity"]
    slot = _weighted_mean(
        (
            F.normalize(prediction.future_slots, dim=-1)
            - F.normalize(output["target_future_slots"], dim=-1)
        ).square().mean(dim=-1),
        activity,
    )
    feature = _weighted_mean(
        (
            F.normalize(predicted_features, dim=-1)
            - F.normalize(
                output["target_future_object_features"],
                dim=-1,
            )
        ).square().mean(dim=-1),
        activity,
    )
    center = _weighted_mean(
        (
            predicted_centers - output["target_future_centers"]
        ).square().mean(dim=-1),
        activity,
    )
    action_fit = F.mse_loss(actions, output["posterior_actions"].detach())
    logit_fit = -(
        responsibility.detach() * F.log_softmax(logits, dim=-1)
    ).sum(dim=-1).mean()
    future = slot + 2.0 * feature + 10.0 * center
    total = future + 0.05 * action_fit + 0.1 * logit_fit
    return total, {
        "total": total,
        "future": future,
        "slot": slot,
        "feature": feature,
        "center": center,
        "action_fit": action_fit,
        "logit_fit": logit_fit,
    }


def _pairwise_weighted_mean(
    value: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity[:, :, None].to(value.dtype)
    return (
        (value * weight).sum(dim=(-1, -2))
        / weight.sum(dim=(-1, -2)).clamp_min(1.0)
    )


def _target_mode_separation(
    target: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    representation = F.normalize(target, dim=-1) if normalize else target
    distance = (
        representation[:, :, None] - representation[:, None, :]
    ).square().mean(dim=(-1, -2, -3))
    off_diagonal = ~torch.eye(
        distance.shape[-1],
        device=distance.device,
        dtype=torch.bool,
    )[None]
    return distance.masked_fill(
        ~off_diagonal,
        torch.inf,
    ).amin(dim=(1, 2))


def _coverage_hinge(
    matched_cost: torch.Tensor,
    separation: torch.Tensor,
    ambiguity: torch.Tensor,
) -> torch.Tensor:
    radius_square = 0.25 * separation
    valid = (
        ambiguity
        & torch.isfinite(radius_square)
        & (radius_square > 1e-8)
    )
    ratio = matched_cost / radius_square[:, None].detach().clamp_min(1e-8)
    weight = valid[:, None].to(ratio.dtype)
    return (
        (F.relu(ratio - 1.0) * weight).sum()
        / (weight.sum() * matched_cost.shape[1]).clamp_min(1.0)
    )


def _exact_effect_mode_set_dynamics_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    assignment_strategy: str,
    action_fit_weight: float,
    canonical_action_fit_weight: float,
    coverage_margin_weight: float,
    posterior_code_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use permutation-invariant future matching for the deploy-time set."""
    prior = model.latent_actions.prior
    if not isinstance(prior, ModeSetActionPrior):
        raise ValueError("joint mode-set loss requires ModeSetActionPrior")
    group_id = batch.get("group_id")
    if group_id is None:
        raise ValueError("exact mode-set matching requires groups")
    ids, counts = torch.unique_consecutive(
        group_id.flatten(),
        return_counts=True,
    )
    if ids.numel() != torch.unique(ids).numel() or not bool(
        (counts == prior.components).all()
    ):
        raise ValueError("exact mode-set matching requires complete groups")
    groups = ids.numel()
    representatives = torch.arange(
        0,
        groups * prior.components,
        prior.components,
        device=group_id.device,
    )
    logits, all_prototypes = prior._distribution(output["prior_context"])
    prototypes = all_prototypes[representatives]
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )[representatives]
    history_scale = signed_gap_scale(
        batch["history_times"][representatives],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"][representatives],
        model.config.gap_reference,
    )
    component_slots = []
    component_centers = []
    component_features = []
    for component in range(prior.components):
        prediction = model.dynamics(
            output["online_history_slots"][representatives],
            history_activity,
            history_scale,
            future_scale,
            prototypes[:, component],
            output["history_mask"][representatives],
            output["online_history_centers"][representatives],
            (
                output["language_condition"][representatives]
                if output.get("language_condition") is not None
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
    target_slots = output["target_future_slots"].reshape(
        groups,
        prior.components,
        *output["target_future_slots"].shape[1:],
    )
    target_centers = output["target_future_centers"].reshape(
        groups,
        prior.components,
        *output["target_future_centers"].shape[1:],
    )
    target_features = output["target_future_object_features"].reshape(
        groups,
        prior.components,
        *output["target_future_object_features"].shape[1:],
    )
    activity = output["target_future_activity"].reshape(
        groups,
        prior.components,
        *output["target_future_activity"].shape[1:],
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
    coverage_slot_cost = (
        F.normalize(predicted_slots[:, None], dim=-1)
        - F.normalize(target_slots[:, :, None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    coverage_feature_cost = (
        F.normalize(predicted_features[:, None], dim=-1)
        - F.normalize(target_features[:, :, None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    coverage_center_cost = (
        predicted_centers[:, None] - target_centers[:, :, None]
    ).square().mean(dim=(-1, -2, -3))
    cost = slot_cost + 2.0 * feature_cost + 10.0 * center_cost
    if assignment_strategy == "posterior_code":
        code_logits = prior.code_assignment_logits(
            output["posterior_actions"]
        ).reshape(groups, prior.components, prior.components)
        assignment = exact_target_component_assignment(
            -code_logits.detach()
        )
        code_assignment = F.cross_entropy(
            code_logits.reshape(-1, prior.components),
            assignment.reshape(-1),
        )
    else:
        assignment = exact_target_component_assignment(cost.detach())
        code_assignment = cost.new_zeros(())
    matched_total = torch.gather(
        cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_slot = torch.gather(
        slot_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_feature = torch.gather(
        feature_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_center = torch.gather(
        center_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_coverage_slot = torch.gather(
        coverage_slot_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_coverage_feature = torch.gather(
        coverage_feature_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    matched_coverage_center = torch.gather(
        coverage_center_cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1)
    future = matched_total.mean()
    ambiguity_value = batch.get("ambiguity")
    ambiguity = (
        torch.ones(groups, device=group_id.device, dtype=torch.bool)
        if ambiguity_value is None
        else ambiguity_value[representatives].bool()
    )
    coverage_margin = (
        _coverage_hinge(
            matched_coverage_slot,
            _target_mode_separation(target_slots, True),
            ambiguity,
        )
        + _coverage_hinge(
            matched_coverage_feature,
            _target_mode_separation(target_features, True),
            ambiguity,
        )
        + _coverage_hinge(
            matched_coverage_center,
            _target_mode_separation(target_centers, False),
            ambiguity,
        )
    )
    matched_actions = select_components(prototypes, assignment)
    posterior_actions = output["posterior_actions"].reshape(
        groups,
        prior.components,
        *output["posterior_actions"].shape[1:],
    )
    action_fit = F.mse_loss(
        matched_actions,
        posterior_actions.detach(),
    )
    canonical_dimensions = (
        6
        if model.config.canonical_semantic_action
        else 3 if model.config.canonical_center_action else 0
    )
    if canonical_action_fit_weight > 0.0 and canonical_dimensions == 0:
        raise ValueError(
            "canonical action fit requires canonical posterior dimensions"
        )
    canonical_action_fit = (
        F.mse_loss(
            matched_actions[..., :canonical_dimensions],
            posterior_actions[..., :canonical_dimensions].detach(),
        )
        if canonical_dimensions > 0
        else action_fit.new_zeros(())
    )
    logit_fit = -F.log_softmax(
        logits[representatives],
        dim=-1,
    ).mean()
    usage = F.one_hot(
        assignment,
        num_classes=prior.components,
    ).to(cost.dtype).mean(dim=(0, 1))
    usage_fit = (
        usage - usage.new_full(usage.shape, 1.0 / prior.components)
    ).square().mean()
    total = (
        future
        + coverage_margin_weight * coverage_margin
        + action_fit_weight * action_fit
        + canonical_action_fit_weight * canonical_action_fit
        + posterior_code_weight * code_assignment
        + 0.1 * (logit_fit + usage_fit)
    )
    return total, {
        "total": total,
        "future": future,
        "slot": matched_slot.mean(),
        "feature": matched_feature.mean(),
        "center": matched_center.mean(),
        "coverage_margin": coverage_margin,
        "action_fit": action_fit,
        "canonical_action_fit": canonical_action_fit,
        "code_assignment": code_assignment,
        "logit_fit": logit_fit,
        "usage_fit": usage_fit,
        "assignment_cost_margin": (
            cost.detach().sort(dim=-1).values[..., 1]
            - cost.detach().sort(dim=-1).values[..., 0]
        ).mean(),
    }


def matched_mode_set_dynamics_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    assignment_strategy: str = "exact_effect",
    action_fit_weight: float = 0.01,
    canonical_action_fit_weight: float = 0.0,
    coverage_margin_weight: float = 0.0,
    posterior_code_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if action_fit_weight < 0.0:
        raise ValueError("action_fit_weight must be non-negative")
    if canonical_action_fit_weight < 0.0:
        raise ValueError("canonical action fit weight must be non-negative")
    if coverage_margin_weight < 0.0:
        raise ValueError("coverage_margin_weight must be non-negative")
    if posterior_code_weight < 0.0:
        raise ValueError("posterior code weight must be non-negative")
    if assignment_strategy == "ordered":
        if coverage_margin_weight > 0.0 or canonical_action_fit_weight > 0.0:
            raise ValueError(
                "coverage and canonical action fit require exact matching"
            )
        return _ordered_mode_set_dynamics_loss(model, batch, output)
    if assignment_strategy in ("exact_effect", "posterior_code"):
        return _exact_effect_mode_set_dynamics_loss(
            model,
            batch,
            output,
            assignment_strategy,
            action_fit_weight,
            canonical_action_fit_weight,
            coverage_margin_weight,
            posterior_code_weight,
        )
    raise ValueError(f"unknown assignment strategy: {assignment_strategy}")
