"""Goal rollout, path consistency, and horizon-stratified diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .change_readout_objective import relative_change_targets
from .diagnostic_statistics import ratio_moments
from .jepa_losses import object_latent_loss, weighted_mean


def prior_flow_loss(model, batch: dict, output: dict) -> torch.Tensor:
    if not model.config.dual_horizon_dynamics:
        return model.latent_actions.prior.loss(
            output["posterior_actions"],
            output["prior_context"],
            batch.get("group_id"),
        )
    valid = output["future_horizon_valid"].all(dim=1)
    if not bool(valid.any()):
        zero = output["prior_context"].sum() * 0.0
        for parameter in model.latent_actions.prior.parameters():
            zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero
    group = batch.get("group_id")
    if group is not None:
        group = group[valid]
    return model.latent_actions.prior.loss(
        output["posterior_actions"][valid],
        output["prior_context"][valid],
        group,
    )


def _sample_feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    weight = valid.float()
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def _change_balanced_dense_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    change_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = error + 0.1 * (
        1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    )
    change_weight = change_target.float() * valid.float()
    static_weight = (1.0 - change_target.float()) * valid.float()
    change_present = (change_weight.sum() > 0).to(error.dtype)
    static_present = (static_weight.sum() > 0).to(error.dtype)
    change_loss = weighted_mean(error, change_weight)
    static_loss = weighted_mean(error, static_weight)
    total = (
        change_loss * change_present + static_loss * static_present
    ) / (change_present + static_present).clamp_min(1.0)
    return total, change_loss, static_loss


def _history_strata(
    batch: dict,
    short_error: torch.Tensor,
    goal_error: torch.Tensor,
) -> dict[str, torch.Tensor]:
    length = batch["history_length"]
    goal_valid = batch["future_horizon_valid"][:, 1].float()
    result = {}
    for value in range(1, 5):
        selected = (length == value).float()
        result.update(
            ratio_moments(
                f"history_h{value}_short_dino_error",
                short_error * selected,
                selected,
            )
        )
        result.update(
            ratio_moments(
                f"history_h{value}_goal_rollout_dino_error",
                goal_error * selected * goal_valid,
                selected * goal_valid,
            )
        )
    return result


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def dual_horizon_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference = output["predicted_future_slots"].sum() * 0.0
    if not model.config.dual_horizon_dynamics:
        return reference, {}
    goal_valid = batch["future_horizon_valid"][:, 1:2]
    observed_goal_valid = batch.get("goal_valid", goal_valid[:, 0]).float()
    validity_metrics = {
        "goal_valid_fraction": observed_goal_valid.mean(),
        "goal_content_valid_fraction": batch.get(
            "goal_content_valid", observed_goal_valid
        ).float().mean(),
        "goal_stability_error_mean": batch.get(
            "goal_stability_error", observed_goal_valid * 0.0
        ).float().mean(),
    }
    direct = output["predicted_future_slots"][:, 1:2]
    target = output["target_future_slots"][:, 1:2]
    activity = output["target_future_activity"][:, 1:2]
    direct_rendered = output["rendered_future_features"][:, 1:2]
    target_rendered = batch["future_features"][:, 1:2]
    target_valid = batch["future_valid"][:, 1:2]
    current = batch["history_features"][:, -1:].float()
    short_error = _sample_feature_error(
        output["rendered_future_features"][:, :1],
        batch["future_features"][:, :1],
        batch["future_valid"][:, :1],
    )[:, 0]

    if "rollout_goal_slots" not in output:
        goal_error = _sample_feature_error(
            direct_rendered, target_rendered, target_valid
        )[:, 0]
        metrics = {
            "goal_direct_dino_error": _masked_mean(
                goal_error, goal_valid[:, 0]
            ),
            **validity_metrics,
        }
        metrics.update(_history_strata(batch, short_error, goal_error))
        return reference, metrics

    rollout = output["rollout_goal_slots"]
    rollout_object = output["rollout_goal_object_features"]
    rollout_rendered = output["rollout_goal_rendered_features"]
    target_object = output["target_future_object_features"][:, 1:2]
    change_target = relative_change_targets(batch)[0][:, 1:2]
    rollout_latent = object_latent_loss(rollout, target, activity)
    rollout_object_loss = object_latent_loss(
        rollout_object, target_object, activity
    )
    rollout_dense, rollout_change, rollout_static = _change_balanced_dense_loss(
        rollout_rendered,
        target_rendered,
        target_valid,
        change_target,
    )
    path_latent = object_latent_loss(rollout, direct.detach(), activity)
    path_object = object_latent_loss(
        rollout_object,
        output["predicted_future_object_features"][:, 1:2].detach(),
        activity,
    )
    path_dense, _, _ = _change_balanced_dense_loss(
        rollout_rendered,
        direct_rendered.detach(),
        target_valid,
        change_target,
    )
    _, direct_change, direct_static = _change_balanced_dense_loss(
        direct_rendered, target_rendered, target_valid, change_target
    )
    goal = 0.5 * rollout_latent + 2.0 * rollout_object_loss + rollout_dense
    path = 0.5 * path_latent + 2.0 * path_object + path_dense
    total = (
        model.config.goal_rollout_weight * goal
        + model.config.path_consistency_weight * path
    )

    direct_error = _sample_feature_error(
        direct_rendered, target_rendered, target_valid
    )[:, 0]
    rollout_error = _sample_feature_error(
        rollout_rendered, target_rendered, target_valid
    )[:, 0]
    persistence_error = _sample_feature_error(
        current.expand_as(target_rendered), target_rendered, target_valid
    )[:, 0]
    metrics = {
        "dual_horizon_total": total,
        "goal_rollout": goal,
        "goal_rollout_latent": rollout_latent,
        "goal_rollout_object_feature": rollout_object_loss,
        "goal_rollout_dino": rollout_dense,
        "goal_direct_dino_change_region": direct_change,
        "goal_direct_dino_static_region": direct_static,
        "goal_rollout_dino_change_region": rollout_change,
        "goal_rollout_dino_static_region": rollout_static,
        "goal_path_consistency": path,
        "goal_path_latent": path_latent,
        "goal_path_object_feature": path_object,
        "goal_path_dino": path_dense,
        "goal_change_target_fraction": change_target.mean(),
        **validity_metrics,
        "short_direct_dino_error": short_error.mean(),
        "goal_direct_dino_error": _masked_mean(direct_error, goal_valid[:, 0]),
        "goal_rollout_dino_error": _masked_mean(rollout_error, goal_valid[:, 0]),
        "goal_persistence_dino_error": _masked_mean(
            persistence_error, goal_valid[:, 0]
        ),
        "goal_direct_gain_over_persistence": _masked_mean(
            persistence_error - direct_error, goal_valid[:, 0]
        ),
        "goal_rollout_gain_over_persistence": _masked_mean(
            persistence_error - rollout_error, goal_valid[:, 0]
        ),
    }
    metrics.update(_history_strata(batch, short_error, rollout_error))
    return total, metrics
