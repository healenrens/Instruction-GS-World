"""Compact root/region JEPA objectives without dense future reconstruction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compact_geometry_objective import (
    path_geometry_error,
    path_region_identity_error,
    region_geometry_error,
    region_identity_error,
    root_geometry_error,
)
from .compact_state_regularization import (
    action_statistics_regularizer,
    feature_statistics_regularizer,
    transient_owner_regularizer,
)


def _cosine_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(predicted.float(), target.float(), dim=-1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _state_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _weighted_mean(_cosine_error(predicted, target), weight)


def _region_owner_weight(owner: torch.Tensor) -> torch.Tensor:
    object_mass = owner[..., :-2].sum(dim=-1)
    scene_mass = owner[..., -2]
    transient_mass = owner[..., -1]
    return object_mass + 0.1 * scene_mass + 0.25 * transient_mass


def _motion_weight(
    target_delta: torch.Tensor,
    base_weight: torch.Tensor,
) -> torch.Tensor:
    change = target_delta.float().square().mean(dim=-1).sqrt().detach()
    scale = (change * base_weight).sum(dim=-1, keepdim=True)
    scale = scale / base_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    relative = (change / scale.clamp_min(1e-4)).clamp_max(4.0)
    return base_weight * (0.25 + relative)


def _delta_state_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    predicted_rms = (
        predicted.float().square().mean(dim=-1).clamp_min(1e-12).sqrt()
    )
    target_rms = target.float().square().mean(dim=-1).clamp_min(1e-12).sqrt()
    moving = weight * (target_rms.detach() > 1e-3).to(weight.dtype)
    direction = _weighted_mean(_cosine_error(predicted, target), moving)
    magnitude = _weighted_mean(
        F.smooth_l1_loss(predicted_rms, target_rms, reduction="none"),
        weight,
    )
    return direction + magnitude


def compact_jepa_loss(model, batch: dict, output: dict, curriculum) -> tuple:
    online = output["online"]
    target = output["target"]
    history_count = online["roots"]["slots"].shape[1]
    current_index = history_count - 1
    current_token = online["token_states"][-1]
    target_native = target["native_features"][:, current_index]
    allocator = _weighted_mean(
        _cosine_error(current_token.reconstructed_features, target_native),
        target["valid"][:, current_index],
    )
    allocator_rate = current_token.budget_fraction.mean()
    mask = online["masked_region_positions"]
    if mask is None or online["masked_region_prediction"] is None:
        raise ValueError("v43 training requires masked-region predictions")
    masked_temporal = _weighted_mean(
        _cosine_error(
            online["masked_region_prediction"],
            target["contextual_regions"][:, :history_count],
        ),
        mask,
    )
    temporal_order = masked_temporal * 0.0
    if history_count > 2:
        permutation = torch.cat(
            (
                torch.arange(
                    history_count - 2,
                    -1,
                    -1,
                    device=online["transformer_inputs"].device,
                ),
                torch.tensor(
                    [history_count - 1],
                    device=online["transformer_inputs"].device,
                ),
            )
        )
        activation = torch.stack(
            [state.activation.squeeze(-1) for state in online["token_states"]],
            dim=1,
        )
        reversed_current = model.region_transformer._run(
            online["transformer_inputs"][:, permutation],
            activation[:, permutation] > 0.5,
        )[:, -1]
        target_current_context = target["contextual_regions"][:, current_index]
        target_previous_context = target["contextual_regions"][:, current_index - 1]
        target_change = _cosine_error(
            target_current_context, target_previous_context
        ).detach()
        temporal_weight = (
            target["regions"]["presence"][:, current_index] * target_change
        )
        ordered = _cosine_error(
            online["contextual_regions"][:, current_index],
            target_current_context,
        )
        reversed_error = _cosine_error(reversed_current, target_current_context)
        temporal_order = _weighted_mean(
            F.relu(ordered + 0.05 - reversed_error), temporal_weight
        )
    online_root = online["roots"]["slots"][:, current_index]
    target_root_current = target["roots"]["slots"][:, current_index]
    current_root = _state_error(
        online_root,
        target_root_current,
        target["roots"]["existence"][:, current_index],
    )
    online_region = online["regions"]["feature"][:, current_index]
    target_region_current = target["regions"]["feature"][:, current_index]
    current_region = _state_error(
        online_region,
        target_region_current,
        target["regions"]["presence"][:, current_index],
    )
    online_owner = online["regions"]["owner"][:, current_index].float()
    target_owner = target["regions"]["owner"][:, current_index].float()
    owner_kl = (
        target_owner
        * (
            target_owner.clamp_min(1e-6).log()
            - online_owner.clamp_min(1e-6).log()
        )
    ).sum(dim=-1)
    owner_alignment = _weighted_mean(
        owner_kl,
        target["regions"]["presence"][:, current_index],
    )
    transient_fraction = _weighted_mean(
        online_owner[..., -1],
        online["regions"]["presence"][:, current_index],
    )
    transient_lifecycle = transient_owner_regularizer(
        online_owner,
        online["regions"]["association_confidence"][:, current_index],
        online["regions"]["presence"][:, current_index],
    )
    owner_regularization = (
        owner_alignment
        + F.relu(transient_fraction - 0.25)
        + 0.25 * transient_lifecycle
    )
    correspondence = _weighted_mean(
        1.0 - online["regions"]["association_confidence"][:, current_index],
        online["regions"]["presence"][:, current_index]
        * online["regions"]["owner"][:, current_index, :, :-2].sum(dim=-1),
    )
    representation = (
        allocator
        + 0.01 * allocator_rate
        + masked_temporal
        + 0.25 * temporal_order
        + current_root
        + current_region
    )
    representation = (
        representation + 0.25 * correspondence + 0.1 * owner_regularization
    )
    region_variance, region_covariance = feature_statistics_regularizer(
        online_region,
        online["regions"]["presence"][:, current_index] > 0.5,
    )
    dino_variance, dino_covariance = feature_statistics_regularizer(
        online["native_features"][:, current_index, ::8],
        target["valid"][:, current_index, ::8],
    )
    representation_variance = region_variance + 0.5 * dino_variance
    representation_covariance = region_covariance + 0.5 * dino_covariance
    zero = representation * 0.0
    if output["root_prediction"] is None:
        total = (
            representation
            + 0.05 * representation_variance
            + 0.01 * representation_covariance
        )
        return total, {
            "total": total,
            "curriculum_step": total.new_tensor(float(curriculum.step)),
            "curriculum_dynamics_weight": total.new_tensor(
                curriculum.dynamics_weight
            ),
            "curriculum_posterior_weight": total.new_tensor(
                curriculum.posterior_weight
            ),
            "loss_allocator": allocator,
            "loss_allocator_rate": allocator_rate,
            "loss_masked_temporal": masked_temporal,
            "loss_temporal_order": temporal_order,
            "loss_current_root": current_root,
            "loss_current_region": current_region,
            "loss_correspondence": correspondence,
            "loss_owner": owner_regularization,
            "loss_transient_lifecycle": transient_lifecycle,
            "loss_short_root": zero,
            "loss_short_region": zero,
            "loss_delta_root": zero,
            "loss_delta_region": zero,
            "loss_geometry_lifecycle": zero,
            "loss_region_identity": zero,
            "loss_goal": zero,
            "loss_path_root": zero,
            "loss_path_region": zero,
            "loss_effect_margin": zero,
            "loss_variance": representation_variance,
            "loss_covariance": representation_covariance,
            "diagnostic_short_persistence_root": zero,
            "diagnostic_short_persistence_region": zero,
            "diagnostic_zero_effect": zero,
            "diagnostic_shuffled_effect": zero,
            "diagnostic_posterior_effect": zero,
            "diagnostic_goal_direct": zero,
            "diagnostic_goal_rollout": zero,
            "diagnostic_goal_persistence": zero,
        }

    root_prediction = output["root_prediction"]
    region_prediction = output["region_prediction"]
    target_short_root = output["target_short_root"]
    target_short_region = output["target_short_region"]
    target_root_delta = target_short_root.slots - target_root_current
    short_root_weight = _motion_weight(
        target_root_delta,
        target_short_root.existence,
    )
    short_root = _state_error(
        root_prediction.future_slots[:, 0],
        target_short_root.slots,
        short_root_weight,
    )
    target_region_delta = target_short_region.feature - target_region_current
    short_region_weight = _motion_weight(
        target_region_delta,
        target_short_region.presence
        * _region_owner_weight(target_short_region.owner),
    )
    short_region = _state_error(
        region_prediction.future_feature[:, 0],
        target_short_region.feature,
        short_region_weight,
    )
    predicted_root_delta = root_prediction.future_slots[:, 0] - online_root
    root_delta = _delta_state_error(
        predicted_root_delta,
        target_root_delta,
        short_root_weight,
    )
    predicted_region_delta = region_prediction.future_feature[:, 0] - online_region
    region_delta = _delta_state_error(
        predicted_region_delta,
        target_region_delta,
        short_region_weight,
    )
    geometry = root_geometry_error(root_prediction, target_short_root, 0)
    geometry = geometry + region_geometry_error(
        region_prediction, target_short_region, 0
    )
    identity = region_identity_error(
        region_prediction, target_short_region, 0
    )
    dynamics = (
        short_root
        + short_region
        + 0.5 * (root_delta + region_delta)
        + geometry
        + identity
    )

    persistence_root = _state_error(
        online_root, target_short_root.slots, short_root_weight
    )
    persistence_region = _state_error(
        online_region, target_short_region.feature, short_region_weight
    )
    if output["rollout_root"] is None or output["rollout_region"] is None:
        total = representation + curriculum.dynamics_weight * dynamics
        total = (
            total
            + 0.05 * representation_variance
            + 0.01 * representation_covariance
        )
        return total, {
            "total": total,
            "curriculum_step": total.new_tensor(float(curriculum.step)),
            "curriculum_dynamics_weight": total.new_tensor(
                curriculum.dynamics_weight
            ),
            "curriculum_posterior_weight": total.new_tensor(
                curriculum.posterior_weight
            ),
            "loss_allocator": allocator,
            "loss_allocator_rate": allocator_rate,
            "loss_masked_temporal": masked_temporal,
            "loss_temporal_order": temporal_order,
            "loss_current_root": current_root,
            "loss_current_region": current_region,
            "loss_correspondence": correspondence,
            "loss_owner": owner_regularization,
            "loss_transient_lifecycle": transient_lifecycle,
            "loss_short_root": short_root,
            "loss_short_region": short_region,
            "loss_delta_root": root_delta,
            "loss_delta_region": region_delta,
            "loss_geometry_lifecycle": geometry,
            "loss_region_identity": identity,
            "loss_goal": zero,
            "loss_path_root": zero,
            "loss_path_region": zero,
            "loss_effect_margin": zero,
            "loss_variance": representation_variance,
            "loss_covariance": representation_covariance,
            "diagnostic_short_persistence_root": persistence_root,
            "diagnostic_short_persistence_region": persistence_region,
            "diagnostic_zero_effect": short_root + short_region,
            "diagnostic_shuffled_effect": short_root + short_region,
            "diagnostic_posterior_effect": short_root + short_region,
            "diagnostic_goal_direct": zero,
            "diagnostic_goal_rollout": zero,
            "diagnostic_goal_persistence": zero,
        }

    goal_valid = output["future_horizon_valid"][:, 1]
    target_goal_root = output["target_goal_root"]
    target_goal_region = output["target_goal_region"]
    goal_root_weight = goal_valid[:, None] * _motion_weight(
        target_goal_root.slots - target_root_current,
        target_goal_root.existence,
    )
    direct_root = _state_error(
        root_prediction.future_slots[:, 1],
        target_goal_root.slots,
        goal_root_weight,
    )
    goal_region_weight = goal_valid[:, None] * _motion_weight(
        target_goal_region.feature - target_region_current,
        target_goal_region.presence
        * _region_owner_weight(target_goal_region.owner),
    )
    direct_region = _state_error(
        region_prediction.future_feature[:, 1],
        target_goal_region.feature,
        goal_region_weight,
    )
    rollout_root = _state_error(
        output["rollout_root"].future_slots[:, 0],
        target_goal_root.slots,
        goal_root_weight,
    )
    rollout_region = _state_error(
        output["rollout_region"].future_feature[:, 0],
        target_goal_region.feature,
        goal_region_weight,
    )
    direct_geometry = root_geometry_error(
        root_prediction, target_goal_root, 1, goal_valid
    )
    direct_geometry = direct_geometry + region_geometry_error(
        region_prediction, target_goal_region, 1, goal_valid
    )
    rollout_geometry = root_geometry_error(
        output["rollout_root"], target_goal_root, 0, goal_valid
    )
    rollout_geometry = rollout_geometry + region_geometry_error(
        output["rollout_region"], target_goal_region, 0, goal_valid
    )
    path_root = _state_error(
        output["rollout_root"].future_slots[:, 0],
        root_prediction.future_slots[:, 1].detach(),
        goal_root_weight,
    )
    path_region = _state_error(
        output["rollout_region"].future_feature[:, 0],
        region_prediction.future_feature[:, 1].detach(),
        goal_region_weight,
    )
    path_geometry = path_geometry_error(
        root_prediction,
        region_prediction,
        output["rollout_root"],
        output["rollout_region"],
        goal_root_weight,
        goal_region_weight,
    )
    direct_identity = region_identity_error(
        region_prediction, target_goal_region, 1, goal_valid
    )
    rollout_identity = region_identity_error(
        output["rollout_region"], target_goal_region, 0, goal_valid
    )
    path_identity = path_region_identity_error(
        region_prediction,
        output["rollout_region"],
        goal_region_weight,
    )
    goal_identity = direct_identity + model.config.goal_rollout_weight * (
        rollout_identity
    )
    goal_identity = goal_identity + model.config.path_consistency_weight * (
        path_identity
    )
    goal = direct_root + direct_region + direct_geometry + direct_identity
    goal = goal + model.config.goal_rollout_weight * (
        rollout_root + rollout_region + rollout_geometry + rollout_identity
    )
    goal = goal + model.config.path_consistency_weight * (
        path_root + path_region + path_geometry + path_identity
    )
    goal_persistence = _state_error(
        online_root,
        target_goal_root.slots,
        goal_root_weight,
    ) + _state_error(
        online_region,
        target_goal_region.feature,
        goal_region_weight,
    )

    matched_short = short_root + short_region
    zero_root = _state_error(
        root_prediction.base_future_slots[:, 0],
        target_short_root.slots,
        short_root_weight,
    )
    zero_region = _state_error(
        region_prediction.base_future_feature[:, 0],
        target_short_region.feature,
        short_region_weight,
    )
    if output["shuffled_root"] is None or output["shuffled_region"] is None:
        raise ValueError("posterior curriculum requires shuffled-effect predictions")
    shuffled_root = _state_error(
        output["shuffled_root"].future_slots[:, 0],
        target_short_root.slots,
        short_root_weight,
    )
    shuffled_region = _state_error(
        output["shuffled_region"].future_feature[:, 0],
        target_short_region.feature,
        short_region_weight,
    )
    zero_effect_error = (zero_root + zero_region).detach()
    shuffled_effect_error = (shuffled_root + shuffled_region).detach()
    effect_margin = F.relu(matched_short + 0.05 - zero_effect_error)
    effect_margin = effect_margin + F.relu(
        matched_short + 0.05 - shuffled_effect_error
    )
    effect_variance, effect_covariance = action_statistics_regularizer(
        torch.cat((output["short_action"], output["tail_action"]), dim=0)
    )
    variance = representation_variance + effect_variance
    covariance = representation_covariance + effect_covariance
    total = representation
    total = total + curriculum.dynamics_weight * dynamics
    total = total + curriculum.posterior_weight * (effect_margin + goal)
    total = total + 0.05 * variance + 0.01 * covariance
    parts = {
        "total": total,
        "curriculum_step": total.new_tensor(float(curriculum.step)),
        "curriculum_dynamics_weight": total.new_tensor(curriculum.dynamics_weight),
        "curriculum_posterior_weight": total.new_tensor(curriculum.posterior_weight),
        "loss_allocator": allocator,
        "loss_allocator_rate": allocator_rate,
        "loss_masked_temporal": masked_temporal,
        "loss_temporal_order": temporal_order,
        "loss_current_root": current_root,
        "loss_current_region": current_region,
        "loss_correspondence": correspondence,
        "loss_owner": owner_regularization,
        "loss_transient_lifecycle": transient_lifecycle,
        "loss_short_root": short_root,
        "loss_short_region": short_region,
        "loss_delta_root": root_delta,
        "loss_delta_region": region_delta,
        "loss_geometry_lifecycle": geometry,
        "loss_region_identity": identity + goal_identity,
        "loss_goal": goal,
        "loss_goal_geometry": direct_geometry + rollout_geometry,
        "loss_path_root": path_root,
        "loss_path_region": path_region,
        "loss_path_geometry": path_geometry,
        "loss_effect_margin": effect_margin,
        "loss_variance": variance,
        "loss_covariance": covariance,
        "diagnostic_short_persistence_root": persistence_root,
        "diagnostic_short_persistence_region": persistence_region,
        "diagnostic_zero_effect": zero_root + zero_region,
        "diagnostic_shuffled_effect": shuffled_root + shuffled_region,
        "diagnostic_posterior_effect": matched_short,
        "diagnostic_goal_direct": direct_root + direct_region,
        "diagnostic_goal_rollout": rollout_root + rollout_region,
        "diagnostic_goal_persistence": goal_persistence,
    }
    return total, parts
