"""Frozen video-detail objectives layered on the compact DINO JEPA loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _detail_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or weight.shape != target.shape[:-1]:
        raise ValueError("video detail objective shapes differ")
    split = target.shape[-1] // 2
    appearance = 1.0 - F.cosine_similarity(
        predicted[..., :split].float(), target[..., :split].float(), dim=-1
    )
    robust = F.smooth_l1_loss(
        predicted.float(), target.float(), reduction="none"
    ).mean(dim=-1)
    return _weighted_mean(robust + 0.25 * appearance, weight)


def _owner_weight(state) -> torch.Tensor:
    object_mass = state.owner[..., :-2].sum(dim=-1)
    scene_mass = state.owner[..., -2]
    transient_mass = state.owner[..., -1]
    return object_mass + 0.25 * scene_mass + transient_mass


def dual_branch_detail_loss(model, output: dict, curriculum) -> tuple:
    online = output["online"]
    current_index = online["regions"]["feature"].shape[1] - 1
    current_feature = online["regions"]["feature"][:, current_index]
    current_target = online["regions"]["detail_latent"][:, current_index]
    current_weight = (
        online["regions"]["detail_valid"][:, current_index]
        * _owner_weight(online["region_states"][current_index])
    )
    current_valid = online["regions"]["detail_valid"][:, current_index]
    current_gate = online["regions"]["detail_gate"][:, current_index]
    decoded_current = model.region_memory.decode_video_feature(current_feature)
    current = _detail_error(decoded_current, current_target, current_weight)
    zero = current * 0.0
    short = zero
    short_persistence = zero
    goal_direct = zero
    goal_rollout = zero
    path = zero
    if output["region_prediction"] is not None:
        target = output["target_short_region"]
        weight = target.detail_valid * _owner_weight(target)
        short_prediction = model.region_memory.decode_video_feature(
            output["region_prediction"].future_feature[:, 0]
        )
        short = _detail_error(short_prediction, target.detail_latent, weight)
        short_persistence = _detail_error(
            decoded_current, target.detail_latent, weight
        )
    if output["rollout_region"] is not None:
        target = output["target_goal_region"]
        valid = output["future_horizon_valid"][:, 1, None]
        weight = target.detail_valid * _owner_weight(target) * valid
        direct_prediction = model.region_memory.decode_video_feature(
            output["region_prediction"].future_feature[:, 1]
        )
        rollout_prediction = model.region_memory.decode_video_feature(
            output["rollout_region"].future_feature[:, 0]
        )
        goal_direct = _detail_error(
            direct_prediction, target.detail_latent, weight
        )
        goal_rollout = _detail_error(
            rollout_prediction, target.detail_latent, weight
        )
        path = _detail_error(
            rollout_prediction, direct_prediction.detach(), weight
        )
    objective = current + curriculum.dynamics_weight * short
    objective = objective + curriculum.posterior_weight * (
        goal_direct
        + model.config.goal_rollout_weight * goal_rollout
        + model.config.path_consistency_weight * path
    )
    parts = {
        "loss_video_current": current,
        "loss_video_short": short,
        "loss_video_goal_direct": goal_direct,
        "loss_video_goal_rollout": goal_rollout,
        "loss_video_path": path,
        "diagnostic_video_short_persistence": short_persistence,
        "diagnostic_video_short_prediction": short,
        "diagnostic_video_short_gain_over_persistence": (
            short_persistence - short
        ) / short_persistence.clamp_min(1e-6),
        "diagnostic_video_gate_prior": torch.sigmoid(
            model.region_memory.video_gate[-1].bias.float()
        ).mean(),
        "diagnostic_video_gate_mean": _weighted_mean(
            current_gate, current_valid
        ),
        "diagnostic_video_gate_open_fraction": _weighted_mean(
            (current_gate > 0.5).float(), current_valid
        ),
    }
    features = output["video_features"]
    for name, value in features.items():
        parts[f"diagnostic_video_{name}_appearance_energy"] = (
            value.appearance_energy
        )
        parts[f"diagnostic_video_{name}_motion_energy"] = value.motion_energy
    return model.config.video_detail_loss_weight * objective, parts
