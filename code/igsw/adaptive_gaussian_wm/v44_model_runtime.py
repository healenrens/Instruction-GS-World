"""Dual DINO/Wan-VAE object-region JEPA runtime with strict causality."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compact_jepa_objective import compact_jepa_loss
from .compact_state_diagnostics import compact_state_diagnostics
from .distributed_statistics import roll_batch_with_grad
from .dual_branch_jepa_objective import dual_branch_detail_loss
from .hierarchical_region_dynamics import region_state_from_prediction
from .hierarchical_world_state import select_region_state
from .scale import signed_gap_scale
from .v43_curriculum import curriculum_at
from .v43_model_runtime import (
    _encode_sequence,
    _rollout_root,
    _root_dynamics,
    _single_region_history,
)


def _video_feature_sequences(
    dino_features,
    history_frames: int,
    video_features: dict,
    target: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = history_frames + (2 if target else 0)
    batch, tokens = dino_features.projected.shape[0], dino_features.projected.shape[2]
    dimension = next(iter(video_features.values())).feature.shape[-1]
    auxiliary = dino_features.projected.new_zeros(
        (batch, frames, tokens, dimension)
    )
    valid = torch.zeros((batch, frames), device=auxiliary.device, dtype=torch.bool)
    current_index = history_frames - 1
    history = video_features["history"].feature
    if history.shape[:2] != (batch, tokens):
        raise ValueError("history VAE features do not match the DINO grid")
    auxiliary[:, current_index] = history.to(auxiliary.dtype)
    valid[:, current_index] = True
    if target and "short" in video_features:
        auxiliary[:, history_frames] = video_features["short"].feature.to(
            auxiliary.dtype
        )
        valid[:, history_frames] = True
    if target and "goal" in video_features:
        auxiliary[:, history_frames + 1] = video_features["goal"].feature.to(
            auxiliary.dtype
        )
        valid[:, history_frames + 1] = True
    return auxiliary, valid


@torch.no_grad()
def _target_paths(model, batch: dict, video_features: dict) -> tuple[dict, torch.Tensor]:
    target_rgb = torch.cat(
        (batch["history_jit_rgb"], batch["future_jit_rgb"]), dim=1
    )
    target_valid = torch.cat(
        (batch["history_jit_valid"], batch["future_jit_valid"]), dim=1
    )
    target_features = model.target_dino(target_rgb, target_valid)
    history_count = batch["history_jit_rgb"].shape[1]
    auxiliary, auxiliary_valid = _video_feature_sequences(
        target_features, history_count, video_features, target=True
    )
    target_times = torch.cat((batch["history_times"], batch["future_times"]), dim=1)
    target = _encode_sequence(
        model,
        target_features,
        target_times,
        target=True,
        masked=False,
        auxiliary_features=auxiliary,
        auxiliary_valid=auxiliary_valid,
    )
    target["valid"] = target_features.valid
    target["native_features"] = target_features.native
    horizon_valid = batch["future_horizon_valid"].clone()
    probe = model.target_dino(
        batch["goal_probe_jit_rgb"], batch["goal_probe_jit_valid"]
    )
    goal = target_features.projected[:, history_count + 1 : history_count + 2]
    goal_valid = target_features.valid[:, history_count + 1 : history_count + 2]
    trajectory = torch.cat((probe.projected, goal), dim=1).float()
    trajectory_valid = torch.cat((probe.valid, goal_valid), dim=1)
    pair_valid = trajectory_valid[:, 1:] & trajectory_valid[:, :-1]
    pair_error = 1.0 - F.cosine_similarity(
        trajectory[:, 1:], trajectory[:, :-1], dim=-1
    )
    support = pair_valid.float().sum(dim=(-1, -2))
    stability = (pair_error * pair_valid).sum(dim=(-1, -2)) / support.clamp_min(1.0)
    goal_pixels = batch["future_jit_rgb"][:, 1].float()
    pixel_valid = batch["future_jit_valid"][:, 1, None].expand_as(goal_pixels)
    pixel_weight = pixel_valid.float()
    pixel_support = pixel_weight.sum(dim=(1, 2, 3))
    mean = (goal_pixels * pixel_weight).sum(dim=(1, 2, 3)) / pixel_support.clamp_min(1.0)
    variance = (
        (goal_pixels - mean[:, None, None, None]).square() * pixel_weight
    ).sum(dim=(1, 2, 3)) / pixel_support.clamp_min(1.0)
    content_valid = (
        (pixel_support > 0)
        & (support > 0)
        & (variance.sqrt() > 2.0)
        & (mean > 2.0)
        & (mean < 253.0)
    )
    horizon_valid[:, 1] &= (
        stability <= model.config.goal_stability_threshold
    ) & content_valid
    target["goal_stability_error"] = stability
    target["goal_content_valid"] = content_valid
    return target, horizon_valid


def _encode_video_features(model, batch: dict, curriculum) -> dict:
    clips = {
        "history": (batch["history_video_rgb"], batch["history_video_valid"]),
    }
    if curriculum.step >= model.config.curriculum_spatial_steps:
        clips["short"] = (
            batch["short_video_rgb"],
            batch["short_video_valid"],
        )
    if curriculum.step >= model.config.curriculum_posterior_steps:
        clips["goal"] = (batch["goal_video_rgb"], batch["goal_video_valid"])
    return model.video_vae.encode_clips(clips)


def forward_v44(model, batch: dict, collect_diagnostics: bool) -> dict:
    required = {
        "history_jit_rgb", "future_jit_rgb", "history_jit_valid",
        "future_jit_valid", "goal_probe_jit_rgb", "goal_probe_jit_valid",
        "history_times", "future_times", "future_horizon_valid",
        "history_length", "history_video_rgb", "history_video_valid",
        "short_video_rgb", "short_video_valid", "goal_video_rgb",
        "goal_video_valid",
    }
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"v44 batch is missing {sorted(missing)}")
    history_frames = batch["history_jit_rgb"].shape[1]
    if not bool((batch["history_length"] == history_frames).all()):
        raise ValueError("v44 batch mixes history lengths")
    curriculum = curriculum_at(int(model.curriculum_step.item()), model.config)
    video_features = _encode_video_features(model, batch, curriculum)
    online_features = model.online_dino(
        batch["history_jit_rgb"], batch["history_jit_valid"]
    )
    online_auxiliary, online_auxiliary_valid = _video_feature_sequences(
        online_features, history_frames, video_features, target=False
    )
    online = _encode_sequence(
        model,
        online_features,
        batch["history_times"],
        target=False,
        masked=True,
        auxiliary_features=online_auxiliary,
        auxiliary_valid=online_auxiliary_valid,
    )
    online["times"] = batch["history_times"]
    target, horizon_valid = _target_paths(model, batch, video_features)
    short_index, goal_index = history_frames, history_frames + 1
    target_short_root = target["root_states"][short_index]
    target_goal_root = target["root_states"][goal_index]
    target_short_region = select_region_state(target["regions"], short_index)
    target_goal_region = select_region_state(target["regions"], goal_index)
    short_action = online["roots"]["slots"].new_zeros(
        len(batch["history_times"]), model.config.action_tokens, model.config.action_dim
    )
    tail_action = torch.zeros_like(short_action)
    composed_action = torch.zeros_like(short_action)
    root_prediction = region_prediction = None
    rollout_root = rollout_region = None
    shuffled_root = shuffled_region = None
    future_scale = signed_gap_scale(batch["future_times"], model.config.gap_reference)
    if curriculum.step >= model.config.curriculum_spatial_steps:
        if curriculum.step >= model.config.curriculum_posterior_steps:
            short_action = model.region_effect_posterior(
                online["last_root"].slots, target_short_root.slots,
                online["last_region"], target_short_region,
            )
            tail_action = model.region_effect_posterior(
                target_short_root.slots, target_goal_root.slots,
                target_short_region, target_goal_region,
            )
            composed_action = model.effect_composer(short_action, tail_action)
        actions = torch.stack((short_action, composed_action), dim=1)
        root_prediction = _root_dynamics(model, online, future_scale, actions)
        region_prediction = model.region_dynamics(
            online["regions"], root_prediction.future_slots,
            root_prediction.future_centers, future_scale, actions,
            base_root_future_slots=root_prediction.base_future_slots,
        )
        if curriculum.step >= model.config.curriculum_posterior_steps:
            rollout_root = _rollout_root(
                model, root_prediction, future_scale[:, :1],
                future_scale[:, 1:2], tail_action,
            )
            rollout_region = model.region_dynamics(
                _single_region_history(region_state_from_prediction(region_prediction, 0)),
                rollout_root.future_slots, rollout_root.future_centers,
                future_scale[:, 1:2], tail_action[:, None],
                base_root_future_slots=rollout_root.base_future_slots,
            )
            shuffled_action = roll_batch_with_grad(short_action)
            shuffled_root = _root_dynamics(
                model, online, future_scale[:, :1], shuffled_action[:, None]
            )
            shuffled_region = model.region_dynamics(
                online["regions"], shuffled_root.future_slots,
                shuffled_root.future_centers, future_scale[:, :1],
                shuffled_action[:, None],
                base_root_future_slots=shuffled_root.base_future_slots,
            )
    output = {
        "online": online, "target": target,
        "target_short_root": target_short_root,
        "target_goal_root": target_goal_root,
        "target_short_region": target_short_region,
        "target_goal_region": target_goal_region,
        "root_prediction": root_prediction, "region_prediction": region_prediction,
        "rollout_root": rollout_root, "rollout_region": rollout_region,
        "shuffled_root": shuffled_root, "shuffled_region": shuffled_region,
        "short_action": short_action, "tail_action": tail_action,
        "composed_action": composed_action,
        "future_horizon_valid": horizon_valid, "curriculum": curriculum,
        "video_features": video_features,
    }
    loss, parts = compact_jepa_loss(model, batch, output, curriculum)
    detail_loss, detail_parts = dual_branch_detail_loss(model, output, curriculum)
    loss = loss + detail_loss
    parts.update(detail_parts)
    parts["total"] = loss
    history_length = int(batch["history_length"][0].item())
    parts[f"history_h{history_length}_short_root_error"] = parts["loss_short_root"].detach()
    parts[f"history_h{history_length}_short_region_error"] = parts["loss_short_region"].detach()
    persistence = parts["diagnostic_short_persistence_root"] + parts["diagnostic_short_persistence_region"]
    predicted = parts["loss_short_root"] + parts["loss_short_region"]
    parts["diagnostic_short_relative_gain_over_persistence"] = (
        persistence - predicted
    ) / persistence.clamp_min(1e-6)
    parts["diagnostic_effect_relative_gain_over_zero"] = (
        parts["diagnostic_zero_effect"] - parts["diagnostic_posterior_effect"]
    ) / parts["diagnostic_zero_effect"].clamp_min(1e-6)
    parts["diagnostic_effect_relative_gain_over_shuffled"] = (
        parts["diagnostic_shuffled_effect"] - parts["diagnostic_posterior_effect"]
    ) / parts["diagnostic_shuffled_effect"].clamp_min(1e-6)
    parts["diagnostic_goal_direct_gain_over_persistence"] = (
        parts["diagnostic_goal_persistence"] - parts["diagnostic_goal_direct"]
    ) / parts["diagnostic_goal_persistence"].clamp_min(1e-6)
    parts["diagnostic_goal_rollout_gain_over_persistence"] = (
        parts["diagnostic_goal_persistence"] - parts["diagnostic_goal_rollout"]
    ) / parts["diagnostic_goal_persistence"].clamp_min(1e-6)
    if collect_diagnostics:
        parts.update(compact_state_diagnostics(model, batch, output))
    output["loss"] = loss
    output["parts"] = parts
    return output
