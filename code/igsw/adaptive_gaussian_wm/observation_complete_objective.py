"""Observation-complete state, latent-effect, and image-goal objectives."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad, gather_batch_without_grad
from .observation_complete_state import scene_basis
from .v47_curriculum import V47Curriculum


def _pairwise_center(center: torch.Tensor) -> torch.Tensor:
    return center[:, :, None] - center[:, None]


def object_state_distance(
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = target["presence"].float().clamp_min(0.05)
    denominator = weight.sum().clamp_min(1.0)
    semantic = 1.0 - F.cosine_similarity(
        prediction["semantic"].float(), target["semantic"].float(), dim=-1, eps=1e-6
    )
    semantic = (semantic * weight).sum() / denominator
    dynamic = 1.0 - F.cosine_similarity(
        prediction["dynamic"].float(), target["dynamic"].float(), dim=-1, eps=1e-6
    )
    dynamic = (dynamic * weight).sum() / denominator
    relative_center = F.smooth_l1_loss(
        _pairwise_center(prediction["center"].float()),
        _pairwise_center(target["center"].float()), reduction="none",
    ).mean(dim=-1)
    pair_weight = weight[:, :, None] * weight[:, None]
    relative_center = (relative_center * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)
    relative_scale = F.smooth_l1_loss(
        prediction["log_scale"].float(), target["log_scale"].float(), reduction="none"
    ).mean(dim=-1)
    relative_scale = (relative_scale * weight).sum() / denominator
    lifecycle = F.smooth_l1_loss(
        torch.stack((prediction["presence"], prediction["visibility"]), dim=-1).float(),
        torch.stack((target["presence"], target["visibility"]), dim=-1).float(),
    )
    total = 0.5 * semantic + dynamic + 0.5 * relative_center + 0.25 * relative_scale + 0.25 * lifecycle
    return total, {
        "semantic": semantic, "dynamic": dynamic,
        "relative_center": relative_center, "relative_scale": relative_scale,
        "lifecycle": lifecycle,
    }


def _query_indices(patch_count: int, query_count: int, device: torch.device) -> torch.Tensor:
    count = min(patch_count, query_count)
    return torch.randperm(patch_count, device=device)[:count]


def _decode_queries(
    state: dict[str, torch.Tensor], coordinates: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    object_assignment = state["assignment"].float().index_select(-1, indices)
    scene_assignment = state["scene_assignment"].float().index_select(-1, indices)
    query_coordinates = coordinates.float().index_select(-2, indices)
    basis = scene_basis(query_coordinates)
    scene_feature = torch.einsum(
        "btqf,btfd->btqd", basis, state["scene_coefficients"].float()
    )
    object_feature = torch.einsum(
        "btkq,btkd->btqd", object_assignment, state["decoded_objects"].float()
    )
    prediction = object_feature + scene_assignment[..., None] * scene_feature
    return prediction, scene_feature, object_assignment, scene_assignment


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _identity_loss(
    model,
    predicted_identity: torch.Tensor,
    observation_identity: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = F.normalize(predicted_identity[:, 1:].float(), dim=-1, eps=1e-6)
    target = F.normalize(
        observation_identity[:, 1:].detach().float(), dim=-1, eps=1e-6
    )
    gathered = gather_batch_without_grad(target)
    batch, steps, count = source.shape[:3]
    candidates = gathered.permute(1, 0, 2, 3).reshape(steps, -1, source.shape[-1])
    logits = torch.einsum("btkd,tqd->btkq", source, candidates)
    logits = logits / model.config.identity_temperature
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    labels = rank * batch * count + torch.arange(
        batch * count, device=logits.device
    ).reshape(batch, count)
    labels = labels[:, None].expand(logits.shape[:-1])
    item_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    loss = _weighted_mean(item_loss, weight)
    accuracy = _weighted_mean((logits.argmax(-1) == labels).float(), weight)
    return loss, accuracy


def _masked_state_loss(
    full: dict[str, torch.Tensor],
    masked: dict[str, torch.Tensor],
    observation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = ~observation_mask
    if not bool(hidden.any()):
        raise ValueError("masked-state objective requires at least one hidden frame")
    keys = (
        "semantic", "dynamic", "center", "log_scale", "presence", "visibility",
    )
    prediction = {name: masked[name][hidden] for name in keys}
    target = {name: full[name][hidden].detach() for name in keys}
    return object_state_distance(prediction, target)


def _state_loss(
    model, output: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    full, masked = output["full_state"], output["masked_state"]
    target = full["patch_target"].float()
    valid = output["valid"]
    indices = _query_indices(target.shape[2], model.config.observation_queries, target.device)
    prediction, scene_prediction, assignment, scene_assignment = _decode_queries(
        full, output["coordinates"], indices
    )
    target_query = target.index_select(2, indices)
    valid_query = valid.index_select(2, indices).float()
    full_error = 1.0 - F.cosine_similarity(prediction, target_query, dim=-1, eps=1e-6)
    scene_error = 1.0 - F.cosine_similarity(
        scene_prediction, target_query, dim=-1, eps=1e-6
    )
    observation = _weighted_mean(full_error, valid_query)
    scene_only = _weighted_mean(scene_error, valid_query)
    object_gain = scene_error - full_error
    motion = torch.zeros_like(full_error)
    if target.shape[1] > 1:
        motion[:, 1:] = 1.0 - F.cosine_similarity(
            target_query[:, 1:], target_query[:, :-1], dim=-1, eps=1e-6
        )
    motion_weight = motion.clamp_min(0.0) * valid_query
    object_improvement = scene_error.detach() - full_error
    object_necessity = _weighted_mean(
        F.relu(model.config.object_gain_margin - object_improvement), motion_weight
    )
    motion_gain = _weighted_mean(object_gain, motion_weight)
    masked_prediction = _decode_queries(masked, output["coordinates"], indices)[0]
    masked_error = 1.0 - F.cosine_similarity(
        masked_prediction, target_query.detach(), dim=-1, eps=1e-6
    )
    masked_weight = (~output["observation_mask"])[..., None].float() * valid_query
    masked_loss = _weighted_mean(masked_error, masked_weight)
    masked_state, masked_state_parts = _masked_state_loss(
        full, masked, output["observation_mask"]
    )
    identity_weight = (
        full["presence"][:, :-1].float()
        * full["observed_visibility"][:, 1:].float()
        * output["observation_mask"][:, 1:, None].float()
    ).detach()
    identity, identity_top1 = _identity_loss(
        model,
        full["predicted_identity"],
        full["observation_identity"],
        identity_weight,
    )
    observed = output["observation_mask"][..., None].float()
    lifecycle_weight = observed.expand_as(full["observed_presence"])
    presence_prediction = _weighted_mean(
        (full["predicted_presence"].float() - full["observed_presence"].detach().float()).abs(),
        lifecycle_weight,
    )
    visibility_prediction = _weighted_mean(
        (full["predicted_visibility"].float() - full["observed_visibility"].detach().float()).abs(),
        lifecycle_weight,
    )
    lifecycle = presence_prediction + visibility_prediction
    object_fraction = assignment.sum(dim=2)
    staticness = torch.exp(
        -motion.detach().clamp_min(0.0) / model.config.object_gain_margin
    )
    gain_shortfall = (
        F.relu(model.config.object_gain_margin - object_gain.detach())
        / model.config.object_gain_margin
    )
    object_grounding = _weighted_mean(
        object_fraction * staticness * gain_shortfall,
        valid_query,
    )
    state_loss = (
        observation
        + 0.5 * masked_loss
        + 0.5 * masked_state
        + 0.25 * identity
        + 0.25 * object_necessity
        + 0.25 * object_grounding
        + 0.1 * lifecycle
    )
    object_mass = assignment.sum(dim=-1)
    valid_count = valid_query.sum(dim=-1, keepdim=True).clamp_min(1.0)
    effective = ((object_mass / valid_count) > 0.01).float().sum(dim=-1).mean()
    slot_fraction = object_mass / valid_count
    slot_utility = (
        (assignment * object_gain.detach()[:, :, None] * valid_query[:, :, None]).sum(dim=-1)
        / object_mass.clamp_min(1e-6)
    )
    supported = (
        (slot_fraction > 0.01)
        & (slot_utility > model.config.object_gain_margin)
    ).float().sum(dim=-1).mean()
    owner = torch.cat((assignment, scene_assignment[:, :, None]), dim=2)
    owner_entropy = -(owner.clamp_min(1e-7) * owner.clamp_min(1e-7).log()).sum(dim=2)
    parts = {
        "loss_state": state_loss,
        "loss_observation_complete": observation,
        "diagnostic_scene_only_error": scene_only,
        "diagnostic_object_gain": scene_only - observation,
        "diagnostic_motion_object_gain": motion_gain,
        "loss_object_necessity": object_necessity,
        "loss_object_grounding": object_grounding,
        "loss_masked_observation": masked_loss,
        "loss_masked_state": masked_state,
        "loss_track_observation_retrieval": identity,
        "track_observation_retrieval_top1": identity_top1,
        "loss_lifecycle_prediction": lifecycle,
        "loss_presence_prediction": presence_prediction,
        "loss_visibility_prediction": visibility_prediction,
        "object_effective_count": effective,
        "object_supported_count": supported,
        "object_owner_fraction": _weighted_mean(object_fraction, valid_query),
        "object_utility": _weighted_mean(slot_utility, slot_fraction),
        "scene_owner_fraction": _weighted_mean(scene_assignment, valid_query),
        "owner_assignment_entropy": _weighted_mean(owner_entropy, valid_query),
        "object_correction_gate": full["correction_gate"].float().mean(),
        "association_entropy": full["association_entropy"].float().mean(),
        "association_unmatched_probability": full["association_unmatched"].float().mean(),
        "association_appearance_similarity": full["association_appearance"].float().mean(),
        "observation_query_count": target.new_tensor(float(len(indices))),
        "masked_query_fraction": masked_weight.mean(),
        "presence_mean": full["presence"].float().mean(),
        "visibility_mean": full["visibility"].float().mean(),
        "presence_visibility_gap": (
            full["presence"].float() - full["visibility"].float()
        ).mean(),
    }
    parts.update({
        f"masked_state_{name}": value
        for name, value in masked_state_parts.items()
    })
    return state_loss, parts


def _intervention_loss(
    correct: torch.Tensor, zero: torch.Tensor, shuffled: torch.Tensor, margin: float
) -> torch.Tensor:
    return F.relu(margin + correct - zero) + F.relu(margin + correct - shuffled)


def _effect_statistics(effect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = gather_batch_with_grad(effect.float()).flatten(1)
    standard_deviation = gathered.std(dim=0, unbiased=False)
    variance = F.relu(0.05 - standard_deviation).mean()
    centered = gathered - gathered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().mean()


def observation_complete_loss(
    model, output: dict, curriculum: V47Curriculum
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state_loss, parts = _state_loss(model, output)
    correct, effect_parts = object_state_distance(
        output["short_prediction"], output["short_target"]
    )
    zero, _ = object_state_distance(output["short_zero"], output["short_target"])
    shuffled, _ = object_state_distance(output["short_shuffled"], output["short_target"])
    intervention = _intervention_loss(correct, zero, shuffled, model.config.intervention_margin)
    variance, covariance = _effect_statistics(output["short_effect"])
    effect_loss = correct + intervention + 0.05 * variance + 0.01 * covariance
    goal_correct, goal_parts = object_state_distance(
        output["goal_prediction"], output["goal_target"]
    )
    goal_zero, _ = object_state_distance(output["goal_zero"], output["goal_target"])
    goal_shuffled, _ = object_state_distance(output["goal_shuffled"], output["goal_target"])
    goal_intervention = _intervention_loss(
        goal_correct, goal_zero, goal_shuffled, model.config.intervention_margin
    )
    goal_alignment = 1.0 - F.cosine_similarity(
        output["goal_effect"].float().flatten(1),
        output["trajectory_effect"].detach().float().flatten(1), dim=-1, eps=1e-6,
    ).mean()
    goal_loss = goal_correct + goal_intervention + 0.5 * goal_alignment
    total = (
        curriculum.state_weight * state_loss
        + curriculum.effect_weight * effect_loss
        + curriculum.goal_weight * goal_loss
    )
    effect = output["short_effect"].float()
    parts.update({
        "loss": total, "loss_effect": effect_loss, "loss_goal": goal_loss,
        "effect_correct_distance": correct, "effect_zero_distance": zero,
        "effect_shuffled_distance": shuffled,
        "effect_relative_gain_over_zero": (zero - correct) / zero.detach().clamp_min(1e-6),
        "effect_relative_gain_over_shuffled": (shuffled - correct) / shuffled.detach().clamp_min(1e-6),
        "effect_intervention_loss": intervention,
        "effect_variance_penalty": variance, "effect_covariance_penalty": covariance,
        "effect_action_std": gather_batch_with_grad(effect).flatten(1).std(dim=0, unbiased=False).mean(),
        "effect_action_norm": effect.norm(dim=-1).mean(),
        "goal_correct_distance": goal_correct, "goal_zero_distance": goal_zero,
        "goal_shuffled_distance": goal_shuffled,
        "goal_relative_gain_over_zero": (goal_zero - goal_correct) / goal_zero.detach().clamp_min(1e-6),
        "goal_effect_alignment": goal_alignment,
        "curriculum_state_weight": total.new_tensor(curriculum.state_weight),
        "curriculum_effect_weight": total.new_tensor(curriculum.effect_weight),
        "curriculum_goal_weight": total.new_tensor(curriculum.goal_weight),
    })
    parts.update({f"effect_state_{name}": value for name, value in effect_parts.items()})
    parts.update({f"goal_state_{name}": value for name, value in goal_parts.items()})
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v47 objective produced a non-finite loss")
    return total, parts
