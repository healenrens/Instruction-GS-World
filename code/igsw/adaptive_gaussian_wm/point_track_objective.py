"""Teacher-student Object State and posterior latent-effect objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .trajectory_component_teacher import LIFECYCLE_OCCLUDED, LIFECYCLE_VISIBLE


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float().clamp(0.0, 1.0)
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())


def owner_supervision(match, evidence) -> tuple[torch.Tensor, dict]:
    prediction = match.sampled_student_assignment.float().clamp_min(1e-6)
    target = match.target_student_owner[:, None].expand_as(prediction)
    visible = evidence.visibility.float()
    loss = weighted_mean(-(target * prediction.log()).sum(dim=-1), visible)
    shuffled_target = target.roll(1, dims=2)
    shuffled = weighted_mean(
        -(shuffled_target * prediction.log()).sum(dim=-1), visible
    )
    accuracy = weighted_mean(
        (prediction.argmax(dim=-1) == target.argmax(dim=-1)).float(), visible
    )
    return loss, {
        "loss_teacher_owner": loss,
        "teacher_owner_top1": accuracy,
        "teacher_owner_shuffled_loss": shuffled,
        "teacher_owner_gain_over_shuffled": (
            shuffled - loss
        ) / shuffled.detach().clamp_min(1e-6),
    }


def identity_supervision(model, state, match) -> tuple[torch.Tensor, torch.Tensor, dict]:
    prediction = F.normalize(
        model.identity_readout(state["identity"].float()), dim=-1, eps=1e-6
    )
    target = match.identity.detach()[:, None]
    distance = 1.0 - (prediction * target).sum(dim=-1)
    visible = match.visibility.detach() * match.component_valid[:, None].float()
    identity = weighted_mean(distance, visible)
    occluded_before = (
        match.lifecycle_state == LIFECYCLE_OCCLUDED
    ).float().cumsum(dim=1) > 0
    reappearance_weight = visible * occluded_before.float()
    reappearance = weighted_mean(distance, reappearance_weight)
    shuffled_target = target.roll(1, dims=2)
    shuffled = weighted_mean(
        1.0 - (prediction * shuffled_target).sum(dim=-1), visible
    )
    return identity, reappearance, {
        "loss_teacher_identity": identity,
        "loss_teacher_reappearance": reappearance,
        "teacher_identity_shuffled_distance": shuffled,
        "teacher_identity_gain_over_shuffled": (
            shuffled - identity
        ) / shuffled.detach().clamp_min(1e-6),
        "teacher_reappearance_weight": reappearance_weight.sum(),
    }


def dynamic_geometry_lifecycle(model, state, match):
    motion_prediction = model.motion_readout(state["dynamic"][:, :-1].float())
    motion_weight = match.motion_valid.detach().float()
    motion = weighted_mean(
        F.smooth_l1_loss(
            motion_prediction.float(), match.motion.detach().float(), reduction="none"
        ).mean(dim=-1),
        motion_weight,
    )
    geometry_weight = match.geometry_valid.detach().float()
    prediction_relative = state["center"][:, :, :, None] - state["center"][:, :, None]
    target_relative = match.center[:, :, :, None] - match.center[:, :, None]
    pair_weight = geometry_weight[:, :, :, None] * geometry_weight[:, :, None]
    center = weighted_mean(
        F.smooth_l1_loss(
            prediction_relative.float(), target_relative.detach().float(), reduction="none"
        ).mean(dim=-1),
        pair_weight,
    )
    prediction_scale = state["log_scale"][:, :, :, None] - state["log_scale"][:, :, None]
    target_scale = match.log_scale[:, :, :, None] - match.log_scale[:, :, None]
    scale = weighted_mean(
        F.smooth_l1_loss(
            prediction_scale.float(), target_scale.detach().float(), reduction="none"
        ),
        pair_weight,
    )
    shape = weighted_mean(
        F.smooth_l1_loss(
            state["support_shape"].float(), match.support_shape.detach().float(), reduction="none"
        ).mean(dim=-1),
        geometry_weight,
    )
    lifecycle_weight = match.lifecycle_known.detach().float()
    visibility = weighted_mean(
        binary_cross_entropy(state["visibility"], match.visibility), lifecycle_weight
    )
    presence = weighted_mean(
        binary_cross_entropy(state["presence"], match.presence), lifecycle_weight
    )
    return motion, center + scale + shape, visibility + presence, {
        "loss_teacher_motion": motion,
        "loss_teacher_relative_center": center,
        "loss_teacher_relative_scale": scale,
        "loss_teacher_support_shape": shape,
        "loss_teacher_visibility": visibility,
        "loss_teacher_presence": presence,
        "teacher_visible_fraction": (match.lifecycle_state == LIFECYCLE_VISIBLE).float().mean(),
        "teacher_occluded_fraction": (match.lifecycle_state == LIFECYCLE_OCCLUDED).float().mean(),
        "teacher_unknown_fraction": (~match.lifecycle_known).float().mean(),
    }


def object_state_objective(model, patches, valid, evidence, teacher, match, output):
    target = F.normalize(patches.float(), dim=-1, eps=1e-6)
    reconstruction = weighted_mean(
        1.0 - (output["reconstruction"].float() * target).sum(dim=-1), valid.float()
    )
    owner, owner_parts = owner_supervision(match, evidence)
    identity, reappearance, identity_parts = identity_supervision(
        model, output["state"], match
    )
    motion, geometry, lifecycle, state_parts = dynamic_geometry_lifecycle(
        model, output["state"], match
    )
    state = output["state"]
    identity_normalized = F.normalize(state["identity"].float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("btkd,btjd->btkj", identity_normalized, identity_normalized)
    diagonal = torch.eye(model.config.object_slots, device=similarity.device, dtype=torch.bool)
    diversity = F.relu(similarity - 0.20).masked_fill(diagonal[None, None], 0.0).mean()
    target_object = match.target_student_owner[..., : model.config.object_slots].sum(dim=-1)
    predicted_object = match.sampled_student_assignment[..., : model.config.object_slots].sum(dim=-1)
    motion_weight = (
        teacher.track_motion[:, None]
        * evidence.visibility.float()
        * target_object[:, None]
    )
    motion_coverage = weighted_mean(F.relu(0.5 - predicted_object), motion_weight)
    total = (
        model.config.reconstruction_weight * reconstruction
        + model.config.track_assignment_weight * owner
        + model.config.identity_weight * identity
        + model.config.reappearance_weight * reappearance
        + model.config.motion_weight * motion
        + model.config.lifecycle_weight * lifecycle
        + model.config.geometry_weight * geometry
        + model.config.diversity_weight * diversity
        + model.config.motion_coverage_weight * motion_coverage
    )
    parts = {
        "loss": total,
        "loss_object_state": total,
        "loss_reconstruction_auxiliary": reconstruction,
        "loss_identity_diversity": diversity,
        "loss_motion_object_coverage": motion_coverage,
        "teacher_component_count": teacher.component_valid.float().sum(dim=-1).mean(),
        "teacher_object_track_fraction": target_object.mean(),
        "student_object_track_fraction": predicted_object.mean(),
        "scene_owner_fraction": match.sampled_student_assignment[..., model.config.object_slots].mean(),
        "transient_owner_fraction": match.sampled_student_assignment[..., model.config.object_slots + 1].mean(),
        "presence_mean": state["presence"].float().mean(),
        "visibility_mean": state["visibility"].float().mean(),
    }
    parts.update(owner_parts)
    parts.update(identity_parts)
    parts.update(state_parts)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v50 Object State objective is non-finite")
    return total, parts


def object_state_distance(prediction, target):
    weight = target["presence"].detach().float().clamp_min(0.05)
    dynamic = 1.0 - F.cosine_similarity(
        prediction["dynamic"].float(), target["dynamic"].detach().float(), dim=-1, eps=1e-6
    )
    prediction_center = prediction["center"][:, :, None] - prediction["center"][:, None]
    target_center = target["center"][:, :, None] - target["center"][:, None]
    pair_weight = weight[:, :, None] * weight[:, None]
    center = weighted_mean(
        F.smooth_l1_loss(prediction_center.float(), target_center.detach().float(), reduction="none").mean(-1),
        pair_weight,
    )
    prediction_scale = prediction["log_scale"][:, :, None] - prediction["log_scale"][:, None]
    target_scale = target["log_scale"][:, :, None] - target["log_scale"][:, None]
    scale = weighted_mean(
        F.smooth_l1_loss(prediction_scale.float(), target_scale.detach().float(), reduction="none"),
        pair_weight,
    )
    shape = weighted_mean(
        F.smooth_l1_loss(prediction["support_shape"].float(), target["support_shape"].detach().float(), reduction="none").mean(-1),
        weight,
    )
    lifecycle = weighted_mean(
        (prediction["presence"] - target["presence"].detach()).abs()
        + (prediction["visibility"] - target["visibility"].detach()).abs(),
        weight,
    )
    parts = {"dynamic": weighted_mean(dynamic, weight), "relative_center": center, "relative_scale": scale, "support_shape": shape, "lifecycle": lifecycle}
    return parts["dynamic"] + 0.5 * center + 0.25 * scale + 0.25 * shape + 0.25 * lifecycle, parts


def effect_regularization(effect):
    gathered = gather_batch_with_grad(effect.float()).flatten(1)
    variance = F.relu(0.05 - gathered.std(dim=0, unbiased=False)).mean()
    centered = gathered - gathered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().mean()


def latent_effect_objective(model, output):
    correct, correct_parts = object_state_distance(output["effect_prediction"], output["effect_target"])
    zero, _ = object_state_distance(output["zero_prediction"], output["effect_target"])
    shuffled, _ = object_state_distance(output["shuffled_prediction"], output["effect_target"])
    intervention = F.relu(model.config.intervention_margin + correct - zero)
    intervention = intervention + F.relu(model.config.intervention_margin + correct - shuffled)
    variance, covariance = effect_regularization(output["effect"])
    total = correct + intervention + 0.05 * variance + 0.01 * covariance
    parts = {
        "loss": total,
        "loss_latent_effect": total,
        "effect_correct_distance": correct,
        "effect_zero_distance": zero,
        "effect_shuffled_distance": shuffled,
        "effect_gain_over_zero": (zero - correct) / zero.detach().clamp_min(1e-6),
        "effect_gain_over_shuffled": (shuffled - correct) / shuffled.detach().clamp_min(1e-6),
        "effect_intervention": intervention,
        "effect_variance_penalty": variance,
        "effect_covariance_penalty": covariance,
        "effect_std": gather_batch_with_grad(output["effect"].float()).std(dim=0, unbiased=False).mean(),
        "effect_norm": output["effect"].float().norm(dim=-1).mean(),
    }
    parts.update({f"effect_state_{name}": value for name, value in correct_parts.items()})
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v50 latent-effect objective is non-finite")
    return total, parts
