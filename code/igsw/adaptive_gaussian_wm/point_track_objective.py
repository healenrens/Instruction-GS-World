"""Component-balanced Object State and posterior latent-effect objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .trajectory_lifecycle import (
    LIFECYCLE_ABSENT,
    LIFECYCLE_OCCLUDED,
    LIFECYCLE_VISIBLE,
)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float().clamp(0.0, 1.0)
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())


def component_set_supervision(model, match, evidence):
    prediction = match.sampled_student_assignment.float().clamp_min(1e-6)
    target = match.target_student_owner[:, None].expand_as(prediction)
    visible = evidence.visibility.float()
    object_count = model.config.object_slots
    object_prediction = prediction[..., :object_count]
    object_target = target[..., :object_count]
    component_weight = match.component_track_weight[:, None] * visible[..., None]
    component_weight = component_weight * match.component_valid[:, None, None].float()
    object_nll = -(object_target * object_prediction.log())
    object_loss = (object_nll * component_weight).sum() / component_weight.sum().clamp_min(1.0)

    nuisance_target = target[..., object_count:]
    nuisance_prediction = prediction[..., object_count:]
    nuisance_weight = nuisance_target.sum(dim=-1) * visible
    nuisance_nll = -(nuisance_target * nuisance_prediction.log()).sum(dim=-1)
    nuisance_loss = weighted_mean(nuisance_nll, nuisance_weight)
    owner_loss = object_loss + nuisance_loss

    coverage_weight = component_weight
    component_coverage = (object_prediction * coverage_weight).sum(dim=(1, 2))
    component_coverage = component_coverage / coverage_weight.sum(dim=(1, 2)).clamp_min(1.0)
    valid_component = match.component_valid.float()
    coverage_loss = weighted_mean(1.0 - component_coverage, valid_component)
    usage = component_coverage * valid_component
    usage_share = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    uniform = valid_component / valid_component.sum(dim=-1, keepdim=True).clamp_min(1.0)
    balance_loss = weighted_mean((usage_share - uniform).square(), valid_component)
    component_set = coverage_loss + balance_loss

    shuffled_target = target.roll(1, dims=2)
    shuffled_component_weight = component_weight.roll(1, dims=2)
    shuffled_object = (
        -(shuffled_target[..., :object_count] * object_prediction.log())
        * shuffled_component_weight
    ).sum() / shuffled_component_weight.sum().clamp_min(1.0)
    shuffled_nuisance_target = shuffled_target[..., object_count:]
    shuffled_nuisance_weight = shuffled_nuisance_target.sum(dim=-1) * visible
    shuffled_nuisance = weighted_mean(
        -(shuffled_nuisance_target * nuisance_prediction.log()).sum(dim=-1),
        shuffled_nuisance_weight,
    )
    shuffled = shuffled_object + shuffled_nuisance
    accuracy = weighted_mean(
        (prediction.argmax(dim=-1) == target.argmax(dim=-1)).float(), visible
    )
    return owner_loss, component_set, {
        "loss_teacher_owner": owner_loss,
        "loss_teacher_object_owner": object_loss,
        "loss_teacher_nuisance_owner": nuisance_loss,
        "loss_component_coverage": coverage_loss,
        "loss_component_set_balance": balance_loss,
        "component_set_mean_coverage": weighted_mean(component_coverage, valid_component),
        "teacher_owner_top1": accuracy,
        "teacher_owner_shuffled_loss": shuffled,
        "teacher_owner_gain_over_shuffled": (
            shuffled - owner_loss
        ) / shuffled.detach().clamp_min(1e-6),
    }


def identity_supervision(model, state, match):
    prediction = F.normalize(
        model.identity_readout(state["identity"].float()), dim=-1, eps=1e-6
    )
    target = match.identity.detach()[:, None]
    distance = 1.0 - (prediction * target).sum(dim=-1)
    visible = match.visibility.detach() * match.component_valid[:, None].float()
    identity = weighted_mean(distance, visible)
    temporal_weight = torch.minimum(visible[:, 1:], visible[:, :-1])
    temporal_distance = 1.0 - F.cosine_similarity(
        state["identity"][:, 1:].float(),
        state["identity"][:, :-1].float(),
        dim=-1,
        eps=1e-6,
    )
    identity_temporal = weighted_mean(temporal_distance, temporal_weight)
    occluded_before = (
        match.lifecycle_state == LIFECYCLE_OCCLUDED
    ).float().cumsum(dim=1) > 0
    reappearance_weight = visible * occluded_before.float()
    reappearance = weighted_mean(distance, reappearance_weight)
    shuffled_target = target.roll(1, dims=2)
    shuffled = weighted_mean(
        1.0 - (prediction * shuffled_target).sum(dim=-1), visible
    )
    return identity, identity_temporal, reappearance, {
        "loss_teacher_identity": identity,
        "loss_identity_temporal_only": identity_temporal,
        "loss_teacher_reappearance": reappearance,
        "teacher_identity_shuffled_distance": shuffled,
        "teacher_identity_gain_over_shuffled": (
            shuffled - identity
        ) / shuffled.detach().clamp_min(1e-6),
        "teacher_reappearance_weight": reappearance_weight.sum(),
    }


def dynamic_geometry_lifecycle(model, state, match):
    batch, frames, slots = state["dynamic"].shape[:3]
    horizon_count = len(model.config.dynamic_horizons)
    motion_prediction = model.motion_readout(state["dynamic"].float()).reshape(
        batch, frames, slots, horizon_count, 2
    )
    motion_weight = match.relative_motion_valid.detach().float()
    motion = weighted_mean(
        F.smooth_l1_loss(
            motion_prediction, match.relative_motion.detach().float(), reduction="none"
        ).mean(dim=-1),
        motion_weight,
    )
    residual_prediction = model.geometry_residual_readout(
        state["dynamic"].float()
    ).reshape(
        batch,
        frames,
        slots,
        horizon_count,
        model.config.dynamic_geometry_dim,
    )
    residual_weight = match.geometry_residual_valid.detach().float()
    geometry_residual = weighted_mean(
        F.smooth_l1_loss(
            residual_prediction,
            match.geometry_residual.detach().float(),
            reduction="none",
        ).mean(dim=-1),
        residual_weight,
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
            state["support_shape"].float(),
            match.support_shape.detach().float(),
            reduction="none",
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
    valid_lifecycle = match.component_valid[:, None].expand_as(match.lifecycle_state)
    lifecycle_total = valid_lifecycle.float().sum().clamp_min(1.0)
    return motion + geometry_residual, center + scale + shape, visibility + presence, {
        "loss_teacher_multihorizon_motion": motion,
        "loss_teacher_multihorizon_geometry_residual": geometry_residual,
        "loss_teacher_relative_center": center,
        "loss_teacher_relative_scale": scale,
        "loss_teacher_support_shape": shape,
        "loss_teacher_visibility": visibility,
        "loss_teacher_presence": presence,
        "teacher_visible_fraction": (
            (match.lifecycle_state == LIFECYCLE_VISIBLE) & valid_lifecycle
        ).float().sum() / lifecycle_total,
        "teacher_occluded_fraction": (
            (match.lifecycle_state == LIFECYCLE_OCCLUDED) & valid_lifecycle
        ).float().sum() / lifecycle_total,
        "teacher_absent_fraction": (
            (match.lifecycle_state == LIFECYCLE_ABSENT) & valid_lifecycle
        ).float().sum() / lifecycle_total,
        "teacher_unknown_fraction": (
            (~match.lifecycle_known) & valid_lifecycle
        ).float().sum() / lifecycle_total,
    }


def object_state_objective(model, patches, valid, evidence, teacher, match, output):
    target = F.normalize(patches.float(), dim=-1, eps=1e-6)
    reconstruction = weighted_mean(
        1.0 - (output["reconstruction"].float() * target).sum(dim=-1), valid.float()
    )
    owner, component_set, owner_parts = component_set_supervision(model, match, evidence)
    identity, identity_temporal, reappearance, identity_parts = identity_supervision(
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
    total = (
        model.config.reconstruction_weight * reconstruction
        + model.config.track_assignment_weight * owner
        + model.config.component_set_weight * component_set
        + model.config.identity_weight * identity
        + model.config.identity_temporal_weight * identity_temporal
        + model.config.reappearance_weight * reappearance
        + model.config.motion_weight * motion
        + model.config.lifecycle_weight * lifecycle
        + model.config.geometry_weight * geometry
        + model.config.diversity_weight * diversity
    )
    parts = {
        "loss": total,
        "loss_object_state": total,
        "loss_reconstruction_auxiliary": reconstruction,
        "loss_identity_diversity": diversity,
        "train_teacher_component_count": teacher.component_valid.float().sum(dim=-1).mean(),
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
        raise RuntimeError("v51 Object State objective is non-finite")
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
        raise RuntimeError("v51 latent-effect objective is non-finite")
    return total, parts
