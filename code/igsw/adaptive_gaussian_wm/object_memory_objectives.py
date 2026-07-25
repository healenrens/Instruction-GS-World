"""Scale-free geometry objectives for persistent object memory."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .jepa_losses import weighted_mean


def object_memory_geometry_loss(
    output: dict,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise observable support geometry without metric-depth claims."""
    reference = output["predicted_future_slots"].sum() * 0.0
    predicted_scale = output.get("predicted_future_relative_scale")
    target_scale = output.get("target_future_relative_scale")
    predicted_relations = output.get("predicted_future_relations")
    target_relations = output.get("target_future_relations")
    if predicted_scale is None:
        return reference, {
            "geometry_relative_scale": reference,
            "geometry_image_plane_relations": reference,
        }
    if target_scale is None or predicted_relations is None or target_relations is None:
        raise ValueError("factorized Dynamics requires target memory geometry")
    activity = output["target_future_activity"].detach()
    scale_error = F.smooth_l1_loss(
        predicted_scale.clamp_min(1e-6).log(),
        target_scale.detach().clamp_min(1e-6).log(),
        beta=0.1,
        reduction="none",
    )
    scale = weighted_mean(scale_error, activity)

    # Only image-plane displacement and scale ratio are observable here.
    relation_error = F.smooth_l1_loss(
        predicted_relations[..., :3],
        target_relations.detach()[..., :3],
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)
    pair_weight = (
        activity.unsqueeze(-1) * activity.unsqueeze(-2)
    )
    relations = weighted_mean(relation_error, pair_weight)
    predicted_existence_logits = output.get("predicted_future_existence_logits")
    if predicted_existence_logits is None:
        raise ValueError("factorized Dynamics has no existence logits")
    existence = F.binary_cross_entropy_with_logits(
        predicted_existence_logits,
        output["target_future_existence"].detach(),
    )
    teacher, teacher_parts = teacher_sidecar_loss(output, batch, reference)
    total = scale + relations + 0.5 * existence + teacher
    parts = {
        "geometry_relative_scale": scale,
        "geometry_image_plane_relations": relations,
        "geometry_existence": existence,
    }
    parts.update(teacher_parts)
    return total, parts


def _pool_dense_teacher(output: dict, batch: dict):
    disparity_targets = []
    visibility_targets = []
    object_validity = []
    confidence_means = []
    correspondence_valid = []
    states = zip(
        output["target_future_token_states"],
        output["target_future_slot_states"],
        strict=True,
    )
    for index, (tokens, slots) in enumerate(states):
        micro_object = slots.assignment.detach() * tokens.activation.detach()
        dense_object = torch.einsum(
            "bmn,bmk->bnk", tokens.assignment.detach(), micro_object
        )
        confidence = batch["teacher_future_confidence"][:, index].float()
        visibility = batch["teacher_future_visibility"][:, index].float()
        correspondence = batch["teacher_future_correspondence"][:, index]
        valid_track = (correspondence >= 0).to(confidence.dtype)
        support = dense_object * (confidence * valid_track)[..., None]
        support_mass = support.sum(dim=1)
        denominator = support_mass.clamp_min(1e-6)
        object_validity.append((support_mass > 1e-5).to(confidence.dtype))
        visibility_targets.append(
            (support * visibility[..., None]).sum(dim=1) / denominator
        )
        visible_support = support * visibility[..., None]
        disparity_targets.append(
            (
                visible_support
                * batch["teacher_future_relative_disparity"][:, index, :, None]
            ).sum(dim=1)
            / visible_support.sum(dim=1).clamp_min(1e-6)
        )
        confidence_means.append(confidence.mean())
        correspondence_valid.append(valid_track.mean())
    return (
        torch.stack(disparity_targets, dim=1),
        torch.stack(visibility_targets, dim=1),
        torch.stack(object_validity, dim=1),
        torch.stack(confidence_means).mean(),
        torch.stack(correspondence_valid).mean(),
    )


def _scale_shift_invariant_disparity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    denominator = weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prediction_mean = (prediction * weight).sum(dim=-1, keepdim=True) / denominator
    target_mean = (target * weight).sum(dim=-1, keepdim=True) / denominator
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    scale = (
        (centered_prediction * centered_target * weight).sum(
            dim=-1, keepdim=True
        )
        / (centered_prediction.square() * weight).sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
    )
    scale = scale.clamp_min(0.0)
    aligned = scale * centered_prediction + target_mean
    return weighted_mean(
        F.smooth_l1_loss(aligned, target, beta=0.05, reduction="none"),
        weight,
    )


def teacher_sidecar_loss(output: dict, batch: dict, reference: torch.Tensor):
    if "teacher_sidecar_present" not in batch:
        return reference, {
            "teacher_sidecar_enabled": reference,
            "teacher_relative_disparity": reference,
            "teacher_visibility": reference,
            "teacher_confidence_mean": reference,
            "teacher_correspondence_valid": reference,
        }
    if not bool(batch["teacher_sidecar_present"].all()):
        raise ValueError("teacher sidecar presence differs within the batch")
    predicted_disparity = output.get("predicted_future_relative_disparity")
    predicted_visibility_logits = output.get("predicted_future_visibility_logits")
    if predicted_disparity is None or predicted_visibility_logits is None:
        raise ValueError("teacher sidecar requires factorized Dynamics outputs")
    target_disparity, target_visibility, object_valid, confidence, correspondence = (
        _pool_dense_teacher(output, batch)
    )
    weight = (target_visibility * object_valid).detach()
    disparity = _scale_shift_invariant_disparity(
        predicted_disparity,
        target_disparity.detach(),
        weight,
    )
    visibility = weighted_mean(
        F.binary_cross_entropy_with_logits(
            predicted_visibility_logits,
            target_visibility.detach(),
            reduction="none",
        ),
        (output["target_future_existence"] * object_valid).detach(),
    )
    return disparity + visibility, {
        "teacher_sidecar_enabled": reference.new_ones(()),
        "teacher_relative_disparity": disparity,
        "teacher_visibility": visibility,
        "teacher_confidence_mean": confidence,
        "teacher_correspondence_valid": correspondence,
    }
