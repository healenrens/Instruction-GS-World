"""Frozen-feature reconstruction and temporal cross-batch slot contrast."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed_statistics import gather_batch_without_grad
from .stable_normalization import stable_unit_normalize


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _directional_contrast(
    source: torch.Tensor,
    target: torch.Tensor,
    pair_valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, transitions, slots = source.shape[:3]
    local_source = stable_unit_normalize(source.flatten(0, 2))
    local_valid = pair_valid.flatten()
    target = stable_unit_normalize(target.detach())
    target_weight = pair_valid[..., None].float()
    prototype = (target * target_weight).sum(dim=1)
    prototype = prototype / target_weight.sum(dim=1).clamp_min(1.0)
    local_target = stable_unit_normalize(prototype.flatten(0, 1))
    local_target_valid = pair_valid.any(dim=1).flatten()
    global_target = gather_batch_without_grad(local_target)
    global_target_valid = gather_batch_without_grad(local_target_valid)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    sequence = torch.arange(batch, device=source.device)[:, None, None]
    slot = torch.arange(slots, device=source.device)[None, None, :]
    local_owner = (sequence * slots + slot).expand(batch, transitions, slots)
    local_owner = local_owner.reshape(-1) + rank * batch * slots
    global_owner = torch.arange(
        len(global_target), device=source.device, dtype=local_owner.dtype
    )
    logits = local_source @ global_target.transpose(0, 1) / temperature
    valid_logits = logits.masked_fill(~global_target_valid[None], -1e4)
    losses = F.cross_entropy(valid_logits, local_owner, reduction="none")
    valid_weight = local_valid.float()
    loss = _weighted_mean(losses, valid_weight)
    prediction = valid_logits.argmax(dim=1)
    retrieved_owner = global_owner[prediction]
    accuracy = _weighted_mean(
        retrieved_owner.eq(local_owner).float(), valid_weight
    )
    return loss, accuracy


def temporal_slot_contrast(
    projected_slots: torch.Tensor,
    activity: torch.Tensor,
    active_threshold: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projected_slots.shape[:3] != activity.shape:
        raise ValueError("slot contrast activity shape differs")
    if projected_slots.shape[1] < 2:
        raise ValueError("slot contrast needs at least two frames")
    pair_valid = (
        (activity[:, :-1] >= active_threshold)
        & (activity[:, 1:] >= active_threshold)
    )
    if not bool(pair_valid.any()):
        raise ValueError("slot contrast has no temporally active positive pair")
    forward_loss, forward_accuracy = _directional_contrast(
        projected_slots[:, :-1],
        projected_slots[:, 1:],
        pair_valid,
        temperature,
    )
    backward_loss, backward_accuracy = _directional_contrast(
        projected_slots[:, 1:],
        projected_slots[:, :-1],
        pair_valid,
        temperature,
    )
    return (
        0.5 * (forward_loss + backward_loss),
        0.5 * (forward_accuracy + backward_accuracy),
    )


def slot_contrast_objective(
    model,
    patches: torch.Tensor,
    valid: torch.Tensor,
    observation_mask: torch.Tensor,
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = F.normalize(patches.float(), dim=-1, eps=1e-6)
    reconstruction = output["reconstruction"].float()
    patch_error = 1.0 - (reconstruction * target).sum(dim=-1)
    patch_valid = valid.float()
    observed_weight = patch_valid * observation_mask[:, :, None].float()
    masked_weight = patch_valid * (~observation_mask)[:, :, None].float()
    loss_reconstruction = _weighted_mean(patch_error, patch_valid)
    loss_observed = _weighted_mean(patch_error, observed_weight)
    loss_masked = _weighted_mean(patch_error, masked_weight)
    loss_contrast, retrieval_top1 = temporal_slot_contrast(
        output["contrast_slots"],
        output["activity"],
        model.config.active_slot_fraction,
        model.config.contrast_temperature,
    )
    loss = (
        loss_reconstruction
        + model.config.masked_reconstruction_weight * loss_masked
        + model.config.contrast_weight * loss_contrast
    )

    count = patch_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    frame_mean = (target * patch_valid[..., None]).sum(dim=2, keepdim=True) / count[..., None]
    frame_mean = F.normalize(frame_mean, dim=-1, eps=1e-6)
    baseline_error = 1.0 - (frame_mean * target).sum(dim=-1)
    loss_frame_mean = _weighted_mean(baseline_error, patch_valid)

    assignment = output["assignment"].float().clamp_min(1e-8)
    entropy = -(assignment * assignment.log()).sum(dim=-1)
    entropy = _weighted_mean(entropy, patch_valid) / torch.log(
        torch.tensor(float(model.config.object_slots), device=patches.device)
    )
    activity = output["activity"].float()
    active_count = (activity >= model.config.active_slot_fraction).float().sum(dim=-1).mean()
    activity_mean = activity.mean(dim=-1, keepdim=True)
    activity_cv = (
        activity.std(dim=-1, unbiased=False) / activity_mean.squeeze(-1).clamp_min(1e-6)
    ).mean()
    normalized_slots = F.normalize(output["slots"].float(), dim=-1, eps=1e-6)
    slot_rms = output["slots"].float().square().mean(dim=-1).sqrt()
    similarity = torch.einsum("btkd,btjd->btkj", normalized_slots, normalized_slots)
    slot_count = model.config.object_slots
    off_diagonal = (similarity.sum(dim=(-1, -2)) - slot_count) / (
        slot_count * (slot_count - 1)
    )
    center_motion = (output["center"][:, 1:] - output["center"][:, :-1]).float().norm(dim=-1)
    pair_activity = torch.minimum(activity[:, 1:], activity[:, :-1])
    temporal_motion = _weighted_mean(center_motion, pair_activity)
    contrast_norm = torch.linalg.vector_norm(
        output["contrast_slots"].float(), dim=-1
    )
    encoder_mass = output["encoder_assignment"].float().sum(dim=2)
    encoder_observed = observation_mask[:, :, None].expand_as(encoder_mass)
    observed_encoder_mass = encoder_mass[encoder_observed]
    low_support = (observed_encoder_mass < 1.0).float().mean()
    parts = {
        "loss_total": loss.detach(),
        "loss_reconstruction": loss_reconstruction.detach(),
        "loss_observed_reconstruction": loss_observed.detach(),
        "loss_masked_reconstruction": loss_masked.detach(),
        "loss_temporal_slot_contrast": loss_contrast.detach(),
        "slot_retrieval_top1": retrieval_top1.detach(),
        "diagnostic_frame_mean_error": loss_frame_mean.detach(),
        "diagnostic_reconstruction_gain_over_frame_mean": (
            loss_frame_mean - loss_reconstruction
        ).detach(),
        "slot_assignment_entropy": entropy.detach(),
        "slot_active_count": active_count.detach(),
        "slot_activity_cv": activity_cv.detach(),
        "slot_pairwise_cosine": off_diagonal.mean().detach(),
        "slot_state_rms_min": slot_rms.min().detach(),
        "slot_state_rms_max": slot_rms.max().detach(),
        "slot_temporal_center_motion": temporal_motion.detach(),
        "slot_activity_min": activity.min().detach(),
        "slot_activity_max": activity.max().detach(),
        "slot_encoder_mass_min": observed_encoder_mass.min().detach(),
        "slot_encoder_low_support_fraction": low_support.detach(),
        "slot_contrast_projection_norm_min": contrast_norm.min().detach(),
        "slot_contrast_projection_norm_mean": contrast_norm.mean().detach(),
    }
    return loss, parts
