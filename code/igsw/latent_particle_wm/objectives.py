"""Training objectives for the latent particle probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import ParticleWorldModel


def mover_mask(target: torch.Tensor, valid: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    return (target[..., :2].norm(dim=-1) > threshold) & valid


def global_effect_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Canonical scene-level effect statistics derived only from future motion."""
    motion = target[..., :3]
    weight = valid.float()[..., None]
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (motion * weight).sum(dim=1) / count
    centered = (motion - mean[:, None]) * weight
    std = torch.sqrt(centered.square().sum(dim=1) / count + 1e-6)
    speed = motion[..., :2].norm(dim=-1)
    mean_speed = (speed * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
    moving_fraction = ((speed > 0.01) & valid).float().sum(dim=1)
    moving_fraction = moving_fraction / valid.float().sum(dim=1).clamp_min(1.0)
    return torch.cat((mean, std, mean_speed[:, None], moving_fraction[:, None]), dim=-1)


def oriented_effect_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    effect = global_effect_target(target, valid)
    scale = effect.new_tensor([0.05, 0.05, 0.10, 0.05, 0.05, 0.10, 0.05, 1.0])
    return (effect / scale).clamp(-4.0, 4.0)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    combined = mask.float()
    if weight is not None:
        combined = combined * weight
    while combined.ndim < value.ndim:
        combined = combined[..., None]
    return (value * combined).sum() / combined.sum().clamp_min(1.0)


def particle_world_model_loss(
    model: ParticleWorldModel,
    batch: dict,
    output: dict,
    kl_weight: float,
    free_bits: float,
    move_weight: float,
    appearance_weight: float,
    visibility_weight: float,
    effect_weight: float,
    usage_weight: float = 0.0,
    usage_margin: float = 0.005,
    alignment_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = batch["target"]
    valid = batch["valid"]
    visible = batch["visible"]
    motion_valid = batch["motion_valid"]
    movers = mover_mask(target, motion_valid)
    token_weight = 1.0 + move_weight * movers.float()

    motion_per = F.smooth_l1_loss(
        output["prediction"][..., :3],
        target[..., :3],
        beta=0.01,
        reduction="none",
    ).mean(dim=-1)
    motion_loss = _masked_mean(motion_per, motion_valid, token_weight)

    appearance_per = F.smooth_l1_loss(
        output["prediction"][..., 3:],
        target[..., 3:],
        beta=0.05,
        reduction="none",
    ).mean(dim=-1)
    appearance_loss = _masked_mean(appearance_per, visible)
    visibility_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["visibility_logits"],
            visible.float(),
            reduction="none",
        ),
        valid,
    )
    kl, kl_parts = model.kl_loss(output, valid, free_bits)
    effect_loss = output["prediction"].sum() * 0.0
    if "local_effect_prediction" in output:
        local_effect_per = F.smooth_l1_loss(
            output["local_effect_prediction"],
            target[..., :3],
            beta=0.02,
            reduction="none",
        ).mean(dim=-1)
        effect_loss = effect_loss + _masked_mean(local_effect_per, motion_valid)
    if "effect_prediction" in output:
        effect_loss = effect_loss + F.smooth_l1_loss(
            output["effect_prediction"],
            global_effect_target(target, motion_valid),
            beta=0.02,
        )
    usage_loss = output["prediction"].sum() * 0.0
    if usage_weight > 0.0 and ("local_z" in output or "global_z" in output) and len(batch["state"]) > 1:
        order = torch.roll(torch.arange(len(batch["state"]), device=target.device), shifts=1)
        shuffled_global = output["global_z"][order] if "global_z" in output else None
        shuffled_local = output["local_z"][order] if "local_z" in output else None
        shuffled_prediction, _ = model.decode_latents(
            output,
            shuffled_local,
            shuffled_global,
        )
        correct_error = (output["prediction"][..., :3] - target[..., :3]).norm(dim=-1)
        shuffled_error = (shuffled_prediction[..., :3] - target[..., :3]).norm(dim=-1)
        usage_per = F.relu(usage_margin + correct_error - shuffled_error)
        usage_loss = _masked_mean(usage_per, motion_valid)
    alignment_loss = output["prediction"].sum() * 0.0
    if alignment_weight > 0.0 and "global_q_mu" in output:
        oriented = oriented_effect_target(target, motion_valid)
        dimensions = min(oriented.shape[-1], output["global_q_mu"].shape[-1])
        alignment_loss = F.smooth_l1_loss(
            output["global_q_mu"][..., :dimensions],
            oriented[..., :dimensions],
            beta=0.1,
        )
    total = (
        motion_loss
        + appearance_weight * appearance_loss
        + visibility_weight * visibility_loss
        + kl_weight * kl
        + effect_weight * effect_loss
        + usage_weight * usage_loss
        + alignment_weight * alignment_loss
    )
    parts = {
        "loss": total.detach(),
        "motion": motion_loss.detach(),
        "appearance": appearance_loss.detach(),
        "visibility": visibility_loss.detach(),
        "kl": kl.detach(),
        "effect": effect_loss.detach(),
        "usage": usage_loss.detach(),
        "alignment": alignment_loss.detach(),
        **kl_parts,
    }
    return total, parts
