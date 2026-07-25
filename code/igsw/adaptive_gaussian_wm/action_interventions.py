"""Counterfactual interventions for object-aligned latent actions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


HISTORY_FIELDS = (
    "history_features",
    "history_coordinates",
    "history_valid",
    "history_times",
    "history_frame_indices",
    "history_control_indices",
    "history_rgb",
    "history_rgb_valid",
)


def replace_history(
    batch: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep targets fixed while replacing every causal history field."""
    result = dict(batch)
    for name in HISTORY_FIELDS:
        if name not in batch:
            continue
        if name not in donor:
            raise ValueError(f"history donor is missing {name}")
        value = donor[name].to(batch[name].device, non_blocking=True)
        if value.shape != batch[name].shape:
            raise ValueError(f"history donor shape differs for {name}")
        result[name] = value
    return result


def component_actions(
    posterior: torch.Tensor,
    canonical_dim: int,
) -> dict[str, torch.Tensor]:
    if posterior.ndim != 4 or not 0 < canonical_dim < posterior.shape[-1]:
        raise ValueError("component actions require [B,Q,K,D] with two parts")
    canonical = posterior.clone()
    canonical[..., canonical_dim:] = 0.0
    residual = posterior.clone()
    residual[..., :canonical_dim] = 0.0
    return {
        "canonical_only": canonical,
        "residual_only": residual,
        "zero_action": torch.zeros_like(posterior),
    }


def swap_one_object_action(
    posterior: torch.Tensor,
    donor: torch.Tensor,
    object_index: int,
) -> torch.Tensor:
    if posterior.shape != donor.shape or posterior.ndim != 4:
        raise ValueError("object action swap requires aligned [B,Q,K,D] tensors")
    if not 0 <= object_index < posterior.shape[2]:
        raise ValueError("object action index is out of range")
    result = posterior.clone()
    result[:, :, object_index] = donor[:, :, object_index]
    return result


def current_object_support(output: dict) -> torch.Tensor:
    """Return normalized current object support on the DINO feature grid."""
    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    if tokens.assignment.ndim != 3 or slots.assignment.ndim != 3:
        raise ValueError("object support requires token and slot assignments")
    if tokens.assignment.shape[:2] != slots.assignment.shape[:2]:
        raise ValueError("token and object assignments do not align")
    micro_object = slots.assignment.float() * tokens.activation.float()
    support = torch.einsum(
        "bmk,bmn->bkn",
        micro_object,
        tokens.assignment.float(),
    )
    return support / support.sum(dim=1, keepdim=True).clamp_min(1e-8)


def support_to_rgb(
    support: torch.Tensor,
    grid_hw: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Interpolate feature-grid object support into the valid RGB content box."""
    if support.ndim != 3 or grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
        raise ValueError("support and feature-grid shapes are invalid")
    if valid.ndim != 4 or valid.shape[0] != support.shape[0]:
        raise ValueError("RGB validity must have shape [B,Q,H,W]")
    if not bool((grid_hw == grid_hw[:1]).all()):
        raise ValueError("feature-grid dimensions differ within a batch")
    height, width = (int(value) for value in grid_hw[0])
    if support.shape[-1] != height * width:
        raise ValueError("object support does not match the feature grid")
    content_height = valid.any(dim=-1).sum(dim=-1)
    content_width = valid.any(dim=-2).sum(dim=-1)
    if not bool((content_height == content_height[:1, :1]).all()):
        raise ValueError("RGB content heights differ within a batch")
    if not bool((content_width == content_width[:1, :1]).all()):
        raise ValueError("RGB content widths differ within a batch")
    resized = F.interpolate(
        support.reshape(support.shape[0], support.shape[1], height, width),
        size=(int(content_height[0, 0]), int(content_width[0, 0])),
        mode="bilinear",
        align_corners=True,
    )
    output = support.new_zeros(
        support.shape[0],
        support.shape[1],
        valid.shape[-2],
        valid.shape[-1],
    )
    output[..., : resized.shape[-2], : resized.shape[-1]] = resized
    return output


def state_locality(
    reference: torch.Tensor,
    intervention: torch.Tensor,
    object_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure total response and the fraction assigned to the changed slot."""
    if reference.shape != intervention.shape or reference.ndim != 4:
        raise ValueError("state locality requires aligned [B,Q,K,D] tensors")
    effect = (reference.float() - intervention.float()).square().mean(dim=-1).sqrt()
    total = effect.sum(dim=-1)
    own = effect[:, :, object_index] / total.clamp_min(1e-8)
    return effect.square().mean(dim=-1).sqrt(), own


def spatial_locality(
    reference: torch.Tensor,
    intervention: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    object_index: int,
) -> dict[str, torch.Tensor]:
    """Measure how much rendered intervention energy falls on one object support."""
    if reference.shape != intervention.shape:
        raise ValueError("spatial locality predictions must align")
    if reference.ndim not in (4, 5):
        raise ValueError("spatial locality expects feature-grid or RGB predictions")
    squared = (reference.float() - intervention.float()).square()
    effect = (
        squared.mean(dim=-1).sqrt()
        if reference.ndim == 4
        else squared.mean(dim=2).sqrt()
    )
    if support.shape[0] != effect.shape[0]:
        raise ValueError("object support batch does not match the prediction")
    if support.ndim == 3:
        selected_support = support[:, object_index, None].expand(-1, effect.shape[1], -1)
    elif support.ndim == 4:
        selected_support = support[:, object_index, None].expand(
            -1, effect.shape[1], -1, -1
        )
    else:
        raise ValueError("object support must be [B,K,N] or [B,K,H,W]")
    if selected_support.shape != effect.shape or valid.shape != effect.shape:
        raise ValueError("support, validity, and spatial effect must align")
    weight = valid.to(effect.dtype)
    dimensions = tuple(range(2, effect.ndim))
    total_effect = (effect * weight).sum(dim=dimensions)
    supported_effect = (effect * selected_support * weight).sum(dim=dimensions)
    valid_area = weight.sum(dim=dimensions).clamp_min(1.0)
    support_area = (selected_support * weight).sum(dim=dimensions)
    mass_fraction = supported_effect / total_effect.clamp_min(1e-8)
    area_fraction = support_area / valid_area
    return {
        "effect_rms": (
            effect.square().mul(weight).sum(dim=dimensions) / valid_area
        ).sqrt(),
        "mass_fraction": mass_fraction,
        "area_fraction": area_fraction,
        "mass_lift": mass_fraction / area_fraction.clamp_min(1e-8),
    }
