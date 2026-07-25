"""Shared object-plus-detail Gaussian readout construction."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .decoder import GaussianReadoutState
from .gpstoken import GPSTokenState
from .object_slots import ObjectSlotState
from .rgb_supervision import current_micro_rgb


@dataclass
class GaussianReadoutContext:
    micro_rgb: torch.Tensor | None
    current_object_rgb: torch.Tensor | None


def residual_future_features(
    future_render: torch.Tensor,
    current_render: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Preserve observed detail while predicting the Gaussian feature change."""
    if future_render.shape != current_render.shape:
        raise ValueError("future and current feature renders must align")
    current = batch["history_features"][:, -1:]
    if current.shape[2:] != future_render.shape[2:]:
        raise ValueError("current features and future render must align")
    return current.expand_as(future_render) + future_render - current_render


def object_rgb_from_micro(
    micro_rgb: torch.Tensor,
    assignment: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    """Pool local RGB into the current object partition."""
    if micro_rgb.shape[:2] != assignment.shape[:2]:
        raise ValueError("micro RGB and object assignment do not align")
    if activation.shape != (*assignment.shape[:2], 1):
        raise ValueError("micro activation must have shape [B,M,1]")
    weight = assignment * activation
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("bmk,bmc->bkc", weight, micro_rgb)


def decode_gaussian_readout(
    model,
    batch: dict[str, torch.Tensor],
    current_tokens: GPSTokenState,
    current_slots: ObjectSlotState,
    predicted_slots: torch.Tensor,
    predicted_centers: torch.Tensor | None = None,
    micro_rgb: torch.Tensor | None = None,
) -> tuple[GaussianReadoutState, GaussianReadoutContext]:
    """Combine predicted object state with current local residual detail."""
    predicted_features = model.object_aggregator.decode_feature(predicted_slots)
    predicted_rgb_logits = None
    current_object_rgb = None
    if model.config.rgb_supervision:
        if micro_rgb is None:
            micro_rgb = current_micro_rgb(current_tokens, batch)
        current_object_rgb = object_rgb_from_micro(
            micro_rgb,
            current_slots.assignment,
            current_tokens.activation,
        )
        predicted_rgb_logits = model.object_aggregator.decode_rgb_logits(
            predicted_slots
        )
    readout = model.gaussian_readout(
        predicted_slots,
        current_tokens,
        current_slots.assignment,
        predicted_features=predicted_features,
        current_object_features=current_slots.feature,
        predicted_centers=predicted_centers,
        current_object_centers=current_slots.center,
        current_rgb=micro_rgb,
        predicted_rgb_logits=predicted_rgb_logits,
        current_object_rgb=current_object_rgb,
    )
    return readout, GaussianReadoutContext(
        micro_rgb=micro_rgb,
        current_object_rgb=current_object_rgb,
    )
