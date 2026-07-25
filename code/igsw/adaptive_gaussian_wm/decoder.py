"""Auxiliary 2D Gaussian readout from predicted object latents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .gaussian_math import mahalanobis_squared_from_precision, precision_2d
from .gpstoken import GPSTokenState


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float(), eps=1e-4)


@dataclass
class GaussianReadoutState:
    feature: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    depth_order: torch.Tensor
    opacity: torch.Tensor
    activation: torch.Tensor
    rgb: torch.Tensor | None = None


class GaussianReadout(nn.Module):
    """Decode auxiliary micro-Gaussian attributes without feeding them back."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        hidden = config.token_dim
        self.feature_dim = config.feature_dim
        self.covariance_floor = config.covariance_floor
        self.current_input = nn.Linear(config.token_dim, hidden)
        self.object_input = nn.Linear(config.object_dim, hidden)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * 2, hidden),
        )
        self.attribute_head = nn.Linear(hidden, 8)
        nn.init.zeros_(self.attribute_head.weight)
        nn.init.zeros_(self.attribute_head.bias)

    def forward(
        self,
        predicted_slots: torch.Tensor,
        current_tokens: GPSTokenState,
        current_object_assignment: torch.Tensor,
        *,
        predicted_features: torch.Tensor,
        current_object_features: torch.Tensor,
        predicted_centers: torch.Tensor | None = None,
        current_object_centers: torch.Tensor | None = None,
        current_rgb: torch.Tensor | None = None,
        predicted_rgb_logits: torch.Tensor | None = None,
        current_object_rgb: torch.Tensor | None = None,
    ) -> GaussianReadoutState:
        if predicted_slots.ndim != 4:
            raise ValueError("predicted_slots must have shape [B,Q,K,D]")
        expected_features = (*predicted_slots.shape[:3], self.feature_dim)
        if predicted_features.shape != expected_features:
            raise ValueError(
                f"predicted_features must have shape {expected_features}"
            )
        if current_object_features.shape != (
            predicted_slots.shape[0],
            predicted_slots.shape[2],
            self.feature_dim,
        ):
            raise ValueError(
                "current_object_features must have shape [B,K,C]"
            )
        object_per_micro = torch.einsum(
            "bmk,bqkd->bqmd",
            current_object_assignment,
            predicted_slots,
        )
        current = self.current_input(current_tokens.latent)[:, None].expand(
            -1,
            predicted_slots.shape[1],
            -1,
            -1,
        )
        hidden = self.fusion(
            torch.cat((current, self.object_input(object_per_micro)), dim=-1)
        )
        predicted_feature_per_micro = torch.einsum(
            "bmk,bqkc->bqmc",
            current_object_assignment,
            predicted_features,
        )
        current_feature_per_micro = torch.einsum(
            "bmk,bkc->bmc",
            current_object_assignment,
            current_object_features,
        )
        feature = (
            current_tokens.decoded_features[:, None]
            + predicted_feature_per_micro
            - current_feature_per_micro[:, None].detach()
        )
        raw = self.attribute_head(hidden)

        center_transport = raw.new_zeros(*raw.shape[:-1], 2)
        if predicted_centers is not None:
            expected = (*predicted_slots.shape[:3], 2)
            if predicted_centers.shape != expected:
                raise ValueError(
                    f"predicted_centers must have shape {expected}"
                )
            if current_object_centers is None:
                raise ValueError(
                    "current_object_centers are required with predicted_centers"
                )
            if current_object_centers.shape != (
                predicted_slots.shape[0],
                predicted_slots.shape[2],
                2,
            ):
                raise ValueError(
                    "current_object_centers must have shape [B,K,2]"
                )
            object_delta = (
                predicted_centers - current_object_centers[:, None]
            )
            center_transport = torch.einsum(
                "bmk,bqkd->bqmd",
                current_object_assignment,
                object_delta,
            )
        center = (
            current_tokens.center[:, None]
            + center_transport
            + 0.25 * torch.tanh(raw[..., :2])
        )
        diagonal_scale = torch.exp(0.5 * torch.tanh(raw[..., 2:4]))
        shear = 0.25 * torch.tanh(raw[..., 4])
        transform = raw.new_zeros(*raw.shape[:-1], 2, 2)
        transform[..., 0, 0] = diagonal_scale[..., 0]
        transform[..., 1, 0] = shear
        transform[..., 1, 1] = diagonal_scale[..., 1]
        base_cholesky = torch.linalg.cholesky(
            current_tokens.covariance.float()
        )[:, None]
        cholesky = transform.float() @ base_cholesky
        covariance = cholesky @ cholesky.transpose(-1, -2)
        identity = torch.eye(
            2,
            device=covariance.device,
            dtype=covariance.dtype,
        )
        covariance = covariance + self.covariance_floor * identity
        depth_order = current_tokens.depth_order[:, None] + raw[..., 5:6]
        opacity_logit = _stable_logit(current_tokens.opacity)[:, None]
        activation_logit = _stable_logit(current_tokens.activation)[:, None]
        opacity = torch.sigmoid(opacity_logit + raw[..., 6:7].float())
        activation = torch.sigmoid(
            activation_logit + raw[..., 7:8].float()
        )
        rgb = None
        rgb_inputs = (
            current_rgb,
            predicted_rgb_logits,
            current_object_rgb,
        )
        if any(value is not None for value in rgb_inputs):
            if not all(value is not None for value in rgb_inputs):
                raise ValueError(
                    "RGB readout requires current, predicted, and object RGB"
                )
            if current_rgb.shape != (
                predicted_slots.shape[0],
                current_tokens.latent.shape[1],
                3,
            ):
                raise ValueError("current_rgb must have shape [B,M,3]")
            expected_object_rgb = (*predicted_slots.shape[:3], 3)
            if predicted_rgb_logits.shape != expected_object_rgb:
                raise ValueError(
                    f"predicted_rgb_logits must have shape {expected_object_rgb}"
                )
            if current_object_rgb.shape != (
                predicted_slots.shape[0],
                predicted_slots.shape[2],
                3,
            ):
                raise ValueError("current_object_rgb must have shape [B,K,3]")
            predicted_rgb_per_micro = torch.einsum(
                "bmk,bqkc->bqmc",
                current_object_assignment,
                predicted_rgb_logits,
            )
            current_object_rgb_per_micro = torch.einsum(
                "bmk,bkc->bmc",
                current_object_assignment,
                current_object_rgb,
            )
            detail_logit = _stable_logit(current_rgb) - _stable_logit(
                current_object_rgb_per_micro
            ).detach()
            rgb = torch.sigmoid(
                detail_logit[:, None] + predicted_rgb_per_micro.float()
            )
        return GaussianReadoutState(
            feature=feature,
            center=center,
            covariance=covariance,
            depth_order=depth_order,
            opacity=opacity,
            activation=activation,
            rgb=rgb,
        )

    @staticmethod
    def splat_features(
        readout: GaussianReadoutState,
        query_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_coordinates.shape[:2] != readout.center.shape[:2]:
            raise ValueError("query_coordinates and readout must share [B,Q]")
        difference = (
            query_coordinates[:, :, None]
            - readout.center[:, :, :, None]
        ).float()
        distance = mahalanobis_squared_from_precision(
            precision_2d(readout.covariance),
            difference,
        )
        weight = torch.exp(-0.5 * distance)
        weight = weight * readout.opacity.squeeze(-1)[..., None]
        weight = weight * readout.activation.squeeze(-1)[..., None]
        coverage = weight.sum(dim=2)
        normalized = weight / coverage[:, :, None].clamp_min(1e-6)
        features = torch.einsum(
            "bqmn,bqmc->bqnc",
            normalized,
            readout.feature.float(),
        )
        return features, coverage
