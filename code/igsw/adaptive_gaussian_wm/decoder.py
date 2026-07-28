"""Auxiliary 2D Gaussian readout from predicted object latents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

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
        self.feature_residual_head = (
            nn.Linear(hidden, config.feature_dim)
            if config.gaussian_feature_residual
            else None
        )
        if self.feature_residual_head is not None:
            nn.init.zeros_(self.feature_residual_head.weight)
            nn.init.zeros_(self.feature_residual_head.bias)

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
        predicted_relative_scale: torch.Tensor | None = None,
        current_relative_scale: torch.Tensor | None = None,
        predicted_relative_disparity: torch.Tensor | None = None,
        current_relative_disparity: torch.Tensor | None = None,
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
        if self.feature_residual_head is not None:
            feature = feature + torch.tanh(
                self.feature_residual_head(hidden).float()
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
        scale_transport = raw.new_zeros(*raw.shape[:-1])
        if predicted_relative_scale is not None:
            expected = predicted_slots.shape[:3]
            if predicted_relative_scale.shape != expected:
                raise ValueError(
                    f"predicted_relative_scale must have shape {expected}"
                )
            if current_relative_scale is None or current_relative_scale.shape != (
                predicted_slots.shape[0], predicted_slots.shape[2]
            ):
                raise ValueError("current_relative_scale must have shape [B,K]")
            object_log_scale = (
                predicted_relative_scale.clamp_min(1e-6).log()
                - current_relative_scale[:, None].clamp_min(1e-6).log()
            ).clamp(-2.0, 2.0)
            scale_transport = torch.einsum(
                "bmk,bqk->bqm", current_object_assignment, object_log_scale
            )
        diagonal_scale = torch.exp(
            scale_transport[..., None]
            + 0.5 * torch.tanh(raw[..., 2:4])
        )
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
        disparity_transport = raw.new_zeros(*raw.shape[:-1])
        if predicted_relative_disparity is not None:
            expected = predicted_slots.shape[:3]
            if predicted_relative_disparity.shape != expected:
                raise ValueError(
                    "predicted_relative_disparity has an invalid shape"
                )
            if (
                current_relative_disparity is None
                or current_relative_disparity.shape
                != (predicted_slots.shape[0], predicted_slots.shape[2])
            ):
                raise ValueError(
                    "current_relative_disparity must have shape [B,K]"
                )
            object_disparity_delta = (
                predicted_relative_disparity
                - current_relative_disparity[:, None]
            )
            disparity_transport = torch.einsum(
                "bmk,bqk->bqm",
                current_object_assignment,
                object_disparity_delta,
            )
        depth_order = (
            current_tokens.depth_order[:, None]
            + disparity_transport[..., None]
            + raw[..., 5:6]
        )
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
        relative_order = torch.softmax(
            readout.depth_order.squeeze(-1).float(), dim=2
        ).to(weight.dtype)
        weight = weight * relative_order[..., None] * readout.center.shape[2]
        coverage = weight.sum(dim=2)
        normalized = weight / coverage[:, :, None].clamp_min(1e-6)
        features = torch.einsum(
            "bqmn,bqmc->bqnc",
            normalized,
            readout.feature.float(),
        )
        return features, coverage
