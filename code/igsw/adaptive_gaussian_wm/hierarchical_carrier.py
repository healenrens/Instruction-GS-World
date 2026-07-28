"""Hierarchical current-detail carrier with exact-zero object deltas."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float(), eps=1e-4)


def _child_seeds(count: int) -> torch.Tensor:
    if count == 1:
        return torch.zeros(1, 2)
    angle = 2.0 * math.pi * torch.arange(count, dtype=torch.float32) / count
    return 0.7 * torch.stack((angle.cos(), angle.sin()), dim=-1)


def _transform(
    diagonal: torch.Tensor,
    shear: torch.Tensor,
) -> torch.Tensor:
    result = diagonal.new_zeros(*diagonal.shape[:-1], 2, 2)
    result[..., 0, 0] = diagonal[..., 0]
    result[..., 1, 0] = shear
    result[..., 1, 1] = diagonal[..., 1]
    return result


class HierarchicalGaussianCarrier(nn.Module):
    """Expand each GPSToken anchor into local children and transport by objects."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.token_dim
        self.children = config.gaussian_children
        self.feature_dim = config.feature_dim
        self.covariance_floor = config.covariance_floor
        self.child_identity = nn.Parameter(
            torch.randn(self.children, hidden) / hidden**0.5
        )
        self.register_buffer("child_seeds", _child_seeds(self.children))
        self.carrier_norm = nn.LayerNorm(hidden)
        self.carrier_body = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden),
        )
        self.carrier_attribute = nn.Linear(hidden, 8)
        self.carrier_feature = nn.Linear(hidden, config.feature_dim)
        self.delta_token = nn.Linear(config.token_dim, hidden)
        self.delta_object = nn.Linear(config.object_dim, hidden)
        self.delta_norm = nn.LayerNorm(hidden)
        self.delta_body = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden),
        )
        self.delta_attribute = nn.Linear(hidden, 8)
        self.delta_feature = nn.Linear(hidden, config.feature_dim)
        for head in (
            self.carrier_attribute,
            self.carrier_feature,
            self.delta_attribute,
            self.delta_feature,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _carrier_hidden(self, token_latent: torch.Tensor) -> torch.Tensor:
        hidden = token_latent[:, :, None] + self.child_identity[None, None]
        return self.carrier_body(self.carrier_norm(hidden))

    def _carrier_tensors(
        self,
        tokens,
        background_feature: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        hidden = self._carrier_hidden(tokens.latent)
        raw = self.carrier_attribute(hidden)
        parent_cholesky = torch.linalg.cholesky(tokens.covariance.float())
        local = self.child_seeds[None, None] + 0.35 * torch.tanh(raw[..., :2])
        offset = torch.einsum("bmij,bmlj->bmli", parent_cholesky, local.float())
        center = tokens.center[:, :, None].float() + offset

        base_scale = self.children**-0.5
        diagonal = base_scale * torch.exp(0.35 * torch.tanh(raw[..., 2:4]))
        transform = _transform(diagonal, 0.2 * torch.tanh(raw[..., 4]))
        cholesky = parent_cholesky[:, :, None] @ transform.float()
        covariance = cholesky @ cholesky.transpose(-1, -2)
        identity = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.covariance_floor * identity

        feature = tokens.decoded_features[:, :, None].float() + 0.25 * torch.tanh(
            self.carrier_feature(hidden).float()
        )
        depth = tokens.depth_order[:, :, None].float() + raw[..., 5:6].float()
        opacity = torch.sigmoid(
            _stable_logit(tokens.opacity)[:, :, None] + raw[..., 6:7].float()
        )
        activation = torch.sigmoid(
            _stable_logit(tokens.activation / self.children)[:, :, None]
            + raw[..., 7:8].float()
        )
        if background_feature is None:
            occupancy = tokens.occupancy.float()
            background_feature = (
                tokens.decoded_features.float() * occupancy
            ).sum(dim=1) / occupancy.sum(dim=1).clamp_min(1e-6)
        expected_background = (tokens.latent.shape[0], self.feature_dim)
        if background_feature.shape != expected_background:
            raise ValueError(
                f"background_feature must have shape {expected_background}"
            )
        return {
            "feature": feature,
            "center": center,
            "covariance": covariance,
            "depth_order": depth,
            "opacity": opacity,
            "activation": activation,
            "background_feature": background_feature.float(),
        }

    def _object_code(
        self,
        object_state: torch.Tensor,
        token_latent: torch.Tensor,
    ) -> torch.Tensor:
        token = self.delta_token(token_latent)[:, None, :, None]
        child = self.child_identity[None, None, None]
        hidden = token + self.delta_object(object_state)[..., None, :] + child
        return self.delta_body(self.delta_norm(hidden))

    @staticmethod
    def _object_transport(
        assignment: torch.Tensor,
        predicted: torch.Tensor | None,
        current: torch.Tensor | None,
        logarithmic: bool = False,
    ) -> torch.Tensor | None:
        if predicted is None:
            return None
        if current is None:
            raise ValueError("current object geometry is required for transport")
        if logarithmic:
            difference = (
                predicted.clamp_min(1e-6).log()
                - current[:, None].clamp_min(1e-6).log()
            ).clamp(-2.0, 2.0)
        else:
            difference = predicted - current[:, None]
        if difference.ndim == 3:
            return torch.einsum("bmk,bqk->bqm", assignment, difference)
        if difference.ndim == 4:
            return torch.einsum("bmk,bqkd->bqmd", assignment, difference)
        raise ValueError("object transport supports scalar or vector geometry")

    @staticmethod
    def _flatten_children(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(2, 3)

    def current_tensors(
        self,
        tokens,
        background_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = self._carrier_tensors(tokens, background_feature)
        result = {
            name: self._flatten_children(value[:, None])
            for name, value in values.items()
            if name != "background_feature"
        }
        result["background_feature"] = values["background_feature"][:, None]
        return result

    def forward_tensors(
        self,
        predicted_slots: torch.Tensor,
        current_tokens,
        assignment: torch.Tensor,
        current_slots: torch.Tensor,
        *,
        background_feature: torch.Tensor | None,
        predicted_centers: torch.Tensor | None,
        current_centers: torch.Tensor | None,
        predicted_scale: torch.Tensor | None,
        current_scale: torch.Tensor | None,
        predicted_disparity: torch.Tensor | None,
        current_disparity: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, queries, objects, object_dim = predicted_slots.shape
        micro_tokens = current_tokens.latent.shape[1]
        if current_slots.shape != (batch, objects, object_dim):
            raise ValueError("current_slots must have shape [B,K,D]")
        if assignment.shape != (batch, micro_tokens, objects):
            raise ValueError("assignment must have shape [B,M,K]")
        geometry = (
            (predicted_centers, (batch, queries, objects, 2)),
            (current_centers, (batch, objects, 2)),
            (predicted_scale, (batch, queries, objects)),
            (current_scale, (batch, objects)),
            (predicted_disparity, (batch, queries, objects)),
            (current_disparity, (batch, objects)),
        )
        for value, expected in geometry:
            if value is not None and value.shape != expected:
                raise ValueError(f"object geometry must have shape {expected}")
        for predicted, current in (
            (predicted_centers, current_centers),
            (predicted_scale, current_scale),
            (predicted_disparity, current_disparity),
        ):
            if predicted is not None and current is None:
                raise ValueError("predicted geometry requires a current reference")
        base = self._carrier_tensors(current_tokens, background_feature)
        predicted_object = torch.einsum(
            "bmk,bqkd->bqmd", assignment, predicted_slots
        )
        current_object = torch.einsum("bmk,bkd->bmd", assignment, current_slots)
        predicted_code = self._object_code(predicted_object, current_tokens.latent)
        current_code = self._object_code(
            current_object[:, None], current_tokens.latent
        )
        raw_delta = self.delta_attribute(predicted_code) - self.delta_attribute(
            current_code
        )
        feature_delta = self.delta_feature(predicted_code) - self.delta_feature(
            current_code
        )

        center_transport = self._object_transport(
            assignment, predicted_centers, current_centers
        )
        if center_transport is None:
            center_transport = raw_delta.new_zeros(*raw_delta.shape[:3], 2)
        child_cholesky = torch.linalg.cholesky(base["covariance"].float())
        local_delta = torch.einsum(
            "bmlij,bqmlj->bqmli",
            child_cholesky,
            0.25 * torch.tanh(raw_delta[..., :2]).float(),
        )
        center = (
            base["center"][:, None]
            + center_transport[..., None, :].float()
            + local_delta
        )

        scale_transport = self._object_transport(
            assignment, predicted_scale, current_scale, logarithmic=True
        )
        if scale_transport is None:
            scale_transport = raw_delta.new_zeros(raw_delta.shape[:3])
        diagonal = torch.exp(
            scale_transport[..., None, None]
            + 0.25 * torch.tanh(raw_delta[..., 2:4])
        )
        transform = _transform(diagonal, 0.1 * torch.tanh(raw_delta[..., 4]))
        cholesky = transform.float() @ child_cholesky[:, None]
        covariance = cholesky @ cholesky.transpose(-1, -2)

        disparity = self._object_transport(
            assignment, predicted_disparity, current_disparity
        )
        if disparity is None:
            disparity = raw_delta.new_zeros(raw_delta.shape[:3])
        feature = base["feature"][:, None] + 0.25 * torch.tanh(
            feature_delta.float()
        )
        depth = (
            base["depth_order"][:, None]
            + disparity[..., None, None].float()
            + raw_delta[..., 5:6].float()
        )
        opacity = torch.sigmoid(
            _stable_logit(base["opacity"])[:, None] + raw_delta[..., 6:7].float()
        )
        activation = torch.sigmoid(
            _stable_logit(base["activation"])[:, None]
            + raw_delta[..., 7:8].float()
        )
        values = {
            "feature": feature,
            "center": center,
            "covariance": covariance,
            "depth_order": depth,
            "opacity": opacity,
            "activation": activation,
        }
        result = {
            name: self._flatten_children(value) for name, value in values.items()
        }
        result["background_feature"] = base["background_feature"][:, None].expand(
            -1, predicted_slots.shape[1], -1
        )
        return result
