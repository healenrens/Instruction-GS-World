"""Coordinate-query dense feature readout with object-conditioned transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .gaussian_math import precision_2d


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


@dataclass
class DenseObjectReadoutState:
    feature: torch.Tensor
    assignment: torch.Tensor
    token_feature: torch.Tensor
    activation: torch.Tensor
    coverage: torch.Tensor
    feature_residual: torch.Tensor
    assignment_residual: torch.Tensor
    background_residual: torch.Tensor


class DenseObjectReadout(nn.Module):
    """Retain learned GPSToken support and transport it with object state."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.dense_readout_dim
        self.feature_dim = config.feature_dim
        self.token_input = nn.Linear(config.token_dim, hidden)
        self.object_input = nn.Linear(config.object_dim, hidden)
        self.object_delta_input = nn.Linear(config.object_dim, hidden, bias=False)
        self.state_norm = nn.LayerNorm(hidden)
        self.state_body = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * 2, hidden),
        )
        self.coordinate_input = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.assignment_query = nn.Linear(hidden, hidden, bias=False)
        self.feature_head = nn.Linear(hidden, config.feature_dim)
        self.activation_head = nn.Linear(hidden, 1)
        self.background_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config.feature_dim),
        )
        for head in (
            self.assignment_query,
            self.feature_head,
            self.activation_head,
            self.background_head[-1],
        ):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    @staticmethod
    def _validate(
        predicted_slots: torch.Tensor,
        current_slots: torch.Tensor,
        token_to_object: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        current_tokens,
    ) -> tuple[int, int, int]:
        if predicted_slots.ndim != 4:
            raise ValueError("predicted_slots must have shape [B,Q,K,D]")
        batch, queries, objects, object_dim = predicted_slots.shape
        if current_slots.shape != (batch, objects, object_dim):
            raise ValueError("current_slots must have shape [B,K,D]")
        micro_tokens = current_tokens.latent.shape[1]
        patches = current_tokens.assignment.shape[-1]
        token_shapes = {
            "latent": (batch, micro_tokens, current_tokens.latent.shape[-1]),
            "assignment": (batch, micro_tokens, patches),
            "decoded_features": (
                batch,
                micro_tokens,
                current_tokens.decoded_features.shape[-1],
            ),
            "center": (batch, micro_tokens, 2),
            "covariance": (batch, micro_tokens, 2, 2),
            "activation": (batch, micro_tokens, 1),
        }
        for name, expected in token_shapes.items():
            if getattr(current_tokens, name).shape != expected:
                raise ValueError(f"current token {name} must have shape {expected}")
        if token_to_object.shape != (batch, micro_tokens, objects):
            raise ValueError("token_to_object must have shape [B,M,K]")
        if coordinates.shape != (batch, queries, patches, 2):
            raise ValueError("coordinates must align with current GPSToken patches")
        if valid.shape != (batch, queries, patches):
            raise ValueError("valid must have shape [B,Q,N]")
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("every dense readout query needs a valid patch")
        return batch, queries, objects

    def _state_hidden(
        self,
        current_tokens,
        token_to_object: torch.Tensor,
        current_slots: torch.Tensor,
        predicted_slots: torch.Tensor,
    ) -> torch.Tensor:
        current_per_token = torch.einsum("bmk,bkd->bmd", token_to_object, current_slots)
        delta_per_token = torch.einsum(
            "bmk,bqkd->bqmd",
            token_to_object,
            predicted_slots - current_slots[:, None],
        )
        token = self.token_input(current_tokens.latent)[:, None]
        hidden = (
            token
            + self.object_input(current_per_token)[:, None]
            + self.object_delta_input(delta_per_token)
        )
        return self.state_body(self.state_norm(hidden))

    @staticmethod
    def _transport_bias(
        current_tokens,
        token_to_object: torch.Tensor,
        coordinates: torch.Tensor,
        predicted_centers: torch.Tensor,
        current_centers: torch.Tensor,
        predicted_scale: torch.Tensor | None,
        current_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        object_delta = predicted_centers - current_centers[:, None]
        center_delta = torch.einsum("bmk,bqkd->bqmd", token_to_object, object_delta)
        predicted_token_center = current_tokens.center[:, None] + center_delta
        log_scale = center_delta.new_zeros(center_delta.shape[:-1])
        if predicted_scale is not None:
            if current_scale is None:
                raise ValueError("predicted scale requires current scale")
            object_log_scale = (
                predicted_scale.clamp_min(1e-6).log()
                - current_scale[:, None].clamp_min(1e-6).log()
            ).clamp(-2.0, 2.0)
            log_scale = torch.einsum("bmk,bqk->bqm", token_to_object, object_log_scale)
        current_difference = (
            coordinates[:, :, None] - current_tokens.center[:, None, :, None]
        ).float()
        predicted_difference = (
            coordinates[:, :, None] - predicted_token_center[..., None, :]
        ).float() * torch.exp(-log_scale)[..., None, None]
        precision = precision_2d(current_tokens.covariance)[:, None].expand(
            -1, coordinates.shape[1], -1, -1, -1
        )
        current_distance = torch.einsum(
            "bqmni,bqmij,bqmnj->bqmn",
            current_difference,
            precision,
            current_difference,
        )
        predicted_distance = torch.einsum(
            "bqmni,bqmij,bqmnj->bqmn",
            predicted_difference,
            precision,
            predicted_difference,
        )
        return (-0.5 * (predicted_distance - current_distance)).clamp(-20.0, 20.0)

    def forward(
        self,
        current_tokens,
        token_to_object: torch.Tensor,
        current_slots: torch.Tensor,
        predicted_slots: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        background_feature: torch.Tensor,
        *,
        current_centers: torch.Tensor,
        predicted_centers: torch.Tensor,
        current_activity: torch.Tensor,
        predicted_visibility: torch.Tensor | None = None,
        current_scale: torch.Tensor | None = None,
        predicted_scale: torch.Tensor | None = None,
    ) -> DenseObjectReadoutState:
        batch, queries, objects = self._validate(
            predicted_slots,
            current_slots,
            token_to_object,
            coordinates,
            valid,
            current_tokens,
        )
        if current_centers.shape != (batch, objects, 2):
            raise ValueError("current_centers must have shape [B,K,2]")
        if predicted_centers.shape != (batch, queries, objects, 2):
            raise ValueError("predicted_centers must have shape [B,Q,K,2]")
        if current_activity.shape != (batch, objects):
            raise ValueError("current_activity must have shape [B,K]")
        if background_feature.shape != (batch, self.feature_dim):
            raise ValueError("background_feature must have shape [B,C]")
        if current_tokens.decoded_features.shape[-1] != self.feature_dim:
            raise ValueError("current token feature dimension differs from readout")
        if predicted_visibility is None:
            predicted_visibility = current_activity[:, None].expand(-1, queries, -1)
        if predicted_visibility.shape != (batch, queries, objects):
            raise ValueError("predicted_visibility must have shape [B,Q,K]")
        if (current_scale is None) != (predicted_scale is None):
            raise ValueError("current and predicted relative scale must be paired")
        if current_scale is not None:
            if current_scale.shape != (batch, objects):
                raise ValueError("current_scale must have shape [B,K]")
            if predicted_scale.shape != (batch, queries, objects):
                raise ValueError("predicted_scale must have shape [B,Q,K]")

        hidden = self._state_hidden(
            current_tokens, token_to_object, current_slots, predicted_slots
        )
        coordinate_key = self.coordinate_input(coordinates.float())
        learned_logits = torch.einsum(
            "bqmd,bqnd->bqmn",
            self.assignment_query(hidden),
            coordinate_key,
        ) / math.sqrt(hidden.shape[-1])
        learned_logits = 2.0 * torch.tanh(learned_logits)
        transport = self._transport_bias(
            current_tokens,
            token_to_object,
            coordinates,
            predicted_centers,
            current_centers,
            predicted_scale,
            current_scale,
        )
        base_assignment = current_tokens.assignment[:, None].expand(-1, queries, -1, -1)
        logits = base_assignment.clamp_min(1e-8).log() + learned_logits + transport
        logits = logits.masked_fill(~valid[:, :, None], torch.finfo(logits.dtype).min)
        assignment = logits.softmax(dim=2) * valid[:, :, None].to(logits.dtype)

        feature_residual = 0.25 * torch.tanh(self.feature_head(hidden).float())
        token_feature = (
            current_tokens.decoded_features[:, None].float() + feature_residual
        )
        current_lifecycle = torch.einsum(
            "bmk,bk->bm", token_to_object, current_activity
        )
        predicted_lifecycle = torch.einsum(
            "bmk,bqk->bqm", token_to_object, predicted_visibility
        )
        lifecycle_delta = (
            _stable_logit(predicted_lifecycle)
            - _stable_logit(current_lifecycle)[:, None]
        )
        activation_logits = (
            _stable_logit(current_tokens.activation)[:, None]
            + self.activation_head(hidden).float()
            + lifecycle_delta[..., None]
        )
        activation = torch.sigmoid(activation_logits)
        active_assignment = assignment * activation
        coverage = active_assignment.sum(dim=2).clamp(0.0, 1.0)
        foreground = torch.einsum(
            "bqmn,bqmc->bqnc", active_assignment.float(), token_feature
        )
        background_residual = 0.25 * torch.tanh(
            self.background_head(hidden.mean(dim=2)).float()
        )
        background = background_feature[:, None].float() + background_residual
        feature = foreground + (1.0 - coverage)[..., None] * background[:, :, None]
        return DenseObjectReadoutState(
            feature=feature,
            assignment=assignment,
            token_feature=token_feature,
            activation=activation,
            coverage=coverage,
            feature_residual=feature_residual,
            assignment_residual=learned_logits,
            background_residual=background_residual,
        )
