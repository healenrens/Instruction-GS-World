"""Object-conditioned local DINO residuals over an exact current feature field."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .gaussian_math import precision_2d


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


@dataclass
class ChangeResidualReadoutState:
    residual: torch.Tensor
    destination_assignment: torch.Tensor
    source_assignment: torch.Tensor
    token_feature_delta: torch.Tensor
    active_change_logits: torch.Tensor
    active_change: torch.Tensor
    active_change_map: torch.Tensor
    destination_coverage: torch.Tensor
    source_coverage: torch.Tensor
    potential_change: torch.Tensor
    assignment_residual: torch.Tensor
    transport_bias: torch.Tensor


class ChangeResidualReadout(nn.Module):
    """Transport only local change carriers; the observed field bypasses decoding."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.change_readout_dim
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
        self.feature_delta = nn.Linear(hidden, config.feature_dim)
        self.active_change = nn.Linear(hidden, 1)
        for head in (self.assignment_query, self.feature_delta, self.active_change):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    @staticmethod
    def _validate(
        current_tokens,
        token_to_object: torch.Tensor,
        potential_change: torch.Tensor,
        current_slots: torch.Tensor,
        predicted_slots: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if predicted_slots.ndim != 4:
            raise ValueError("predicted_slots must have shape [B,Q,K,D]")
        batch, queries, objects, object_dim = predicted_slots.shape
        if current_slots.shape != (batch, objects, object_dim):
            raise ValueError("current_slots must have shape [B,K,D]")
        micro_tokens = current_tokens.latent.shape[1]
        patches = current_tokens.assignment.shape[-1]
        if token_to_object.shape != (batch, micro_tokens, objects):
            raise ValueError("token_to_object must have shape [B,M,K]")
        if potential_change.shape != (batch, micro_tokens):
            raise ValueError("potential_change must have shape [B,M]")
        if coordinates.shape != (batch, queries, patches, 2):
            raise ValueError("coordinates must have shape [B,Q,N,2]")
        if valid.shape != (batch, queries, patches):
            raise ValueError("valid must have shape [B,Q,N]")
        if current_tokens.decoded_features.shape != (
            batch,
            micro_tokens,
            current_tokens.decoded_features.shape[-1],
        ):
            raise ValueError("current token features have an invalid shape")
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("every readout query requires a valid patch")
        return batch, queries, objects, micro_tokens, patches

    @staticmethod
    def _conditional_object_assignment(
        token_to_object: torch.Tensor,
        potential_change: torch.Tensor,
    ) -> torch.Tensor:
        return token_to_object / potential_change[..., None].clamp_min(1e-6)

    def _state_hidden(
        self,
        current_tokens,
        object_assignment: torch.Tensor,
        current_slots: torch.Tensor,
        predicted_slots: torch.Tensor,
    ) -> torch.Tensor:
        current_per_token = torch.einsum(
            "bmk,bkd->bmd", object_assignment, current_slots
        )
        delta_per_token = torch.einsum(
            "bmk,bqkd->bqmd",
            object_assignment,
            predicted_slots - current_slots[:, None],
        )
        hidden = (
            self.token_input(current_tokens.latent)[:, None]
            + self.object_input(current_per_token)[:, None]
            + self.object_delta_input(delta_per_token)
        )
        return self.state_body(self.state_norm(hidden))

    @staticmethod
    def _transport_bias(
        current_tokens,
        object_assignment: torch.Tensor,
        coordinates: torch.Tensor,
        current_centers: torch.Tensor,
        predicted_centers: torch.Tensor,
        current_scale: torch.Tensor | None,
        predicted_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        object_delta = predicted_centers - current_centers[:, None]
        center_delta = torch.einsum(
            "bmk,bqkd->bqmd", object_assignment, object_delta
        )
        predicted_token_center = current_tokens.center[:, None] + center_delta
        log_scale = center_delta.new_zeros(center_delta.shape[:-1])
        if predicted_scale is not None:
            if current_scale is None:
                raise ValueError("predicted scale requires current scale")
            object_log_scale = (
                predicted_scale.clamp_min(1e-6).log()
                - current_scale[:, None].clamp_min(1e-6).log()
            ).clamp(-2.0, 2.0)
            log_scale = torch.einsum(
                "bmk,bqk->bqm", object_assignment, object_log_scale
            )
        current_difference = (
            coordinates[:, :, None] - current_tokens.center[:, None, :, None]
        ).float()
        predicted_difference = (
            coordinates[:, :, None] - predicted_token_center[..., None, :]
        ).float() * torch.exp(-log_scale)[..., None, None]
        precision = precision_2d(current_tokens.covariance.float())[:, None]
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
        potential_change: torch.Tensor,
        current_slots: torch.Tensor,
        predicted_slots: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        *,
        current_centers: torch.Tensor,
        predicted_centers: torch.Tensor,
        current_activity: torch.Tensor,
        predicted_visibility: torch.Tensor | None = None,
        current_scale: torch.Tensor | None = None,
        predicted_scale: torch.Tensor | None = None,
    ) -> ChangeResidualReadoutState:
        batch, queries, objects, _, _ = self._validate(
            current_tokens,
            token_to_object,
            potential_change,
            current_slots,
            predicted_slots,
            coordinates,
            valid,
        )
        if current_centers.shape != (batch, objects, 2):
            raise ValueError("current_centers must have shape [B,K,2]")
        if predicted_centers.shape != (batch, queries, objects, 2):
            raise ValueError("predicted_centers must have shape [B,Q,K,2]")
        if current_activity.shape != (batch, objects):
            raise ValueError("current_activity must have shape [B,K]")
        if current_tokens.decoded_features.shape[-1] != self.feature_dim:
            raise ValueError("token feature dimension differs from DINO readout")
        if predicted_visibility is None:
            predicted_visibility = current_activity[:, None].expand(-1, queries, -1)
        if predicted_visibility.shape != (batch, queries, objects):
            raise ValueError("predicted_visibility must have shape [B,Q,K]")
        if (current_scale is None) != (predicted_scale is None):
            raise ValueError("current and predicted scales must be paired")

        object_assignment = self._conditional_object_assignment(
            token_to_object, potential_change
        )
        hidden = self._state_hidden(
            current_tokens, object_assignment, current_slots, predicted_slots
        )
        coordinate_key = self.coordinate_input(coordinates.float())
        assignment_residual = torch.einsum(
            "bqmd,bqnd->bqmn",
            self.assignment_query(hidden),
            coordinate_key,
        ) / math.sqrt(hidden.shape[-1])
        assignment_residual = 2.0 * torch.tanh(assignment_residual)
        transport = self._transport_bias(
            current_tokens,
            object_assignment,
            coordinates,
            current_centers,
            predicted_centers,
            current_scale,
            predicted_scale,
        )
        source_assignment = current_tokens.assignment[:, None].expand(
            -1, queries, -1, -1
        )
        logits = source_assignment.clamp_min(1e-8).log() + assignment_residual + transport
        logits = logits.masked_fill(~valid[:, :, None], torch.finfo(logits.dtype).min)
        destination_assignment = logits.softmax(dim=2)
        destination_assignment = destination_assignment * valid[:, :, None].to(
            destination_assignment.dtype
        )

        current_lifecycle = torch.einsum(
            "bmk,bk->bm", object_assignment, current_activity
        )
        predicted_lifecycle = torch.einsum(
            "bmk,bqk->bqm", object_assignment, predicted_visibility
        )
        base_gate = (
            potential_change * current_tokens.activation.squeeze(-1)
        ).clamp(1e-4, 1.0 - 1e-4)
        active_change_logits = (
            _stable_logit(base_gate)[:, None]
            + self.active_change(hidden).squeeze(-1).float()
            + _stable_logit(predicted_lifecycle)
            - _stable_logit(current_lifecycle)[:, None]
        )
        active_change = torch.sigmoid(active_change_logits)
        token_feature_delta = 0.25 * torch.tanh(self.feature_delta(hidden).float())
        current_feature = current_tokens.decoded_features[:, None].float()
        future_feature = current_feature + token_feature_delta
        source_weight = source_assignment.float() * active_change[..., None]
        destination_weight = destination_assignment.float() * active_change[..., None]
        source_field = torch.einsum(
            "bqmn,bqmc->bqnc", source_weight, current_feature.expand_as(future_feature)
        )
        destination_field = torch.einsum(
            "bqmn,bqmc->bqnc", destination_weight, future_feature
        )
        source_coverage = source_weight.sum(dim=2).clamp(0.0, 1.0)
        destination_coverage = destination_weight.sum(dim=2).clamp(0.0, 1.0)
        return ChangeResidualReadoutState(
            residual=destination_field - source_field,
            destination_assignment=destination_assignment,
            source_assignment=source_assignment,
            token_feature_delta=token_feature_delta,
            active_change_logits=active_change_logits,
            active_change=active_change,
            active_change_map=torch.maximum(source_coverage, destination_coverage),
            destination_coverage=destination_coverage,
            source_coverage=source_coverage,
            potential_change=potential_change,
            assignment_residual=assignment_residual,
            transport_bias=transport,
        )
