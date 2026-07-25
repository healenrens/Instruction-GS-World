"""Action-free object prediction plus a strictly residual latent-effect branch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .relative_geometry import (
    RelativeGeometryEncoder,
    pairwise_relative_geometry,
)


@dataclass
class FactorizedDynamicsOutput:
    future_slots: torch.Tensor
    history_slots: torch.Tensor
    future_centers: torch.Tensor
    history_centers: torch.Tensor
    future_relative_scale: torch.Tensor
    future_relative_disparity: torch.Tensor
    future_visibility_logits: torch.Tensor
    future_existence_logits: torch.Tensor
    future_visibility: torch.Tensor
    future_existence: torch.Tensor
    future_relations: torch.Tensor
    base_future_slots: torch.Tensor
    action_slot_residual: torch.Tensor


class BiasedDynamicsBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, dim = tokens.shape
        if attention_bias.shape != (batch, self.heads, count, count):
            raise ValueError("relative attention bias has an invalid shape")
        normalized = self.norm_attention(tokens)
        qkv = self.qkv(normalized).reshape(
            batch,
            count,
            3,
            self.heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        score = query @ key.transpose(-1, -2) * self.scale
        score = score + attention_bias.to(score.dtype)
        weight = torch.softmax(score.float(), dim=-1).to(score.dtype)
        attended = (weight @ value).transpose(1, 2).reshape(batch, count, dim)
        tokens = tokens + self.dropout(self.projection(attended))
        return tokens + self.dropout(self.mlp(self.norm_mlp(tokens)))


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.float().clamp(1e-4, 1.0 - 1e-4))


class FactorizedObjectDynamics(nn.Module):
    """Model predictable evolution separately from future-specific effects."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.model_dim
        self.config = config
        self.slot_input = nn.Linear(config.object_dim, dim)
        self.identity_input = nn.Linear(config.object_dim, dim)
        self.activity_input = nn.Linear(1, dim)
        self.geometry_input = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_input = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.object_identity = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        self.history_type = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.future_type = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.history_mask_token = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.future_query = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.relative_geometry = RelativeGeometryEncoder(config)
        self.blocks = nn.ModuleList(
            [
                BiasedDynamicsBlock(dim, config.heads, config.dropout)
                for _ in range(config.dynamics_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.base_slot_output = nn.Linear(dim, config.object_dim)
        self.base_geometry_output = nn.Linear(dim, 4)
        self.base_lifecycle_output = nn.Linear(dim, 2)

        route_dim = min(256, dim)
        self.routing_query = nn.Linear(config.object_dim, route_dim, bias=False)
        self.factor_keys = nn.Parameter(
            torch.randn(config.action_tokens, route_dim) / route_dim**0.5
        )
        self.action_input = nn.Linear(config.action_dim, dim, bias=False)
        self.action_slot_basis = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, config.object_dim),
        )
        self.action_slot_gate = nn.Linear(dim, config.object_dim, bias=False)
        self.action_geometry_basis = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 6),
        )
        self.action_geometry_gate = nn.Linear(dim, 6, bias=False)
        nn.init.normal_(self.base_slot_output.weight, std=1e-3)
        nn.init.zeros_(self.base_slot_output.bias)
        nn.init.zeros_(self.base_geometry_output.weight)
        nn.init.zeros_(self.base_geometry_output.bias)
        nn.init.zeros_(self.base_lifecycle_output.weight)
        nn.init.zeros_(self.base_lifecycle_output.bias)

    def _relation_bias(
        self,
        history_relations: torch.Tensor,
        future_count: int,
    ) -> torch.Tensor:
        batch, history_count, object_count = history_relations.shape[:3]
        group_count = history_count + future_count
        history_bias = self.relative_geometry.bias(history_relations)
        latest = history_bias[:, -1]
        rows = []
        for row in range(group_count):
            columns = []
            for column in range(group_count):
                block = (
                    history_bias[:, row]
                    if row == column and row < history_count
                    else latest
                )
                columns.append(block)
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2).reshape(
            batch,
            self.config.heads,
            group_count * object_count,
            group_count * object_count,
        )

    def _geometry_defaults(
        self,
        history_centers: torch.Tensor,
        history_activity: torch.Tensor,
        history_relative_scale: torch.Tensor | None,
        history_relative_disparity: torch.Tensor | None,
        history_relations: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (
            torch.ones_like(history_activity)
            if history_relative_scale is None
            else history_relative_scale
        )
        disparity = (
            torch.zeros_like(history_activity)
            if history_relative_disparity is None
            else history_relative_disparity
        )
        relations = (
            pairwise_relative_geometry(
                history_centers,
                scale,
                disparity,
                history_activity,
            )
            if history_relations is None
            else history_relations
        )
        return scale, disparity, relations

    def forward(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        history_scale: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        history_centers: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        history_relative_scale: torch.Tensor | None = None,
        history_relative_disparity: torch.Tensor | None = None,
        history_relations: torch.Tensor | None = None,
        history_existence: torch.Tensor | None = None,
    ) -> FactorizedDynamicsOutput:
        if condition is not None:
            raise ValueError("factorized object Dynamics is language-free")
        if history_centers is None:
            raise ValueError("factorized Dynamics requires image-plane centers")
        batch, history_count, object_count, _ = history_slots.shape
        future_count = future_scale.shape[1]
        expected_action = (
            batch,
            future_count,
            self.config.action_tokens,
            self.config.action_dim,
        )
        if actions.shape != expected_action:
            raise ValueError(f"actions must have shape {expected_action}")
        if history_activity.shape != history_slots.shape[:3]:
            raise ValueError("history_activity must have shape [B,T,K]")
        if history_centers.shape != (*history_slots.shape[:3], 2):
            raise ValueError("history_centers must have shape [B,T,K,2]")
        if history_scale.shape != history_slots.shape[:2]:
            raise ValueError("history_scale must have shape [B,T]")
        if history_mask is None:
            history_mask = torch.zeros_like(history_activity, dtype=torch.bool)
        if history_mask.shape != history_slots.shape[:3]:
            raise ValueError("history_mask must have shape [B,T,K]")
        relative_scale, disparity, relations = self._geometry_defaults(
            history_centers,
            history_activity,
            history_relative_scale,
            history_relative_disparity,
            history_relations,
        )
        existence = (
            history_activity
            if history_existence is None
            else history_existence
        )
        geometry = torch.cat(
            (
                history_centers,
                relative_scale[..., None].clamp_min(1e-6).log(),
                disparity[..., None],
                history_activity[..., None],
                existence[..., None],
            ),
            dim=-1,
        )
        identity = self.object_identity[None, None]
        history_tokens = (
            self.slot_input(history_slots)
            + self.activity_input(history_activity[..., None])
            + self.geometry_input(geometry)
            + self.time_input(history_scale[..., None])[:, :, None]
            + identity
            + self.history_type
        )
        masked = self.history_mask_token + identity + self.history_type
        history_tokens = torch.where(
            history_mask[..., None], masked, history_tokens
        )
        current_identity = self.identity_input(history_slots[:, -1])
        future_tokens = (
            self.future_query
            + self.future_type
            + self.object_identity[None, None]
            + current_identity[:, None]
            + self.time_input(future_scale[..., None])[:, :, None]
        )
        tokens = torch.cat(
            (
                history_tokens.flatten(1, 2),
                future_tokens.flatten(1, 2),
            ),
            dim=1,
        )
        attention_bias = self._relation_bias(relations, future_count)
        for block in self.blocks:
            tokens = block(tokens, attention_bias)
        hidden = self.output_norm(tokens)
        history_end = history_count * object_count
        history_hidden = hidden[:, :history_end].reshape(
            batch, history_count, object_count, -1
        )
        future_hidden = hidden[:, history_end:].reshape(
            batch, future_count, object_count, -1
        )
        predicted_history = history_slots + self.base_slot_output(history_hidden)
        base_future = history_slots[:, -1, None] + self.base_slot_output(
            future_hidden
        )

        route_query = F.normalize(
            self.routing_query(history_slots[:, -1]), dim=-1
        )
        route_keys = F.normalize(self.factor_keys, dim=-1)
        routing = torch.einsum("bkd,rd->bkr", route_query, route_keys).softmax(
            dim=-1
        )
        object_effect = torch.einsum("bkr,bqrd->bqkd", routing, actions)
        action_hidden = self.action_input(object_effect)
        action_slot_residual = torch.tanh(
            self.action_slot_basis(future_hidden)
        ) * torch.tanh(self.action_slot_gate(action_hidden))
        future_slots = base_future + action_slot_residual

        base_geometry = self.base_geometry_output(future_hidden)
        action_geometry = torch.tanh(
            self.action_geometry_basis(future_hidden)
        ) * torch.tanh(self.action_geometry_gate(action_hidden))
        current_center = history_centers[:, -1, None]
        future_centers = current_center + 0.5 * torch.tanh(
            base_geometry[..., :2] + action_geometry[..., :2]
        )
        current_scale = relative_scale[:, -1, None]
        future_relative_scale = current_scale * torch.exp(
            0.25 * torch.tanh(base_geometry[..., 2] + action_geometry[..., 2])
        )
        current_disparity = disparity[:, -1, None]
        future_relative_disparity = current_disparity + 0.25 * torch.tanh(
            base_geometry[..., 3] + action_geometry[..., 3]
        )
        lifecycle = self.base_lifecycle_output(future_hidden)
        current_visibility = history_activity[:, -1, None]
        current_existence = existence[:, -1, None]
        future_visibility_logits = (
            _stable_logit(current_visibility)
            + lifecycle[..., 0]
            + action_geometry[..., 4]
        )
        future_existence_logits = (
            _stable_logit(current_existence)
            + lifecycle[..., 1]
            + action_geometry[..., 5]
        )
        future_visibility = torch.sigmoid(future_visibility_logits)
        future_existence = torch.sigmoid(future_existence_logits)
        future_relations = pairwise_relative_geometry(
            future_centers,
            future_relative_scale,
            future_relative_disparity,
            future_visibility,
        )
        return FactorizedDynamicsOutput(
            future_slots=future_slots,
            history_slots=predicted_history,
            future_centers=future_centers,
            history_centers=history_centers,
            future_relative_scale=future_relative_scale,
            future_relative_disparity=future_relative_disparity,
            future_visibility_logits=future_visibility_logits,
            future_existence_logits=future_existence_logits,
            future_visibility=future_visibility,
            future_existence=future_existence,
            future_relations=future_relations,
            base_future_slots=base_future,
            action_slot_residual=action_slot_residual,
        )
