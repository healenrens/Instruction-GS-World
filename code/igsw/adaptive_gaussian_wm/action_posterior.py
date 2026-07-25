"""Future-conditioned latent action posterior modules."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .observed_action import rgb_logit_action


def weighted_slot_pool(
    slots: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity.to(slots.dtype)[..., None]
    return (slots * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1e-6)


class ActionPosterior(nn.Module):
    """Legacy pooled posterior retained for checkpoint compatibility."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.model_dim
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.history_input = nn.Linear(config.object_dim * 2, dim)
        self.future_input = nn.Linear(config.object_dim, dim)
        self.gap_input = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, config.action_tokens * config.action_dim),
        )
        self.output_norm = (
            nn.LayerNorm(config.action_dim)
            if config.normalize_posterior
            else nn.Identity()
        )

    def history_context(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
    ) -> torch.Tensor:
        pooled = weighted_slot_pool(history_slots, history_activity)
        summary = pooled.mean(dim=1)
        current = pooled[:, -1]
        return self.history_input(torch.cat((summary, current), dim=-1))

    def forward(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        future_slots: torch.Tensor,
        future_activity: torch.Tensor,
        future_scale: torch.Tensor,
        history_centers: torch.Tensor | None = None,
        future_centers: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del history_centers, future_centers
        history = self.history_context(history_slots, history_activity)
        future = self.future_input(
            weighted_slot_pool(future_slots, future_activity)
        )
        gap = self.gap_input(future_scale[..., None])
        if condition is not None:
            if condition.shape != history.shape:
                raise ValueError("condition must have shape [B,D]")
            history = history + condition
            future = future + condition[:, None]
        history = history[:, None].expand(-1, future.shape[1], -1)
        raw = self.posterior(torch.cat((history, future, gap), dim=-1))
        return self.output_norm(
            raw.reshape(
                raw.shape[0],
                raw.shape[1],
                self.action_tokens,
                self.action_dim,
            )
        )


class ObjectDeltaActionPosterior(nn.Module):
    """Attend to aligned object changes with an optional center anchor."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.model_dim
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.delta_only = config.delta_only_posterior
        self.object_aligned = config.object_aligned_actions
        self.canonical_center_action = config.canonical_center_action
        self.canonical_semantic_action = config.canonical_semantic_action
        self.canonical_activity_gate = config.canonical_activity_gate
        self.canonical_activity_power = config.canonical_activity_power
        self.learned_semantic_action_basis = (
            config.learned_semantic_action_basis
        )
        self.rgb_semantic_action = config.rgb_semantic_action
        self.residual_dim = (
            config.action_residual_dim
            if self.canonical_semantic_action
            else config.action_dim
        )
        if self.canonical_semantic_action and not self.rgb_semantic_action:
            row = torch.arange(config.object_dim, dtype=torch.float32)[:, None] + 1
            column = torch.arange(3, dtype=torch.float32)[None] + 1
            projection = torch.sin(row * column * 0.017)
            projection = projection / projection.norm(dim=0, keepdim=True)
            if self.learned_semantic_action_basis:
                self.semantic_projection = nn.Parameter(projection)
            else:
                self.register_buffer("semantic_projection", projection)
        else:
            self.semantic_projection = None
        self.current_input = (
            None
            if self.delta_only
            else nn.Linear(config.object_dim, dim)
        )
        self.future_input = (
            None
            if self.delta_only
            else nn.Linear(config.object_dim, dim)
        )
        self.delta_input = nn.Linear(config.object_dim, dim)
        self.activity_input = nn.Linear(1, dim)
        self.history_input = (
            None
            if self.delta_only
            else nn.Linear(config.object_dim * 2, dim)
        )
        self.gap_input = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.center_delta_input = (
            nn.Sequential(
                nn.Linear(3, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            if config.center_conditioned_posterior
            else None
        )
        self.queries = nn.Parameter(
            torch.randn(config.action_tokens, dim) / dim**0.5
        )
        self.attention = nn.MultiheadAttention(
            dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim * 2, self.residual_dim),
            )
            if self.residual_dim > 0
            else None
        )
        self.output_norm = (
            (
                nn.LayerNorm(self.residual_dim)
                if config.normalize_posterior
                else nn.Identity()
            )
            if self.residual_dim > 0
            else None
        )

    def semantic_basis(self) -> torch.Tensor:
        if self.semantic_projection is None:
            raise ValueError("semantic basis is not initialized")
        if self.learned_semantic_action_basis:
            return F.normalize(self.semantic_projection, dim=0)
        return self.semantic_projection

    def forward(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        future_slots: torch.Tensor,
        future_activity: torch.Tensor,
        future_scale: torch.Tensor,
        history_centers: torch.Tensor | None = None,
        future_centers: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        current_object_rgb: torch.Tensor | None = None,
        future_object_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = history_slots[:, -1]
        batch, future_count, object_count, _ = future_slots.shape
        current = current[:, None].expand(-1, future_count, -1, -1)
        slot_delta = future_slots - current
        object_tokens = self.delta_input(slot_delta)
        object_tokens = (
            object_tokens + self.activity_input(future_activity[..., None])
        )
        center_geometry = None
        if self.center_delta_input is not None:
            if history_centers is None or future_centers is None:
                raise ValueError("center-conditioned posterior requires centers")
            center_delta = (
                future_centers - history_centers[:, -1, None]
            )
            center_geometry = torch.cat(
                (center_delta, center_delta.norm(dim=-1, keepdim=True)),
                dim=-1,
            )
            object_tokens = (
                object_tokens + self.center_delta_input(center_geometry)
            )
        query = (
            self.queries[None, None]
            + self.gap_input(future_scale[..., None])[:, :, None]
        )
        if condition is not None:
            if condition.shape != (batch, query.shape[-1]):
                raise ValueError("condition must have shape [B,D]")
            query = query + condition[:, None, None]
        if not self.delta_only:
            pooled = weighted_slot_pool(history_slots, history_activity)
            history = self.history_input(
                torch.cat((pooled.mean(dim=1), pooled[:, -1]), dim=-1)
            )
            query = query + history[:, None, None]
            object_tokens = (
                object_tokens
                + self.current_input(current)
                + self.future_input(future_slots)
            )
        if self.object_aligned:
            if object_count != self.action_tokens:
                raise ValueError(
                    "object-aligned posterior requires one action per object"
                )
            query = query + object_tokens
        object_tokens = object_tokens.reshape(
            batch * future_count,
            object_count,
            -1,
        )
        query = query.reshape(
            batch * future_count,
            self.action_tokens,
            -1,
        )
        attended, attention = self.attention(
            query,
            object_tokens,
            object_tokens,
            need_weights=(
                self.canonical_center_action and not self.object_aligned
            ),
        )
        encoded = query + attended
        learned_actions = (
            self.output_norm(self.output(encoded))
            if self.output is not None and self.output_norm is not None
            else encoded.new_empty((*encoded.shape[:-1], 0))
        )
        if self.canonical_center_action:
            if center_geometry is None:
                raise ValueError("canonical center action requires center geometry")
            geometry = center_geometry.reshape(
                batch * future_count,
                object_count,
                3,
            ).detach()
            if self.object_aligned:
                canonical = torch.tanh(geometry / 0.25)
            else:
                if attention is None:
                    raise ValueError("global canonical action requires attention")
                canonical = torch.tanh(attention @ geometry / 0.25)
            if self.canonical_semantic_action:
                if self.rgb_semantic_action:
                    if current_object_rgb is None or future_object_rgb is None:
                        raise ValueError(
                            "RGB semantic action requires object RGB targets"
                        )
                    semantic = rgb_logit_action(
                        current_object_rgb,
                        future_object_rgb,
                    ).reshape(batch * future_count, object_count, 3)
                else:
                    if self.semantic_projection is None:
                        raise ValueError("semantic projection is not initialized")
                    semantic_geometry = (
                        slot_delta @ self.semantic_basis().to(slot_delta.dtype)
                    ).reshape(
                        batch * future_count,
                        object_count,
                        3,
                    )
                    if not self.learned_semantic_action_basis:
                        semantic_geometry = semantic_geometry.detach()
                    if self.object_aligned:
                        semantic = torch.tanh(semantic_geometry / 0.25)
                    else:
                        semantic = torch.tanh(
                            attention @ semantic_geometry / 0.25
                        )
                if self.canonical_activity_gate:
                    confidence = (
                        (
                            history_activity[:, -1, None]
                            * future_activity
                        ).float().clamp(0.0, 1.0)
                    ).pow(self.canonical_activity_power)
                    confidence = confidence.reshape(
                        batch * future_count, object_count, 1
                    ).detach()
                    canonical = canonical * confidence
                    semantic = semantic * confidence
                actions = torch.cat(
                    (canonical[..., :3], semantic, learned_actions),
                    dim=-1,
                )
            else:
                dimensions = min(3, self.action_dim)
                actions = torch.cat(
                    (
                        canonical[..., :dimensions],
                        learned_actions[..., dimensions:],
                    ),
                    dim=-1,
                )
        else:
            actions = learned_actions
        if actions.shape[-1] != self.action_dim:
            raise ValueError("Posterior action layout does not match action_dim")
        return actions.reshape(
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
