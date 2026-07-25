"""Future-conditioned latent actions and a strictly history-only flow prior."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .action_posterior import (
    ActionPosterior,
    ObjectDeltaActionPosterior,
    weighted_slot_pool,
)
from .config import AdaptiveGaussianWMConfig
from .continuous_effect import ContinuousEffectPosterior
from .flow_matching import flow_training_objective, sample_flow_source
from .source_lifted_prior import SourceLiftedJointFlowPrior
from .mode_set_prior import ModeSetActionPrior
from .prior_token_conditioning import PriorTokenConditioner

def _action_context(
    context: torch.Tensor,
    action_tokens: int,
) -> torch.Tensor:
    if context.ndim == 3:
        return context[:, :, None].expand(-1, -1, action_tokens, -1)
    if context.ndim == 4 and context.shape[2] == action_tokens:
        return context
    raise ValueError("context must have shape [B,Q,D] or [B,Q,A,D]")


def _repeat_context(context: torch.Tensor, sample_count: int) -> torch.Tensor:
    return context[None].expand(sample_count, *context.shape).reshape(
        sample_count * context.shape[0],
        *context.shape[1:],
    )

class StructuredConditionalFlowPrior(nn.Module):
    """Conditional flow matching over a set of correlated action tokens."""
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.steps = config.flow_steps
        self.endpoint_prediction = config.flow_endpoint_prediction
        self.correlated_source = config.correlated_flow_source
        self.source_scale = config.flow_source_scale
        context_dim = config.model_dim
        hidden = config.flow_hidden_dim
        self.action_identity = nn.Parameter(
            torch.randn(config.action_tokens, config.action_dim)
            / config.action_dim**0.5
        )
        self.velocity = nn.Sequential(
            nn.Linear(config.action_dim * 2 + context_dim + 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.action_dim),
        )
    @staticmethod
    def _flow_time_features(time: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                time,
                time.square(),
                torch.sin(math.pi * time),
                torch.cos(math.pi * time),
                torch.sin(2.0 * math.pi * time),
            ),
            dim=-1,
        )
    def forward(
        self,
        latent: torch.Tensor,
        flow_time: torch.Tensor,
        context: torch.Tensor,
        endpoint_output: bool = False,
    ) -> torch.Tensor:
        if latent.shape[:2] != context.shape[:2]:
            raise ValueError("latent and context must share [B,Q]")
        if flow_time.shape != latent.shape[:2]:
            raise ValueError("flow_time must have shape [B,Q]")
        identity = self.action_identity[None, None].expand(
            latent.shape[0],
            latent.shape[1],
            -1,
            -1,
        )
        expanded_context = _action_context(context, self.action_tokens)
        expanded_time = self._flow_time_features(flow_time)[:, :, None].expand(
            -1,
            -1,
            self.action_tokens,
            -1,
        )
        output = self.velocity(
            torch.cat((latent, identity, expanded_context, expanded_time), dim=-1)
        )
        if self.endpoint_prediction and not endpoint_output:
            remaining = (1.0 - flow_time)[..., None, None].clamp_min(1e-4)
            return (output - latent) / remaining
        return output
    def loss(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flow_training_objective(self, target, context, group_id)[0]
    def sample(
        self,
        context: torch.Tensor,
        sample_count: int = 1,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        batch, future_count = context.shape[:2]
        expanded_context = _repeat_context(context, sample_count)
        latent = sample_flow_source(
            self,
            sample_count * batch,
            future_count,
            context.device,
            context.dtype,
        )
        if not stochastic:
            latent.zero_()
        step_size = 1.0 / self.steps
        for index in range(self.steps):
            flow_fraction = (
                index * step_size
                if self.endpoint_prediction
                else (index + 0.5) * step_size
            )
            flow_time = latent.new_full(
                latent.shape[:2],
                flow_fraction,
            )
            latent = latent + step_size * self(latent, flow_time, expanded_context)
        return latent.reshape(
            sample_count,
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
class JointConditionalFlowPrior(nn.Module):
    """Joint velocity field over every future and action token."""
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.steps = config.flow_steps
        self.mix_tokens = config.joint_flow
        self.endpoint_prediction = config.flow_endpoint_prediction
        self.correlated_source = config.correlated_flow_source
        self.source_scale = config.flow_source_scale
        hidden = config.flow_hidden_dim
        self.action_identity = nn.Parameter(
            torch.randn(config.action_tokens, config.action_dim)
            / config.action_dim**0.5
        )
        self.input_projection = nn.Linear(
            config.action_dim * 2 + config.model_dim + 5,
            hidden,
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=config.heads,
                    dim_feedforward=hidden * 2,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.flow_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_projection = nn.Linear(hidden, config.action_dim)
    def forward(
        self,
        latent: torch.Tensor,
        flow_time: torch.Tensor,
        context: torch.Tensor,
        endpoint_output: bool = False,
    ) -> torch.Tensor:
        if latent.shape[:2] != context.shape[:2]:
            raise ValueError("latent and context must share [B,Q]")
        if flow_time.shape != latent.shape[:2]:
            raise ValueError("flow_time must have shape [B,Q]")
        identity = self.action_identity[None, None].expand(
            latent.shape[0],
            latent.shape[1],
            -1,
            -1,
        )
        expanded_context = _action_context(context, self.action_tokens)
        time_features = StructuredConditionalFlowPrior._flow_time_features(
            flow_time
        )[:, :, None].expand(-1, -1, self.action_tokens, -1)
        hidden = self.input_projection(
            torch.cat(
                (latent, identity, expanded_context, time_features),
                dim=-1,
            )
        )
        shape = hidden.shape
        if self.mix_tokens:
            hidden = hidden.flatten(1, 2)
        else:
            hidden = hidden.reshape(-1, 1, hidden.shape[-1])
        for block in self.blocks:
            hidden = block(hidden)
        velocity = self.output_projection(self.output_norm(hidden))
        output = velocity.reshape(
            shape[0],
            shape[1],
            shape[2],
            self.action_dim,
        )
        if self.endpoint_prediction and not endpoint_output:
            remaining = (1.0 - flow_time)[..., None, None].clamp_min(1e-4)
            return (output - latent) / remaining
        return output
    def loss(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flow_training_objective(self, target, context, group_id)[0]
    def sample(
        self,
        context: torch.Tensor,
        sample_count: int = 1,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        batch, future_count = context.shape[:2]
        expanded_context = _repeat_context(context, sample_count)
        latent = sample_flow_source(
            self,
            sample_count * batch,
            future_count,
            context.device,
            context.dtype,
        )
        if not stochastic:
            latent.zero_()
        step_size = 1.0 / self.steps
        for index in range(self.steps):
            flow_fraction = (
                index * step_size
                if self.endpoint_prediction
                else (index + 0.5) * step_size
            )
            flow_time = latent.new_full(
                latent.shape[:2],
                flow_fraction,
            )
            latent = latent + step_size * self(
                latent,
                flow_time,
                expanded_context,
            )
        return latent.reshape(
            sample_count,
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
class LatentActionModel(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.action_tokens = config.action_tokens
        self.multi_query_prior_context = config.multi_query_prior_context
        self.prior_query_residual = config.prior_query_residual
        if config.continuous_effect_action:
            self.posterior = ContinuousEffectPosterior(config)
        elif config.structured_action:
            self.posterior = ObjectDeltaActionPosterior(config)
        else:
            self.posterior = ActionPosterior(config)
        self.object_aligned_actions = config.object_aligned_actions
        self.prior_token_conditioner = (
            PriorTokenConditioner(config)
            if config.token_conditioned_prior
            else None
        )
        self.prior_history_input = nn.Linear(config.object_dim * 2, config.model_dim)
        if config.structured_action:
            self.prior_slot_input = nn.Linear(
                config.object_dim,
                config.model_dim,
            )
            self.prior_center_input = (
                nn.Linear(2, config.model_dim)
                if config.center_conditioned_posterior
                else None
            )
            self.prior_history_scale_input = (
                nn.Sequential(
                    nn.Linear(1, config.model_dim),
                    nn.SiLU(),
                    nn.Linear(config.model_dim, config.model_dim),
                )
                if config.temporal_prior_context
                else None
            )
            self.prior_query = nn.Parameter(
                torch.randn(config.model_dim) / config.model_dim**0.5
            )
            self.prior_attention = nn.MultiheadAttention(
                config.model_dim,
                config.heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.prior_norm = nn.LayerNorm(config.model_dim)
        else:
            self.prior_slot_input = None
            self.prior_center_input = None
            self.prior_history_scale_input = None
            self.prior_query = None
            self.prior_attention = None
            self.prior_norm = None
        self.prior_gap_input = nn.Sequential(
            nn.Linear(1, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.effect_head = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, config.object_dim),
        )
        self.center_effect_head = (
            nn.Sequential(
                nn.LayerNorm(
                    config.action_dim
                    if config.object_aligned_actions
                    else config.action_tokens * config.action_dim
                ),
                nn.Linear(
                    config.action_dim
                    if config.object_aligned_actions
                    else config.action_tokens * config.action_dim,
                    config.model_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model_dim,
                    2 if config.object_aligned_actions else config.object_slots * 2,
                ),
            )
            if config.center_conditioned_posterior
            else None
        )
        if config.mode_set_prior:
            self.prior = ModeSetActionPrior(config)
        elif config.flow_source_components > 1:
            self.prior = SourceLiftedJointFlowPrior(config)
        elif config.joint_flow or config.structured_action:
            self.prior = JointConditionalFlowPrior(config)
        else:
            self.prior = StructuredConditionalFlowPrior(config)
    def prior_context(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        future_scale: torch.Tensor,
        history_centers: torch.Tensor | None = None,
        history_scale: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        condition_tokens: torch.Tensor | None = None,
        condition_token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = weighted_slot_pool(
            history_slots.detach(),
            history_activity.detach(),
        )
        if self.prior_attention is None:
            summary = pooled.mean(dim=1)
            current = pooled[:, -1]
            history = self.prior_history_input(
                torch.cat((summary, current), dim=-1)
            )
            if condition is not None:
                if condition.shape != history.shape:
                    raise ValueError("condition must have shape [B,D]")
                history = history + condition
        else:
            slot_tokens = self.prior_slot_input(history_slots.detach())
            if self.prior_center_input is not None:
                if history_centers is None:
                    raise ValueError("center-conditioned prior requires centers")
                slot_tokens = (
                    slot_tokens
                    + self.prior_center_input(history_centers.detach())
                )
            if self.prior_history_scale_input is not None:
                if history_scale is None:
                    raise ValueError("temporal prior requires history_scale")
                slot_tokens = slot_tokens + self.prior_history_scale_input(
                    history_scale.detach()[..., None]
                )[:, :, None]
            current_queries = slot_tokens[:, -1]
            slot_tokens = slot_tokens.flatten(1, 2)
            if self.object_aligned_actions:
                query = (
                    current_queries
                    + self.prior_query[None, None]
                )
            elif self.multi_query_prior_context:
                identity = F.one_hot(
                    torch.arange(self.action_tokens, device=slot_tokens.device),
                    num_classes=slot_tokens.shape[-1],
                ).to(slot_tokens.dtype)
                query = (
                    self.prior_query[None, None] + identity[None]
                ).expand(slot_tokens.shape[0], -1, -1)
            else:
                query = self.prior_query[None, None].expand(
                    slot_tokens.shape[0],
                    1,
                    -1,
                )
            if self.prior_token_conditioner is not None:
                if condition_tokens is None or condition_token_valid is None:
                    raise ValueError(
                        "token-conditioned Prior requires instruction tokens"
                    )
                query = query + self.prior_token_conditioner(
                    query,
                    condition_tokens,
                    condition_token_valid,
                )
            if condition is not None:
                if condition.shape != (slot_tokens.shape[0], slot_tokens.shape[-1]):
                    raise ValueError("condition must have shape [B,D]")
                query = query + condition[:, None]
            attended = self.prior_attention(
                query,
                slot_tokens,
                slot_tokens,
                need_weights=False,
            )[0]
            history = (
                query + attended
                if self.prior_query_residual
                else attended
            )
            if not (
                self.object_aligned_actions
                or self.multi_query_prior_context
            ):
                history = history[:, 0]
            history = self.prior_norm(history)
        gap = self.prior_gap_input(future_scale[..., None])
        if self.object_aligned_actions or self.multi_query_prior_context:
            return history[:, None] + gap[:, :, None]
        return history[:, None] + gap
    def predict_effect(self, actions: torch.Tensor) -> torch.Tensor:
        return self.effect_head(actions.mean(dim=-2))
    def predict_object_effect(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.object_aligned_actions:
            raise ValueError("object effect requires object-aligned actions")
        return self.effect_head(actions)
    def predict_center_effect(self, actions: torch.Tensor) -> torch.Tensor:
        if self.center_effect_head is None:
            raise ValueError("center effect head is disabled")
        if self.object_aligned_actions:
            return self.center_effect_head(actions)
        shape = actions.shape[:2]
        return self.center_effect_head(actions.flatten(-2)).reshape(
            *shape,
            -1,
            2,
        )
    def prior_condition_parameters(self) -> list[nn.Parameter]:
        parameters = [
            *self.prior_history_input.parameters(),
            *self.prior_gap_input.parameters(),
        ]
        if self.prior_slot_input is not None:
            parameters.extend(self.prior_slot_input.parameters())
            if self.prior_center_input is not None:
                parameters.extend(self.prior_center_input.parameters())
            if self.prior_history_scale_input is not None:
                parameters.extend(self.prior_history_scale_input.parameters())
            parameters.extend(self.prior_attention.parameters())
            parameters.extend(self.prior_norm.parameters())
            parameters.append(self.prior_query)
        if self.prior_token_conditioner is not None:
            parameters.extend(self.prior_token_conditioner.parameters())
        return parameters
