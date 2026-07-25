"""State-conditioned source-lifted flow prior for multimodal latent actions."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig


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


class SourceLiftedJointFlowPrior(nn.Module):
    """One shared flow over action coordinates and orthogonal source handles."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.steps = config.flow_steps
        self.mix_tokens = config.joint_flow
        self.endpoint_prediction = config.flow_endpoint_prediction
        self.source_components = config.flow_source_components
        self.lift_scale = config.flow_lift_scale
        self.responsibility_floor = config.flow_responsibility_floor
        self.source_fit_weight = config.flow_source_fit_weight
        self.source_min_scale = config.flow_source_min_scale
        self.sample_noise_scale = 1.0
        self.balanced_assignment = config.flow_balanced_source_assignment
        self.assignment_temperature = config.flow_assignment_temperature
        self.state_dim = self.action_dim + self.source_components
        hidden = config.flow_hidden_dim

        self.action_identity = nn.Parameter(
            torch.randn(config.action_tokens, config.action_dim)
            / config.action_dim**0.5
        )
        self.input_projection = nn.Linear(
            self.state_dim + config.action_dim + config.model_dim + 5,
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
        self.output_projection = nn.Linear(hidden, self.state_dim)

        self.source_context = nn.Linear(config.model_dim, hidden)
        self.source_handles = nn.Parameter(
            torch.randn(self.source_components, hidden) / hidden**0.5
        )
        self.source_norm = nn.LayerNorm(hidden)
        self.source_output = nn.Linear(hidden, 2 * self.action_dim)
        self.source_logits = nn.Linear(config.model_dim, self.source_components)
        nn.init.normal_(self.source_output.weight, std=1e-3)
        nn.init.zeros_(self.source_output.bias)
        nn.init.zeros_(self.source_logits.weight)
        nn.init.zeros_(self.source_logits.bias)
        initial_scale = max(0.25 - self.source_min_scale, 1e-4)
        inverse_softplus = math.log(math.expm1(initial_scale))
        with torch.no_grad():
            self.source_output.bias[self.action_dim :].fill_(
                inverse_softplus
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

    def _source_distribution(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_context = _action_context(context, self.action_tokens)
        hidden = (
            self.source_context(action_context)[:, None]
            + self.source_handles[None, :, None, None]
        )
        parameters = self.source_output(self.source_norm(F.silu(hidden)))
        mean, raw_scale = parameters.split(self.action_dim, dim=-1)
        scale = self.source_min_scale + F.softplus(raw_scale)
        pooled = action_context.mean(dim=(1, 2))
        logits = self.source_logits(pooled)
        return logits, mean, scale

    def _lift_anchors(
        self,
        component: torch.Tensor,
        future_count: int,
    ) -> torch.Tensor:
        anchor = F.one_hot(
            component,
            num_classes=self.source_components,
        ).to(dtype=self.action_identity.dtype)
        return (
            self.lift_scale
            * anchor[..., None, None, :].expand(
                *anchor.shape[:-1],
                future_count,
                self.action_tokens,
                self.source_components,
            )
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
        if latent.shape[-1] != self.state_dim:
            raise ValueError("lifted latent has the wrong state dimension")
        if flow_time.shape != latent.shape[:2]:
            raise ValueError("flow_time must have shape [B,Q]")
        identity = self.action_identity[None, None].expand(
            latent.shape[0],
            latent.shape[1],
            -1,
            -1,
        )
        expanded_context = _action_context(context, self.action_tokens)
        time_features = self._flow_time_features(
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
        output = self.output_projection(self.output_norm(hidden)).reshape(
            shape[0],
            shape[1],
            shape[2],
            self.state_dim,
        )
        if self.endpoint_prediction and not endpoint_output:
            remaining = (1.0 - flow_time)[..., None, None].clamp_min(1e-4)
            return (output - latent) / remaining
        return output

    def _responsibilities(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference = (target[:, None] - mean) / scale
        log_probability = -0.5 * (
            difference.square()
            + 2.0 * scale.log()
            + math.log(2.0 * math.pi)
        ).sum(dim=(2, 3, 4))
        log_joint = F.log_softmax(logits, dim=-1) + log_probability
        dimensions = target.shape[1] * target.shape[2] * target.shape[3]
        if self.balanced_assignment and group_id is not None:
            ids, counts = torch.unique_consecutive(
                group_id.flatten(),
                return_counts=True,
            )
            if ids.numel() != torch.unique(ids).numel() or not bool(
                (counts == self.source_components).all()
            ):
                raise ValueError(
                    "balanced source assignment requires complete groups"
                )
            log_assignment = (
                log_probability.detach().reshape(
                    ids.numel(),
                    self.source_components,
                    self.source_components,
                )
                / self.assignment_temperature
            )
            for _ in range(8):
                log_assignment = log_assignment - torch.logsumexp(
                    log_assignment,
                    dim=2,
                    keepdim=True,
                )
                log_assignment = log_assignment - torch.logsumexp(
                    log_assignment,
                    dim=1,
                    keepdim=True,
                )
            responsibility = F.softmax(log_assignment, dim=2).reshape_as(
                log_probability
            )
            action_fit = -(
                responsibility.detach() * log_probability
            ).sum(dim=-1).mean() / dimensions
            logit_fit = -(
                responsibility.detach() * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1).mean()
            source_fit = action_fit + logit_fit
        else:
            responsibility = F.softmax(log_joint, dim=-1)
            source_fit = (
                -torch.logsumexp(log_joint, dim=-1).mean() / dimensions
            )
        return responsibility, source_fit

    def training_objective(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, future_count = target.shape[:2]
        logits, mean, scale = self._source_distribution(context)
        responsibility, source_fit = self._responsibilities(
            target,
            logits,
            mean,
            scale,
            group_id,
        )
        source_action = mean + scale * torch.randn_like(mean)
        component = torch.arange(
            self.source_components,
            device=target.device,
        )[None].expand(batch, -1)
        lift = self._lift_anchors(component, future_count)
        source = torch.cat((source_action, lift), dim=-1)
        target_lift = target.new_zeros(
            batch,
            self.source_components,
            future_count,
            self.action_tokens,
            self.source_components,
        )
        lifted_target = torch.cat(
            (
                target[:, None].expand(-1, self.source_components, -1, -1, -1),
                target_lift,
            ),
            dim=-1,
        )
        source = source.reshape(
            batch * self.source_components,
            future_count,
            self.action_tokens,
            self.state_dim,
        )
        lifted_target = lifted_target.reshape_as(source)
        expanded_context = context[:, None].expand(
            batch,
            self.source_components,
            *context.shape[1:],
        ).reshape(
            batch * self.source_components,
            *context.shape[1:],
        )
        flow_time = torch.rand(
            batch * self.source_components,
            1,
            device=target.device,
            dtype=target.dtype,
        ).expand(-1, future_count)
        interpolation = flow_time[..., None, None]
        latent = (
            (1.0 - interpolation) * source
            + interpolation * lifted_target
        )
        target_velocity = lifted_target - source
        prediction = self(
            latent,
            flow_time,
            expanded_context,
            self.endpoint_prediction,
        )
        training_target = (
            lifted_target if self.endpoint_prediction else target_velocity
        )
        endpoint = (
            prediction
            if self.endpoint_prediction
            else latent + (1.0 - interpolation) * prediction
        )
        per_component = (prediction - training_target).square().mean(
            dim=(1, 2, 3)
        ).reshape(batch, self.source_components)
        weights = (
            (1.0 - self.responsibility_floor) * responsibility
            + self.responsibility_floor / self.source_components
        )
        flow = (weights.detach() * per_component).sum(dim=-1).mean()
        loss = flow + self.source_fit_weight * source_fit
        endpoint_action = endpoint[..., : self.action_dim].reshape(
            batch,
            self.source_components,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
        weighted_endpoint = (
            weights[..., None, None, None] * endpoint_action
        ).sum(dim=1)
        return loss, weighted_endpoint, target.detach()

    def loss(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.training_objective(target, context, group_id)[0]

    def sample(
        self,
        context: torch.Tensor,
        sample_count: int = 1,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        batch, future_count = context.shape[:2]
        logits, mean, scale = self._source_distribution(context)
        if stochastic:
            component = torch.multinomial(
                F.softmax(logits, dim=-1),
                sample_count,
                replacement=True,
            ).transpose(0, 1)
        else:
            component = logits.argmax(dim=-1)[None].expand(sample_count, -1)
        gather = component[..., None, None, None, None].expand(
            sample_count,
            batch,
            1,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
        expanded_mean = mean[None].expand(sample_count, -1, -1, -1, -1, -1)
        expanded_scale = scale[None].expand_as(expanded_mean)
        selected_mean = torch.gather(expanded_mean, 2, gather).squeeze(2)
        selected_scale = torch.gather(expanded_scale, 2, gather).squeeze(2)
        noise = (
            self.sample_noise_scale * torch.randn_like(selected_mean)
            if stochastic
            else torch.zeros_like(selected_mean)
        )
        source_action = selected_mean + selected_scale * noise
        lift = self._lift_anchors(component, future_count).to(context.dtype)
        latent = torch.cat((source_action, lift), dim=-1).reshape(
            sample_count * batch,
            future_count,
            self.action_tokens,
            self.state_dim,
        )
        expanded_context = _repeat_context(context, sample_count)
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
        return latent[..., : self.action_dim].reshape(
            sample_count,
            batch,
            future_count,
            self.action_tokens,
            self.action_dim,
        )
