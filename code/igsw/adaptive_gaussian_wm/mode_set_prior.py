"""History-conditioned set prediction for discrete future hypotheses."""
from __future__ import annotations

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


def ordered_group_responsibility(
    target_centers: torch.Tensor,
    current_centers: torch.Tensor,
    components: int,
    group_id: torch.Tensor | None,
) -> torch.Tensor:
    """Rank same-context futures along an oriented principal motion axis."""
    if group_id is None:
        raise ValueError("ordered mode-set assignment requires groups")
    _, counts = torch.unique_consecutive(
        group_id.flatten(),
        return_counts=True,
    )
    if not bool((counts == components).all()):
        raise ValueError("ordered mode-set assignment requires complete groups")
    groups = counts.numel()
    center_effect = (
        target_centers - current_centers[:, None]
    ).reshape(groups, components, -1)
    centered_effect = center_effect - center_effect.mean(dim=1, keepdim=True)
    principal = torch.linalg.svd(
        centered_effect.detach(),
        full_matrices=False,
    ).Vh[:, 0]
    anchor = principal.abs().argmax(dim=-1, keepdim=True)
    orientation = torch.gather(principal, 1, anchor).sign()
    orientation = torch.where(
        orientation == 0,
        torch.ones_like(orientation),
        orientation,
    )
    principal = principal * orientation
    score = torch.einsum(
        "gcd,gd->gc",
        centered_effect.detach(),
        principal,
    )
    component = score.argsort(dim=-1).argsort(dim=-1)
    return F.one_hot(
        component,
        num_classes=components,
    ).to(target_centers.dtype).reshape(-1, components)


class ModeSetActionPrior(nn.Module):
    """Predict a calibrated finite set of latent-action hypotheses."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.action_tokens = config.action_tokens
        self.action_dim = config.action_dim
        self.components = config.flow_source_components
        self.assignment_temperature = config.flow_assignment_temperature
        self.responsibility_floor = config.flow_responsibility_floor
        self.fit_weight = config.flow_source_fit_weight
        self.geometry_weight = config.mode_set_geometry_weight
        self.ordered_assignment = config.mode_set_ordered_assignment
        self.normalize_prototypes = config.mode_set_normalize_prototypes
        self.transformer_enabled = config.mode_set_transformer
        self.global_codebook = config.mode_set_global_codebook
        self.balanced_assignment = config.flow_balanced_source_assignment
        hidden = config.flow_hidden_dim
        self.context_input = nn.Linear(config.model_dim, hidden)
        self.mode_handles = nn.Parameter(
            torch.randn(self.components, hidden) / hidden**0.5
        )
        self.action_handles = nn.Parameter(
            torch.randn(config.action_tokens, hidden) / hidden**0.5
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=config.heads,
                    dim_feedforward=hidden * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.flow_layers)
            ]
            if self.transformer_enabled
            else []
        )
        self.prototype = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * 2, config.action_dim),
        )
        self.mode_logits = nn.Linear(config.model_dim, self.components)
        code_assignment_dim = (
            6
            if config.canonical_semantic_action
            else 3 if config.canonical_center_action else config.action_dim
        )
        self.code_assignment_dim = code_assignment_dim
        self.code_assignment = (
            nn.Sequential(
                nn.LayerNorm(code_assignment_dim),
                nn.Linear(code_assignment_dim, hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden, self.components),
            )
            if self.global_codebook
            else None
        )

    def code_assignment_logits(
        self,
        posterior_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.code_assignment is None:
            raise ValueError("code assignment requires a global codebook")
        if posterior_actions.ndim != 4:
            raise ValueError(
                "posterior actions must have shape [B,Q,A,D]"
            )
        canonical = posterior_actions[..., : self.code_assignment_dim]
        return self.code_assignment(canonical.mean(dim=(1, 2)))

    def _distribution(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_context = _action_context(context, self.action_tokens)
        if self.global_codebook:
            hidden = (
                self.mode_handles[None, :, None, None]
                + self.action_handles[None, None, None]
            ).expand(
                action_context.shape[0],
                -1,
                action_context.shape[1],
                -1,
                -1,
            )
        else:
            hidden = (
                self.context_input(action_context)[:, None]
                + self.mode_handles[None, :, None, None]
                + self.action_handles[None, None, None]
            )
        if self.transformer_enabled:
            shape = hidden.shape
            hidden = hidden.flatten(1, 3)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = hidden.reshape(shape)
        prototypes = self.prototype(hidden)
        if self.normalize_prototypes:
            prototypes = F.layer_norm(
                prototypes,
                (self.action_dim,),
            )
        logits = self.mode_logits(action_context.mean(dim=(1, 2)))
        return logits, prototypes

    def _responsibilities(
        self,
        cost: torch.Tensor,
        logits: torch.Tensor,
        group_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.balanced_assignment and group_id is not None:
            ids, counts = torch.unique_consecutive(
                group_id.flatten(),
                return_counts=True,
            )
            if ids.numel() != torch.unique(ids).numel() or not bool(
                (counts == self.components).all()
            ):
                raise ValueError(
                    "balanced mode-set assignment requires complete groups"
                )
            log_assignment = (
                -cost.detach().reshape(
                    ids.numel(),
                    self.components,
                    self.components,
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
            return F.softmax(log_assignment, dim=2).reshape_as(cost)
        return F.softmax(
            F.log_softmax(logits, dim=-1)
            - cost.detach() / self.assignment_temperature,
            dim=-1,
        )

    def training_objective(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        group_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, prototypes = self._distribution(context)
        cost = (prototypes - target[:, None]).square().mean(dim=(2, 3, 4))
        responsibility = self._responsibilities(cost, logits, group_id)
        weights = (
            (1.0 - self.responsibility_floor) * responsibility
            + self.responsibility_floor / self.components
        )
        reconstruction = (weights.detach() * cost).sum(dim=-1).mean()
        logit_fit = -(
            responsibility.detach() * F.log_softmax(logits, dim=-1)
        ).sum(dim=-1).mean()
        usage = responsibility.mean(dim=0)
        usage_loss = (
            usage - usage.new_full(usage.shape, 1.0 / self.components)
        ).square().mean()
        loss = reconstruction + self.fit_weight * (logit_fit + usage_loss)
        assignment = F.one_hot(
            responsibility.argmax(dim=-1),
            num_classes=self.components,
        ).to(prototypes.dtype)
        endpoint = (
            assignment[..., None, None, None] * prototypes
        ).sum(dim=1)
        return loss, endpoint, target.detach()

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
        logits, prototypes = self._distribution(context)
        if stochastic:
            probability = F.softmax(logits, dim=-1)
            offset = torch.rand(
                probability.shape[0],
                device=probability.device,
                dtype=probability.dtype,
            )
            positions = (
                torch.arange(
                    sample_count,
                    device=probability.device,
                    dtype=probability.dtype,
                )[:, None]
                + offset[None]
            ) / sample_count
            cumulative = probability.cumsum(dim=-1)
            component = (
                positions[..., None] > cumulative[None]
            ).sum(dim=-1).clamp_max(self.components - 1)
        else:
            component = logits.argmax(dim=-1)[None].expand(sample_count, -1)
        gather = component[..., None, None, None, None].expand(
            sample_count,
            prototypes.shape[0],
            1,
            *prototypes.shape[2:],
        )
        return torch.gather(
            prototypes[None].expand(sample_count, *prototypes.shape),
            2,
            gather,
        ).squeeze(2)
