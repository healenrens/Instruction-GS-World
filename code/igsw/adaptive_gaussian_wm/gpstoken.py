"""Learnable current-only allocation of dense features to micro Gaussian tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig


@dataclass
class GPSTokenState:
    latent: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    depth_order: torch.Tensor
    opacity: torch.Tensor
    activation: torch.Tensor
    assignment: torch.Tensor
    occupancy: torch.Tensor
    decoded_features: torch.Tensor
    reconstructed_features: torch.Tensor
    activation_logits: torch.Tensor
    density_mode: str
    fixed_token_fraction: float
    active_count: torch.Tensor
    budget_fraction: torch.Tensor
    budget_logit: torch.Tensor
    hard_token_gate: bool


class LearnableGPSTokenAllocator(nn.Module):
    """Soft partition of a feature grid into an adaptive set of 2D Gaussians."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.token_dim
        count = config.max_micro_tokens
        self.config = config
        self.feature_input = nn.Sequential(
            nn.Linear(config.feature_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.coordinate_input = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.queries = nn.Parameter(torch.randn(count, dim) / dim**0.5)
        side = math.ceil(count**0.5)
        axis = torch.linspace(-0.85, 0.85, side)
        seed_y, seed_x = torch.meshgrid(axis, axis, indexing="ij")
        seeds = torch.stack((seed_x, seed_y), dim=-1).reshape(-1, 2)[:count]
        self.spatial_seeds = nn.Parameter(seeds)
        self.spatial_precision = nn.Parameter(torch.full((count,), 4.0))
        self.center_offset = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )
        self.context_to_query = nn.Linear(dim, dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim)
        self.token_norm = nn.LayerNorm(dim)
        self.activation_head = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        self.budget_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.depth_head = nn.Linear(dim, 1)
        self.opacity_head = nn.Linear(dim, 1)
        self.feature_decoder = nn.Linear(dim, config.feature_dim)
        nn.init.zeros_(self.feature_decoder.weight)
        nn.init.zeros_(self.feature_decoder.bias)

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid.to(value.dtype)[..., None]
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _fixed_budget_activation(self, logits: torch.Tensor) -> torch.Tensor:
        target = logits.new_full(
            (logits.shape[0], 1, 1),
            self.config.max_micro_tokens * self.config.fixed_token_fraction,
        )
        detached = logits.detach()
        lower = detached.amin(dim=1, keepdim=True) - 20.0
        upper = detached.amax(dim=1, keepdim=True) + 20.0
        for _ in range(24):
            threshold = 0.5 * (lower + upper)
            count = torch.sigmoid(detached - threshold).sum(dim=1, keepdim=True)
            lower = torch.where(count > target, threshold, lower)
            upper = torch.where(count > target, upper, threshold)
        return torch.sigmoid(logits - 0.5 * (lower + upper))

    def _hard_adaptive_activation(
        self,
        logits: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        soft = torch.sigmoid(logits)
        budget_logit = self.budget_head(context).squeeze(-1)
        budget_fraction = torch.sigmoid(0.25 * budget_logit)
        available = self.config.max_micro_tokens - self.config.min_active_tokens
        continuous_count = (
            self.config.min_active_tokens + available * budget_fraction
        )
        active_count = (
            self.config.min_active_tokens
            + torch.round(available * budget_fraction.detach()).long()
        ).clamp(
            self.config.min_active_tokens,
            self.config.max_micro_tokens,
        )
        scores = logits.squeeze(-1)
        order = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        indices = torch.arange(
            scores.shape[1],
            device=scores.device,
        )[None].expand_as(order)
        ranks.scatter_(1, order, indices)
        hard = (ranks < active_count[:, None]).to(soft.dtype)[..., None]
        if self.config.object_region_memory:
            rank_gate = torch.sigmoid(
                continuous_count[:, None] - ranks.to(continuous_count.dtype) - 0.5
            )[..., None]
            surrogate = soft * rank_gate
            activation = hard.detach() + surrogate - surrogate.detach()
        else:
            activation = hard.detach() + soft - soft.detach()
        return activation, active_count, budget_fraction, budget_logit

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> GPSTokenState:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,N,C]")
        if coordinates.shape != (*features.shape[:2], 2):
            raise ValueError("coordinates must have shape [B,N,2]")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("each sample must contain at least one valid grid position")

        grid = self.feature_input(features) + self.coordinate_input(coordinates)
        context = self._masked_mean(grid, valid_mask)
        queries = self.queries[None] + self.context_to_query(context)[:, None]
        query_centers = (
            self.spatial_seeds[None]
            + 0.5 * torch.tanh(self.center_offset(queries))
        ).clamp(-1.0, 1.0)
        logits = torch.einsum(
            "bmd,bnd->bmn",
            self.query_projection(queries),
            self.key_projection(grid),
        ) / self.config.token_dim**0.5
        spatial_distance = (
            coordinates[:, None] - query_centers[:, :, None]
        ).square().sum(dim=-1)
        precision = (
            self.config.token_spatial_precision_floor
            + torch.nn.functional.softplus(self.spatial_precision)
        )[None, :, None]
        logits = logits - precision * spatial_distance
        logits = logits.masked_fill(
            ~valid_mask[:, None],
            torch.finfo(logits.dtype).min,
        )
        assignment = logits.softmax(dim=1)
        assignment = assignment * valid_mask[:, None].to(assignment.dtype)
        mass = assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        occupancy = mass / valid_mask.sum(dim=-1, keepdim=True)[:, None].clamp_min(1)
        normalized = assignment / mass

        values = self.value_projection(grid)
        pooled = torch.einsum("bmn,bnd->bmd", normalized, values)
        pooled_features = torch.einsum(
            "bmn,bnc->bmc",
            normalized,
            features,
        )
        latent = self.token_norm(pooled + queries)
        dispersion = torch.einsum(
            "bmn,bmnd->bmd",
            normalized,
            (values[:, None] - pooled[:, :, None]).square(),
        ).mean(dim=-1, keepdim=True)
        center = torch.einsum("bmn,bnd->bmd", normalized, coordinates)
        difference = coordinates[:, None] - center[:, :, None]
        covariance = torch.einsum(
            "bmn,bmni,bmnj->bmij",
            normalized,
            difference,
            difference,
        )
        identity = torch.eye(
            2,
            device=covariance.device,
            dtype=covariance.dtype,
        )
        covariance = covariance + self.config.covariance_floor * identity

        activation_logits = self.activation_head(
            torch.cat((latent, occupancy, dispersion), dim=-1)
        )
        if self.config.density_mode == "fixed":
            activation = self._fixed_budget_activation(activation_logits)
            active_count = activation.detach().sum(dim=1).squeeze(-1)
            budget_fraction = activation.mean(dim=1).squeeze(-1)
            budget_logit = torch.logit(
                budget_fraction.float().clamp(1e-5, 1.0 - 1e-5)
            )
        elif self.config.hard_token_gate:
            activation, active_count, budget_fraction, budget_logit = (
                self._hard_adaptive_activation(activation_logits, context)
            )
        else:
            activation = torch.sigmoid(activation_logits)
            active_count = activation.detach().sum(dim=1).squeeze(-1)
            budget_fraction = activation.mean(dim=1).squeeze(-1)
            budget_logit = torch.logit(
                budget_fraction.float().clamp(1e-5, 1.0 - 1e-5)
            )
        opacity = torch.sigmoid(self.opacity_head(latent))
        depth_order = torch.tanh(self.depth_head(latent))
        decoded = pooled_features + self.feature_decoder(latent)
        active_assignment = assignment * activation
        coverage = active_assignment.sum(dim=1).clamp(0.0, 1.0)
        if self.config.density_mode == "legacy":
            reconstruction = torch.einsum(
                "bmn,bmc->bnc",
                active_assignment / coverage[:, None].clamp_min(1e-6),
                decoded,
            )
        else:
            active_reconstruction = torch.einsum(
                "bmn,bmc->bnc",
                active_assignment,
                decoded,
            )
            background = self._masked_mean(features, valid_mask)[:, None]
            reconstruction = (
                active_reconstruction
                + (1.0 - coverage)[..., None] * background
            )
        return GPSTokenState(
            latent=latent,
            center=center,
            covariance=covariance,
            depth_order=depth_order,
            opacity=opacity,
            activation=activation,
            assignment=assignment,
            occupancy=occupancy,
            decoded_features=decoded,
            reconstructed_features=reconstruction,
            activation_logits=activation_logits,
            density_mode=self.config.density_mode,
            fixed_token_fraction=self.config.fixed_token_fraction,
            active_count=active_count,
            budget_fraction=budget_fraction,
            budget_logit=budget_logit,
            hard_token_gate=self.config.hard_token_gate,
        )
