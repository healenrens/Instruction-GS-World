"""Conditional flow-matching prior for continuous global latent actions."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowPrior(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int, steps: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.velocity = nn.Sequential(
            nn.Linear(latent_dim + context_dim + 5, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, latent_dim),
        )

    @staticmethod
    def time_features(time: torch.Tensor) -> torch.Tensor:
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

    def forward(self, latent: torch.Tensor, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.velocity(torch.cat((latent, context, self.time_features(time)), dim=-1))

    def loss(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        source = torch.randn_like(target)
        time = torch.rand(len(target), device=target.device, dtype=target.dtype)
        interpolated = (1.0 - time[:, None]) * source + time[:, None] * target
        velocity_target = target - source
        return F.mse_loss(self(interpolated, time, context), velocity_target)

    def sample(self, context: torch.Tensor, stochastic: bool) -> torch.Tensor:
        latent = torch.randn(
            len(context),
            self.latent_dim,
            device=context.device,
            dtype=context.dtype,
        )
        if not stochastic:
            latent.zero_()
        step_size = 1.0 / self.steps
        for index in range(self.steps):
            time = latent.new_full((len(latent),), (index + 0.5) * step_size)
            latent = latent + step_size * self(latent, time, context)
        return latent
