"""Causal spatial-temporal transformer over adaptive GPSToken regions."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig


class CausalRegionTransformer(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.region_dim
        if dim != config.token_dim:
            raise ValueError("v43 region and GPSToken dimensions must match")
        spatial_layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        spatial_layers = max(1, config.region_temporal_layers // 2)
        self.spatial = nn.TransformerEncoder(spatial_layer, spatial_layers)
        self.temporal = nn.TransformerEncoder(
            temporal_layer, config.region_temporal_layers
        )
        self.projected_input = nn.Linear(config.region_dim, dim)
        self.geometry_input = nn.Sequential(
            nn.Linear(7, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_input = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.mask_predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )

    @staticmethod
    def _time_features(times: torch.Tensor) -> torch.Tensor:
        return torch.stack((times, times.abs(), torch.tanh(times)), dim=-1)

    @staticmethod
    def _covariance_features(covariance: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                covariance[..., 0, 0],
                covariance[..., 0, 1],
                covariance[..., 1, 1],
            ),
            dim=-1,
        )

    def prepare_inputs(
        self,
        latent: torch.Tensor,
        projected: torch.Tensor,
        center: torch.Tensor,
        covariance: torch.Tensor,
        activation: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError("region latent must have shape [B,T,R,D]")
        if projected.shape != latent.shape:
            raise ValueError("projected region features must match latent shape")
        geometry = torch.cat(
            (
                center,
                self._covariance_features(covariance),
                activation[..., None],
                torch.logit(activation.clamp(1e-4, 1.0 - 1e-4))[..., None],
            ),
            dim=-1,
        )
        tokens = (
            latent
            + self.projected_input(projected)
            + self.geometry_input(geometry)
            + self.time_input(self._time_features(times))[:, :, None]
        )
        return self.input_norm(tokens)

    def _run(
        self,
        tokens: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, regions, dim = tokens.shape
        spatial = self.spatial(
            tokens.reshape(batch * frames, regions, dim),
            src_key_padding_mask=~active.reshape(batch * frames, regions),
        ).reshape(batch, frames, regions, dim)
        spatial = torch.where(active[..., None], spatial, torch.zeros_like(spatial))
        temporal = spatial.permute(0, 2, 1, 3).reshape(
            batch * regions, frames, dim
        )
        temporal_active = active.permute(0, 2, 1).reshape(
            batch * regions, frames
        )
        safe_temporal_active = temporal_active.clone()
        # A zero sentinel at t=0 prevents all-masked causal attention rows.
        safe_temporal_active[:, 0] = True
        causal_mask = torch.triu(
            torch.ones(frames, frames, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        contextual = self.temporal(
            temporal,
            mask=causal_mask,
            src_key_padding_mask=~safe_temporal_active,
        )
        contextual = torch.where(
            temporal_active[..., None],
            contextual,
            torch.zeros_like(contextual),
        )
        contextual = contextual.reshape(batch, regions, frames, dim).permute(
            0, 2, 1, 3
        )
        contextual = self.output_norm(contextual)
        return torch.where(
            active[..., None], contextual, torch.zeros_like(contextual)
        )

    def forward(
        self,
        latent: torch.Tensor,
        projected: torch.Tensor,
        center: torch.Tensor,
        covariance: torch.Tensor,
        activation: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.prepare_inputs(
            latent, projected, center, covariance, activation, times
        )
        active = activation > 0.5
        return self._run(inputs, active), inputs

    def masked_prediction(
        self,
        inputs: torch.Tensor,
        activation: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 < ratio < 1.0:
            raise ValueError("masked-region ratio must be in (0, 1)")
        active = activation > 0.5
        mask = (torch.rand_like(activation) < ratio) & active
        missing = ~mask.flatten(1).any(dim=1)
        if bool(missing.any()):
            first_active = active[missing].float().flatten(1).argmax(dim=1)
            flat = mask[missing].flatten(1)
            flat.scatter_(1, first_active[:, None], True)
            mask[missing] = flat.reshape_as(mask[missing])
        masked = torch.where(mask[..., None], self.mask_token, inputs)
        return self.mask_predictor(self._run(masked, active)), mask
