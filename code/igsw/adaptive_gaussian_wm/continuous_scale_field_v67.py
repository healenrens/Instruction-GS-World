"""Deployable RGB-only continuous multi-scale feature field for v67."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .continuous_field_sampling_v67 import sample_native_multiscale_crops_v67


@dataclass(frozen=True)
class ContinuousFieldSamplesV67:
    features: torch.Tensor
    coordinates: torch.Tensor
    scales: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CausalFieldHistoryV67:
    sequence: torch.Tensor
    current: torch.Tensor
    valid: torch.Tensor


def fourier_features_v67(value: torch.Tensor, bands: int) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(
        bands, device=value.device, dtype=value.dtype
    )
    phase = value[..., None] * frequencies
    encoded = torch.cat((value[..., None], phase.sin(), phase.cos()), dim=-1)
    return encoded.flatten(-2)


class ContinuousScaleFieldEncoderV67(nn.Module):
    """Map native RGB neighborhoods at arbitrary coordinates/scales to a field."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config.local_channels
        dim = config.field_dim
        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        scalar_width = 1 + 2 * config.fourier_bands
        coordinate_width = 2 * scalar_width
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(coordinate_width, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.scale_embedding = nn.Sequential(
            nn.Linear(scalar_width, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(scalar_width, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.scale_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        video_rgb: torch.Tensor,
        video_pixel_valid: torch.Tensor,
        native_hw: torch.Tensor,
        coordinates: torch.Tensor,
        scales: torch.Tensor,
        frame_times: torch.Tensor,
    ) -> ContinuousFieldSamplesV67:
        crops = sample_native_multiscale_crops_v67(
            video_rgb,
            video_pixel_valid,
            native_hw,
            coordinates,
            scales,
            self.config.scale_multipliers,
            self.config.crop_side,
        )
        batch, frames, points, levels = crops.rgb.shape[:4]
        flat = crops.rgb.reshape(
            batch * frames * points * levels,
            3,
            self.config.crop_side,
            self.config.crop_side,
        )
        encoded = self.local_encoder(flat)
        encoded = encoded.flatten(1).reshape(
            batch, frames, points, levels, self.config.field_dim
        )
        log_scale = crops.effective_scales.clamp(
            self.config.minimum_scale, self.config.maximum_scale
        ).log()
        scale_input = fourier_features_v67(log_scale[..., None], self.config.fourier_bands)
        scale_feature = self.scale_embedding(
            scale_input.to(dtype=self.scale_embedding[0].weight.dtype)
        )
        level_feature = encoded + scale_feature
        score = self.scale_score(level_feature)[..., 0].float()
        level_valid = crops.valid_fraction > 0.05
        score = score.masked_fill(~level_valid, -30.0)
        weight = score.softmax(dim=3) * level_valid.float()
        weight = weight / weight.sum(dim=3, keepdim=True).clamp_min(1e-6)
        mixed = (level_feature.float() * weight[..., None]).sum(dim=3)

        coordinate_input = fourier_features_v67(
            coordinates.float(), self.config.fourier_bands
        )
        coordinate_feature = self.coordinate_embedding(
            coordinate_input.to(dtype=self.coordinate_embedding[0].weight.dtype)
        )
        relative_time = frame_times.float() - frame_times[:, -1:]
        time_input = fourier_features_v67(
            relative_time[..., None], self.config.fourier_bands
        )
        time_feature = self.time_embedding(
            time_input.to(dtype=self.time_embedding[0].weight.dtype)
        )
        hidden = mixed.to(coordinate_feature.dtype) + coordinate_feature
        hidden = hidden + time_feature[:, :, None]
        hidden = hidden + self.output(hidden)
        valid = level_valid.any(dim=3)
        hidden = hidden * valid[..., None]
        return ContinuousFieldSamplesV67(
            features=hidden,
            coordinates=coordinates.float(),
            scales=scales.float(),
            valid=valid,
        )


class CausalSpatiotemporalFieldMixerV67(nn.Module):
    """Propagate a queryable field through observed history without future input."""

    def __init__(self, config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            config.field_dim,
            config.temporal_heads,
            dim_feedforward=config.field_dim * config.temporal_ffn_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, config.temporal_layers)
        self.output_norm = nn.LayerNorm(config.field_dim)

    def forward(self, samples: ContinuousFieldSamplesV67) -> CausalFieldHistoryV67:
        batch, frames, points, dim = samples.features.shape
        tokens = samples.features.reshape(batch, frames * points, dim)
        valid = samples.valid.reshape(batch, frames * points)
        time_index = torch.arange(frames, device=tokens.device).repeat_interleave(points)
        causal_mask = time_index[None] > time_index[:, None]
        mixed = self.layers(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=~valid,
        )
        mixed = self.output_norm(mixed).reshape(batch, frames, points, dim)
        return CausalFieldHistoryV67(
            sequence=mixed,
            current=mixed[:, -1],
            valid=samples.valid,
        )
