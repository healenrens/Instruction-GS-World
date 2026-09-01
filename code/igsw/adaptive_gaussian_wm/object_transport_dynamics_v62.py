"""Soft-transport plus residual object Dynamics for v62 E1."""

from __future__ import annotations

import torch
import torch.nn as nn

from .object_effect_posterior_v62 import state_tokens_without_identity_v62
from .teacher_object_codec_v62 import TeacherObjectStateV62


class ObjectTransportDynamicsV62(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.state_dim
        self.state_input = nn.Sequential(nn.LayerNorm(dim + 8), nn.Linear(dim + 8, dim))
        self.effect_input = nn.Linear(config.effect_dim, dim)
        self.time_input = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.target_queries = nn.Parameter(
            torch.randn(config.carrier_count, dim) * 0.02
        )
        self.transport_query = nn.Linear(dim, dim, bias=False)
        self.transport_key = nn.Linear(dim, dim, bias=False)
        self.effect_attention = nn.MultiheadAttention(
            dim, config.effect_heads, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            dim,
            config.effect_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refine = nn.TransformerEncoder(layer, num_layers=3)
        self.feature_residual = nn.Linear(dim, dim)
        self.center_residual = nn.Linear(dim, 2)
        self.covariance_residual = nn.Linear(dim, 3)
        self.presence_residual = nn.Linear(dim, 1)
        self.visibility_residual = nn.Linear(dim, 1)
        self.lifecycle_residual = nn.Linear(dim, 3)

    @staticmethod
    def _probability(source, residual):
        source_logits = torch.logit(source.float().clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(source_logits + residual.float())

    def _covariance(self, source, residual):
        source = source.float()
        diagonal = source.diagonal(dim1=-2, dim2=-1).clamp_min(1e-5).sqrt()
        diagonal = diagonal * torch.exp(0.25 * torch.tanh(residual[..., :2].float()))
        correlation = (
            source[..., 0, 1]
            / (source[..., 0, 0] * source[..., 1, 1]).clamp_min(1e-8).sqrt()
        )
        correlation = torch.tanh(
            torch.atanh(correlation.clamp(-0.95, 0.95))
            + 0.25 * torch.tanh(residual[..., 2].float())
        )
        covariance = torch.zeros(
            *diagonal.shape[:-1], 2, 2, device=source.device, dtype=source.dtype
        )
        covariance[..., 0, 0] = diagonal[..., 0].square()
        covariance[..., 1, 1] = diagonal[..., 1].square()
        cross = correlation * diagonal[..., 0] * diagonal[..., 1]
        covariance[..., 0, 1] = cross
        covariance[..., 1, 0] = cross
        eye = torch.eye(2, device=source.device, dtype=source.dtype)
        return covariance + self.config.covariance_floor * eye

    def forward(self, source, effect, delta_seconds):
        source_tokens = state_tokens_without_identity_v62(source).to(
            dtype=self.state_input[1].weight.dtype
        )
        source_hidden = self.state_input(source_tokens)
        effect_hidden = self.effect_input(
            effect.value.to(dtype=self.effect_input.weight.dtype)
        )
        time_value = torch.log1p(delta_seconds.float())[:, None]
        time = self.time_input(
            time_value.to(dtype=self.time_input[0].weight.dtype)
        )[:, None]
        target_hidden = self.target_queries[None].expand(len(source_hidden), -1, -1)
        target_hidden = target_hidden + time
        effect_context, _ = self.effect_attention(
            target_hidden, effect_hidden, effect_hidden, need_weights=False
        )
        target_hidden = target_hidden + effect_context
        transport_logits = (
            torch.einsum(
                "bkd,bjd->bkj",
                self.transport_query(target_hidden),
                self.transport_key(source_hidden),
            )
            / self.config.state_dim**0.5
        )
        transport = torch.softmax(transport_logits.float(), dim=-1)
        transported_hidden = torch.einsum("bkj,bjd->bkd", transport, source_hidden)
        hidden = self.refine(target_hidden + transported_hidden)
        transported_carriers = torch.einsum(
            "bkj,bjd->bkd", transport, source.carriers.float()
        )
        transported_center = torch.einsum(
            "bkj,bjd->bkd", transport, source.center.float()
        )
        transported_covariance = torch.einsum(
            "bkj,bjmn->bkmn", transport, source.covariance.float()
        )
        transported_presence = torch.einsum(
            "bkj,bj->bk", transport, source.presence.float()
        )
        transported_visibility = torch.einsum(
            "bkj,bj->bk", transport, source.visibility.float()
        )
        assignment = torch.einsum("bkj,bjp->bkp", transport, source.assignment.float())
        lifecycle_base = source.lifecycle_logits.float()
        lifecycle = lifecycle_base + self.lifecycle_residual(hidden.mean(dim=1)).float()
        return TeacherObjectStateV62(
            carriers=transported_carriers + self.feature_residual(hidden),
            identity=source.identity,
            center=transported_center
            + 0.25 * torch.tanh(self.center_residual(hidden).float()),
            covariance=self._covariance(
                transported_covariance, self.covariance_residual(hidden)
            ),
            presence=self._probability(
                transported_presence, self.presence_residual(hidden)[..., 0]
            ),
            visibility=self._probability(
                transported_visibility, self.visibility_residual(hidden)[..., 0]
            ),
            lifecycle_logits=lifecycle,
            assignment=assignment,
        ), transport
