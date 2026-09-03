"""Query-conditioned relation field and predictive object code for v67."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_scale_field_v67 import fourier_features_v67


@dataclass(frozen=True)
class QueryRelationFieldV67:
    logits: torch.Tensor
    probability: torch.Tensor
    response: torch.Tensor
    visibility_logits: torch.Tensor
    log_uncertainty: torch.Tensor
    anchor_coordinates: torch.Tensor
    anchor_scales: torch.Tensor
    anchor_valid: torch.Tensor


@dataclass(frozen=True)
class PredictiveObjectCodeV67:
    sample: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor
    identity: torch.Tensor
    dynamic: torch.Tensor
    rate: torch.Tensor
    point_mean: torch.Tensor
    point_log_variance: torch.Tensor
    point_rate: torch.Tensor
    support_mass: torch.Tensor


@dataclass(frozen=True)
class DecodedPredictiveFieldV67:
    support_logits: torch.Tensor
    dino: torch.Tensor
    siglip: torch.Tensor
    response: torch.Tensor
    visibility_logits: torch.Tensor
    log_uncertainty: torch.Tensor


class QueryRelationFieldNetworkV67(nn.Module):
    """Evaluate a soft same-persistent-entity predicate at arbitrary points."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.field_dim
        scalar_width = 1 + 2 * config.fourier_bands
        geometry_width = 3 * scalar_width
        self.pair = nn.Sequential(
            nn.LayerNorm(dim * 3 + geometry_width),
            nn.Linear(dim * 3 + geometry_width, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.relation = nn.Linear(dim, 1)
        self.response = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.dynamic_dim),
        )
        self.visibility = nn.Linear(dim, 1)
        self.uncertainty = nn.Linear(dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        scales: torch.Tensor,
        valid: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> QueryRelationFieldV67:
        anchor = features.index_select(1, query_indices)
        anchor_coordinates = coordinates.index_select(1, query_indices)
        anchor_scales = scales.index_select(1, query_indices)
        query = anchor[:, :, None]
        candidate = features[:, None]
        pair_feature = torch.cat(
            (
                query + candidate,
                (query - candidate).abs(),
                query * candidate,
            ),
            dim=-1,
        )
        coordinate_delta = (
            anchor_coordinates[:, :, None] - coordinates[:, None]
        ).abs()
        scale_delta = (
            anchor_scales[:, :, None] / scales[:, None].clamp_min(1e-6)
        ).log().abs()
        geometry = torch.cat((coordinate_delta, scale_delta[..., None]), dim=-1)
        geometry = fourier_features_v67(geometry, self.config.fourier_bands)
        hidden = self.pair(
            torch.cat((pair_feature, geometry.to(pair_feature.dtype)), dim=-1)
        )
        logits = self.relation(hidden)[..., 0].float()
        pair_valid = valid[:, None] & valid.index_select(1, query_indices)[:, :, None]
        logits = logits.masked_fill(~pair_valid, -12.0)
        return QueryRelationFieldV67(
            logits=logits,
            probability=torch.sigmoid(logits),
            response=self.response(hidden),
            visibility_logits=self.visibility(hidden)[..., 0].float(),
            log_uncertainty=self.uncertainty(hidden)[..., 0].float().clamp(-5.0, 3.0),
            anchor_coordinates=anchor_coordinates.float(),
            anchor_scales=anchor_scales.float(),
            anchor_valid=valid.index_select(1, query_indices),
        )


def gaussian_rate_v67(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        mean.float().square() + log_variance.float().exp() - log_variance.float() - 1.0
    ).sum(dim=-1)


class PredictiveObjectCodeNetworkV67(nn.Module):
    """Compress a query-conditioned context field into a stochastic state."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.field_dim
        width = dim * 2 + config.dynamic_dim + 1
        self.response_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.dynamic_dim),
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.code_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.code_dim * 2, config.code_dim * 2),
        )
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.code_dim * 2),
        )

    def _encode(
        self,
        features: torch.Tensor,
        support: torch.Tensor,
        response: torch.Tensor,
        valid: torch.Tensor,
        context_mask: torch.Tensor,
        query_indices: torch.Tensor,
        sample: bool,
    ) -> PredictiveObjectCodeV67:
        weight = support.float() * valid[:, None].float()
        weight = weight * context_mask[:, None].float()
        denominator = weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weight / denominator
        aggregate_feature = torch.einsum("bqp,bpd->bqd", normalized, features.float())
        aggregate_response = torch.einsum(
            "bqp,bqpd->bqd", normalized, response.float()
        )
        anchor = features.index_select(1, query_indices).float()
        support_mass = weight.sum(dim=-1) / context_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        encoded = self.encoder(
            torch.cat(
                (anchor, aggregate_feature, aggregate_response, support_mass[..., None]),
                dim=-1,
            ).to(dtype=self.encoder[1].weight.dtype)
        )
        mean, log_variance = encoded.chunk(2, dim=-1)
        log_variance = log_variance.float().clamp(-6.0, 2.0)
        mean = mean.float()
        if sample:
            value = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            value = mean
        point = self.point_encoder(features)
        point_mean, point_log_variance = point.chunk(2, dim=-1)
        point_mean = point_mean.float()
        point_log_variance = point_log_variance.float().clamp(-6.0, 2.0)
        identity = F.normalize(value[..., : self.config.identity_dim], dim=-1, eps=1e-6)
        dynamic = value[..., self.config.identity_dim :]
        return PredictiveObjectCodeV67(
            sample=value,
            mean=mean,
            log_variance=log_variance,
            identity=identity,
            dynamic=dynamic,
            rate=gaussian_rate_v67(mean, log_variance),
            point_mean=point_mean,
            point_log_variance=point_log_variance,
            point_rate=gaussian_rate_v67(point_mean, point_log_variance),
            support_mass=support_mass,
        )

    def forward(
        self,
        features: torch.Tensor,
        relation: QueryRelationFieldV67,
        valid: torch.Tensor,
        context_mask: torch.Tensor,
        query_indices: torch.Tensor,
        sample: bool,
    ) -> PredictiveObjectCodeV67:
        projected = self.response_projection(features)
        response = relation.response + projected[:, None]
        return self._encode(
            features,
            relation.probability,
            response,
            valid,
            context_mask,
            query_indices,
            sample,
        )

    def from_external_support(
        self,
        features: torch.Tensor,
        support: torch.Tensor,
        valid: torch.Tensor,
        context_mask: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> PredictiveObjectCodeV67:
        response = self.response_projection(features)
        response = response[:, None].expand(-1, support.shape[1], -1, -1)
        return self._encode(
            features,
            support,
            response,
            valid,
            context_mask,
            query_indices,
            False,
        )


class PredictiveObjectFieldDecoderV67(nn.Module):
    """Decode one predictive object code at continuous output coordinates."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.code_dim
        scalar_width = 1 + 2 * config.fourier_bands
        trunk_width = 4 * scalar_width
        self.branch = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.trunk = nn.Sequential(
            nn.Linear(trunk_width, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.interaction = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.support = nn.Linear(dim, 1)
        self.dino = nn.Linear(dim, config.semantic_dim)
        self.siglip = nn.Linear(dim, config.semantic_dim)
        self.response = nn.Linear(dim, config.dynamic_dim)
        self.visibility = nn.Linear(dim, 1)
        self.uncertainty = nn.Linear(dim, 1)

    def forward(
        self,
        code: torch.Tensor,
        anchor_coordinates: torch.Tensor,
        output_coordinates: torch.Tensor,
        output_scales: torch.Tensor,
    ) -> DecodedPredictiveFieldV67:
        relative = output_coordinates[:, None].float() - anchor_coordinates[:, :, None].float()
        distance = relative.square().sum(dim=-1, keepdim=True).sqrt()
        log_scale = output_scales[:, None, :, None].float().clamp_min(1e-6).log()
        geometry = torch.cat((relative, distance, log_scale), dim=-1)
        trunk_input = fourier_features_v67(geometry, self.config.fourier_bands)
        trunk = self.trunk(trunk_input.to(dtype=self.trunk[0].weight.dtype))
        branch = self.branch(code.to(dtype=self.branch[1].weight.dtype))[:, :, None]
        hidden = branch + trunk + branch * trunk
        hidden = hidden + self.interaction(hidden)
        return DecodedPredictiveFieldV67(
            support_logits=self.support(hidden)[..., 0].float(),
            dino=F.normalize(self.dino(hidden).float(), dim=-1, eps=1e-6),
            siglip=F.normalize(self.siglip(hidden).float(), dim=-1, eps=1e-6),
            response=self.response(hidden).float(),
            visibility_logits=self.visibility(hidden)[..., 0].float(),
            log_uncertainty=self.uncertainty(hidden)[..., 0].float().clamp(-5.0, 3.0),
        )


class PointObservableDecoderV67(nn.Module):
    """Independent-point rate/distortion reference, never the object prediction."""

    def __init__(self, config):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(config.code_dim),
            nn.Linear(config.code_dim, config.code_dim),
            nn.GELU(approximate="tanh"),
        )
        self.dino = nn.Linear(config.code_dim, config.semantic_dim)
        self.siglip = nn.Linear(config.code_dim, config.semantic_dim)

    def forward(self, point_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(point_code.to(dtype=self.body[1].weight.dtype))
        return (
            F.normalize(self.dino(hidden).float(), dim=-1, eps=1e-6),
            F.normalize(self.siglip(hidden).float(), dim=-1, eps=1e-6),
        )
