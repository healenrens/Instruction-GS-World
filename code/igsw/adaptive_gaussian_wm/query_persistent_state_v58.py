"""RGB-history-only persistent state for one queried visual entity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QueryPersistentState:
    support_logits: torch.Tensor
    support: torch.Tensor
    identity: torch.Tensor
    identity_sequence: torch.Tensor
    dynamic: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    visibility_logits: torch.Tensor
    visibility: torch.Tensor
    pooled_semantic: torch.Tensor


class QueryPersistentStateEncoder(nn.Module):
    """Track one current query backward through observed RGB features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.query_appearance = nn.Linear(config.patch_dim, config.model_dim)
        self.query_position = nn.Sequential(
            nn.Linear(2, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.query_update = nn.GRUCell(config.model_dim, config.model_dim)
        self.identity = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.identity_dim),
        )
        self.dynamic = nn.Sequential(
            nn.LayerNorm(2 * config.model_dim + 4),
            nn.Linear(2 * config.model_dim + 4, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.dynamic_dim),
        )
        self.support_bias = nn.Parameter(torch.tensor(0.0))
        self.visibility = nn.Sequential(
            nn.LayerNorm(2 * config.model_dim + 1),
            nn.Linear(2 * config.model_dim + 1, config.model_dim // 2),
            nn.GELU(),
            nn.Linear(config.model_dim // 2, 1),
        )

    @staticmethod
    def _query_feature(patches, coordinates, valid, query_coordinate, sigma):
        distance = (coordinates[:, -1] - query_coordinate[:, None]).square().sum(-1)
        weight = torch.exp(-distance / (2.0 * sigma**2)) * valid[:, -1].float()
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bn,bnd->bd", weight.to(patches.dtype), patches[:, -1])

    @staticmethod
    def _geometry(weight, coordinates):
        center = torch.einsum("btn,btnd->btd", weight, coordinates.float())
        offset = coordinates.float() - center[:, :, None]
        covariance = torch.einsum("btn,btni,btnj->btij", weight, offset, offset)
        eye = torch.eye(2, device=coordinates.device, dtype=torch.float32)
        return center, covariance + 1e-4 * eye

    def forward(self, patches, coordinates, valid, frame_times, query_coordinate):
        if patches.ndim != 4 or patches.shape[-1] != self.config.patch_dim:
            raise ValueError("v58 patches must have shape [B,T,N,patch_dim]")
        if coordinates.shape != (*patches.shape[:3], 2):
            raise ValueError("v58 coordinate layout differs from patches")
        if valid.shape != patches.shape[:3]:
            raise ValueError("v58 validity layout differs from patches")
        if frame_times.shape != patches.shape[:2]:
            raise ValueError("v58 frame_times layout differs from patches")
        if query_coordinate.shape != (patches.shape[0], 2):
            raise ValueError("v58 query_coordinate must have shape [B,2]")

        encoded = self.patch(patches)
        query_patch = self._query_feature(
            patches.float(), coordinates, valid, query_coordinate, self.config.spatial_sigma
        )
        query = self.query_appearance(query_patch) + self.query_position(
            query_coordinate.float()
        )
        keys, values = self.key(encoded), self.value(encoded)
        supports, pooled, recurrent = [], [], []
        centers = query_coordinate.float()
        for frame in range(patches.shape[1] - 1, -1, -1):
            semantic = torch.einsum(
                "bd,bnd->bn",
                F.normalize(query.float(), dim=-1),
                F.normalize(keys[:, frame].float(), dim=-1),
            )
            distance = (coordinates[:, frame].float() - centers[:, None]).square().sum(-1)
            logits = semantic / self.config.support_temperature
            logits = logits - distance / (2.0 * self.config.spatial_sigma**2)
            logits = (logits + self.support_bias.float()).masked_fill(~valid[:, frame], -30.0)
            probability = torch.sigmoid(logits) * valid[:, frame].float()
            weight = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            frame_value = torch.einsum("bn,bnd->bd", weight.to(values.dtype), values[:, frame])
            query = self.query_update(frame_value, query)
            centers = torch.einsum("bn,bnd->bd", weight, coordinates[:, frame].float())
            supports.append(logits)
            pooled.append(frame_value)
            recurrent.append(query)

        support_logits = torch.stack(supports[::-1], dim=1)
        pooled_state = torch.stack(pooled[::-1], dim=1)
        recurrent_state = torch.stack(recurrent[::-1], dim=1)
        support = torch.sigmoid(support_logits) * valid.float()
        normalized = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        center, covariance = self._geometry(normalized, coordinates)
        temporal_mean = pooled_state.mean(dim=1, keepdim=True)
        geometry = torch.cat((center, covariance.diagonal(dim1=-2, dim2=-1)), dim=-1)
        dynamic = self.dynamic(
            torch.cat((pooled_state, pooled_state - temporal_mean, geometry), dim=-1)
        ).float()
        identity_sequence = F.normalize(
            self.identity((pooled_state + recurrent_state).float()), dim=-1, eps=1e-6
        )
        identity = F.normalize(identity_sequence.mean(dim=1), dim=-1, eps=1e-6)
        support_peak = support.amax(dim=-1, keepdim=True)
        visibility_input = torch.cat(
            (recurrent_state.float(), pooled_state.float(), support_peak.float()), dim=-1
        )
        visibility_logits = self.visibility(visibility_input).squeeze(-1).float()
        visibility = torch.sigmoid(visibility_logits)
        pooled_semantic = torch.einsum(
            "btn,btnd->btd", normalized.to(patches.dtype), patches
        ).float()
        tensors = (
            support_logits, identity, identity_sequence, dynamic, center,
            covariance, visibility_logits, pooled_semantic,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("v58 query persistent state contains non-finite values")
        return QueryPersistentState(
            support_logits=support_logits,
            support=support,
            identity=identity,
            identity_sequence=identity_sequence,
            dynamic=dynamic,
            center=center,
            covariance=covariance,
            visibility_logits=visibility_logits,
            visibility=visibility,
            pooled_semantic=pooled_semantic,
        )
