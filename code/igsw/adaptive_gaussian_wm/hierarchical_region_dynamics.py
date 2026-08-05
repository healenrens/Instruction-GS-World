"""Hierarchical region Dynamics conditioned on predicted object roots."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .hierarchical_world_state import RegionMemoryState


@dataclass
class RegionDynamicsOutput:
    future_feature: torch.Tensor
    future_center: torch.Tensor
    future_covariance: torch.Tensor
    future_owner: torch.Tensor
    future_relative_center: torch.Tensor
    future_activation: torch.Tensor
    future_presence: torch.Tensor
    future_visibility: torch.Tensor
    future_identity_key: torch.Tensor
    base_future_feature: torch.Tensor
    base_future_center: torch.Tensor
    base_future_presence: torch.Tensor
    base_future_visibility: torch.Tensor
    action_feature_residual: torch.Tensor
    action_geometry_residual: torch.Tensor


def _stable_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1.0 - 1e-4))


class HierarchicalRegionDynamics(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.region_dim
        self.config = config
        self.region_input = nn.Linear(dim, dim)
        self.geometry_input = nn.Sequential(
            nn.Linear(9, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.owner_input = nn.Linear(config.region_owners, dim)
        self.root_input = nn.Linear(config.object_dim, dim)
        self.time_input = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.future_query = nn.Parameter(torch.randn(dim) / dim**0.5)
        layer = nn.TransformerEncoderLayer(
            dim,
            config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.region_dynamics_layers)
        self.output_norm = nn.LayerNorm(dim)
        self.base_feature_head = nn.Linear(dim, dim)
        self.base_geometry_head = nn.Linear(dim, 5)
        self.base_owner_head = nn.Linear(dim, config.region_owners)
        self.base_lifecycle_head = nn.Linear(dim, 2)
        self.identity_head = nn.Linear(dim, config.region_identity_dim)
        self.route_query = nn.Linear(dim, dim // 2, bias=False)
        self.factor_keys = nn.Parameter(
            torch.randn(config.action_tokens, dim // 2) / (dim // 2) ** 0.5
        )
        self.action_input = nn.Linear(config.action_dim, dim, bias=False)
        self.action_feature_basis = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.action_feature_gate = nn.Linear(dim, dim, bias=False)
        self.action_geometry_basis = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 5)
        )
        self.action_geometry_gate = nn.Linear(dim, 5, bias=False)
        self.action_owner_gate = nn.Linear(dim, config.region_owners, bias=False)
        self.action_lifecycle_gate = nn.Linear(dim, 2, bias=False)
        for head in (
            self.base_feature_head,
            self.base_geometry_head,
            self.base_owner_head,
            self.base_lifecycle_head,
            self.identity_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _covariance_features(covariance: torch.Tensor) -> torch.Tensor:
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
        correlation = covariance[..., 0, 1] / torch.sqrt(
            diagonal[..., 0] * diagonal[..., 1]
        )
        return torch.cat((diagonal.log(), correlation[..., None]), dim=-1)

    @staticmethod
    def _covariance_update(
        covariance: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
        diagonal = diagonal * torch.exp(0.25 * torch.tanh(residual[..., :2]))
        old_correlation = covariance[..., 0, 1] / torch.sqrt(
            covariance[..., 0, 0].clamp_min(1e-6)
            * covariance[..., 1, 1].clamp_min(1e-6)
        )
        correlation = (
            old_correlation + 0.1 * torch.tanh(residual[..., 2])
        ).clamp(-0.95, 0.95)
        off_diagonal = correlation * torch.sqrt(diagonal[..., 0] * diagonal[..., 1])
        row0 = torch.stack((diagonal[..., 0], off_diagonal), dim=-1)
        row1 = torch.stack((off_diagonal, diagonal[..., 1]), dim=-1)
        return torch.stack((row0, row1), dim=-2)

    def _root_context(
        self,
        owner: torch.Tensor,
        root_slots: torch.Tensor,
    ) -> torch.Tensor:
        object_owner = owner[..., : self.config.object_slots]
        object_context = torch.einsum(
            "brk,bqkd->bqrd", object_owner, root_slots
        )
        return self.root_input(object_context)

    def _enforce_scene_quota(
        self,
        owner: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        scene_score = owner[..., -2]
        order = scene_score.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        indices = torch.arange(owner.shape[-2], device=owner.device)
        rank.scatter_(
            -1,
            order,
            indices.view(1, 1, -1).expand_as(order),
        )
        active = presence > 0.5
        allowance = torch.floor(
            active.sum(dim=-1).float() * self.config.region_scene_fraction
        ).long().clamp_min(1)
        allowed = (rank < allowance[..., None]) & active
        scene = scene_score * allowed.to(owner.dtype)
        transient = owner[..., -1] + scene_score - scene
        return torch.cat(
            (owner[..., :-2], scene[..., None], transient[..., None]), dim=-1
        )

    def _hard_presence(self, value: torch.Tensor) -> torch.Tensor:
        count = torch.round(value.detach().sum(dim=-1)).long().clamp(
            self.config.min_active_tokens,
            self.config.max_micro_tokens,
        )
        order = value.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        indices = torch.arange(value.shape[-1], device=value.device)
        rank.scatter_(
            -1,
            order,
            indices.view(1, 1, -1).expand_as(order),
        )
        hard = (rank < count[..., None]).to(value.dtype)
        return hard.detach() + value - value.detach()

    def forward(
        self,
        history_regions: dict[str, torch.Tensor],
        root_future_slots: torch.Tensor,
        root_future_centers: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
        base_root_future_slots: torch.Tensor | None = None,
    ) -> RegionDynamicsOutput:
        current_feature = history_regions["feature"][:, -1]
        current_center = history_regions["center"][:, -1]
        current_covariance = history_regions["covariance"][:, -1]
        current_owner = history_regions["owner"][:, -1]
        current_relative = history_regions["relative_center"][:, -1]
        current_presence = history_regions["presence"][:, -1]
        current_visibility = history_regions["visibility"][:, -1]
        current_identity = history_regions["identity_key"][:, -1]
        batch, regions, dim = current_feature.shape
        queries = future_scale.shape[1]
        expected_action = (
            batch,
            queries,
            self.config.action_tokens,
            self.config.action_dim,
        )
        if actions.shape != expected_action:
            raise ValueError(f"region actions must have shape {expected_action}")
        if base_root_future_slots is None:
            base_root_future_slots = root_future_slots
        if base_root_future_slots.shape != root_future_slots.shape:
            raise ValueError("base and action-conditioned root shapes differ")
        geometry = torch.cat(
            (
                current_center,
                self._covariance_features(current_covariance),
                current_relative,
                current_presence[..., None],
                current_visibility[..., None],
            ),
            dim=-1,
        )
        base_input = (
            self.region_input(current_feature)
            + self.geometry_input(geometry)
            + self.owner_input(current_owner)
        )
        tokens = (
            base_input[:, None]
            + self._root_context(current_owner, base_root_future_slots)
            + self.time_input(future_scale[..., None])[:, :, None]
            + self.future_query
        )
        active = current_presence > 0.5
        key_padding = (~active)[:, None].expand(-1, queries, -1).reshape(
            batch * queries, regions
        )
        hidden = self.output_norm(
            self.blocks(
                tokens.reshape(batch * queries, regions, dim),
                src_key_padding_mask=key_padding,
            )
        ).reshape(batch, queries, regions, dim)
        hidden = hidden * active[:, None, :, None].to(hidden.dtype)
        base_feature = current_feature[:, None] + self.base_feature_head(hidden)
        base_geometry = self.base_geometry_head(hidden)
        base_center = current_center[:, None] + 0.25 * torch.tanh(
            base_geometry[..., :2]
        )

        route = torch.einsum(
            "brd,kd->brk",
            F.normalize(self.route_query(current_feature), dim=-1),
            F.normalize(self.factor_keys, dim=-1),
        ).softmax(dim=-1)
        region_effect = torch.einsum("brk,bqkd->bqrd", route, actions)
        root_effect = self._root_context(
            current_owner,
            root_future_slots - base_root_future_slots,
        )
        action_hidden = self.action_input(region_effect) + root_effect
        action_feature = torch.tanh(
            self.action_feature_basis(hidden)
        ) * torch.tanh(self.action_feature_gate(action_hidden))
        action_geometry = torch.tanh(
            self.action_geometry_basis(hidden)
        ) * torch.tanh(self.action_geometry_gate(action_hidden))
        future_feature = base_feature + action_feature
        future_center = base_center + 0.25 * torch.tanh(action_geometry[..., :2])
        covariance_residual = base_geometry[..., 2:] + action_geometry[..., 2:]
        future_covariance = self._covariance_update(
            current_covariance[:, None], covariance_residual
        )
        owner_logits = (
            current_owner[:, None].clamp_min(1e-6).log()
            + self.base_owner_head(hidden)
            + self.action_owner_gate(action_hidden)
        )
        base_lifecycle = self.base_lifecycle_head(hidden)
        action_lifecycle = self.action_lifecycle_gate(action_hidden)
        base_presence_soft = torch.sigmoid(
            _stable_logit(current_presence[:, None]) + base_lifecycle[..., 0]
        )
        base_visibility_soft = torch.sigmoid(
            _stable_logit(current_visibility[:, None]) + base_lifecycle[..., 1]
        )
        future_presence_soft = torch.sigmoid(
            _stable_logit(base_presence_soft) + action_lifecycle[..., 0]
        )
        future_visibility_soft = torch.sigmoid(
            _stable_logit(base_visibility_soft) + action_lifecycle[..., 1]
        )
        base_presence = self._hard_presence(base_presence_soft)
        future_presence = self._hard_presence(future_presence_soft)
        base_visibility = base_visibility_soft * base_presence
        future_visibility = future_visibility_soft * future_presence
        future_owner = self._enforce_scene_quota(
            torch.softmax(owner_logits, dim=-1),
            future_presence,
        )
        object_owner = future_owner[..., : self.config.object_slots]
        owner_mass = object_owner.sum(dim=-1, keepdim=True)
        owner_center = torch.einsum(
            "bqrk,bqkd->bqrd",
            object_owner / owner_mass.clamp_min(1e-6),
            root_future_centers,
        )
        future_relative = owner_mass * (future_center - owner_center)
        future_relative = future_relative + (1.0 - owner_mass) * future_center
        future_identity = F.normalize(
            current_identity[:, None].float() + 0.05 * self.identity_head(hidden).float(),
            dim=-1,
        ).to(future_feature.dtype)
        object_mass = future_owner[..., : self.config.object_slots].sum(dim=-1)
        future_identity = future_identity * object_mass[..., None]
        return RegionDynamicsOutput(
            future_feature=future_feature,
            future_center=future_center,
            future_covariance=future_covariance,
            future_owner=future_owner,
            future_relative_center=future_relative,
            future_activation=future_presence,
            future_presence=future_presence,
            future_visibility=future_visibility,
            future_identity_key=future_identity,
            base_future_feature=base_feature,
            base_future_center=base_center,
            base_future_presence=base_presence,
            base_future_visibility=base_visibility,
            action_feature_residual=action_feature,
            action_geometry_residual=action_geometry,
        )


def region_state_from_prediction(
    output: RegionDynamicsOutput,
    index: int,
) -> RegionMemoryState:
    feature = output.future_feature[:, index]
    batch, regions = feature.shape[:2]
    identity = torch.eye(
        regions, device=feature.device, dtype=feature.dtype
    )[None].expand(batch, -1, -1)
    presence = output.future_presence[:, index]
    visibility = output.future_visibility[:, index]
    return RegionMemoryState(
        feature=feature,
        center=output.future_center[:, index],
        covariance=output.future_covariance[:, index],
        owner=output.future_owner[:, index],
        relative_center=output.future_relative_center[:, index],
        activation=presence,
        presence=presence,
        visibility=visibility,
        identity_key=output.future_identity_key[:, index],
        observed=torch.zeros_like(visibility),
        update_gate=torch.zeros_like(visibility),
        association=identity,
        association_confidence=torch.zeros_like(visibility),
    )
