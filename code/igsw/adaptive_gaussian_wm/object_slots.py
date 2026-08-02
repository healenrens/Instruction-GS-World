"""Identity-anchored competitive aggregation of GPSTokens into object slots."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig
from .gpstoken import GPSTokenState


@dataclass
class ObjectSlotState:
    slots: torch.Tensor
    tracking_slots: torch.Tensor
    assignment: torch.Tensor
    background_assignment: torch.Tensor
    potential_change: torch.Tensor
    potential_change_logits: torch.Tensor
    activity: torch.Tensor
    center: torch.Tensor
    feature: torch.Tensor
    decoded_center: torch.Tensor
    decoded_feature: torch.Tensor
    auxiliary_enabled: bool
    center_auxiliary_enabled: bool


class ObjectSlotAggregator(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.object_dim
        self.iterations = config.slot_iterations
        self.aggregation_mode = config.aggregation_mode
        self.auxiliary_enabled = config.slot_auxiliary
        self.decoupled_jepa_slots = config.decoupled_jepa_slots
        self.persistent_identity_key = config.persistent_identity_key
        self.use_background_dustbin = config.persistent_object_memory
        self.center_auxiliary_enabled = config.slot_auxiliary
        self.feature_dim = config.feature_dim
        self.token_input = nn.Linear(config.token_dim, dim)
        self.appearance_input = nn.Linear(config.feature_dim, dim)
        self.geometry_input = nn.Sequential(
            nn.Linear(8, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initial_slots = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        if config.spatial_slot_attention:
            side = math.ceil(config.object_slots**0.5)
            axis = torch.linspace(-0.45, 0.45, side)
            seed_y, seed_x = torch.meshgrid(axis, axis, indexing="ij")
            seeds = torch.stack((seed_x, seed_y), dim=-1).reshape(-1, 2)
            self.initial_slot_centers = nn.Parameter(
                seeds[: config.object_slots]
            )
            self.slot_spatial_precision = nn.Parameter(
                torch.full((config.object_slots,), 2.0)
            )
        else:
            self.initial_slot_centers = None
            self.slot_spatial_precision = None
        self.context_update = nn.Linear(dim, dim)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.identity_anchor_projection = (
            nn.Linear(dim, dim, bias=False)
            if self.persistent_identity_key
            else None
        )
        if self.identity_anchor_projection is not None:
            nn.init.zeros_(self.identity_anchor_projection.weight)
        self.background_score = (
            nn.Linear(dim, 1)
            if self.use_background_dustbin
            else None
        )
        self.update = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.activity_head = nn.Linear(dim, 1)
        self.geometry_fusion = (
            nn.Sequential(
                nn.LayerNorm(config.feature_dim + 2),
                nn.Linear(config.feature_dim + 2, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            if config.slot_geometry_fusion
            else None
        )
        self.center_fusion = (
            nn.Linear(2, dim, bias=False)
            if config.slot_center_fusion
            else None
        )
        self.feature_fusion = (
            nn.Linear(config.feature_dim, dim, bias=False)
            if config.slot_feature_fusion
            else None
        )
        if self.decoupled_jepa_slots:
            self.semantic_feature_input = nn.Sequential(
                nn.LayerNorm(config.feature_dim),
                nn.Linear(config.feature_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.semantic_identity = nn.Parameter(
                torch.randn(config.object_slots, dim) / dim**0.5
            )
            self.semantic_norm = nn.LayerNorm(dim)
        else:
            self.semantic_feature_input = None
            self.semantic_identity = None
            self.semantic_norm = None
        if self.auxiliary_enabled:
            self.center_head = (
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, 2),
                )
                if self.center_auxiliary_enabled
                else None
            )
            self.feature_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, config.feature_dim),
            )
        else:
            self.center_head = None
            self.feature_head = None
        self.rgb_head = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 3),
            )
            if config.rgb_supervision
            else None
        )

    def decode_center(self, slots: torch.Tensor) -> torch.Tensor:
        if self.center_head is None:
            return torch.tanh(slots[..., :2])
        return torch.tanh(self.center_head(slots))

    def decode_feature(self, slots: torch.Tensor) -> torch.Tensor:
        if self.feature_head is None:
            if slots.shape[-1] >= self.feature_dim:
                return slots[..., : self.feature_dim]
            return torch.nn.functional.pad(
                slots,
                (0, self.feature_dim - slots.shape[-1]),
            )
        return self.feature_head(slots)

    def decode_rgb_logits(self, slots: torch.Tensor) -> torch.Tensor:
        if self.rgb_head is None:
            raise ValueError("RGB slot decoding is disabled in the model config")
        return self.rgb_head(slots)

    def forward(
        self,
        tokens: GPSTokenState,
        anchor_slots: torch.Tensor | None = None,
        anchor_centers: torch.Tensor | None = None,
        anchor_identity: torch.Tensor | None = None,
    ) -> ObjectSlotState:
        covariance = torch.stack(
            (
                tokens.covariance[..., 0, 0],
                tokens.covariance[..., 0, 1],
                tokens.covariance[..., 1, 1],
            ),
            dim=-1,
        )
        geometry = torch.cat(
            (
                tokens.center,
                covariance,
                tokens.depth_order,
                tokens.opacity,
                tokens.activation,
            ),
            dim=-1,
        )
        values = self.norm_tokens(
            self.token_input(tokens.latent)
            + self.appearance_input(tokens.decoded_features)
            + self.geometry_input(geometry)
        )
        batch, micro_count, dim = values.shape
        active = tokens.activation.squeeze(-1)
        pooled = (values * active[..., None]).sum(dim=1)
        pooled = pooled / active.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if anchor_slots is None:
            if anchor_identity is not None:
                raise ValueError("anchor identity requires tracking slots")
            slots = self.initial_slots[None].expand(batch, -1, -1)
        else:
            expected = (batch, self.initial_slots.shape[0], dim)
            if anchor_slots.shape != expected:
                raise ValueError(
                    f"anchor_slots must have shape {expected}, got {anchor_slots.shape}"
                )
            slots = anchor_slots
            if anchor_identity is not None:
                if self.identity_anchor_projection is None:
                    raise ValueError("identity anchor is disabled")
                if anchor_identity.shape != expected:
                    raise ValueError(
                        f"anchor_identity must have shape {expected}"
                    )
                slots = slots + self.identity_anchor_projection(anchor_identity)
        slots = slots + 0.1 * self.context_update(pooled)[:, None]
        slot_centers = None
        if self.initial_slot_centers is not None:
            if anchor_centers is None:
                slot_centers = self.initial_slot_centers[None].expand(
                    batch, -1, -1
                )
            else:
                expected = (batch, self.initial_slots.shape[0], 2)
                if anchor_centers.shape != expected:
                    raise ValueError(
                        f"anchor_centers must have shape {expected}"
                    )
                slot_centers = anchor_centers

        keys = self.key(values)
        projected_values = self.value(values)
        assignment = values.new_zeros(
            batch,
            micro_count,
            self.initial_slots.shape[0],
        )
        background_assignment = values.new_zeros(batch, micro_count)
        potential_change_logits = values.new_full((batch, micro_count), 20.0)
        for _ in range(self.iterations):
            queries = self.query(self.norm_slots(slots))
            logits = torch.einsum("bmd,bkd->bmk", keys, queries) / dim**0.5
            if slot_centers is not None:
                distance = (
                    tokens.center[:, :, None] - slot_centers[:, None]
                ).square().sum(dim=-1)
                precision = torch.nn.functional.softplus(
                    self.slot_spatial_precision
                )[None, None]
                logits = logits - precision * distance
            if self.aggregation_mode == "competitive":
                if self.background_score is not None:
                    background = self.background_score(values)
                    full_assignment = torch.cat(
                        (logits, background),
                        dim=-1,
                    ).softmax(dim=-1)
                    assignment = full_assignment[..., :-1]
                    background_assignment = full_assignment[..., -1]
                    potential_change_logits = (
                        torch.logsumexp(logits.float(), dim=-1)
                        - background.squeeze(-1).float()
                    )
                else:
                    assignment = logits.softmax(dim=-1)
                update_assignment = assignment
            elif self.aggregation_mode == "independent":
                update_assignment = logits.softmax(dim=1)
                assignment = update_assignment / update_assignment.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-6)
            else:
                shared = logits.mean(dim=-1, keepdim=True).softmax(dim=1)
                update_assignment = shared.expand(-1, -1, slots.shape[1])
                assignment = torch.full_like(
                    update_assignment,
                    1.0 / slots.shape[1],
                )
            weighted = update_assignment * active[..., None]
            weights = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-6)
            updates = torch.einsum("bmk,bmd->bkd", weights, projected_values)
            if slot_centers is not None:
                slot_centers = torch.einsum(
                    "bmk,bmd->bkd",
                    weights,
                    tokens.center,
                )
            slots = self.update(
                updates.reshape(-1, dim),
                slots.reshape(-1, dim),
            ).reshape_as(slots)
            slots = slots + self.mlp(slots)

        mass = (assignment * active[..., None]).sum(dim=1)
        mass = mass / active.sum(dim=1, keepdim=True).clamp_min(1e-6)
        slot_weight = assignment * active[..., None]
        slot_weight = slot_weight / slot_weight.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        center = torch.einsum("bmk,bmd->bkd", slot_weight, tokens.center)
        feature = torch.einsum(
            "bmk,bmc->bkc",
            slot_weight,
            tokens.decoded_features,
        )
        tracking_slots = slots
        if self.decoupled_jepa_slots:
            slots = self.semantic_norm(
                self.semantic_feature_input(feature)
                + self.semantic_identity[None]
            )
        elif self.geometry_fusion is not None:
            slots = slots + 0.5 * self.geometry_fusion(
                torch.cat((center, feature), dim=-1)
            )
        if self.center_fusion is not None:
            slots = slots + 0.5 * self.center_fusion(center)
        if self.feature_fusion is not None:
            slots = slots + 0.5 * self.feature_fusion(feature)
        learned_activity = torch.sigmoid(
            self.activity_head(tracking_slots)
        ).squeeze(-1)
        learned_activity = 0.5 + 0.5 * learned_activity
        activity = learned_activity * (
            mass * self.initial_slots.shape[0]
        ).clamp_max(1.0)
        potential_change = assignment.sum(dim=-1).clamp(0.0, 1.0)
        return ObjectSlotState(
            slots=slots,
            tracking_slots=tracking_slots,
            assignment=assignment,
            background_assignment=background_assignment,
            potential_change=potential_change,
            potential_change_logits=potential_change_logits,
            activity=activity,
            center=center,
            feature=feature,
            decoded_center=self.decode_center(
                tracking_slots if self.decoupled_jepa_slots else slots
            ),
            decoded_feature=self.decode_feature(slots),
            auxiliary_enabled=self.auxiliary_enabled,
            center_auxiliary_enabled=self.center_auxiliary_enabled,
        )
