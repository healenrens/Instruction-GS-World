"""Posterior latent effects and frozen-state carrier Dynamics for v61."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_carrier_state_v61 import (
    CarrierStateV61,
    ContinuousObjectStateV61,
    ObjectRootStateV61,
)


EFFECT_CAPACITIES = {
    "4x32_global": (4, 32, "global"),
    "4x32_bound": (4, 32, "learned"),
    "8x64_bound": (8, 64, "learned"),
    "16x32_root": (16, 32, "root"),
}


@dataclass(frozen=True)
class ObjectBoundLatentEffectV61:
    value: torch.Tensor
    activation: torch.Tensor
    owner: torch.Tensor


def zero_effect_v61(effect: ObjectBoundLatentEffectV61):
    return ObjectBoundLatentEffectV61(
        value=torch.zeros_like(effect.value),
        activation=torch.zeros_like(effect.activation),
        owner=effect.owner,
    )


def shuffled_effect_v61(effect: ObjectBoundLatentEffectV61):
    if len(effect.value) > 1:
        return ObjectBoundLatentEffectV61(
            value=effect.value.roll(1, dims=0),
            activation=effect.activation.roll(1, dims=0),
            owner=effect.owner.roll(1, dims=0),
        )
    return ObjectBoundLatentEffectV61(
        value=effect.value.flip(1),
        activation=effect.activation.flip(1),
        owner=effect.owner.flip(1),
    )


def frame_state_v61(state: ContinuousObjectStateV61, index: int):
    def take(value):
        return value[:, index : index + 1]

    return ContinuousObjectStateV61(
        carriers=CarrierStateV61(
            **{name: take(value) for name, value in vars(state.carriers).items()}
        ),
        roots=ObjectRootStateV61(
            **{name: take(value) for name, value in vars(state.roots).items()}
        ),
    )


def _carrier_tokens(state: ContinuousObjectStateV61):
    carrier = state.carriers
    return torch.cat(
        (
            carrier.feature,
            carrier.identity,
            carrier.dynamic,
            carrier.center,
            carrier.covariance.flatten(-2),
            carrier.presence[..., None],
            carrier.visibility[..., None],
        ),
        dim=-1,
    )[:, 0]


def _root_tokens(state: ContinuousObjectStateV61):
    root = state.roots
    return torch.cat(
        (
            root.feature,
            root.identity,
            root.dynamic,
            root.center,
            root.relative_scale[..., None],
            root.presence[..., None],
            root.visibility[..., None],
        ),
        dim=-1,
    )[:, 0]


class ContinuousCarrierEffectPosteriorV61(nn.Module):
    """Extract a bounded latent effect without explicit action or center deltas."""

    def __init__(self, config, capacity: str):
        super().__init__()
        effect_factors, effect_dim, binding = EFFECT_CAPACITIES[capacity]
        dim = config.student_dim
        carrier_width = dim + config.identity_dim + config.dynamic_dim + 8
        root_width = dim + config.identity_dim + config.dynamic_dim + 5
        self.effect_factors = effect_factors
        self.effect_dim = effect_dim
        self.binding = binding
        self.total_owners = config.total_owners
        self.carrier_input = nn.Sequential(
            nn.LayerNorm(carrier_width), nn.Linear(carrier_width, dim)
        )
        self.root_input = nn.Sequential(
            nn.LayerNorm(root_width), nn.Linear(root_width, dim)
        )
        self.source_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.target_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(effect_factors, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, effect_dim),
            nn.Tanh(),
        )
        self.activation = nn.Linear(dim, 1)
        self.owner = (
            nn.Linear(dim, config.total_owners) if binding == "learned" else None
        )

    def _tokens(self, state):
        return torch.cat(
            (
                self.carrier_input(_carrier_tokens(state)),
                self.root_input(_root_tokens(state)),
            ),
            dim=1,
        )

    def forward(self, source, target):
        source_tokens = self._tokens(source) + self.source_type
        target_tokens = self._tokens(target) + self.target_type
        context = torch.cat((source_tokens, target_tokens), dim=1)
        query = self.queries[None].expand(len(context), -1, -1)
        hidden, _ = self.attention(query, context, context, need_weights=False)
        value = self.output(hidden)
        activation = torch.sigmoid(self.activation(hidden)[..., 0])
        if self.binding == "global":
            owner = value.new_full(
                (*value.shape[:2], self.total_owners), 1.0 / self.total_owners
            )
        elif self.binding == "root":
            owner = F.one_hot(
                torch.arange(self.effect_factors, device=value.device),
                self.total_owners,
            ).to(value.dtype)
            owner = owner[None].expand(len(value), -1, -1)
        else:
            owner = torch.softmax(self.owner(hidden), dim=-1)
        return ObjectBoundLatentEffectV61(value, activation, owner)


class EffectConditionedCarrierDynamicsV61(nn.Module):
    def __init__(self, config, effect_dim: int):
        super().__init__()
        self.config = config
        dim = config.student_dim
        carrier_width = dim + config.identity_dim + config.dynamic_dim + 8
        root_width = dim + config.identity_dim + config.dynamic_dim + 5
        self.carrier_input = nn.Sequential(
            nn.LayerNorm(carrier_width), nn.Linear(carrier_width, dim)
        )
        self.root_input = nn.Sequential(
            nn.LayerNorm(root_width), nn.Linear(root_width, dim)
        )
        self.effect_input = nn.Linear(effect_dim, dim)
        self.time_input = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.root_carrier = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.carrier_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.root_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.carrier_feature = nn.Linear(dim, config.student_dim)
        self.carrier_identity = nn.Linear(dim, config.identity_dim)
        self.carrier_dynamic = nn.Linear(dim, config.dynamic_dim)
        self.carrier_center = nn.Linear(dim, 2)
        self.carrier_covariance = nn.Linear(dim, 3)
        self.carrier_presence = nn.Linear(dim, 1)
        self.carrier_visibility = nn.Linear(dim, 1)
        self.owner = nn.Linear(dim, config.total_owners)
        self.root_feature = nn.Linear(dim, config.student_dim)
        self.root_identity = nn.Linear(dim, config.identity_dim)
        self.root_dynamic = nn.Linear(dim, config.dynamic_dim)
        self.root_center = nn.Linear(dim, 2)
        self.root_scale = nn.Linear(dim, 1)
        self.root_presence = nn.Linear(dim, 1)
        self.root_visibility = nn.Linear(dim, 1)

    @staticmethod
    def _probability(source, residual):
        logits = torch.logit(source.float().clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(logits + residual.float())

    def _covariance(self, source, residual):
        source = source.float()
        source_diagonal = source.diagonal(dim1=-2, dim2=-1).clamp_min(1e-5).sqrt()
        source_correlation = source[..., 0, 1] / (
            source_diagonal[..., 0] * source_diagonal[..., 1]
        ).clamp_min(1e-5)
        diagonal = source_diagonal
        diagonal = diagonal * torch.exp(0.25 * torch.tanh(residual[..., :2].float()))
        correlation = (
            source_correlation + 0.25 * torch.tanh(residual[..., 2].float())
        ).clamp(-0.95, 0.95)
        off_diagonal = correlation * diagonal[..., 0] * diagonal[..., 1]
        return torch.stack(
            (
                torch.stack((diagonal[..., 0].square(), off_diagonal), dim=-1),
                torch.stack((off_diagonal, diagonal[..., 1].square()), dim=-1),
            ),
            dim=-2,
        )

    def forward(self, source, effect, delta_seconds):
        carrier = self.carrier_input(_carrier_tokens(source))
        root = self.root_input(_root_tokens(source))
        effect_tokens = self.effect_input(effect.value.float())
        effect_tokens = effect_tokens * effect.activation[..., None].float()
        time = self.time_input(torch.log1p(delta_seconds.float())[:, None])[:, None]
        carrier = carrier + time
        root = root + time
        carrier_owner = source.roots.owner[:, 0].float()
        carrier_binding = torch.einsum(
            "bqo,bko->bqk", carrier_owner, effect.owner.float()
        )
        carrier_binding = carrier_binding / carrier_binding.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        carrier = carrier + torch.einsum("bqk,bkd->bqd", carrier_binding, effect_tokens)
        carrier = carrier + self.carrier_mlp(carrier)
        root_owner = F.one_hot(
            torch.arange(self.config.object_roots, device=root.device),
            self.config.total_owners,
        ).to(root.dtype)
        root_binding = torch.einsum("ro,bko->brk", root_owner, effect.owner.float())
        root_binding = root_binding / root_binding.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        root = root + torch.einsum("brk,bkd->brd", root_binding, effect_tokens)
        root_update, _ = self.root_carrier(root, carrier, carrier, need_weights=False)
        root = root + root_update + self.root_mlp(root + root_update)
        source_carrier, source_root = source.carriers, source.roots
        owner = torch.softmax(
            source_root.owner[:, 0].float() + self.owner(carrier), dim=-1
        )
        carrier_state = CarrierStateV61(
            feature=F.normalize(
                source_carrier.feature[:, 0] + self.carrier_feature(carrier),
                dim=-1,
                eps=1e-6,
            )[:, None],
            identity=F.normalize(
                source_carrier.identity[:, 0] + self.carrier_identity(carrier),
                dim=-1,
                eps=1e-6,
            )[:, None],
            dynamic=(source_carrier.dynamic[:, 0] + self.carrier_dynamic(carrier))[
                :, None
            ],
            center=(
                source_carrier.center[:, 0]
                + 0.5 * torch.tanh(self.carrier_center(carrier))
            )[:, None],
            covariance=self._covariance(
                source_carrier.covariance[:, 0], self.carrier_covariance(carrier)
            )[:, None],
            presence=self._probability(
                source_carrier.presence[:, 0], self.carrier_presence(carrier)[..., 0]
            )[:, None],
            visibility=self._probability(
                source_carrier.visibility[:, 0],
                self.carrier_visibility(carrier)[..., 0],
            )[:, None],
            support=source_carrier.support,
        )
        root_state = ObjectRootStateV61(
            feature=F.normalize(
                source_root.feature[:, 0] + self.root_feature(root), dim=-1, eps=1e-6
            )[:, None],
            identity=F.normalize(
                source_root.identity[:, 0] + self.root_identity(root), dim=-1, eps=1e-6
            )[:, None],
            dynamic=(source_root.dynamic[:, 0] + self.root_dynamic(root))[:, None],
            center=(
                source_root.center[:, 0] + 0.5 * torch.tanh(self.root_center(root))
            )[:, None],
            relative_scale=(
                source_root.relative_scale[:, 0]
                * torch.exp(0.25 * torch.tanh(self.root_scale(root)[..., 0]))
            )[:, None],
            presence=self._probability(
                source_root.presence[:, 0], self.root_presence(root)[..., 0]
            )[:, None],
            visibility=self._probability(
                source_root.visibility[:, 0], self.root_visibility(root)[..., 0]
            )[:, None],
            owner=owner[:, None],
        )
        return ContinuousObjectStateV61(carriers=carrier_state, roots=root_state)
