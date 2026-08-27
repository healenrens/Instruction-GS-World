"""Continuous carriers and persistent object roots derived from one token field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CarrierStateV61:
    feature: torch.Tensor
    identity: torch.Tensor
    dynamic: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    presence: torch.Tensor
    visibility: torch.Tensor
    support: torch.Tensor


@dataclass(frozen=True)
class ObjectRootStateV61:
    feature: torch.Tensor
    identity: torch.Tensor
    dynamic: torch.Tensor
    center: torch.Tensor
    relative_scale: torch.Tensor
    presence: torch.Tensor
    visibility: torch.Tensor
    owner: torch.Tensor


@dataclass(frozen=True)
class ContinuousObjectStateV61:
    carriers: CarrierStateV61
    roots: ObjectRootStateV61


class ContinuousCarrierExtractorV61(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, count = config.student_dim, config.carrier_count
        self.config = config
        self.queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        axis = torch.linspace(-0.9, 0.9, int(count**0.5) + 1)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        seeds = torch.stack((x, y), dim=-1).reshape(-1, 2)[:count]
        self.spatial_seeds = nn.Parameter(seeds)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, dim)
        self.memory = nn.GRUCell(dim, dim)
        self.center_gate = nn.Linear(dim * 2, 1)
        self.identity = nn.Linear(dim, config.identity_dim)
        self.dynamic = nn.Linear(dim, config.dynamic_dim)
        self.presence = nn.Linear(dim, 1)
        self.visibility = nn.Linear(dim, 1)

    def _observe(self, field):
        features = field.features.float()
        coordinates = field.coordinates.float()
        batch, frames, tokens, dim = features.shape
        query = (
            self.queries[None, None] + self.context(field.pooled.float())[:, :, None]
        )
        logits = (
            torch.einsum("btqd,btnd->btqn", self.query(query), self.key(features))
            / dim**0.5
        )
        spatial = coordinates[:, :, None] - self.spatial_seeds[None, None, :, None]
        logits = logits - spatial.square().sum(dim=-1) / (
            2.0 * self.config.spatial_temperature**2
        )
        logits = logits.masked_fill(~field.valid[:, :, None], -torch.inf)
        support = torch.softmax(logits, dim=-1)
        observed = torch.einsum("btqn,btnd->btqd", support, self.value(features))
        center = torch.einsum("btqn,btnd->btqd", support, coordinates)
        offset = coordinates[:, :, None] - center[:, :, :, None]
        covariance = torch.einsum("btqn,btqni,btqnj->btqij", support, offset, offset)
        return observed, center, covariance, support

    def forward(self, field) -> CarrierStateV61:
        observed, observed_center, covariance, support = self._observe(field)
        features, centers = [], []
        previous_feature = self.queries[None].expand(observed.shape[0], -1, -1)
        previous_center = self.spatial_seeds[None].expand(observed.shape[0], -1, -1)
        for frame in range(observed.shape[1]):
            updated = self.memory(
                observed[:, frame].flatten(0, 1),
                previous_feature.flatten(0, 1),
            ).reshape_as(previous_feature)
            gate = torch.sigmoid(
                self.center_gate(torch.cat((previous_feature, updated), dim=-1))
            )
            center = torch.lerp(previous_center, observed_center[:, frame], gate)
            features.append(updated)
            centers.append(center)
            previous_feature, previous_center = updated, center
        feature = torch.stack(features, dim=1)
        center = torch.stack(centers, dim=1)
        return CarrierStateV61(
            feature=feature,
            identity=F.normalize(self.identity(feature), dim=-1, eps=1e-6),
            dynamic=self.dynamic(feature),
            center=center,
            covariance=covariance,
            presence=torch.sigmoid(self.presence(feature)[..., 0]),
            visibility=torch.sigmoid(self.visibility(feature)[..., 0]),
            support=support,
        )


class PersistentObjectRootsV61(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, roots = config.student_dim, config.object_roots
        self.config = config
        self.object_queries = nn.Parameter(torch.randn(roots, dim) * 0.02)
        self.scene_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.memory = nn.GRUCell(dim, dim)
        self.identity = nn.Linear(dim, config.identity_dim)
        self.dynamic = nn.Linear(dim, config.dynamic_dim)
        self.presence = nn.Linear(dim, 1)
        self.visibility = nn.Linear(dim, 1)

    def _owners(self, carriers: CarrierStateV61):
        queries = torch.cat((self.object_queries, self.scene_query), dim=0)
        logits = (
            torch.einsum(
                "md,btqd->btqm", self.query(queries), self.key(carriers.feature)
            )
            / self.config.student_dim**0.5
        )
        distance = carriers.center[:, :, :, None] - carriers.center[:, :, None]
        neighborhood = torch.exp(-distance.square().sum(dim=-1) / 0.08).mean(dim=-1)
        logits[..., -1] = logits[..., -1] + (1.0 - neighborhood)
        return torch.softmax(logits / self.config.root_temperature, dim=-1)

    def forward(self, carriers: CarrierStateV61) -> ObjectRootStateV61:
        owner = self._owners(carriers)
        object_owner = owner[..., : self.config.object_roots]
        weight = object_owner * carriers.presence[..., None]
        weight = weight / weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        observed = torch.einsum("btqm,btqd->btmd", weight, carriers.feature)
        center = torch.einsum("btqm,btqd->btmd", weight, carriers.center)
        offset = carriers.center[:, :, :, None] - center[:, :, None]
        scale = torch.einsum("btqm,btqmd->btm", weight, offset.square().sum(dim=-1))
        features = []
        previous = self.object_queries[None].expand(observed.shape[0], -1, -1)
        for frame in range(observed.shape[1]):
            previous = self.memory(
                observed[:, frame].flatten(0, 1), previous.flatten(0, 1)
            ).reshape_as(previous)
            features.append(previous)
        feature = torch.stack(features, dim=1)
        carrier_visibility = carriers.visibility[..., None]
        visibility = (weight * carrier_visibility).sum(dim=2)
        carrier_presence = carriers.presence[..., None]
        observed_presence = (object_owner * carrier_presence).sum(dim=2)
        observed_presence = observed_presence / object_owner.sum(dim=2).clamp_min(1e-6)
        return ObjectRootStateV61(
            feature=feature,
            identity=F.normalize(self.identity(feature), dim=-1, eps=1e-6),
            dynamic=self.dynamic(feature),
            center=center,
            relative_scale=scale.clamp_min(1e-6).sqrt(),
            presence=torch.sigmoid(self.presence(feature)[..., 0]) * observed_presence,
            visibility=torch.sigmoid(self.visibility(feature)[..., 0]) * visibility,
            owner=owner,
        )


class ContinuousCarrierObjectStateEncoderV61(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.carriers = ContinuousCarrierExtractorV61(config)
        self.roots = PersistentObjectRootsV61(config)

    def forward(self, field) -> ContinuousObjectStateV61:
        carriers = self.carriers(field)
        return ContinuousObjectStateV61(carriers=carriers, roots=self.roots(carriers))
