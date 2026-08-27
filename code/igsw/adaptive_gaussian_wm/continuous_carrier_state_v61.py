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

    def forward(self, field) -> CarrierStateV61:
        token_features = field.features.float()
        coordinates = field.coordinates.float()
        batch, frames, _, dim = token_features.shape
        features, centers, covariances, supports = [], [], [], []
        previous_feature = self.queries[None].expand(batch, -1, -1)
        previous_center = self.spatial_seeds[None].expand(batch, -1, -1)
        for frame in range(frames):
            query = (
                previous_feature + self.context(field.pooled[:, frame].float())[:, None]
            )
            logits = (
                torch.einsum(
                    "bqd,bnd->bqn",
                    self.query(query),
                    self.key(token_features[:, frame]),
                )
                / dim**0.5
            )
            spatial = coordinates[:, frame, None] - previous_center[:, :, None]
            logits = logits - spatial.square().sum(dim=-1) / (
                2.0 * self.config.spatial_temperature**2
            )
            logits = logits.masked_fill(~field.valid[:, frame, None], -torch.inf)
            support = torch.softmax(logits, dim=-1)
            observed = torch.einsum(
                "bqn,bnd->bqd", support, self.value(token_features[:, frame])
            )
            observed_center = torch.einsum(
                "bqn,bnd->bqd", support, coordinates[:, frame]
            )
            offset = coordinates[:, frame, None] - observed_center[:, :, None]
            covariance = torch.einsum("bqn,bqni,bqnj->bqij", support, offset, offset)
            updated = self.memory(
                observed.flatten(0, 1),
                previous_feature.flatten(0, 1),
            ).reshape_as(previous_feature)
            gate = torch.sigmoid(
                self.center_gate(torch.cat((previous_feature, updated), dim=-1))
            ).float()
            center = torch.lerp(previous_center.float(), observed_center.float(), gate)
            features.append(updated)
            centers.append(center)
            covariances.append(covariance)
            supports.append(support)
            previous_feature, previous_center = updated, center
        feature = torch.stack(features, dim=1)
        center = torch.stack(centers, dim=1)
        return CarrierStateV61(
            feature=feature,
            identity=F.normalize(self.identity(feature), dim=-1, eps=1e-6),
            dynamic=self.dynamic(feature),
            center=center,
            covariance=torch.stack(covariances, dim=1),
            presence=torch.sigmoid(self.presence(feature)[..., 0]),
            visibility=torch.sigmoid(self.visibility(feature)[..., 0]),
            support=torch.stack(supports, dim=1),
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

    def forward(self, carriers: CarrierStateV61) -> ObjectRootStateV61:
        batch, frames = carriers.feature.shape[:2]
        features, centers, scales = [], [], []
        owners, visibilities, observed_presences = [], [], []
        previous = self.object_queries[None].expand(batch, -1, -1)
        for frame in range(frames):
            queries = torch.cat(
                (previous, self.scene_query[None].expand(batch, -1, -1)), dim=1
            )
            logits = (
                torch.einsum(
                    "bmd,bqd->bqm",
                    self.query(queries),
                    self.key(carriers.feature[:, frame]),
                )
                / self.config.student_dim**0.5
            )
            distance = (
                carriers.center[:, frame, :, None] - carriers.center[:, frame, None]
            )
            neighborhood = torch.exp(-distance.square().sum(dim=-1) / 0.08).mean(dim=-1)
            logits[..., -1] = logits[..., -1] + (1.0 - neighborhood)
            owner = torch.softmax(logits / self.config.root_temperature, dim=-1)
            object_owner = owner[..., : self.config.object_roots]
            weight = object_owner * carriers.presence[:, frame, :, None]
            weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            observed = torch.einsum("bqm,bqd->bmd", weight, carriers.feature[:, frame])
            center = torch.einsum("bqm,bqd->bmd", weight, carriers.center[:, frame])
            offset = carriers.center[:, frame, :, None] - center[:, None]
            squared_distance = offset.square().sum(dim=-1)
            scale = torch.einsum("bqm,bqm->bm", weight, squared_distance)
            previous = self.memory(
                observed.flatten(0, 1), previous.flatten(0, 1)
            ).reshape_as(previous)
            features.append(previous)
            centers.append(center)
            scales.append(scale)
            owners.append(owner)
            visibilities.append(
                (weight * carriers.visibility[:, frame, :, None]).sum(dim=1)
            )
            observed_presence = (
                object_owner * carriers.presence[:, frame, :, None]
            ).sum(dim=1)
            observed_presence = observed_presence / object_owner.sum(dim=1).clamp_min(
                1e-6
            )
            observed_presences.append(observed_presence)
        feature = torch.stack(features, dim=1)
        observed_presence = torch.stack(observed_presences, dim=1)
        return ObjectRootStateV61(
            feature=feature,
            identity=F.normalize(self.identity(feature), dim=-1, eps=1e-6),
            dynamic=self.dynamic(feature),
            center=torch.stack(centers, dim=1),
            relative_scale=torch.stack(scales, dim=1).clamp_min(1e-6).sqrt(),
            presence=torch.sigmoid(self.presence(feature)[..., 0]) * observed_presence,
            visibility=torch.sigmoid(self.visibility(feature)[..., 0])
            * torch.stack(visibilities, dim=1),
            owner=torch.stack(owners, dim=1),
        )


class ContinuousCarrierObjectStateEncoderV61(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.carriers = ContinuousCarrierExtractorV61(config)
        self.roots = PersistentObjectRootsV61(config)

    def forward(self, field) -> ContinuousObjectStateV61:
        carriers = self.carriers(field)
        return ContinuousObjectStateV61(carriers=carriers, roots=self.roots(carriers))
