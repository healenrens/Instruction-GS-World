"""Relation-supervised continuous object codec for the v62 E0 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TeacherObjectStateV62:
    carriers: torch.Tensor
    identity: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    presence: torch.Tensor
    visibility: torch.Tensor
    lifecycle_logits: torch.Tensor
    assignment: torch.Tensor


def frame_observation_v62(observation, frame: int):
    return {
        "coordinates": observation.coordinates[:, frame],
        "dino": observation.dino[:, frame],
        "siglip": observation.siglip[:, frame],
        "support": observation.support[:, frame],
        "visibility": observation.visibility[:, frame],
        "membership": observation.membership,
        "lifecycle": observation.lifecycle[:, frame],
        "object_valid": observation.object_valid,
    }


class TeacherObjectCodecV62(nn.Module):
    """Compress one externally selected object into persistent local carriers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.state_dim
        point_width = config.semantic_dim * 2 + 4
        self.point_input = nn.Sequential(
            nn.LayerNorm(point_width),
            nn.Linear(point_width, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.queries = nn.Parameter(torch.randn(config.carrier_count, dim) * 0.02)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim)
        self.carrier_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )
        self.identity = nn.Sequential(
            nn.LayerNorm(config.semantic_dim * 2),
            nn.Linear(config.semantic_dim * 2, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, config.identity_dim),
        )
        self.presence = nn.Linear(dim, 1)
        self.visibility = nn.Linear(dim, 1)
        self.lifecycle = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, frame: dict[str, torch.Tensor]) -> TeacherObjectStateV62:
        coordinates = frame["coordinates"].float()
        support = frame["support"].float()
        visibility = frame["visibility"].float()
        point_input = torch.cat(
            (
                frame["dino"].float(),
                frame["siglip"].float(),
                coordinates,
                support[..., None],
                visibility[..., None],
            ),
            dim=-1,
        )
        points = self.point_input(point_input)
        queries = self.queries[None].expand(len(points), -1, -1)
        logits = (
            torch.einsum("bkd,bpd->bkp", self.query(queries), self.key(points))
            / self.config.state_dim**0.5
        )
        evidence = (support * visibility).clamp_min(1e-4)
        logits = logits + evidence[:, None].log()
        assignment = torch.softmax(logits.float(), dim=-1)
        observed = torch.einsum("bkp,bpd->bkd", assignment, self.value(points))
        carriers = queries + observed
        carriers = carriers + self.carrier_update(carriers)
        center = torch.einsum("bkp,bpd->bkd", assignment, coordinates)
        offset = coordinates[:, None] - center[:, :, None]
        covariance = torch.einsum("bkp,bkpi,bkpj->bkij", assignment, offset, offset)
        eye = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.config.covariance_floor * eye
        identity_weight = support * visibility
        identity_input = torch.cat((frame["dino"], frame["siglip"]), dim=-1).float()
        pooled = (identity_input * identity_weight[..., None]).sum(dim=1)
        pooled = pooled / identity_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        identity = F.normalize(self.identity(pooled), dim=-1, eps=1e-6)
        observed_mass = evidence.sum(dim=-1, keepdim=True) / evidence.shape[-1]
        presence = torch.sigmoid(self.presence(carriers)[..., 0])
        presence = presence * frame["object_valid"][:, None].float()
        visibility_prediction = torch.sigmoid(self.visibility(carriers)[..., 0])
        visibility_prediction = visibility_prediction * observed_mass.clamp(0.0, 1.0)
        lifecycle_logits = self.lifecycle(carriers.mean(dim=1))
        return TeacherObjectStateV62(
            carriers=carriers,
            identity=identity,
            center=center,
            covariance=covariance,
            presence=presence,
            visibility=visibility_prediction,
            lifecycle_logits=lifecycle_logits,
            assignment=assignment,
        )
