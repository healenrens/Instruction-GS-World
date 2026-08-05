"""Typed root/region state containers for the v43 compact world state."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RegionMemoryState:
    feature: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    owner: torch.Tensor
    relative_center: torch.Tensor
    activation: torch.Tensor
    presence: torch.Tensor
    visibility: torch.Tensor
    identity_key: torch.Tensor
    observed: torch.Tensor
    update_gate: torch.Tensor
    association: torch.Tensor
    association_confidence: torch.Tensor
    detail_latent: torch.Tensor
    detail_valid: torch.Tensor
    detail_gate: torch.Tensor

    def validate(self, object_count: int) -> None:
        batch, region_count, feature_dim = self.feature.shape
        expected = {
            "center": (batch, region_count, 2),
            "covariance": (batch, region_count, 2, 2),
            "owner": (batch, region_count, object_count + 2),
            "relative_center": (batch, region_count, 2),
            "activation": (batch, region_count),
            "presence": (batch, region_count),
            "visibility": (batch, region_count),
            "identity_key": (batch, region_count, 128),
            "observed": (batch, region_count),
            "update_gate": (batch, region_count),
            "association": (batch, region_count, region_count),
            "association_confidence": (batch, region_count),
            "detail_valid": (batch, region_count),
            "detail_gate": (batch, region_count),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"region {name} must have shape {shape}")
        if feature_dim <= 0:
            raise ValueError("region feature dimension must be positive")
        if self.detail_latent.shape[:2] != (batch, region_count):
            raise ValueError("region detail latent has invalid leading dimensions")


def stack_region_states(states: list[RegionMemoryState]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot stack an empty region sequence")
    names = (
        "feature",
        "center",
        "covariance",
        "owner",
        "relative_center",
        "activation",
        "presence",
        "visibility",
        "identity_key",
        "observed",
        "update_gate",
        "association_confidence",
        "detail_latent",
        "detail_valid",
        "detail_gate",
    )
    return {
        name: torch.stack([getattr(state, name) for state in states], dim=1)
        for name in names
    }


def select_region_state(
    sequence: dict[str, torch.Tensor],
    index: int,
) -> RegionMemoryState:
    feature = sequence["feature"][:, index]
    batch, regions = feature.shape[:2]
    association = torch.eye(
        regions, device=feature.device, dtype=feature.dtype
    )[None].expand(batch, -1, -1)
    return RegionMemoryState(
        feature=feature,
        center=sequence["center"][:, index],
        covariance=sequence["covariance"][:, index],
        owner=sequence["owner"][:, index],
        relative_center=sequence["relative_center"][:, index],
        activation=sequence["activation"][:, index],
        presence=sequence["presence"][:, index],
        visibility=sequence["visibility"][:, index],
        identity_key=sequence["identity_key"][:, index],
        observed=sequence["observed"][:, index],
        update_gate=sequence["update_gate"][:, index],
        association=association,
        association_confidence=sequence["association_confidence"][:, index],
        detail_latent=sequence["detail_latent"][:, index],
        detail_valid=sequence["detail_valid"][:, index],
        detail_gate=sequence["detail_gate"][:, index],
    )
