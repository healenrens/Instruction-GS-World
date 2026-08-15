"""Independent per-owner feature fields composed by spatial mask competition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_slot_state import _coordinate_basis
from .stable_normalization import stable_unit_normalize
from .v49_config import TrajectoryObjectStateConfig


class CompositionalObjectDecoder(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.config = config
        rank = config.decoder_spatial_rank
        self.mask_position = nn.Sequential(
            nn.Linear(21, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.object_mask = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.nuisance_mask = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.object_coefficients = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, rank * config.patch_dim),
        )
        self.scene_coefficients = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, rank * config.patch_dim),
        )
        self.transient_coefficients = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, rank * config.patch_dim),
        )
        self.spatial_basis = nn.Sequential(
            nn.Linear(21, 2 * rank),
            nn.GELU(),
            nn.Linear(2 * rank, rank),
        )

    def forward(
        self,
        state: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        object_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = state["identity"]
        dynamic = state["dynamic"]
        if identity.ndim != 3 or dynamic.shape[:2] != identity.shape[:2]:
            raise ValueError("v49 decoder expects per-frame object states")
        if coordinates.ndim != 3 or valid.shape != coordinates.shape[:2]:
            raise ValueError("v49 decoder coordinate shapes differ")
        object_state = torch.cat((identity, dynamic), dim=-1)
        scene = state["scene"][:, None]
        transient = state["transient"][:, None]
        position_basis = _coordinate_basis(coordinates)
        position = self.mask_position(position_basis)
        object_logits = torch.einsum(
            "bkd,bnd->bnk",
            self.object_mask(object_state),
            position,
        ) / self.config.state_dim**0.5
        distance = (
            coordinates[:, :, None].float() - state["center"][:, None].float()
        ).square().sum(dim=-1)
        variance = (2.0 * state["log_scale"].float()).exp()[:, None].clamp_min(1e-3)
        object_logits = object_logits.float() - 0.25 * distance / variance
        object_logits = object_logits + state["presence"].float().clamp_min(0.01).log()[:, None]
        if object_valid is not None:
            if object_valid.shape != identity.shape[:2]:
                raise ValueError("v49 decoder object validity shape differs")
            object_logits = object_logits.masked_fill(~object_valid[:, None], -1e4)
        nuisance_state = torch.cat((scene, transient), dim=1)
        nuisance_logits = torch.einsum(
            "bod,bnd->bno",
            self.nuisance_mask(nuisance_state),
            position,
        ) / self.config.state_dim**0.5
        logits = torch.cat((object_logits, nuisance_logits.float()), dim=-1)
        logits = logits.masked_fill(~valid[..., None], -1e4)
        assignment = logits.softmax(dim=-1) * valid[..., None].float()

        rank = self.config.decoder_spatial_rank
        shape = (*object_state.shape[:2], rank, self.config.patch_dim)
        object_coefficients = self.object_coefficients(object_state).reshape(shape)
        scene_coefficients = self.scene_coefficients(scene[:, 0]).reshape(
            len(scene), rank, self.config.patch_dim
        )
        transient_coefficients = self.transient_coefficients(transient[:, 0]).reshape(
            len(scene), rank, self.config.patch_dim
        )
        basis = self.spatial_basis(position_basis)
        object_feature = torch.einsum(
            "bnk,bnr,bkrd->bnd",
            assignment[:, :, : self.config.object_slots].to(object_coefficients.dtype),
            basis.to(object_coefficients.dtype),
            object_coefficients,
        )
        scene_feature = torch.einsum(
            "bn,bnr,brd->bnd",
            assignment[:, :, self.config.object_slots].to(scene_coefficients.dtype),
            basis.to(scene_coefficients.dtype),
            scene_coefficients,
        )
        transient_feature = torch.einsum(
            "bn,bnr,brd->bnd",
            assignment[:, :, self.config.object_slots + 1].to(transient_coefficients.dtype),
            basis.to(transient_coefficients.dtype),
            transient_coefficients,
        )
        reconstruction = stable_unit_normalize(
            object_feature.float() + scene_feature.float() + transient_feature.float()
        )
        return reconstruction, assignment

