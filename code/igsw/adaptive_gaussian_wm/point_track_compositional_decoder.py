"""Independent object feature fields with lifecycle-aware mask competition."""

from __future__ import annotations

import torch
import torch.nn as nn

from .recurrent_slot_state import _coordinate_basis
from .stable_normalization import stable_unit_normalize


class PointTrackCompositionalDecoder(nn.Module):
    def __init__(self, config):
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
            nn.Linear(21, 2 * rank), nn.GELU(), nn.Linear(2 * rank, rank)
        )

    def forward(self, state, coordinates, valid, object_valid=None):
        identity, dynamic = state["identity"], state["dynamic"]
        object_state = torch.cat((identity, dynamic), dim=-1)
        scene, transient = state["scene"][:, None], state["transient"][:, None]
        position_basis = _coordinate_basis(coordinates)
        position = self.mask_position(position_basis)
        logits = torch.einsum(
            "bkd,bnd->bnk", self.object_mask(object_state), position
        ) / self.config.state_dim**0.5
        offset = coordinates[:, :, None].float() - state["center"][:, None].float()
        angle = 0.5 * torch.atan2(
            state["support_shape"][..., 2], state["support_shape"][..., 1]
        )
        cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
        major = offset[..., 0] * cosine + offset[..., 1] * sine
        minor = -offset[..., 0] * sine + offset[..., 1] * cosine
        aspect = state["support_shape"][..., 0].exp()[:, None]
        variance = (2.0 * state["log_scale"]).exp()[:, None].clamp_min(1e-3)
        distance = major.square() / (variance * aspect) + minor.square() / (variance / aspect)
        observable = state["presence"].float() * state["visibility"].float()
        logits = logits.float() - 0.25 * distance
        logits = logits + observable.clamp_min(1e-5).log()[:, None]
        if object_valid is not None:
            logits = logits.masked_fill(~object_valid[:, None], -1e4)
        nuisance = torch.cat((scene, transient), dim=1)
        nuisance_logits = torch.einsum(
            "bod,bnd->bno", self.nuisance_mask(nuisance), position
        ) / self.config.state_dim**0.5
        owner_logits = torch.cat((logits, nuisance_logits.float()), dim=-1)
        assignment = owner_logits.masked_fill(~valid[..., None], -1e4).softmax(dim=-1)
        assignment = assignment * valid[..., None].float()
        rank = self.config.decoder_spatial_rank
        coefficients = self.object_coefficients(object_state).reshape(
            *object_state.shape[:2], rank, self.config.patch_dim
        )
        scene_coefficients = self.scene_coefficients(scene[:, 0]).reshape(
            len(scene), rank, self.config.patch_dim
        )
        transient_coefficients = self.transient_coefficients(transient[:, 0]).reshape(
            len(scene), rank, self.config.patch_dim
        )
        basis = self.spatial_basis(position_basis)
        object_feature = torch.einsum(
            "bnk,bnr,bkrd->bnd",
            assignment[..., : self.config.object_slots].to(coefficients.dtype),
            basis.to(coefficients.dtype),
            coefficients,
        )
        scene_feature = torch.einsum(
            "bn,bnr,brd->bnd",
            assignment[..., self.config.object_slots].to(scene_coefficients.dtype),
            basis.to(scene_coefficients.dtype),
            scene_coefficients,
        )
        transient_feature = torch.einsum(
            "bn,bnr,brd->bnd",
            assignment[..., self.config.object_slots + 1].to(transient_coefficients.dtype),
            basis.to(transient_coefficients.dtype),
            transient_coefficients,
        )
        reconstruction = stable_unit_normalize(
            object_feature.float() + scene_feature.float() + transient_feature.float()
        )
        return reconstruction, assignment
