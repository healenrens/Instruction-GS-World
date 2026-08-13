"""Minimal pure-video object-state learner for v48."""

from __future__ import annotations

import torch
import torch.nn as nn

from .recurrent_slot_state import RecurrentSlotAttention
from .slot_contrast_objective import slot_contrast_objective
from .slot_mixer_decoder import PositionConditionedSlotMixer
from .v48_config import SlotContrastConfig


class SlotContrastObjectWorldModel(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = RecurrentSlotAttention(config)
        self.decoder = PositionConditionedSlotMixer(config)
        self.contrast_projector = nn.Sequential(
            nn.LayerNorm(config.slot_dim),
            nn.Linear(config.slot_dim, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.contrast_dim),
        )

    def decode_frame(
        self,
        slots: torch.Tensor,
        coordinates: torch.Tensor,
        slot_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(slots, coordinates, slot_valid)

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> dict:
        slots, encoder_assignment = self.state_encoder(
            patches, coordinates, valid, frame_times, observation_mask
        )
        reconstructions, assignments = [], []
        for index in range(patches.shape[1]):
            decoded, assignment = self.decode_frame(
                slots[:, index], coordinates[:, index]
            )
            reconstructions.append(decoded)
            assignments.append(assignment)
        reconstruction = torch.stack(reconstructions, dim=1)
        assignment = torch.stack(assignments, dim=1)
        weighted = assignment * valid[..., None].float()
        mass = weighted.sum(dim=2)
        center = torch.einsum("btnk,btnd->btkd", weighted, coordinates.float())
        center = center / mass[..., None].clamp_min(1e-6)
        offset = coordinates[:, :, :, None].float() - center[:, :, None]
        variance = torch.einsum("btnk,btnkd->btkd", weighted, offset.square())
        variance = variance / mass[..., None].clamp_min(1e-6)
        valid_count = valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        activity = mass / valid_count
        output = {
            "slots": slots,
            "contrast_slots": self.contrast_projector(slots),
            "encoder_assignment": encoder_assignment,
            "assignment": assignment,
            "reconstruction": reconstruction,
            "mass": mass,
            "activity": activity,
            "center": center,
            "relative_scale": variance.mean(dim=-1).clamp_min(1e-8).sqrt(),
        }
        loss, parts = slot_contrast_objective(
            self, patches, valid, observation_mask, output
        )
        output.update(loss=loss, parts=parts)
        return output
