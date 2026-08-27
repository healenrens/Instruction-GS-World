"""Single-Student continuous-carrier Object State model for v61."""

from __future__ import annotations

import torch.nn as nn

from .carrier_objective_v61 import continuous_carrier_objective_v61
from .continuous_carrier_state_v61 import ContinuousCarrierObjectStateEncoderV61
from .student_visual_encoder_v61 import StudentVisualEncoderV61


class ContinuousCarrierObjectWorldModelV61(nn.Module):
    def __init__(
        self,
        config,
        dino_checkpoint: str,
        siglip2_checkpoint: str,
        student_frame_batch: int,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.student = StudentVisualEncoderV61(
            config,
            dino_checkpoint,
            siglip2_checkpoint,
            student_frame_batch,
        )
        self.state_encoder = ContinuousCarrierObjectStateEncoderV61(config)
        self.motion_readout = nn.Linear(
            config.dynamic_dim, len(config.dynamic_horizons) * 2
        )
        self.identity_to_dino = (
            nn.Sequential(
                nn.LayerNorm(config.identity_dim),
                nn.Linear(config.identity_dim, config.student_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.student_dim, config.teacher_projection_dim),
            )
            if config.uses_dino_alignment
            else None
        )
        self.root_to_siglip = (
            nn.Sequential(
                nn.LayerNorm(config.identity_dim),
                nn.Linear(config.identity_dim, config.student_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.student_dim, config.teacher_projection_dim),
            )
            if config.uses_object_semantics
            else None
        )

    def encode_student(self, video_rgb, video_pixel_valid):
        field = self.student(video_rgb, video_pixel_valid)
        return field, self.state_encoder(field)

    def forward(self, batch, evidence, relation, components):
        field, state = self.encode_student(
            batch["video_rgb"], batch["video_pixel_valid"]
        )
        loss, parts, carrier_assignment, root_assignment = (
            continuous_carrier_objective_v61(
                self, field, state, evidence, relation, components
            )
        )
        return {
            "loss": loss,
            "parts": parts,
            "field": field,
            "state": state,
            "carrier_assignment": carrier_assignment,
            "root_assignment": root_assignment,
        }
