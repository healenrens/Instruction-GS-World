"""Point-track supervised Object State and latent-effect world model."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from .causal_student_tracklets import CausalStudentTrackletEncoder
from .point_track_alignment import match_trajectory_teacher_to_student
from .point_track_compositional_decoder import PointTrackCompositionalDecoder
from .point_track_effect_dynamics import PointTrackEffectDynamics, PointTrackEffectPosterior
from .point_track_object_state import PointTrackObjectStateEncoder
from .point_track_objective import latent_effect_objective, object_state_objective
from .trajectory_component_teacher import build_trajectory_component_teacher
from .v50_config import STAGES


STATE_NAMES = (
    "identity", "dynamic", "center", "log_scale", "support_shape", "presence", "visibility"
)


def frame_state(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {name: state[name][:, index] for name in STATE_NAMES}


class PointTrackObjectWorldModel(nn.Module):
    def __init__(self, config, stage: str):
        super().__init__()
        config.validate()
        self.config = config
        self.student_tracklets = CausalStudentTrackletEncoder(config)
        self.state_encoder = PointTrackObjectStateEncoder(config)
        self.decoder = PointTrackCompositionalDecoder(config)
        self.identity_readout = nn.Linear(config.identity_dim, config.patch_dim)
        self.motion_readout = nn.Linear(config.dynamic_dim, 2)
        self.effect_posterior = PointTrackEffectPosterior(config)
        self.dynamics = PointTrackEffectDynamics(config)
        self.stage = ""
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unsupported v50 stage: {stage}")
        self.stage = stage
        state_trainable = stage == "object_state"
        self.student_tracklets.requires_grad_(state_trainable)
        self.state_encoder.requires_grad_(state_trainable)
        self.decoder.requires_grad_(state_trainable)
        self.identity_readout.requires_grad_(state_trainable)
        self.motion_readout.requires_grad_(state_trainable)
        self.effect_posterior.requires_grad_(not state_trainable)
        self.dynamics.requires_grad_(not state_trainable)

    def decode_sequence(self, state, coordinates, valid):
        batch, frames = state["identity"].shape[:2]
        flat = {
            name: state[name].flatten(0, 1)
            for name in (
                "identity", "dynamic", "center", "log_scale", "support_shape",
                "presence", "visibility", "scene", "transient",
            )
        }
        decoded, assignment = self.decoder(
            flat, coordinates.flatten(0, 1), valid.flatten(0, 1)
        )
        return (
            decoded.reshape(batch, frames, *decoded.shape[1:]),
            assignment.reshape(batch, frames, *assignment.shape[1:]),
        )

    def encode_student(self, patches, coordinates, valid, frame_times):
        """Deployment path: frozen DINO features in, persistent Object State out."""
        tracklets = self.student_tracklets(patches, coordinates, valid)
        state = self.state_encoder(
            tracklets.features, coordinates, valid, frame_times
        )
        return tracklets, state

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        point_tracks,
        grid_hw: tuple[int, int],
    ) -> dict:
        state_context = torch.no_grad() if self.stage == "latent_effect" else nullcontext()
        with state_context:
            tracklets, state = self.encode_student(
                patches, coordinates, valid, frame_times
            )
        if self.stage == "object_state":
            if point_tracks is None:
                raise ValueError("v50 Object State training requires point-track teacher evidence")
            teacher = build_trajectory_component_teacher(point_tracks, self.config)
            match = match_trajectory_teacher_to_student(
                state, point_tracks, teacher, grid_hw, self.config.object_slots
            )
            reconstruction, decoder_assignment = self.decode_sequence(
                state, coordinates, valid
            )
            output = {
                "state": state,
                "student_tracklets": tracklets,
                "teacher": teacher,
                "match": match,
                "reconstruction": reconstruction,
                "decoder_assignment": decoder_assignment,
            }
            loss, parts = object_state_objective(
                self, patches, valid, point_tracks, teacher, match, output
            )
        else:
            source = frame_state(state, 0)
            target = frame_state(state, -1)
            effect = self.effect_posterior(source, target)
            delta_time = frame_times[:, -1] - frame_times[:, 0]
            correct = self.dynamics(source, effect, delta_time)
            zero = self.dynamics(source, torch.zeros_like(effect), delta_time)
            shuffled_effect = effect.roll(1, dims=0 if len(effect) > 1 else 1)
            shuffled = self.dynamics(source, shuffled_effect, delta_time)
            output = {
                "state": state,
                "student_tracklets": tracklets,
                "effect": effect,
                "effect_target": target,
                "effect_prediction": correct,
                "zero_prediction": zero,
                "shuffled_prediction": shuffled,
            }
            loss, parts = latent_effect_objective(self, output)
        output.update(loss=loss, parts=parts)
        return output
