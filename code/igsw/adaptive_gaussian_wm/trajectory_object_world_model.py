"""Trajectory-anchored object state with posterior-conditioned world dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from .compositional_object_decoder import CompositionalObjectDecoder
from .disentangled_object_state import DisentangledObjectStateEncoder
from .trajectory_effect_dynamics import (
    EffectConditionedObjectDynamics,
    TrajectoryEffectPosterior,
)
from .trajectory_object_objective import trajectory_object_state_loss
from .trajectory_state_alignment import align_endpoint_state, build_slot_trajectory
from .trajectory_teacher import TrajectoryEvidence
from .v49_config import TrajectoryObjectStateConfig


def _frame_state(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {
        name: state[name][:, index]
        for name in (
            "identity",
            "dynamic",
            "center",
            "log_scale",
            "presence",
            "visibility",
        )
    }


def _flatten_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch, frames = state["identity"].shape[:2]
    result = {}
    for name in (
        "identity",
        "dynamic",
        "center",
        "log_scale",
        "presence",
        "visibility",
        "scene",
        "transient",
    ):
        result[name] = state[name].flatten(0, 1)
    result["batch"] = batch
    result["frames"] = frames
    return result


class TrajectoryObjectWorldModel(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = DisentangledObjectStateEncoder(config)
        self.decoder = CompositionalObjectDecoder(config)
        self.effect_posterior = TrajectoryEffectPosterior(config)
        self.dynamics = EffectConditionedObjectDynamics(config)

    def decode_sequence(
        self,
        state: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = _flatten_state(state)
        decoded, assignment = self.decoder(
            flat,
            coordinates.flatten(0, 1),
            valid.flatten(0, 1),
        )
        batch, frames = flat["batch"], flat["frames"]
        return (
            decoded.reshape(batch, frames, *decoded.shape[1:]),
            assignment.reshape(batch, frames, *assignment.shape[1:]),
        )

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
        trajectory_evidence: TrajectoryEvidence,
        effect_weight: float,
    ) -> dict:
        full_mask = torch.ones_like(observation_mask)
        full_state = self.state_encoder(
            patches, coordinates, valid, frame_times, full_mask
        )
        masked_state = self.state_encoder(
            patches, coordinates, valid, frame_times, observation_mask
        )
        reconstruction, decoder_assignment = self.decode_sequence(
            full_state, coordinates, valid
        )
        slot_trajectory = build_slot_trajectory(
            full_state["assignment"],
            trajectory_evidence,
            self.config.object_slots,
        )
        source = _frame_state(full_state, 0)
        target = align_endpoint_state(full_state, slot_trajectory.endpoint)
        effect = self.effect_posterior(source, target)
        delta_time = frame_times[:, -1] - frame_times[:, 0]
        effect_prediction = self.dynamics(source, effect, delta_time)
        zero_prediction = self.dynamics(source, torch.zeros_like(effect), delta_time)
        if len(effect) > 1:
            shuffled_effect = effect.roll(1, dims=0)
        else:
            shuffled_effect = effect.roll(1, dims=1)
        shuffled_prediction = self.dynamics(source, shuffled_effect, delta_time)
        output = {
            "full_state": full_state,
            "masked_state": masked_state,
            "reconstruction": reconstruction,
            "decoder_assignment": decoder_assignment,
            "coordinates": coordinates,
            "slot_trajectory": slot_trajectory,
            "effect": effect,
            "effect_target": target,
            "effect_prediction": effect_prediction,
            "zero_prediction": zero_prediction,
            "shuffled_prediction": shuffled_prediction,
        }
        loss, parts = trajectory_object_state_loss(
            self,
            patches,
            valid,
            observation_mask,
            trajectory_evidence,
            slot_trajectory,
            output,
            effect_weight,
        )
        output.update(loss=loss, parts=parts)
        return output

