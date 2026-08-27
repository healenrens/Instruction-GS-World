"""Frozen Object State plus posterior-conditioned future carrier prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .carrier_dynamics_objective_v61 import carrier_dynamics_objective_v61
from .carrier_objective_v61 import track_assignments_v61
from .continuous_carrier_dynamics_v61 import (
    EFFECT_CAPACITIES,
    ContinuousCarrierEffectPosteriorV61,
    EffectConditionedCarrierDynamicsV61,
    frame_state_v61,
    shuffled_effect_v61,
    zero_effect_v61,
)


class ContinuousCarrierDynamicsModelV61(nn.Module):
    def __init__(self, state_model, effect_capacity: str):
        super().__init__()
        if effect_capacity not in EFFECT_CAPACITIES:
            raise ValueError(f"unsupported v61 effect capacity: {effect_capacity}")
        if not state_model.config.uses_dino_alignment:
            raise ValueError("v61 Dynamics requires a DINO-aligned Object State")
        self.config = state_model.config
        self.effect_capacity = effect_capacity
        self.state_model = state_model
        self.state_model.requires_grad_(False)
        self.state_model.eval()
        self.effect_posterior = ContinuousCarrierEffectPosteriorV61(
            self.config, effect_capacity
        )
        _, effect_dim, _ = EFFECT_CAPACITIES[effect_capacity]
        self.dynamics = EffectConditionedCarrierDynamicsV61(self.config, effect_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.state_model.eval()
        return self

    @torch.no_grad()
    def encode_state(self, batch):
        return self.state_model.encode_student(
            batch["video_rgb"], batch["video_pixel_valid"]
        )

    def forward(self, batch, evidence, relation):
        field, state = self.encode_state(batch)
        source = frame_state_v61(state, 0)
        target = frame_state_v61(state, -1)
        carrier_assignment, root_assignment = track_assignments_v61(
            state, field, evidence, self.config
        )
        effect = self.effect_posterior(source, target)
        delta = batch["frame_times"][:, -1] - batch["frame_times"][:, 0]
        correct = self.dynamics(source, effect, delta)
        zero = self.dynamics(source, zero_effect_v61(effect), delta)
        shuffled = self.dynamics(source, shuffled_effect_v61(effect), delta)
        loss, parts = carrier_dynamics_objective_v61(
            self,
            source,
            target,
            correct,
            zero,
            shuffled,
            effect,
            carrier_assignment,
            root_assignment,
            evidence,
            relation,
        )
        return {
            "loss": loss,
            "parts": parts,
            "effect": effect,
            "source": source,
            "target": target,
            "correct": correct,
            "zero": zero,
            "shuffled": shuffled,
        }
