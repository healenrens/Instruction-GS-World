"""E1 teacher-state oracle for falsifying the v62 transition objective."""

from __future__ import annotations

import torch
import torch.nn as nn

from .continuous_object_decoder_v62 import ContinuousObjectDecoderV62
from .object_effect_posterior_v62 import (
    DeterministicObjectEffectPosteriorV62,
    zero_object_effect_v62,
)
from .object_transition_objective_v62 import (
    matched_shuffle_effect_v62,
    object_change_magnitude_v62,
    object_transition_objective_v62,
)
from .object_transport_dynamics_v62 import ObjectTransportDynamicsV62
from .teacher_object_codec_v62 import TeacherObjectCodecV62, frame_observation_v62


class TeacherTransitionOracleV62(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.codec = TeacherObjectCodecV62(config)
        self.decoder = ContinuousObjectDecoderV62(config)
        self.posterior = DeterministicObjectEffectPosteriorV62(config)
        self.dynamics = ObjectTransportDynamicsV62(config)
        self.codec.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        self.decoder.eval()
        return self

    def load_codec_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        codec = {
            name.removeprefix("codec."): value
            for name, value in state_dict.items()
            if name.startswith("codec.")
        }
        decoder = {
            name.removeprefix("decoder."): value
            for name, value in state_dict.items()
            if name.startswith("decoder.")
        }
        self.codec.load_state_dict(codec, strict=True)
        self.decoder.load_state_dict(decoder, strict=True)

    def encode_frame(self, observation, frame_index: int):
        with torch.no_grad():
            return self.codec(frame_observation_v62(observation, frame_index))

    def forward(self, observation, frame_times, source_index):
        source = self.encode_frame(observation, 0)
        target = self.encode_frame(observation, 1)
        delta_seconds = frame_times[:, 1] - frame_times[:, 0]
        effect = self.posterior(source, target, delta_seconds)
        correct, transport = self.dynamics(source, effect, delta_seconds)
        zero, _ = self.dynamics(source, zero_object_effect_v62(effect), delta_seconds)
        magnitude = object_change_magnitude_v62(source, target)
        shuffled_effect, matched_fraction = matched_shuffle_effect_v62(
            effect, source_index, magnitude
        )
        shuffled, _ = self.dynamics(source, shuffled_effect, delta_seconds)
        output = {
            "source": source,
            "target": target,
            "effect": effect,
            "correct": correct,
            "zero": zero,
            "shuffled": shuffled,
            "transport": transport,
            "matched_shuffle_fraction": matched_fraction,
        }
        target_frame = frame_observation_v62(observation, 1)
        loss, parts = object_transition_objective_v62(self, output, target_frame)
        return {"loss": loss, "parts": parts, **output}
