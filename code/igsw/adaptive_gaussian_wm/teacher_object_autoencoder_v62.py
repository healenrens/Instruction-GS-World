"""E0 model for compressed continuous teacher-object state."""

from __future__ import annotations

import torch.nn as nn

from .continuous_object_decoder_v62 import ContinuousObjectDecoderV62
from .teacher_object_codec_objective_v62 import teacher_object_codec_objective_v62
from .teacher_object_codec_v62 import TeacherObjectCodecV62, frame_observation_v62


class TeacherObjectAutoencoderV62(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.codec = TeacherObjectCodecV62(config)
        self.decoder = ContinuousObjectDecoderV62(config)

    def encode_frame(self, observation, frame: int):
        return self.codec(frame_observation_v62(observation, frame))

    def forward(self, observation):
        losses, states, decoded_fields = [], [], []
        accumulated = {}
        frame_count = observation.coordinates.shape[1]
        for frame_index in range(frame_count):
            frame = frame_observation_v62(observation, frame_index)
            state = self.codec(frame)
            decoded = self.decoder(state, frame["coordinates"])
            loss, parts = teacher_object_codec_objective_v62(
                state, decoded, frame, self.config
            )
            losses.append(loss)
            states.append(state)
            decoded_fields.append(decoded)
            for name, value in parts.items():
                accumulated[name] = accumulated.get(name, value * 0.0) + value
        parts = {name: value / frame_count for name, value in accumulated.items()}
        return {
            "loss": sum(losses) / frame_count,
            "parts": parts,
            "states": tuple(states),
            "decoded": tuple(decoded_fields),
        }
