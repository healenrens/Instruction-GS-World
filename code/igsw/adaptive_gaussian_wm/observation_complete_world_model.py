"""Pure-video world model built on observation-complete object states."""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributed_statistics import roll_batch_with_grad
from .observation_complete_objective import observation_complete_loss
from .observation_complete_state import ObservationCompleteObjectState
from .v45_effect_models import ImageGoalEffectPredictor, VideoEffectPosterior
from .v45_object_dynamics import EffectConditionedObjectDynamics
from .v47_config import ObservationCompleteConfig
from .v47_curriculum import curriculum_at


_STATE_KEYS = (
    "semantic", "dynamic", "center", "log_scale", "presence", "visibility"
)


def _frame(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {name: state[name][:, index] for name in _STATE_KEYS}


def _prefix(state: dict[str, torch.Tensor], stop: int) -> dict[str, torch.Tensor]:
    return {name: state[name][:, :stop] for name in _STATE_KEYS}


def _detach_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in state.items()}


class ObservationCompleteWorldModel(nn.Module):
    def __init__(self, config: ObservationCompleteConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = ObservationCompleteObjectState(config)
        self.effect_posterior = VideoEffectPosterior(config)
        self.goal_effect_predictor = ImageGoalEffectPredictor(config)
        self.dynamics = EffectConditionedObjectDynamics(config)

    def encode_image_goal(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patches.shape[1] != 1:
            raise ValueError("v47 image-goal encoder expects exactly one frame")
        batch = len(patches)
        times = torch.zeros(batch, 1, device=patches.device)
        observed = torch.ones(batch, 1, dtype=torch.bool, device=patches.device)
        return self.state_encoder(patches, coordinates, valid, times, observed)

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
        global_step: int,
    ) -> dict:
        full_mask = torch.ones_like(observation_mask)
        full_state = self.state_encoder(
            patches, coordinates, valid, frame_times, full_mask
        )
        masked_state = self.state_encoder(
            patches, coordinates, valid, frame_times, observation_mask
        )
        independent_goal = self.encode_image_goal(
            patches[:, -1:], coordinates[:, -1:], valid[:, -1:]
        )
        midpoint = max(1, patches.shape[1] // 2)
        current = _detach_state(_frame(full_state, 0))
        short_target = _detach_state(_frame(full_state, midpoint))
        goal_target = _detach_state(_frame(independent_goal, 0))
        short_effect = self.effect_posterior(
            _detach_state(_prefix(full_state, midpoint + 1))
        )
        trajectory_effect = self.effect_posterior(_detach_state(full_state))
        goal_effect = self.goal_effect_predictor(current, goal_target)
        short_time = frame_times[:, midpoint] - frame_times[:, 0]
        goal_time = frame_times[:, -1] - frame_times[:, 0]
        zero_short = torch.zeros_like(short_effect)
        zero_goal = torch.zeros_like(goal_effect)
        output = {
            "full_state": full_state,
            "masked_state": masked_state,
            "independent_goal_state": independent_goal,
            "observation_mask": observation_mask,
            "coordinates": coordinates,
            "valid": valid,
            "short_effect": short_effect,
            "trajectory_effect": trajectory_effect,
            "goal_effect": goal_effect,
            "short_target": short_target,
            "goal_target": goal_target,
            "short_prediction": self.dynamics(current, short_effect, short_time),
            "short_zero": self.dynamics(current, zero_short, short_time),
            "short_shuffled": self.dynamics(
                current, roll_batch_with_grad(short_effect), short_time
            ),
            "goal_prediction": self.dynamics(current, goal_effect, goal_time),
            "goal_zero": self.dynamics(current, zero_goal, goal_time),
            "goal_shuffled": self.dynamics(
                current, roll_batch_with_grad(goal_effect), goal_time
            ),
        }
        curriculum = curriculum_at(global_step, self.config)
        loss, parts = observation_complete_loss(self, output, curriculum)
        output.update(loss=loss, parts=parts, curriculum=curriculum)
        return output
