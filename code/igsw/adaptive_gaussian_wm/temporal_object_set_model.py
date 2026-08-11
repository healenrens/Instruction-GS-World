"""Pure-video Temporal Object Set world model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .effect_conditioned_object_dynamics import EffectConditionedObjectDynamics
from .object_effect_models import GoalEffectPredictor, VideoEffectPosterior
from .temporal_object_objective import temporal_object_set_loss
from .temporal_object_tokenizer import TemporalObjectTokenizer
from .v44_config import TemporalObjectSetConfig
from .v44_curriculum import curriculum_at
from .video_correspondence import build_video_correspondence


def _frame(
    state: dict[str, torch.Tensor], index: int, object_slots: int
) -> dict[str, torch.Tensor]:
    keys = ("semantic", "dynamic", "center", "log_scale", "presence", "visibility")
    return {name: state[name][:, index, :object_slots] for name in keys}


def _prefix(state: dict[str, torch.Tensor], stop: int) -> dict[str, torch.Tensor]:
    keys = ("semantic", "dynamic", "center", "log_scale", "presence", "visibility")
    return {name: state[name][:, :stop] for name in keys}


def _detach_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in state.items()}


def _roll_batch(value: torch.Tensor) -> torch.Tensor:
    from .distributed_statistics import roll_batch_with_grad

    return roll_batch_with_grad(value)


class TemporalObjectSetWorldModel(nn.Module):
    def __init__(self, config: TemporalObjectSetConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.tokenizer = TemporalObjectTokenizer(config)
        self.effect_posterior = VideoEffectPosterior(config)
        self.goal_effect_predictor = GoalEffectPredictor(config)
        self.dynamics = EffectConditionedObjectDynamics(config)

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
        global_step: int,
    ) -> dict:
        correspondence = build_video_correspondence(
            patches.detach(),
            coordinates,
            valid,
            self.config.correspondence_temperature,
            self.config.correspondence_spatial_sigma,
        )
        full_mask = torch.ones_like(observation_mask)
        full_state = self.tokenizer(
            patches, coordinates, valid, frame_times, full_mask, correspondence
        )
        masked_state = self.tokenizer(
            patches, coordinates, valid, frame_times, observation_mask, correspondence
        )
        midpoint = max(1, patches.shape[1] // 2)
        current = _detach_state(_frame(full_state, 0, self.config.object_slots))
        short_target = _detach_state(
            _frame(full_state, midpoint, self.config.object_slots)
        )
        goal_target = _detach_state(_frame(full_state, -1, self.config.object_slots))
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
            "correspondence": correspondence,
            "observation_mask": observation_mask,
            "short_effect": short_effect,
            "trajectory_effect": trajectory_effect,
            "goal_effect": goal_effect,
            "short_target": short_target,
            "goal_target": goal_target,
            "short_prediction": self.dynamics(current, short_effect, short_time),
            "short_zero": self.dynamics(current, zero_short, short_time),
            "short_shuffled": self.dynamics(
                current, _roll_batch(short_effect), short_time
            ),
            "goal_prediction": self.dynamics(current, goal_effect, goal_time),
            "goal_zero": self.dynamics(current, zero_goal, goal_time),
            "goal_shuffled": self.dynamics(
                current, _roll_batch(goal_effect), goal_time
            ),
        }
        curriculum = curriculum_at(global_step, self.config)
        loss, parts = temporal_object_set_loss(self, output, curriculum)
        output.update(loss=loss, parts=parts, curriculum=curriculum)
        return output
