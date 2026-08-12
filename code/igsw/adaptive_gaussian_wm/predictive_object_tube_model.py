"""Pure-video predictive Object Tube JEPA world model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributed_statistics import roll_batch_with_grad
from .predictive_object_tokenizer import PredictiveObjectTubeTokenizer
from .predictive_object_tube_objective import predictive_object_tube_loss
from .v45_config import PredictiveObjectTubeConfig
from .v45_curriculum import curriculum_at
from .v45_effect_models import ImageGoalEffectPredictor, VideoEffectPosterior
from .v45_object_dynamics import EffectConditionedObjectDynamics
from .video_correspondence import VideoCorrespondence, build_video_correspondence


_STATE_KEYS = (
    "semantic",
    "dynamic",
    "center",
    "log_scale",
    "presence",
    "visibility",
)


def _frame(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {name: state[name][:, index] for name in _STATE_KEYS}


def _prefix(state: dict[str, torch.Tensor], stop: int) -> dict[str, torch.Tensor]:
    return {name: state[name][:, :stop] for name in _STATE_KEYS}


def _detach_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in state.items()}


def _empty_correspondence(
    patches: torch.Tensor, coordinates: torch.Tensor
) -> VideoCorrespondence:
    batch, _, patch_count = patches.shape[:3]
    empty_matrix = patches.new_empty(batch, 0, patch_count, patch_count).float()
    empty_vector = coordinates.new_empty(batch, 0, patch_count).float()
    return VideoCorrespondence(
        forward=empty_matrix,
        backward=empty_matrix,
        cycle_error=empty_vector,
        residual_motion=empty_vector,
    )


class PredictiveObjectTubeWorldModel(nn.Module):
    def __init__(self, config: PredictiveObjectTubeConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.tokenizer = PredictiveObjectTubeTokenizer(config)
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
            raise ValueError("image-goal encoder expects exactly one frame")
        batch = len(patches)
        times = torch.zeros(batch, 1, device=patches.device)
        observed = torch.ones(batch, 1, dtype=torch.bool, device=patches.device)
        return self.tokenizer(
            patches,
            coordinates,
            valid,
            times,
            observed,
            _empty_correspondence(patches, coordinates),
        )

    def predict_image_goal(
        self,
        current: dict[str, torch.Tensor],
        goal: dict[str, torch.Tensor],
        delta_time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        effect = self.goal_effect_predictor(current, goal)
        return effect, self.dynamics(current, effect, delta_time)

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
                current, roll_batch_with_grad(short_effect), short_time
            ),
            "goal_prediction": self.dynamics(current, goal_effect, goal_time),
            "goal_zero": self.dynamics(current, zero_goal, goal_time),
            "goal_shuffled": self.dynamics(
                current, roll_batch_with_grad(goal_effect), goal_time
            ),
        }
        curriculum = curriculum_at(global_step, self.config)
        loss, parts = predictive_object_tube_loss(self, output, curriculum)
        output.update(loss=loss, parts=parts, curriculum=curriculum)
        return output
