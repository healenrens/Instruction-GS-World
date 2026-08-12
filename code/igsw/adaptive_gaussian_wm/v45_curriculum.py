"""Single-run curriculum for v45 object tubes, effects, and image goals."""

from __future__ import annotations

from dataclasses import dataclass

from .v45_config import PredictiveObjectTubeConfig


@dataclass(frozen=True)
class V45Curriculum:
    phase: str
    object_weight: float
    effect_weight: float
    goal_weight: float


def _ramp(step: int, start: int, width: int) -> float:
    return max(0.0, min(1.0, (step - start) / width))


def curriculum_at(step: int, config: PredictiveObjectTubeConfig) -> V45Curriculum:
    if step < 0:
        raise ValueError("curriculum step cannot be negative")
    effect = _ramp(step, config.object_phase_steps, config.curriculum_ramp_steps)
    goal = _ramp(step, config.effect_phase_steps, config.curriculum_ramp_steps)
    phase = (
        "object_tube"
        if step < config.object_phase_steps
        else "effect"
        if step < config.effect_phase_steps
        else "image_goal"
    )
    return V45Curriculum(phase, 1.0, effect, goal)
