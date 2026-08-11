"""Single-run three-stage curriculum for v44."""

from __future__ import annotations

from dataclasses import dataclass

from .v44_config import TemporalObjectSetConfig


@dataclass(frozen=True)
class V44Curriculum:
    phase: str
    object_weight: float
    effect_weight: float
    goal_weight: float


def _ramp(step: int, start: int, width: int) -> float:
    return max(0.0, min(1.0, (step - start) / width))


def curriculum_at(step: int, config: TemporalObjectSetConfig) -> V44Curriculum:
    if step < 0:
        raise ValueError("curriculum step cannot be negative")
    effect = _ramp(step, config.object_phase_steps, config.curriculum_ramp_steps)
    goal = _ramp(step, config.effect_phase_steps, config.curriculum_ramp_steps)
    phase = (
        "object"
        if step < config.object_phase_steps
        else "effect"
        if step < config.effect_phase_steps
        else "goal"
    )
    return V44Curriculum(phase, 1.0, effect, goal)
