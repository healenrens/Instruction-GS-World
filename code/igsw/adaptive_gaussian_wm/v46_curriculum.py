"""Hard representation boundary for v46 state and transition learning."""

from __future__ import annotations

from dataclasses import dataclass

from .v46_config import ObservationCompleteConfig


@dataclass(frozen=True)
class V46Curriculum:
    phase: str
    state_weight: float
    effect_weight: float
    goal_weight: float


def _ramp_after(step: int, start: int, width: int) -> float:
    return max(0.0, min(1.0, (step - start + 1) / width))


def curriculum_at(step: int, config: ObservationCompleteConfig) -> V46Curriculum:
    if step < 0:
        raise ValueError("curriculum step cannot be negative")
    if step < config.state_phase_steps:
        return V46Curriculum("state", 1.0, 0.0, 0.0)
    effect = _ramp_after(
        step, config.state_phase_steps, config.curriculum_ramp_steps
    )
    goal = _ramp_after(step, config.goal_phase_steps, config.curriculum_ramp_steps)
    phase = "effect" if step < config.goal_phase_steps else "image_goal"
    return V46Curriculum(phase, 0.0, effect, goal)
