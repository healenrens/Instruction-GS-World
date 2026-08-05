"""Single-run curriculum schedule for Object-Region JEPA v43."""
from __future__ import annotations

from dataclasses import dataclass

from .config import AdaptiveGaussianWMConfig


@dataclass(frozen=True)
class V43Curriculum:
    step: int
    phase: str
    representation_weight: float
    dynamics_weight: float
    posterior_weight: float


def _ramp(step: int, boundary: int, width: int) -> float:
    return min(1.0, max(0.0, (step - boundary) / float(width)))


def curriculum_at(step: int, config: AdaptiveGaussianWMConfig) -> V43Curriculum:
    if step < 0:
        raise ValueError("curriculum step must be non-negative")
    dynamics = _ramp(
        step, config.curriculum_spatial_steps, config.curriculum_ramp_steps
    )
    posterior = _ramp(
        step, config.curriculum_posterior_steps, config.curriculum_ramp_steps
    )
    if step < config.curriculum_spatial_steps:
        phase = "spatial_temporal_state"
    elif step < config.curriculum_posterior_steps:
        phase = "action_free_short_dynamics"
    else:
        phase = "posterior_dual_horizon"
    return V43Curriculum(
        step=step,
        phase=phase,
        representation_weight=1.0,
        dynamics_weight=dynamics,
        posterior_weight=posterior,
    )
