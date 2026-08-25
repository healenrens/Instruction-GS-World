"""Configuration for calibrated gated-residual object transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v59_config import ObjectTransitionConfig


CHECKPOINT_VERSION = 60
ARCHITECTURE = "gated_residual_object_transition_v1"
STAGE = "object_transition_calibration"


@dataclass(frozen=True)
class GatedResidualTransitionConfig(ObjectTransitionConfig):
    change_noise_floor: float = 0.02
    change_scale: float = 0.08
    low_change_threshold: float = 0.25
    high_change_threshold: float = 0.60
    base_anchor_weight: float = 0.5
    gate_calibration_weight: float = 0.5
    no_change_consistency_weight: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.change_noise_floor < 0.0 or self.change_scale <= 0.0:
            raise ValueError("v60 change calibration scales are invalid")
        if not 0.0 < self.low_change_threshold < self.high_change_threshold < 1.0:
            raise ValueError("v60 change thresholds must be ordered within (0,1)")
        weights = (
            self.base_anchor_weight,
            self.gate_calibration_weight,
            self.no_change_consistency_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v60 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
