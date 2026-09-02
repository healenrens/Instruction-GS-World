"""Configuration for the larger-sample G0 object motion-field audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v65_config import ReliableNativeTransitionConfigV65


AUDIT_VERSION = 66
CONTRACT = "object_motion_field_g0_v66"


@dataclass(frozen=True)
class ObjectMotionFieldAuditConfigV66(ReliableNativeTransitionConfigV65):
    local_motion_modes: int = 3
    local_mode_width_scale: float = 0.75
    minimum_local_mode_scale: float = 0.04
    motion_field_ridge: float = 3e-3
    audit_items_per_source: int = 256
    bootstrap_samples: int = 2000
    high_change_quantile: float = 0.50
    minimum_high_change_samples: int = 32

    def validate(self) -> None:
        super().validate()
        if not 1 <= self.local_motion_modes < self.core_tracks:
            raise ValueError("v66 local modes must be fewer than core tracks")
        if self.local_mode_width_scale <= 0.0:
            raise ValueError("v66 local mode width scale must be positive")
        if self.minimum_local_mode_scale <= 0.0:
            raise ValueError("v66 minimum local mode scale must be positive")
        if self.motion_field_ridge <= 0.0:
            raise ValueError("v66 motion field ridge must be positive")
        if self.audit_items_per_source < 128:
            raise ValueError("v66 G0 audit needs at least 128 items per source")
        if self.bootstrap_samples < 1000:
            raise ValueError("v66 bootstrap sample count is too small")
        if not 0.0 < self.high_change_quantile < 1.0:
            raise ValueError("v66 high-change quantile is invalid")
        if self.minimum_high_change_samples < 16:
            raise ValueError("v66 high-change stratum is too small")

    def to_dict(self) -> dict:
        return asdict(self)
