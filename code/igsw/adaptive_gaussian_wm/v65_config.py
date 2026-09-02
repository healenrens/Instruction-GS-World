"""Configuration for reliable native-resolution object-transition audit v65."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v64_config import ObjectBoundTransitionConfigV64


TEACHER_VERSION = 65
CONTRACT = "native_reliable_multitrack_object_transition_v65"


@dataclass(frozen=True)
class ReliableNativeTransitionConfigV65(ObjectBoundTransitionConfigV64):
    tracker_grid_side: int = 16
    native_tile_size: int = 224
    native_tile_stride: int = 168
    local_radii_pixels: tuple[float, ...] = (14.0, 28.0, 56.0)
    local_tokens_per_scale: int = 8
    tracker_relay_sigma: float = 0.04
    tracker_min_joint_fraction: float = 0.35
    tracker_reliability_floor: float = 0.20
    appearance_reliability_sigma: float = 0.20
    core_tracks: int = 6
    core_candidate_tracks: int = 12
    minimum_holdout_tracks: int = 2
    transition_huber_delta: float = 0.03
    transition_irls_steps: int = 3
    minimum_audit_valid_fraction: float = 0.50

    def validate(self) -> None:
        super().validate()
        if self.native_tile_size < 32:
            raise ValueError("v65 native tile size is too small")
        if not 0 < self.native_tile_stride <= self.native_tile_size:
            raise ValueError("v65 native tile stride must overlap or touch")
        if not self.local_radii_pixels or min(self.local_radii_pixels) <= 0.0:
            raise ValueError("v65 local radii must be positive")
        if self.local_tokens_per_scale < 1:
            raise ValueError("v65 local pooling needs at least one token")
        if self.tracker_relay_sigma <= 0.0:
            raise ValueError("v65 relay sigma must be positive")
        if not 0.0 < self.tracker_min_joint_fraction <= 1.0:
            raise ValueError("v65 joint visibility fraction is invalid")
        if not 0.0 <= self.tracker_reliability_floor < 1.0:
            raise ValueError("v65 tracker reliability floor is invalid")
        if self.core_tracks < self.minimum_component_tracks:
            raise ValueError("v65 core must satisfy minimum component tracks")
        if self.core_candidate_tracks < self.core_tracks:
            raise ValueError("v65 core candidate pool is smaller than the core")
        if self.minimum_holdout_tracks < 1:
            raise ValueError("v65 needs held-out component tracks")
        if self.transition_huber_delta <= 0.0:
            raise ValueError("v65 Huber delta must be positive")
        if self.transition_irls_steps < 1:
            raise ValueError("v65 robust transition needs an IRLS step")
        if not 0.0 < self.minimum_audit_valid_fraction <= 1.0:
            raise ValueError("v65 audit coverage threshold is invalid")

    def to_dict(self) -> dict:
        return asdict(self)
