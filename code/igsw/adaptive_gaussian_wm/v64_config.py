"""Configuration for the object-bound transition teacher audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v62_config import ObjectTransitionConfigV62


TEACHER_VERSION = 64
CONTRACT = "temporally_held_object_bound_transition_teacher_v64"


@dataclass(frozen=True)
class ObjectBoundTransitionConfigV64(ObjectTransitionConfigV62):
    transition_horizons: tuple[int, ...] = (1, 2, 4)
    transition_ridge: float = 1e-3
    transition_refinement_steps: int = 2
    component_claim_ratio: float = 0.5
    minimum_component_tracks: int = 4
    minimum_audit_visible_tracks: int = 4

    def validate(self) -> None:
        super().validate()
        if self.transition_ridge <= 0.0:
            raise ValueError("v64 transition ridge must be positive")
        if not self.transition_horizons or min(self.transition_horizons) < 1:
            raise ValueError("v64 transition horizons must be positive")
        if tuple(sorted(set(self.transition_horizons))) != self.transition_horizons:
            raise ValueError("v64 transition horizons must be unique and sorted")
        if self.transition_refinement_steps < 1:
            raise ValueError("v64 needs at least one transition refinement")
        if not 0.0 < self.component_claim_ratio < 1.0:
            raise ValueError("v64 component claim ratio must be between zero and one")
        if self.minimum_component_tracks < 4:
            raise ValueError("v64 affine components need at least four tracks")
        if self.minimum_audit_visible_tracks < 4:
            raise ValueError("v64 affine audit needs at least four visible tracks")

    def to_dict(self) -> dict:
        return asdict(self)
