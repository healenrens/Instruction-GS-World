"""Configuration for the isolated object-transition objective experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v58_config import QueryPersistentObjectStateConfig


CHECKPOINT_VERSION = 59
ARCHITECTURE = "query_object_transition_v1"
STAGE = "object_transition_objective"


@dataclass(frozen=True)
class ObjectTransitionConfig(QueryPersistentObjectStateConfig):
    effect_factors: int = 4
    effect_dim: int = 32
    transition_layers: int = 2
    target_geometry_dim: int = 5
    semantic_loss_weight: float = 1.0
    geometry_loss_weight: float = 1.0
    lifecycle_loss_weight: float = 0.25
    zero_anchor_weight: float = 0.5
    intervention_weight: float = 1.0
    effect_variance_weight: float = 0.05
    intervention_margin: float = 0.10
    minimum_effect_std: float = 0.10
    motion_center_threshold: float = 0.025
    motion_shape_threshold: float = 0.015
    motion_semantic_threshold: float = 0.02

    def validate(self) -> None:
        super().validate()
        dimensions = (
            self.effect_factors,
            self.effect_dim,
            self.transition_layers,
            self.target_geometry_dim,
        )
        if min(dimensions) < 1:
            raise ValueError("v59 dimensions must be positive")
        if self.target_geometry_dim != 5:
            raise ValueError(
                "v59 geometry is relative center plus symmetric covariance"
            )
        weights = (
            self.semantic_loss_weight,
            self.geometry_loss_weight,
            self.lifecycle_loss_weight,
            self.zero_anchor_weight,
            self.intervention_weight,
            self.effect_variance_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v59 objective weights cannot be negative")
        positive = (
            self.intervention_margin,
            self.minimum_effect_std,
            self.motion_center_threshold,
            self.motion_shape_threshold,
            self.motion_semantic_threshold,
        )
        if min(positive) <= 0.0:
            raise ValueError("v59 margins and motion thresholds must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
