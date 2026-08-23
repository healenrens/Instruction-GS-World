"""Configuration for query-conditioned persistent object state learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 58
ARCHITECTURE = "query_persistent_object_state_v1"
STAGE = "query_persistent_state"


@dataclass(frozen=True)
class QueryPersistentObjectStateConfig:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    tracker_image_size: int = 224
    tracker_grid_side: int = 8
    tracker_anchor_fractions: tuple[float, ...] = (0.0, 0.5)
    tracker_bidirectional: bool = True
    tracker_include_observed_current_anchor: bool = True
    dynamic_horizons: tuple[int, ...] = (1, 2, 4)
    model_dim: int = 256
    identity_dim: int = 128
    dynamic_dim: int = 128
    heads: int = 8
    dropout: float = 0.0

    relation_same_floor: float = 0.55
    relation_different_floor: float = 0.58
    minimum_query_object_confidence: float = 0.05
    minimum_query_relation_confidence: float = 0.05
    support_temperature: float = 0.10
    spatial_sigma: float = 0.35
    identity_negative_margin: float = 0.20
    support_overlap_margin: float = 0.20
    group_distance_sigma: float = 0.12
    group_locality_sigma: float = 0.40
    relation_motion_sigma: float = 0.08
    relation_confidence_floor: float = 0.10
    object_motion_floor: float = 0.12

    teacher_future_frames: int = 4
    minimum_condition_query_fraction: float = 0.25
    minimum_condition_trainable_fraction: float = 0.10
    minimum_aggregate_trainable_fraction: float = 0.25
    minimum_aggregate_occluded_fraction: float = 0.001

    heldout_track_weight: float = 1.0
    seed_identity_weight: float = 0.5
    seed_support_weight: float = 0.5
    query_separation_weight: float = 0.5
    visibility_weight: float = 1.0
    identity_persistence_weight: float = 0.25
    semantic_consistency_weight: float = 0.25
    compactness_weight: float = 0.05
    dynamic_motion_weight: float = 0.5
    geometry_motion_weight: float = 0.25

    def validate(self) -> None:
        dimensions = (
            self.patch_dim,
            self.model_dim,
            self.identity_dim,
            self.dynamic_dim,
            self.heads,
            self.dino_image_size,
            self.tracker_image_size,
            self.tracker_grid_side,
            self.teacher_future_frames,
            *self.dynamic_horizons,
        )
        if min(dimensions) < 1:
            raise ValueError("v58 dimensions must be positive")
        if self.model_dim % self.heads:
            raise ValueError("v58 model_dim must be divisible by heads")
        probabilities = (
            self.relation_same_floor,
            self.relation_different_floor,
            self.minimum_query_object_confidence,
            self.minimum_query_relation_confidence,
            self.identity_negative_margin,
            self.support_overlap_margin,
            self.relation_confidence_floor,
            self.object_motion_floor,
            self.minimum_condition_query_fraction,
            self.minimum_condition_trainable_fraction,
            self.minimum_aggregate_trainable_fraction,
            self.minimum_aggregate_occluded_fraction,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v58 probability-like values must stay within [0,1]")
        scales = (
            self.support_temperature,
            self.spatial_sigma,
            self.group_distance_sigma,
            self.group_locality_sigma,
            self.relation_motion_sigma,
        )
        if min(scales) <= 0.0:
            raise ValueError("v58 temperatures and scales must be positive")
        if not self.tracker_anchor_fractions:
            raise ValueError("v58 requires tracker anchor fractions")
        if min(self.tracker_anchor_fractions) < 0.0 or max(self.tracker_anchor_fractions) > 1.0:
            raise ValueError("v58 tracker anchor fractions must stay within [0,1]")
        if tuple(sorted(set(self.dynamic_horizons))) != self.dynamic_horizons:
            raise ValueError("v58 dynamic horizons must be unique and increasing")
        weights = (
            self.heldout_track_weight,
            self.seed_identity_weight,
            self.seed_support_weight,
            self.query_separation_weight,
            self.visibility_weight,
            self.identity_persistence_weight,
            self.semantic_consistency_weight,
            self.compactness_weight,
            self.dynamic_motion_weight,
            self.geometry_motion_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v58 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tracker_queries(self) -> int:
        return self.tracker_grid_side**2 * len(self.tracker_anchor_fractions)
