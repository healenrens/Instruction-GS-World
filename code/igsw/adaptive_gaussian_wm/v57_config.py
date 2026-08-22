"""Configuration for query-conditioned object binding."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 57
ARCHITECTURE = "query_conditioned_object_state_v1"
STAGE = "single_query_binding"


@dataclass(frozen=True)
class QueryConditionedObjectStateConfig:
    patch_dim: int = 1024
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

    heldout_track_weight: float = 1.0
    seed_identity_weight: float = 0.5
    seed_support_weight: float = 0.5
    query_separation_weight: float = 0.5
    semantic_consistency_weight: float = 0.25
    compactness_weight: float = 0.05

    def validate(self) -> None:
        dimensions = (
            self.patch_dim,
            self.model_dim,
            self.identity_dim,
            self.dynamic_dim,
            self.heads,
        )
        if min(dimensions) < 1:
            raise ValueError("v57 dimensions must be positive")
        if self.model_dim % self.heads:
            raise ValueError("v57 model_dim must be divisible by heads")
        probabilities = (
            self.relation_same_floor,
            self.relation_different_floor,
            self.minimum_query_object_confidence,
            self.minimum_query_relation_confidence,
            self.identity_negative_margin,
            self.support_overlap_margin,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v57 probability-like values must stay within [0,1]")
        if min(self.support_temperature, self.spatial_sigma) <= 0.0:
            raise ValueError("v57 temperatures and scales must be positive")
        weights = (
            self.heldout_track_weight,
            self.seed_identity_weight,
            self.seed_support_weight,
            self.query_separation_weight,
            self.semantic_consistency_weight,
            self.compactness_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v57 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
