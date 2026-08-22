"""Configuration for externally verified relation Object State learning."""

from __future__ import annotations

from dataclasses import dataclass

from .v54_config import RelationSemanticObjectStateConfig


CHECKPOINT_VERSION = 56
ARCHITECTURE = "verified_relation_object_state_v1"
STAGE = "object_state"


@dataclass(frozen=True)
class VerifiedRelationObjectStateConfig(RelationSemanticObjectStateConfig):
    """Learn only from relation and lifecycle evidence that video can support."""

    reconstruction_weight: float = 0.03
    track_cycle_weight: float = 0.0
    relation_weight: float = 0.0
    owner_evidence_weight: float = 0.0
    lifecycle_weight: float = 0.25
    decoder_support_weight: float = 0.02
    assignment_entropy_weight: float = 0.0
    root_complexity_weight: float = 0.0
    dominant_root_weight: float = 0.0

    relation_partition_weight: float = 1.0
    contrastive_cycle_weight: float = 1.0
    object_support_weight: float = 0.5
    relation_confidence_floor: float = 0.03
    cycle_margin: float = 0.20
    cycle_temperature: float = 0.10

    minimum_teacher_batch_fraction: float = 0.125
    minimum_same_edge_fraction: float = 0.005
    minimum_different_edge_fraction: float = 0.005
    minimum_negative_track_fraction: float = 0.10
    minimum_object_support_fraction: float = 0.05
    minimum_real_target_collapse_margin: float = 0.02

    def validate(self) -> None:
        super().validate()
        disabled = (
            self.track_cycle_weight,
            self.relation_weight,
            self.owner_evidence_weight,
            self.assignment_entropy_weight,
            self.root_complexity_weight,
            self.dominant_root_weight,
        )
        if any(value != 0.0 for value in disabled):
            raise ValueError("v56 forbids legacy owner, cycle, and root-count losses")
        positive = (
            self.relation_partition_weight,
            self.contrastive_cycle_weight,
            self.object_support_weight,
            self.cycle_margin,
            self.cycle_temperature,
            self.minimum_teacher_batch_fraction,
            self.minimum_real_target_collapse_margin,
        )
        if min(positive) <= 0.0:
            raise ValueError("v56 relation objective values must be positive")
        probabilities = (
            self.relation_confidence_floor,
            self.minimum_teacher_batch_fraction,
            self.minimum_same_edge_fraction,
            self.minimum_different_edge_fraction,
            self.minimum_negative_track_fraction,
            self.minimum_object_support_fraction,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v56 evidence thresholds must stay within [0,1]")
