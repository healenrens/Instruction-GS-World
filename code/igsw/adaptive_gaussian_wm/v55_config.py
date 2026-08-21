"""Configuration for relation-graph component Object State learning."""

from __future__ import annotations

from dataclasses import dataclass

from .v54_config import RelationSemanticObjectStateConfig


CHECKPOINT_VERSION = 55
ARCHITECTURE = "relation_graph_component_object_state_v1"
STAGE = "object_state"


@dataclass(frozen=True)
class RelationComponentObjectStateConfig(RelationSemanticObjectStateConfig):
    """Use graph factorization instead of fixed root-count regularization."""

    relation_weight: float = 0.0
    root_complexity_weight: float = 0.0
    dominant_root_weight: float = 0.0
    component_utilization_weight: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.component_utilization_weight <= 0.0:
            raise ValueError("v55 component utilization weight must be positive")
        if self.relation_weight != 0.0:
            raise ValueError("v55 disables the legacy pairwise relation loss")
        if self.root_complexity_weight != 0.0:
            raise ValueError("v55 forbids active-root count minimization")
        if self.dominant_root_weight != 0.0:
            raise ValueError("v55 utilization must come only from the relation graph")
