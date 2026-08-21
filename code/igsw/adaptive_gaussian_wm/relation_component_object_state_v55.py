"""v55 deployable Object State with graph-supported components."""

from __future__ import annotations

from .relation_component_objective_v55 import (
    relation_component_object_state_terms,
)
from .relation_semantic_object_state_v54 import RelationSemanticObjectStateModel


class RelationComponentObjectStateModel(RelationSemanticObjectStateModel):
    def objective_terms(self, prediction, semantic_identity, teacher, evidence):
        return relation_component_object_state_terms(
            prediction,
            semantic_identity,
            teacher,
            evidence,
            self.config,
        )
