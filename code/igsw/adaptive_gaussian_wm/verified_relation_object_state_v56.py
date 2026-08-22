"""Deployable RGB-only Object State with verified training relations."""

from __future__ import annotations

import torch

from .relation_semantic_object_state_v54 import RelationSemanticObjectStateModel
from .trajectory_relation_teacher_v56 import (
    build_trajectory_relation_teacher_v56,
)
from .verified_relation_objective_v56 import (
    verified_relation_object_state_terms,
)


class VerifiedRelationObjectStateModel(RelationSemanticObjectStateModel):
    def build_teacher(self, point_tracks, frame_times):
        return build_trajectory_relation_teacher_v56(
            point_tracks, self.config, frame_times
        )

    def objective_terms(self, prediction, semantic_identity, teacher, evidence):
        return verified_relation_object_state_terms(
            prediction,
            semantic_identity,
            teacher,
            evidence,
            self.config,
        )

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        teacher = output["teacher"]
        points = teacher.same_confidence.shape[-1]
        off_diagonal = ~torch.eye(
            points, device=teacher.same_confidence.device, dtype=torch.bool
        )[None]
        same_known = (teacher.same_confidence > 0.0) & off_diagonal
        different_known = (teacher.different_confidence > 0.0) & off_diagonal
        relation_known = same_known | different_known
        output["parts"].update(
            {
                "teacher_same_edge_fraction": same_known.float().mean(),
                "teacher_different_edge_fraction": different_known.float().mean(),
                "teacher_known_relation_fraction": relation_known.float().mean(),
                "teacher_negative_track_fraction": (
                    teacher.different_confidence.amax(dim=-1) > 0.0
                )
                .float()
                .mean(),
                "teacher_object_support_fraction": (
                    teacher.object_confidence
                    >= self.config.minimum_object_support_fraction
                )
                .float()
                .mean(),
                "teacher_object_support_mean": teacher.object_confidence.mean(),
                "teacher_scene_target_maximum": teacher.scene_confidence.amax(),
                "teacher_transient_target_maximum": teacher.transient_confidence.amax(),
            }
        )
        return output
