"""Expose frozen v62 teacher components for structural audits."""

from __future__ import annotations

from dataclasses import dataclass

from .continuous_object_observation_v62 import (
    build_continuous_object_observation_v62,
)
from .object_transition_teacher_runtime_v62 import (
    ObjectTransitionTeacherRuntimeV62,
)
from .trajectory_relation_teacher_v56 import (
    build_trajectory_relation_teacher_v56,
)


@dataclass(frozen=True)
class ObjectTransitionTeacherBundleV62:
    dino: object
    siglip: object
    evidence: object
    relation: object
    observation: object


class ObjectTransitionAuditRuntimeV62:
    """Run the exact training teachers while retaining their intermediate evidence."""

    def __init__(
        self,
        config,
        device,
        amp,
        dino_checkpoint,
        siglip_checkpoint,
        tracker_checkpoint,
        dino_frame_batch,
        siglip_frame_batch,
    ):
        self.runtime = ObjectTransitionTeacherRuntimeV62(
            config,
            device,
            amp,
            dino_checkpoint,
            siglip_checkpoint,
            tracker_checkpoint,
            dino_frame_batch,
            siglip_frame_batch,
        )

    def __call__(self, batch) -> ObjectTransitionTeacherBundleV62:
        dino = self.runtime.dino(batch)
        evidence = self.runtime.tracker(batch, dino.patches, dino.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence,
            self.runtime.config,
            batch["frame_times"],
        )
        siglip = self.runtime.siglip(batch)
        observation = build_continuous_object_observation_v62(
            batch,
            evidence,
            relation,
            siglip,
            self.runtime.config,
        )
        return ObjectTransitionTeacherBundleV62(
            dino=dino,
            siglip=siglip,
            evidence=evidence,
            relation=relation,
            observation=observation,
        )
