"""Frozen training-only inputs for the v64 object-bound teacher."""

from __future__ import annotations

from dataclasses import dataclass

from .continuous_object_observation_v62 import SiglipVideoFeaturesV62
from .fixed_teacher_projection_v61 import fixed_group_projection_v61
from .object_transition_teacher_runtime_v62 import ObjectTransitionTeacherRuntimeV62
from .point_track_teacher import sample_patch_field
from .trajectory_relation_teacher_v56 import build_trajectory_relation_teacher_v56


@dataclass(frozen=True)
class ObjectBoundPointFeaturesV64:
    dino: object
    siglip: object


@dataclass(frozen=True)
class ObjectBoundTeacherBundleV64:
    evidence: object
    observation: ObjectBoundPointFeaturesV64
    relation: object


class ObjectBoundTransitionAuditRuntimeV64:
    """Run frozen perception teachers without constructing a pseudo object mask."""

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
        loaders = ObjectTransitionTeacherRuntimeV62(
            config,
            device,
            amp,
            dino_checkpoint,
            siglip_checkpoint,
            tracker_checkpoint,
            dino_frame_batch,
            siglip_frame_batch,
        )
        self.config = config
        self.dino = loaders.dino
        self.siglip = loaders.siglip
        self.tracker = loaders.tracker

    def __call__(self, batch):
        dino = self.dino(batch)
        evidence = self.tracker(batch, dino.patches, dino.grid_hw)
        siglip: SiglipVideoFeaturesV62 = self.siglip(batch)
        siglip_points = sample_patch_field(
            siglip.patches,
            evidence.coordinates,
            siglip.grid_hw,
        )
        observation = ObjectBoundPointFeaturesV64(
            dino=fixed_group_projection_v61(
                evidence.sampled_features, self.config.semantic_dim
            ).detach(),
            siglip=fixed_group_projection_v61(
                siglip_points, self.config.semantic_dim
            ).detach(),
        )
        relation = build_trajectory_relation_teacher_v56(
            evidence,
            self.config,
            batch["frame_times"],
        )
        return ObjectBoundTeacherBundleV64(
            evidence=evidence,
            observation=observation,
            relation=relation,
        )
