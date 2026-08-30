"""Frozen external teacher bundle used by both v62 experiments."""

from __future__ import annotations

from .continuous_object_observation_v62 import (
    FrozenSiglipVideoRuntimeV62,
    build_continuous_object_observation_v62,
)
from .frozen_video_encoder import FrozenDinoVideoRuntime
from .point_track_teacher import FrozenPointTrackerRuntime
from .trajectory_relation_teacher_v56 import build_trajectory_relation_teacher_v56


class ObjectTransitionTeacherRuntimeV62:
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
        self.config = config
        self.dino = FrozenDinoVideoRuntime(
            config, device, amp, dino_frame_batch, dino_checkpoint
        )
        self.siglip = FrozenSiglipVideoRuntimeV62(
            siglip_checkpoint, device, siglip_frame_batch
        )
        self.tracker = FrozenPointTrackerRuntime(
            config, device, tracker_checkpoint, sequence_batch=1
        )

    def __call__(self, batch):
        dino = self.dino(batch)
        evidence = self.tracker(batch, dino.patches, dino.grid_hw)
        relation = build_trajectory_relation_teacher_v56(
            evidence, self.config, batch["frame_times"]
        )
        siglip = self.siglip(batch)
        return build_continuous_object_observation_v62(
            batch, evidence, relation, siglip, self.config
        )
