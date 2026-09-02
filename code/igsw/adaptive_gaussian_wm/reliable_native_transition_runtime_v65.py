"""Frozen native-resolution observation runtime for v65 audit."""

from __future__ import annotations

from dataclasses import dataclass

from .native_local_feature_field_v65 import (
    NativeLocalFeatureFieldV65,
    NativeTiledPerceptionRuntimeV65,
    pool_native_local_features_v65,
)
from .reliable_point_tracker_v65 import (
    NativeRelayPointTrackerRuntimeV65,
    ReliablePointTrackEvidenceV65,
    add_appearance_reliability_v65,
)


@dataclass(frozen=True)
class ReliableNativePointObservationV65:
    dino: object
    siglip: object
    dino_valid: object
    siglip_valid: object


@dataclass(frozen=True)
class ReliableNativeTransitionBundleV65:
    field: NativeLocalFeatureFieldV65
    evidence: ReliablePointTrackEvidenceV65
    observation: ReliableNativePointObservationV65


class ReliableNativeTransitionAuditRuntimeV65:
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
        self.perception = NativeTiledPerceptionRuntimeV65(
            config,
            device,
            amp,
            dino_checkpoint,
            siglip_checkpoint,
            dino_frame_batch,
            siglip_frame_batch,
        )
        self.tracker = NativeRelayPointTrackerRuntimeV65(
            config, device, tracker_checkpoint
        )

    def __call__(self, batch):
        evidence = self.tracker(batch)
        field = self.perception(batch)
        dino = pool_native_local_features_v65(
            field.dino,
            evidence.coordinates,
            self.config.local_radii_pixels,
            self.config.local_tokens_per_scale,
        )
        siglip = pool_native_local_features_v65(
            field.siglip,
            evidence.coordinates,
            self.config.local_radii_pixels,
            self.config.local_tokens_per_scale,
        )
        evidence = add_appearance_reliability_v65(
            evidence,
            dino,
            siglip,
            self.config.appearance_reliability_sigma,
            self.config.tracker_reliability_floor,
        )
        observation = ReliableNativePointObservationV65(
            dino=dino.features.detach(),
            siglip=siglip.features.detach(),
            dino_valid=dino.valid.detach(),
            siglip_valid=siglip.valid.detach(),
        )
        return ReliableNativeTransitionBundleV65(
            field=field,
            evidence=evidence,
            observation=observation,
        )
