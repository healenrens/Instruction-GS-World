"""Configuration for learning-objective-first Object State training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v51_config import PointTrackObjectStateConfig


CHECKPOINT_VERSION = 52
ARCHITECTURE = "relation_supervised_rgb_object_state_v1"
STAGES = ("object_state",)


@dataclass(frozen=True)
class LearningObjectiveObjectStateConfig(PointTrackObjectStateConfig):
    """v52 keeps the RGB-only Student and replaces hard component targets."""

    # Continuous pair evidence. Ambiguous pairs remain unsupervised.
    relation_motion_sigma: float = 0.08
    relation_same_floor: float = 0.62
    relation_different_floor: float = 0.58
    object_motion_floor: float = 0.15
    scene_evidence_scale: float = 0.20
    transient_visible_fraction: float = 0.20

    # Objective topology. These weights combine independently falsifiable terms.
    reconstruction_weight: float = 0.10
    track_cycle_weight: float = 1.00
    relation_weight: float = 1.00
    owner_evidence_weight: float = 1.00
    identity_weight: float = 0.50
    motion_weight: float = 0.50
    lifecycle_weight: float = 0.50
    geometry_weight: float = 0.25
    decoder_support_weight: float = 0.10
    assignment_entropy_weight: float = 0.02
    root_complexity_weight: float = 0.02
    dominant_root_weight: float = 0.10
    dominant_root_limit: float = 0.75
    identity_negative_margin: float = 0.20
    objective_falsification_margin: float = 0.01
    promotion_step: int = 22_500

    def validate(self) -> None:
        super().validate()
        if self.object_slots < 4:
            raise ValueError("v52 requires capacity for multiple persistent objects")
        probabilities = (
            self.relation_same_floor,
            self.relation_different_floor,
            self.object_motion_floor,
            self.scene_evidence_scale,
            self.transient_visible_fraction,
            self.dominant_root_limit,
            self.identity_negative_margin,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v52 probability-like configuration values must stay in [0,1]")
        if self.relation_motion_sigma <= 0.0:
            raise ValueError("v52 relation motion scale must be positive")
        if self.promotion_step != 22_500:
            raise ValueError("v52 Object State promotion is fixed at step 22500")
        weights = (
            self.reconstruction_weight,
            self.track_cycle_weight,
            self.relation_weight,
            self.owner_evidence_weight,
            self.identity_weight,
            self.motion_weight,
            self.lifecycle_weight,
            self.geometry_weight,
            self.decoder_support_weight,
            self.assignment_entropy_weight,
            self.root_complexity_weight,
            self.dominant_root_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v52 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
