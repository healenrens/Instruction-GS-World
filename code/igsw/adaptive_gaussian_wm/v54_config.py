"""Configuration for relation-anchored semantic Object State learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 54
ARCHITECTURE = "track_relation_semantic_object_state_v1"
STAGE = "object_state"


@dataclass(frozen=True)
class RelationSemanticObjectStateConfig:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    tracker_image_size: int = 224
    tracker_grid_side: int = 8
    tracker_anchor_fractions: tuple[float, ...] = (0.0, 0.5)

    object_slots: int = 16
    identity_dim: int = 128
    dynamic_dim: int = 128
    state_dim: int = 256
    support_shape_dim: int = 3
    heads: int = 8
    bptt_span: int = 4
    decoder_spatial_rank: int = 4
    identity_update_rate: float = 0.20
    lifecycle_visible_track_fraction: float = 0.20
    dynamic_horizons: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.0

    student_tracklet_temperature: float = 0.10
    student_tracklet_spatial_sigma: float = 0.35
    group_distance_sigma: float = 0.12
    group_locality_sigma: float = 0.40
    group_appearance_floor: float = 0.20
    relation_motion_sigma: float = 0.08
    relation_same_floor: float = 0.55
    relation_different_floor: float = 0.58
    object_motion_floor: float = 0.12
    object_persistence_floor: float = 0.45
    scene_evidence_scale: float = 0.20
    transient_visible_fraction: float = 0.20

    reconstruction_weight: float = 0.03
    track_cycle_weight: float = 1.00
    relation_weight: float = 1.00
    owner_evidence_weight: float = 0.75
    identity_weight: float = 0.50
    semantic_alignment_weight: float = 0.50
    motion_weight: float = 0.50
    lifecycle_weight: float = 0.50
    geometry_weight: float = 0.25
    decoder_support_weight: float = 0.05
    assignment_entropy_weight: float = 0.01
    root_complexity_weight: float = 0.01
    dominant_root_weight: float = 0.10
    dominant_root_limit: float = 0.75
    identity_negative_margin: float = 0.20
    objective_falsification_margin: float = 0.01

    @property
    def owner_count(self) -> int:
        return self.object_slots + 2

    @property
    def tracker_queries(self) -> int:
        return self.tracker_grid_side**2 * len(self.tracker_anchor_fractions)

    @property
    def memory_state_token_dim(self) -> int:
        return self.state_dim + 2 + 1 + self.support_shape_dim + 3

    def validate(self) -> None:
        if self.object_slots < 4:
            raise ValueError("v54 requires multiple persistent object roots")
        if self.state_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v54 state_dim must equal identity_dim plus dynamic_dim")
        if self.state_dim % self.heads:
            raise ValueError("v54 state_dim must be divisible by heads")
        if self.support_shape_dim != 3:
            raise ValueError("v54 support shape must encode aspect and orientation")
        dimensions = (
            self.dino_image_size, self.patch_dim, self.tracker_image_size,
            self.tracker_grid_side, self.object_slots, self.identity_dim,
            self.dynamic_dim, self.state_dim, self.heads, self.bptt_span,
            self.decoder_spatial_rank, *self.dynamic_horizons,
        )
        if min(dimensions) < 1:
            raise ValueError("v54 architectural dimensions must be positive")
        if tuple(sorted(set(self.dynamic_horizons))) != self.dynamic_horizons:
            raise ValueError("v54 dynamic horizons must be unique and increasing")
        if not self.tracker_anchor_fractions:
            raise ValueError("v54 requires point-track anchors")
        if min(self.tracker_anchor_fractions) < 0.0 or max(self.tracker_anchor_fractions) > 1.0:
            raise ValueError("v54 tracker anchors must stay within [0,1]")
        probabilities = (
            self.group_appearance_floor, self.relation_same_floor,
            self.relation_different_floor, self.object_motion_floor,
            self.object_persistence_floor, self.scene_evidence_scale,
            self.transient_visible_fraction, self.lifecycle_visible_track_fraction,
            self.dominant_root_limit, self.identity_negative_margin,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v54 probability-like values must stay in [0,1]")
        scales = (
            self.student_tracklet_temperature, self.student_tracklet_spatial_sigma,
            self.group_distance_sigma, self.group_locality_sigma,
            self.relation_motion_sigma,
        )
        if min(scales) <= 0.0:
            raise ValueError("v54 relation scales must be positive")
        weights = (
            self.reconstruction_weight, self.track_cycle_weight,
            self.relation_weight, self.owner_evidence_weight,
            self.identity_weight, self.semantic_alignment_weight,
            self.motion_weight, self.lifecycle_weight, self.geometry_weight,
            self.decoder_support_weight, self.assignment_entropy_weight,
            self.root_complexity_weight, self.dominant_root_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v54 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
