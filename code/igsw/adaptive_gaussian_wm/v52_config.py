"""Configuration for learning-objective-first Object State training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 52
ARCHITECTURE = "relation_supervised_rgb_object_state_v1"
STAGES = ("object_state",)


@dataclass(frozen=True)
class LearningObjectiveObjectStateConfig:
    """v52 keeps the RGB-only Student and replaces hard component targets."""

    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    tracker_image_size: int = 224
    tracker_grid_side: int = 8
    tracker_anchor_fractions: tuple[float, ...] = (0.0, 0.5)
    student_tracklet_temperature: float = 0.10
    student_tracklet_spatial_sigma: float = 0.35
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

    # Appearance and relative trajectory evidence; none of these yields an object ID.
    group_distance_sigma: float = 0.12
    group_locality_sigma: float = 0.40
    group_appearance_floor: float = 0.20

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
            raise ValueError("v52 requires capacity for multiple persistent objects")
        if self.state_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v52 state_dim must equal identity_dim plus dynamic_dim")
        if self.state_dim % self.heads:
            raise ValueError("v52 state_dim must be divisible by heads")
        if self.support_shape_dim != 3:
            raise ValueError("v52 support shape must use aspect and orientation")
        dimensions = (
            self.dino_image_size, self.patch_dim, self.tracker_image_size,
            self.tracker_grid_side, self.object_slots, self.identity_dim,
            self.dynamic_dim, self.state_dim, self.heads, self.bptt_span,
            self.decoder_spatial_rank, *self.dynamic_horizons,
        )
        if min(dimensions) < 1:
            raise ValueError("v52 architectural dimensions must be positive")
        if tuple(sorted(set(self.dynamic_horizons))) != self.dynamic_horizons:
            raise ValueError("v52 dynamic horizons must be unique and increasing")
        if not self.tracker_anchor_fractions:
            raise ValueError("v52 needs at least one point-track anchor")
        if min(self.tracker_anchor_fractions) < 0.0 or max(self.tracker_anchor_fractions) > 1.0:
            raise ValueError("v52 tracker anchors must stay within [0,1]")
        probabilities = (
            self.group_appearance_floor,
            self.relation_same_floor,
            self.relation_different_floor,
            self.object_motion_floor,
            self.scene_evidence_scale,
            self.transient_visible_fraction,
            self.lifecycle_visible_track_fraction,
            self.dominant_root_limit,
            self.identity_negative_margin,
        )
        if min(probabilities) < 0.0 or max(probabilities) > 1.0:
            raise ValueError("v52 probability-like configuration values must stay in [0,1]")
        if self.relation_motion_sigma <= 0.0:
            raise ValueError("v52 relation motion scale must be positive")
        positive_scales = (
            self.student_tracklet_temperature,
            self.student_tracklet_spatial_sigma,
            self.group_distance_sigma,
            self.group_locality_sigma,
        )
        if min(positive_scales) <= 0.0:
            raise ValueError("v52 relation and Student scales must be positive")
        if not 0.0 < self.identity_update_rate <= 1.0:
            raise ValueError("v52 identity update rate is invalid")
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
