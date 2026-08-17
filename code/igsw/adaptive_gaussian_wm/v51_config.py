"""Configuration for component-balanced point-track Object State learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 51
ARCHITECTURE = "component_teacher_rgb_student_object_state_v2"
STAGES = ("object_state", "latent_effect")


@dataclass(frozen=True)
class PointTrackObjectStateConfig:
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
    presence_half_life_seconds: float = 2.0

    # Teacher grouping is appearance, locality and trajectory-geometry based.
    group_distance_sigma: float = 0.12
    group_locality_sigma: float = 0.40
    group_appearance_floor: float = 0.20
    component_affinity_threshold: float = 0.42
    component_min_visible_frames: int = 3
    component_min_visible_fraction: float = 0.35
    component_min_tracks: int = 2
    component_max_track_fraction: float = 0.40

    # Lifecycle is inferred from gaps and sustained non-observation.
    lifecycle_occlusion_grace_frames: int = 2
    lifecycle_absent_gap_frames: int = 4
    lifecycle_visible_track_fraction: float = 0.20
    lifecycle_absent_track_fraction: float = 0.75

    # Dynamic state is supervised at multiple temporal scales.
    dynamic_horizons: tuple[int, ...] = (1, 2, 4, 8)
    dynamic_geometry_dim: int = 6

    reconstruction_weight: float = 0.20
    track_assignment_weight: float = 1.0
    component_set_weight: float = 0.50
    identity_weight: float = 0.50
    identity_temporal_weight: float = 0.25
    reappearance_weight: float = 0.50
    motion_weight: float = 0.50
    lifecycle_weight: float = 0.50
    geometry_weight: float = 0.25
    diversity_weight: float = 0.05
    effect_factors: int = 4
    effect_dim: int = 32
    dynamics_layers: int = 4
    intervention_margin: float = 0.05
    dropout: float = 0.0

    @property
    def owner_count(self) -> int:
        return self.object_slots + 2

    @property
    def tracker_queries(self) -> int:
        return self.tracker_grid_side**2 * len(self.tracker_anchor_fractions)

    @property
    def state_token_dim(self) -> int:
        return self.state_dim + 2 + 1 + self.support_shape_dim + 3

    def validate(self) -> None:
        if self.object_slots < 2:
            raise ValueError("v51 requires at least two persistent object slots")
        if self.state_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v51 state_dim must equal identity_dim + dynamic_dim")
        if self.state_dim % self.heads:
            raise ValueError("v51 state_dim must be divisible by heads")
        if self.support_shape_dim != 3 or self.dynamic_geometry_dim != 6:
            raise ValueError("v51 geometry dimensions differ from the state contract")
        integer_scales = (
            self.tracker_grid_side,
            self.bptt_span,
            self.decoder_spatial_rank,
            self.effect_factors,
            self.effect_dim,
            self.dynamics_layers,
            self.component_min_visible_frames,
            self.component_min_tracks,
            self.lifecycle_occlusion_grace_frames,
            self.lifecycle_absent_gap_frames,
            *self.dynamic_horizons,
        )
        if min(integer_scales) < 1:
            raise ValueError("v51 architectural dimensions must be positive")
        if tuple(sorted(set(self.dynamic_horizons))) != self.dynamic_horizons:
            raise ValueError("v51 dynamic horizons must be unique and increasing")
        if not self.tracker_anchor_fractions:
            raise ValueError("v51 needs at least one point-track anchor")
        if min(self.tracker_anchor_fractions) < 0.0 or max(self.tracker_anchor_fractions) > 1.0:
            raise ValueError("v51 tracker anchor fractions must stay within [0,1]")
        fractions = (
            self.component_min_visible_fraction,
            self.component_max_track_fraction,
            self.lifecycle_visible_track_fraction,
            self.lifecycle_absent_track_fraction,
        )
        if min(fractions) <= 0.0 or max(fractions) > 1.0:
            raise ValueError("v51 teacher fractions must stay within (0,1]")
        positive = (
            self.presence_half_life_seconds,
            self.student_tracklet_temperature,
            self.student_tracklet_spatial_sigma,
            self.group_distance_sigma,
            self.group_locality_sigma,
        )
        if min(positive) <= 0.0:
            raise ValueError("v51 scales and temperatures must be positive")
        if not 0.0 < self.identity_update_rate <= 1.0:
            raise ValueError("v51 identity update rate is invalid")

    def to_dict(self) -> dict:
        return asdict(self)
