"""Configuration for point-track supervised object-state learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 50
ARCHITECTURE = "trajectory_teacher_rgb_student_object_state_v1"
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
    group_motion_sigma: float = 0.10
    group_distance_sigma: float = 0.12
    group_appearance_floor: float = 0.20
    reconstruction_weight: float = 0.20
    track_assignment_weight: float = 1.0
    identity_weight: float = 0.50
    reappearance_weight: float = 0.50
    motion_weight: float = 0.25
    lifecycle_weight: float = 0.50
    geometry_weight: float = 0.25
    diversity_weight: float = 0.05
    motion_coverage_weight: float = 0.25
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
        return self.state_dim + 2 + 1 + self.support_shape_dim + 2

    def validate(self) -> None:
        if self.object_slots < 2:
            raise ValueError("v50 requires at least two persistent object slots")
        if self.state_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v50 state_dim must equal identity_dim + dynamic_dim")
        if self.state_dim % self.heads:
            raise ValueError("v50 state_dim must be divisible by heads")
        if self.support_shape_dim != 3:
            raise ValueError("v50 support shape is [log aspect, cos 2theta, sin 2theta]")
        if min(
            self.tracker_grid_side,
            self.bptt_span,
            self.decoder_spatial_rank,
            self.effect_factors,
            self.effect_dim,
            self.dynamics_layers,
        ) < 1:
            raise ValueError("v50 architectural dimensions must be positive")
        if not self.tracker_anchor_fractions:
            raise ValueError("v50 needs at least one point-track anchor")
        if min(self.tracker_anchor_fractions) < 0.0 or max(
            self.tracker_anchor_fractions
        ) > 1.0:
            raise ValueError("v50 tracker anchor fractions must stay within [0,1]")
        positive = (
            self.presence_half_life_seconds,
            self.student_tracklet_temperature,
            self.student_tracklet_spatial_sigma,
            self.group_motion_sigma,
            self.group_distance_sigma,
        )
        if min(positive) <= 0.0:
            raise ValueError("v50 scales and temperatures must be positive")
        if not 0.0 < self.identity_update_rate <= 1.0:
            raise ValueError("v50 identity update rate is invalid")

    def to_dict(self) -> dict:
        return asdict(self)
