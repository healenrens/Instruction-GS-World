"""Configuration for the falsifiable v62 teacher-state transition study."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 62
ARCHITECTURE = "continuous_teacher_object_transition_v1"
E0_STAGE = "teacher_object_codec"
E1_STAGE = "teacher_transition_oracle"
STAGES = (E0_STAGE, E1_STAGE)


@dataclass(frozen=True)
class ObjectTransitionConfigV62:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    dino_dim: int = 1024
    siglip_image_size: int = 224
    siglip_patch_size: int = 16
    siglip_dim: int = 768
    semantic_dim: int = 256
    state_dim: int = 256
    identity_dim: int = 256
    carrier_count: int = 16
    carrier_heads: int = 8
    effect_factors: int = 8
    effect_dim: int = 32
    effect_heads: int = 8
    tracker_image_size: int = 224
    tracker_grid_side: int = 8
    tracker_anchor_fractions: tuple[float, ...] = (0.0, 0.5)
    tracker_bidirectional: bool = True
    tracker_include_observed_current_anchor: bool = False
    teacher_future_frames: int = 0
    dynamic_horizons: tuple[int, ...] = (1,)
    relation_confidence_floor: float = 0.10
    group_distance_sigma: float = 0.10
    group_locality_sigma: float = 0.35
    relation_motion_sigma: float = 0.05
    object_motion_floor: float = 0.10
    covariance_floor: float = 1e-3
    support_weight: float = 1.0
    semantic_weight: float = 1.0
    visibility_weight: float = 0.25
    lifecycle_weight: float = 0.25
    geometry_weight: float = 0.25
    capacity_weight: float = 0.02
    intervention_weight: float = 0.25
    intervention_margin: float = 0.05

    @property
    def patch_dim(self) -> int:
        return self.dino_dim

    @property
    def effect_width(self) -> int:
        return self.effect_factors * self.effect_dim

    def validate(self) -> None:
        dimensions = (
            self.dino_image_size,
            self.dino_dim,
            self.siglip_image_size,
            self.siglip_patch_size,
            self.siglip_dim,
            self.semantic_dim,
            self.state_dim,
            self.identity_dim,
            self.carrier_count,
            self.carrier_heads,
            self.effect_factors,
            self.effect_dim,
            self.effect_heads,
            self.tracker_grid_side,
        )
        if min(dimensions) < 1:
            raise ValueError("v62 dimensions must be positive")
        if self.state_dim % self.carrier_heads:
            raise ValueError("v62 state dimension must divide carrier heads")
        if self.state_dim % self.effect_heads:
            raise ValueError("v62 state dimension must divide effect heads")
        if self.dino_dim % self.semantic_dim:
            raise ValueError("v62 DINO dimension must divide semantic dimension")
        if self.siglip_dim % self.semantic_dim:
            raise ValueError("v62 SigLIP dimension must divide semantic dimension")
        if self.siglip_image_size % self.siglip_patch_size:
            raise ValueError("v62 SigLIP image size must divide patch size")
        if self.covariance_floor <= 0.0:
            raise ValueError("v62 covariance floor must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
