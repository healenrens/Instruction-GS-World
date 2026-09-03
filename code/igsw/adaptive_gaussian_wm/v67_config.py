"""Configuration for the continuous predictive object field v67."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 67
ARCHITECTURE = "continuous_predictive_object_field_v1"
STATE_STAGE = "predictive_state"
DYNAMICS_STAGE = "posterior_dynamics"
STAGES = (STATE_STAGE, DYNAMICS_STAGE)


@dataclass(frozen=True)
class ContinuousPredictiveObjectFieldConfigV67:
    # Frozen training-only observation teachers.
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    dino_dim: int = 1024
    siglip_image_size: int = 224
    siglip_patch_size: int = 16
    siglip_dim: int = 768
    semantic_dim: int = 256
    native_tile_size: int = 224
    native_tile_stride: int = 168
    local_radii_pixels: tuple[float, ...] = (14.0, 28.0, 56.0)
    local_tokens_per_scale: int = 8

    # Continuous training coordinates. They are quadrature points, not tokens.
    tracker_grid_side: int = 16
    tracker_anchor_fractions: tuple[float, ...] = (3.0 / 7.0,)
    tracker_relay_sigma: float = 0.04
    tracker_min_joint_fraction: float = 0.35
    tracker_reliability_floor: float = 0.10
    appearance_reliability_sigma: float = 0.25
    coordinate_jitter_fraction: float = 0.35
    query_count: int = 32
    context_fraction: float = 0.50
    crop_side: int = 9
    base_query_scale: float = 0.08
    scale_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    minimum_scale: float = 0.015
    maximum_scale: float = 0.50

    # Deployable RGB-only continuous field.
    field_dim: int = 256
    local_channels: int = 96
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ffn_multiplier: int = 4
    identity_dim: int = 128
    dynamic_dim: int = 192
    code_dim: int = 320
    effect_dim: int = 256
    operator_heads: int = 8
    operator_layers: int = 4
    fourier_bands: int = 8

    # Eight frames at 100 ms: four observed, one midpoint, one terminal target.
    clip_frames: int = 8
    source_frame: int = 3
    midpoint_frame: int = 5
    target_frame: int = 7

    # Soft relation evidence and predictive rate-distortion.
    relation_motion_sigma: float = 0.08
    relation_temperature: float = 0.10
    semantic_weight: float = 1.0
    relation_weight: float = 1.0
    visibility_weight: float = 0.25
    identity_weight: float = 0.50
    point_reconstruction_weight: float = 0.25
    rate_weight: float = 0.005
    uncertainty_weight: float = 0.10
    symmetry_weight: float = 0.10
    transitivity_weight: float = 0.05
    continuity_weight: float = 0.05
    variance_weight: float = 0.05

    # Posterior-conditioned field Dynamics.
    effect_rate_weight: float = 0.002
    effect_variance_weight: float = 0.05
    state_prediction_weight: float = 1.0
    field_prediction_weight: float = 1.0
    short_prediction_weight: float = 1.0
    path_consistency_weight: float = 0.25
    intervention_weight: float = 0.50
    intervention_margin: float = 0.10
    target_ema_momentum: float = 0.996

    @property
    def patch_dim(self) -> int:
        return self.dino_dim

    @property
    def candidate_count(self) -> int:
        return self.tracker_grid_side**2

    @property
    def observable_dim(self) -> int:
        return self.semantic_dim * 2

    def validate(self) -> None:
        positive = (
            self.dino_image_size,
            self.dino_dim,
            self.siglip_image_size,
            self.siglip_patch_size,
            self.siglip_dim,
            self.semantic_dim,
            self.native_tile_size,
            self.native_tile_stride,
            self.tracker_grid_side,
            self.query_count,
            self.crop_side,
            self.field_dim,
            self.identity_dim,
            self.dynamic_dim,
            self.code_dim,
            self.effect_dim,
            self.temporal_layers,
            self.operator_layers,
        )
        if min(positive) < 1:
            raise ValueError("v67 dimensions must be positive")
        if self.code_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v67 code dimension must equal identity plus dynamic")
        if self.query_count > self.candidate_count:
            raise ValueError("v67 query count exceeds coordinate count")
        if self.field_dim % self.temporal_heads:
            raise ValueError("v67 field dimension must divide temporal heads")
        if self.code_dim % self.operator_heads:
            raise ValueError("v67 code dimension must divide operator heads")
        if self.dino_dim % self.semantic_dim or self.siglip_dim % self.semantic_dim:
            raise ValueError("v67 teacher dimensions must divide semantic projection")
        if self.crop_side < 5 or not self.crop_side % 2:
            raise ValueError("v67 crop side must be odd and at least five")
        if not 0.0 < self.context_fraction < 1.0:
            raise ValueError("v67 context fraction must be inside (0,1)")
        if not 0.0 < self.minimum_scale < self.maximum_scale:
            raise ValueError("v67 continuous scale interval is invalid")
        if not self.scale_multipliers or min(self.scale_multipliers) <= 0.0:
            raise ValueError("v67 scale multipliers must be positive")
        if not (
            0 <= self.source_frame < self.midpoint_frame < self.target_frame
            and self.target_frame < self.clip_frames
        ):
            raise ValueError("v67 temporal frame contract is invalid")
        if not 0.0 < self.target_ema_momentum < 1.0:
            raise ValueError("v67 EMA momentum must be inside (0,1)")

    def to_dict(self) -> dict:
        return asdict(self)
