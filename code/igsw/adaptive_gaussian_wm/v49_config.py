"""Configuration for trajectory-anchored object-state learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 49
ARCHITECTURE = "trajectory_disentangled_object_state_v1"


@dataclass(frozen=True)
class TrajectoryObjectStateConfig:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    object_slots: int = 16
    identity_dim: int = 128
    dynamic_dim: int = 128
    state_dim: int = 256
    heads: int = 8
    slot_iterations: int = 3
    bptt_span: int = 4
    decoder_spatial_rank: int = 4
    correspondence_temperature: float = 0.07
    correspondence_spatial_sigma: float = 0.35
    trajectory_confidence_floor: float = 0.10
    common_fate_queries: int = 64
    common_fate_motion_sigma: float = 0.12
    common_fate_spatial_sigma: float = 0.75
    identity_temperature: float = 0.10
    identity_update_rate: float = 0.20
    presence_half_life_seconds: float = 2.0
    reconstruction_weight: float = 0.25
    trajectory_weight: float = 1.0
    common_fate_weight: float = 0.25
    identity_weight: float = 0.25
    masked_state_weight: float = 0.50
    lifecycle_weight: float = 0.10
    diversity_weight: float = 0.05
    motion_coverage_weight: float = 0.25
    transient_budget: float = 0.25
    effect_factors: int = 4
    effect_dim: int = 32
    dynamics_layers: int = 4
    intervention_margin: float = 0.05
    effect_start_step: int = 30_000
    effect_ramp_steps: int = 2_000
    dropout: float = 0.0

    @property
    def owner_count(self) -> int:
        return self.object_slots + 2

    def validate(self) -> None:
        if self.object_slots < 2:
            raise ValueError("v49 requires at least two persistent object slots")
        if self.state_dim != self.identity_dim + self.dynamic_dim:
            raise ValueError("v49 state_dim must equal identity_dim + dynamic_dim")
        if self.state_dim % self.heads:
            raise ValueError("v49 state_dim must be divisible by heads")
        if min(
            self.slot_iterations,
            self.bptt_span,
            self.decoder_spatial_rank,
            self.common_fate_queries,
            self.effect_factors,
            self.effect_dim,
            self.dynamics_layers,
        ) < 1:
            raise ValueError("v49 architectural dimensions must be positive")
        positive = (
            self.correspondence_temperature,
            self.correspondence_spatial_sigma,
            self.common_fate_motion_sigma,
            self.common_fate_spatial_sigma,
            self.identity_temperature,
            self.presence_half_life_seconds,
        )
        if min(positive) <= 0.0:
            raise ValueError("v49 temperatures and scales must be positive")
        if not 0.0 <= self.trajectory_confidence_floor < 1.0:
            raise ValueError("v49 trajectory confidence floor is invalid")
        if not 0.0 < self.identity_update_rate <= 1.0:
            raise ValueError("v49 identity update rate is invalid")
        if not 0.0 <= self.transient_budget < 1.0:
            raise ValueError("v49 transient budget is invalid")
        if self.effect_start_step < 1 or self.effect_ramp_steps < 1:
            raise ValueError("v49 effect curriculum is invalid")

    def to_dict(self) -> dict:
        return asdict(self)

