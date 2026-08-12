"""Configuration for the predictive Object Tube JEPA world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 45
ARCHITECTURE = "predictive_object_tube_v1"


@dataclass(frozen=True)
class PredictiveObjectTubeConfig:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    model_dim: int = 512
    semantic_dim: int = 256
    dynamic_dim: int = 256
    object_slots: int = 16
    identity_dim: int = 128
    action_tokens: int = 4
    action_dim: int = 32
    heads: int = 8
    dropout: float = 0.0
    correspondence_temperature: float = 0.07
    correspondence_spatial_sigma: float = 0.45
    semantic_update_rate: float = 0.10
    observation_mass_tau: float = 1.0
    identity_temperature: float = 0.10
    object_phase_steps: int = 10_000
    effect_phase_steps: int = 30_000
    total_steps: int = 50_000
    curriculum_ramp_steps: int = 2_000
    intervention_margin: float = 0.05

    @property
    def total_slots(self) -> int:
        return self.object_slots

    def validate(self) -> None:
        if self.semantic_dim + self.dynamic_dim != self.model_dim:
            raise ValueError("semantic and dynamic dimensions must sum to model_dim")
        if self.object_slots < 2:
            raise ValueError("v45 requires at least two object slots")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by attention heads")
        if not 0.0 < self.semantic_update_rate <= 1.0:
            raise ValueError("semantic update rate must be in (0,1]")
        if self.observation_mass_tau <= 0.0:
            raise ValueError("observation mass tau must be positive")
        if self.identity_temperature <= 0.0:
            raise ValueError("identity temperature must be positive")
        if not 0 < self.object_phase_steps < self.effect_phase_steps < self.total_steps:
            raise ValueError("v45 curriculum boundaries are invalid")
        if self.curriculum_ramp_steps < 1:
            raise ValueError("v45 curriculum ramp must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
