"""Configuration for observation-complete object-state learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 47
ARCHITECTURE = "grounded_object_state_v2"


@dataclass(frozen=True)
class ObservationCompleteConfig:
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
    scene_basis_dim: int = 6
    observation_queries: int = 128
    semantic_update_rate: float = 0.10
    scene_update_rate: float = 0.05
    observation_mass_tau: float = 1.0
    association_temperature: float = 0.10
    association_sinkhorn_iterations: int = 8
    association_dustbin_logit: float = -0.5
    identity_temperature: float = 0.10
    object_gain_margin: float = 0.05
    presence_half_life_seconds: float = 4.0
    intervention_margin: float = 0.05
    state_phase_steps: int = 15_000
    goal_phase_steps: int = 30_000
    total_steps: int = 50_000
    curriculum_ramp_steps: int = 2_000

    @property
    def total_slots(self) -> int:
        return self.object_slots

    def validate(self) -> None:
        if self.semantic_dim + self.dynamic_dim != self.model_dim:
            raise ValueError("semantic and dynamic dimensions must sum to model_dim")
        if self.object_slots < 2:
            raise ValueError("v47 requires at least two object slots")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by attention heads")
        if self.scene_basis_dim != 6:
            raise ValueError("v47 scene decoder uses the six-term spatial basis")
        if self.observation_queries < 1:
            raise ValueError("observation query count must be positive")
        if not 0.0 < self.semantic_update_rate <= 1.0:
            raise ValueError("semantic update rate must be in (0,1]")
        if not 0.0 < self.scene_update_rate <= 1.0:
            raise ValueError("scene update rate must be in (0,1]")
        if self.observation_mass_tau <= 0.0:
            raise ValueError("observation mass tau must be positive")
        if self.association_temperature <= 0.0:
            raise ValueError("association temperature must be positive")
        if self.association_sinkhorn_iterations < 1:
            raise ValueError("association Sinkhorn iterations must be positive")
        if self.identity_temperature <= 0.0:
            raise ValueError("identity temperature must be positive")
        if self.object_gain_margin <= 0.0:
            raise ValueError("object gain margin must be positive")
        if self.presence_half_life_seconds <= 0.0:
            raise ValueError("presence half-life must be positive")
        if not 0 < self.state_phase_steps < self.goal_phase_steps < self.total_steps:
            raise ValueError("v47 curriculum boundaries are invalid")
        if self.curriculum_ramp_steps < 1:
            raise ValueError("v47 curriculum ramp must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
