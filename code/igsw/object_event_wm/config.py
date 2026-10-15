"""Configuration shared by the matched object-effect experiments."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    state_layout: str = "object"
    view_size: int = 0
    feature_dim: int = 768
    width: int = 384
    heads: int = 8
    object_slots: int = 16
    local_tokens: int = 4
    observation_iterations: int = 3
    dynamics_layers: int = 6
    effect_dim: int = 64
    effect_mode: str = "object_step"
    feature_weight: float = 1.0
    rgb_weight: float = 1.0
    change_weight: float = 0.5
    effect_kl_weight: float = 0.001
    update_weight: float = 0.01
    rgb_loss_frames: int = 4

    @property
    def slots(self):
        return self.object_slots + 1
