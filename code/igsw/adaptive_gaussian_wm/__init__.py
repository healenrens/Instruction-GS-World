"""Adaptive GPSToken object-latent world model."""

from .config import AdaptiveGaussianWMConfig
from .losses import adaptive_world_model_loss
from .loss_weights import AdaptiveGaussianLossWeights
from .model import AdaptiveGaussianObjectWorldModel
from .synthetic import make_oracle_mode_actions, make_synthetic_batch

__all__ = [
    "AdaptiveGaussianLossWeights",
    "AdaptiveGaussianObjectWorldModel",
    "AdaptiveGaussianWMConfig",
    "adaptive_world_model_loss",
    "make_oracle_mode_actions",
    "make_synthetic_batch",
]
