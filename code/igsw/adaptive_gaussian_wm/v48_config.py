"""Configuration for the pure-video Slot Contrast object-state model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 48
ARCHITECTURE = "videosaur_slot_contrast_v1"


@dataclass(frozen=True)
class SlotContrastConfig:
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    slot_dim: int = 256
    object_slots: int = 16
    heads: int = 8
    slot_iterations: int = 3
    decoder_layers: int = 2
    contrast_dim: int = 128
    contrast_temperature: float = 0.10
    contrast_weight: float = 0.20
    masked_reconstruction_weight: float = 0.25
    active_slot_fraction: float = 0.02
    dropout: float = 0.0

    def validate(self) -> None:
        if self.object_slots < 2:
            raise ValueError("v48 requires at least two slots")
        if self.slot_dim % self.heads:
            raise ValueError("slot_dim must be divisible by attention heads")
        if min(self.slot_iterations, self.decoder_layers, self.contrast_dim) < 1:
            raise ValueError("v48 depth and contrast dimensions must be positive")
        if self.contrast_temperature <= 0.0:
            raise ValueError("contrast temperature must be positive")
        if self.contrast_weight <= 0.0:
            raise ValueError("contrast weight must be positive")
        if self.masked_reconstruction_weight <= 0.0:
            raise ValueError("masked reconstruction weight must be positive")
        if not 0.0 < self.active_slot_fraction < 1.0:
            raise ValueError("active slot fraction must be in (0,1)")

    def to_dict(self) -> dict:
        return asdict(self)
