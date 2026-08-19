"""Configuration for semantic object tokenization and latent object dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_VERSION = 53
ARCHITECTURE = "semantic_object_latent_dynamics_v1"
DECODER_CONTRACT = "separable_low_rank_fp32_v1"
STAGES = ("tokenizer", "dynamics")


@dataclass(frozen=True)
class SemanticObjectWorldModelConfig:
    decoder_contract: str = DECODER_CONTRACT
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    patch_dim: int = 1024
    object_slots: int = 12
    scene_slots: int = 1
    slot_dim: int = 256
    slot_iterations: int = 3
    decoder_rank: int = 8
    temporal_temperature: float = 0.07
    assignment_temperature: float = 0.50
    feature_reconstruction_weight: float = 0.10
    temporal_affinity_weight: float = 1.00
    slot_diversity_weight: float = 0.02
    action_dim: int = 32
    dynamics_depth: int = 6
    dynamics_heads: int = 8
    dynamics_mlp_ratio: int = 4
    action_kl_weight: float = 1e-4
    counterfactual_weight: float = 0.50
    counterfactual_margin: float = 0.10
    delta_state_weight: float = 0.50
    sinkhorn_temperature: float = 0.10
    sinkhorn_iterations: int = 5
    dropout: float = 0.0

    @property
    def total_slots(self) -> int:
        return self.object_slots + self.scene_slots

    def validate(self) -> None:
        if self.decoder_contract != DECODER_CONTRACT:
            raise ValueError("v53 decoder contract differs")
        dimensions = (
            self.dino_image_size,
            self.patch_dim,
            self.object_slots,
            self.scene_slots,
            self.slot_dim,
            self.slot_iterations,
            self.decoder_rank,
            self.action_dim,
            self.dynamics_depth,
            self.dynamics_heads,
            self.dynamics_mlp_ratio,
            self.sinkhorn_iterations,
        )
        if min(dimensions) < 1:
            raise ValueError("v53 architectural dimensions must be positive")
        if self.slot_dim % self.dynamics_heads:
            raise ValueError("v53 slot dimension must be divisible by dynamics heads")
        positive_scales = (
            self.temporal_temperature,
            self.assignment_temperature,
            self.sinkhorn_temperature,
            self.counterfactual_margin,
        )
        if min(positive_scales) <= 0.0:
            raise ValueError("v53 temperatures and margins must be positive")
        weights = (
            self.feature_reconstruction_weight,
            self.temporal_affinity_weight,
            self.slot_diversity_weight,
            self.action_kl_weight,
            self.counterfactual_weight,
            self.delta_state_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v53 objective weights cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
