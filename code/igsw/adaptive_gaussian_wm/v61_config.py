"""Configuration for the single-encoder continuous-carrier Object State study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


CHECKPOINT_VERSION = 61
ARCHITECTURE = "single_encoder_continuous_carrier_object_state_v1"
STAGE = "object_state"
VARIANTS = (
    "dino",
    "siglip2",
    "siglip2_dino",
    "siglip2_dino_object",
)


@dataclass(frozen=True)
class ContinuousCarrierObjectStateConfig:
    variant: str = "siglip2_dino_object"
    dino_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    dino_image_size: int = 224
    dino_dim: int = 1024
    siglip2_image_size: int = 224
    siglip2_patch_size: int = 16
    siglip2_max_patches: int = 256
    siglip2_dim: int = 768
    student_dim: int = 256
    carrier_count: int = 128
    object_roots: int = 16
    identity_dim: int = 128
    dynamic_dim: int = 128
    carrier_heads: int = 8
    root_heads: int = 8
    student_trainable_blocks: int = 4
    carrier_temperature: float = 0.10
    root_temperature: float = 0.10
    spatial_temperature: float = 0.20
    relation_confidence_floor: float = 0.10
    group_distance_sigma: float = 0.10
    group_locality_sigma: float = 0.35
    relation_motion_sigma: float = 0.05
    object_motion_floor: float = 0.10
    dynamic_horizons: tuple[int, ...] = (1, 2, 4)
    tracker_image_size: int = 224
    tracker_grid_side: int = 8
    tracker_anchor_fractions: tuple[float, ...] = (0.0, 0.5)
    tracker_bidirectional: bool = True
    tracker_include_observed_current_anchor: bool = False
    teacher_future_frames: int = 0
    track_assignment_weight: float = 1.0
    relation_weight: float = 1.0
    geometry_weight: float = 0.5
    lifecycle_weight: float = 0.25
    identity_weight: float = 0.5
    dino_alignment_weight: float = 0.0
    object_semantic_weight: float = 0.0
    carrier_diversity_weight: float = 0.02
    root_balance_weight: float = 0.01
    temporal_identity_weight: float = 0.25

    @property
    def student_encoder(self) -> str:
        return "dino" if self.variant == "dino" else "siglip2"

    @property
    def uses_dino_alignment(self) -> bool:
        return self.variant in ("siglip2_dino", "siglip2_dino_object")

    @property
    def uses_object_semantics(self) -> bool:
        return self.variant == "siglip2_dino_object"

    @property
    def total_owners(self) -> int:
        return self.object_roots + 1

    @property
    def patch_dim(self) -> int:
        return self.dino_dim

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unsupported v61 variant: {self.variant}")
        dimensions = (
            self.dino_image_size,
            self.dino_dim,
            self.siglip2_image_size,
            self.siglip2_patch_size,
            self.siglip2_max_patches,
            self.siglip2_dim,
            self.student_dim,
            self.carrier_count,
            self.object_roots,
            self.identity_dim,
            self.dynamic_dim,
            self.carrier_heads,
            self.root_heads,
            self.student_trainable_blocks,
        )
        if min(dimensions) < 1:
            raise ValueError("v61 dimensions must be positive")
        if self.student_dim % self.carrier_heads:
            raise ValueError("student dimension must divide carrier heads")
        if self.student_dim % self.root_heads:
            raise ValueError("student dimension must divide root heads")
        if self.siglip2_image_size % self.siglip2_patch_size:
            raise ValueError("SigLIP2 image size must divide its patch size")
        if self.carrier_count < self.object_roots:
            raise ValueError("carrier count must cover all object roots")
        positive = (
            self.carrier_temperature,
            self.root_temperature,
            self.spatial_temperature,
            self.group_distance_sigma,
            self.group_locality_sigma,
            self.relation_motion_sigma,
        )
        if min(positive) <= 0.0:
            raise ValueError("v61 temperatures must be positive")
        if not 0.0 <= self.relation_confidence_floor < 1.0:
            raise ValueError("relation confidence floor must stay within [0,1)")
        expected_dino = 0.5 if self.uses_dino_alignment else 0.0
        expected_object = 0.5 if self.uses_object_semantics else 0.0
        if self.dino_alignment_weight != expected_dino:
            raise ValueError("variant and DINO alignment weight differ")
        if self.object_semantic_weight != expected_object:
            raise ValueError("variant and object semantic weight differ")

    def to_dict(self) -> dict:
        return asdict(self)


def config_for_variant(variant: str) -> ContinuousCarrierObjectStateConfig:
    config = ContinuousCarrierObjectStateConfig(variant=variant)
    config = replace(
        config,
        dino_alignment_weight=(
            0.5 if variant in ("siglip2_dino", "siglip2_dino_object") else 0.0
        ),
        object_semantic_weight=(0.5 if variant == "siglip2_dino_object" else 0.0),
    )
    config.validate()
    return config
