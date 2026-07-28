"""Configuration for the adaptive GPSToken object-latent world model."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AdaptiveGaussianWMConfig:
    feature_dim: int
    token_dim: int = 128
    object_dim: int = 128
    model_dim: int = 128
    max_micro_tokens: int = 32
    object_slots: int = 8
    slot_iterations: int = 3
    dynamics_layers: int = 4
    heads: int = 8
    action_tokens: int = 4
    action_dim: int = 32
    flow_hidden_dim: int = 256
    flow_steps: int = 16
    dropout: float = 0.0
    gap_reference: float = 1.0
    history_mask_ratio: float = 0.5
    covariance_floor: float = 1e-3
    target_momentum: float = 0.996
    aggregation_mode: str = "competitive"
    density_mode: str = "legacy"
    fixed_token_fraction: float = 0.5
    joint_flow: bool = False
    normalize_posterior: bool = False
    flow_layers: int = 2
    slot_auxiliary: bool = False
    structured_action: bool = False
    center_conditioned_posterior: bool = False
    delta_only_posterior: bool = False
    slot_geometry_fusion: bool = False
    temporal_prior_context: bool = False
    slot_center_fusion: bool = False
    object_aligned_actions: bool = False
    slot_feature_fusion: bool = False
    decoupled_jepa_slots: bool = False
    action_query_modulation: bool = False
    action_film_modulation: bool = False
    kinematic_action_modulation: bool = False
    learned_velocity_baseline: bool = False
    spatial_slot_attention: bool = False
    token_spatial_precision_floor: float = 0.0
    flow_endpoint_prediction: bool = False
    prior_effect_weight: float = 0.0
    correlated_flow_source: bool = False
    flow_source_scale: float = 1.0
    multi_query_prior_context: bool = False
    prior_query_residual: bool = False
    flow_source_components: int = 1
    flow_lift_scale: float = 1.0
    flow_responsibility_floor: float = 0.05
    flow_source_fit_weight: float = 0.1
    flow_source_min_scale: float = 0.05
    flow_balanced_source_assignment: bool = False
    flow_assignment_temperature: float = 0.05
    canonical_center_action: bool = False
    canonical_center_gate: float = 1.0
    canonical_semantic_action: bool = False
    canonical_activity_gate: bool = False
    canonical_activity_power: float = 0.5
    learned_semantic_action_basis: bool = False
    rgb_semantic_action: bool = False
    semantic_action_basis_weight: float = 0.0
    bounded_residual_action: bool = False
    action_residual_gate: float = 1.0
    action_residual_dropout: float = 0.0
    mode_set_prior: bool = False
    mode_set_geometry_weight: float = 0.0
    mode_set_ordered_assignment: bool = False
    mode_set_normalize_prototypes: bool = False
    mode_set_transformer: bool = False
    mode_set_global_codebook: bool = False
    condition_dim: int = 0
    token_conditioned_prior: bool = False
    rgb_supervision: bool = False
    rgb_short_side: int = 256
    rgb_pad_multiple: int = 16
    rgb_render_chunk: int = 8192
    rgb_loss_weight: float = 0.5
    rgb_ssim_weight: float = 0.2
    rgb_change_loss_weight: float = 0.0
    rgb_change_threshold: float = 0.04
    language_effect_weight: float = 0.0
    zero_action_margin_weight: float = 0.0
    zero_action_relative_margin: float = 0.01
    architecture: str = "legacy"
    persistent_object_memory: bool = False
    relative_geometry: bool = False
    hard_token_gate: bool = False
    min_active_tokens: int = 1
    memory_motion_scale: float = 0.1
    memory_relation_dim: int = 64
    continuous_effect_action: bool = False
    factorized_dynamics: bool = False
    gaussian_feature_residual: bool = False
    gaussian_children: int = 1
    hierarchical_gaussian_carrier: bool = False

    def __post_init__(self) -> None:
        positive = {
            "feature_dim": self.feature_dim,
            "token_dim": self.token_dim,
            "object_dim": self.object_dim,
            "model_dim": self.model_dim,
            "max_micro_tokens": self.max_micro_tokens,
            "object_slots": self.object_slots,
            "slot_iterations": self.slot_iterations,
            "dynamics_layers": self.dynamics_layers,
            "heads": self.heads,
            "action_tokens": self.action_tokens,
            "action_dim": self.action_dim,
            "flow_hidden_dim": self.flow_hidden_dim,
            "flow_steps": self.flow_steps,
            "flow_layers": self.flow_layers,
            "flow_source_components": self.flow_source_components,
            "min_active_tokens": self.min_active_tokens,
            "memory_relation_dim": self.memory_relation_dim,
            "gaussian_children": self.gaussian_children,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.architecture not in ("legacy", "object_memory_v1"):
            raise ValueError("architecture must be legacy or object_memory_v1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.gap_reference <= 0.0:
            raise ValueError("gap_reference must be positive")
        if not 0.0 <= self.history_mask_ratio < 1.0:
            raise ValueError("history_mask_ratio must be in [0, 1)")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive")
        if self.token_spatial_precision_floor < 0.0:
            raise ValueError("token_spatial_precision_floor must be non-negative")
        if self.min_active_tokens > self.max_micro_tokens:
            raise ValueError("min_active_tokens cannot exceed max_micro_tokens")
        if self.gaussian_children not in (1, 2, 4, 8):
            raise ValueError("gaussian_children must be one of 1, 2, 4, or 8")
        if self.hierarchical_gaussian_carrier != (self.gaussian_children > 1):
            raise ValueError(
                "hierarchical_gaussian_carrier must match gaussian_children > 1"
            )
        if self.memory_motion_scale <= 0.0:
            raise ValueError("memory_motion_scale must be positive")
        if self.prior_effect_weight < 0.0:
            raise ValueError("prior_effect_weight must be non-negative")
        if self.flow_source_scale <= 0.0:
            raise ValueError("flow_source_scale must be positive")
        if self.flow_lift_scale < 0.0:
            raise ValueError("flow_lift_scale must be non-negative")
        if not 0.0 <= self.flow_responsibility_floor <= 1.0:
            raise ValueError("flow_responsibility_floor must be in [0, 1]")
        if self.flow_source_fit_weight < 0.0:
            raise ValueError("flow_source_fit_weight must be non-negative")
        if self.flow_source_min_scale <= 0.0:
            raise ValueError("flow_source_min_scale must be positive")
        if self.flow_assignment_temperature <= 0.0:
            raise ValueError("flow_assignment_temperature must be positive")
        if not 0.0 <= self.target_momentum < 1.0:
            raise ValueError("target_momentum must be in [0, 1)")
        if self.aggregation_mode not in ("competitive", "independent", "global"):
            raise ValueError(
                "aggregation_mode must be competitive, independent, or global"
            )
        if self.density_mode not in ("legacy", "adaptive", "fixed"):
            raise ValueError("density_mode must be legacy, adaptive, or fixed")
        if not 0.0 < self.fixed_token_fraction < 1.0:
            raise ValueError("fixed_token_fraction must be in (0, 1)")
        if self.object_aligned_actions and self.action_tokens != self.object_slots:
            raise ValueError(
                "object-aligned actions require action_tokens == object_slots"
            )
        if self.object_aligned_actions and not self.structured_action:
            raise ValueError("object-aligned actions require structured_action")
        if self.prior_query_residual and not self.structured_action:
            raise ValueError("prior query residual requires structured_action")
        if self.decoupled_jepa_slots and (
            self.slot_geometry_fusion
            or self.slot_center_fusion
            or self.slot_feature_fusion
        ):
            raise ValueError("decoupled JEPA slots cannot use slot fusion")
        if self.kinematic_action_modulation and not self.decoupled_jepa_slots:
            raise ValueError("kinematic action modulation requires decoupled slots")
        if self.learned_velocity_baseline and not self.kinematic_action_modulation:
            raise ValueError("learned velocity baseline requires kinematic modulation")
        if self.canonical_center_action and not self.center_conditioned_posterior:
            raise ValueError(
                "canonical center action requires a center-conditioned posterior"
            )
        if not 0.0 < self.canonical_center_gate <= 1.0:
            raise ValueError("canonical center gate must be in (0, 1]")
        if not self.canonical_center_action and self.canonical_center_gate != 1.0:
            raise ValueError("canonical center gate requires center actions")
        if self.canonical_semantic_action and (
            not self.canonical_center_action or self.action_dim < 6
        ):
            raise ValueError(
                "canonical semantic action requires center action and action_dim >= 6"
            )
        if self.canonical_activity_gate and (
            not self.canonical_semantic_action or not self.object_aligned_actions
        ):
            raise ValueError(
                "canonical activity gate requires object-aligned semantic actions"
            )
        if self.canonical_activity_power <= 0.0:
            raise ValueError("canonical activity power must be positive")
        if not self.canonical_activity_gate and self.canonical_activity_power != 0.5:
            raise ValueError("canonical activity power requires its gate")
        if self.learned_semantic_action_basis and not self.canonical_semantic_action:
            raise ValueError("learned semantic basis requires semantic actions")
        if self.rgb_semantic_action and (
            not self.canonical_semantic_action
            or not self.object_aligned_actions
            or not self.rgb_supervision
        ):
            raise ValueError(
                "RGB semantic actions require object alignment and RGB supervision"
            )
        if self.rgb_semantic_action and self.learned_semantic_action_basis:
            raise ValueError("RGB semantic actions cannot use a slot basis")
        if self.semantic_action_basis_weight < 0.0:
            raise ValueError("semantic action basis weight must be non-negative")
        if self.learned_semantic_action_basis != (
            self.semantic_action_basis_weight > 0.0
        ):
            raise ValueError("learned semantic basis requires a positive weight")
        if self.bounded_residual_action and (
            not self.object_aligned_actions
            or not self.canonical_semantic_action
        ):
            raise ValueError(
                "bounded action embedding requires canonical Object-Slot actions"
            )
        if not 0.0 < self.action_residual_gate <= 1.0:
            raise ValueError("action_residual_gate must be in (0, 1]")
        if not self.bounded_residual_action and self.action_residual_gate != 1.0:
            raise ValueError("residual gate requires bounded residual action")
        if self.action_residual_dim == 0 and self.action_residual_gate != 1.0:
            raise ValueError("canonical-only actions cannot gate a residual")
        if not 0.0 <= self.action_residual_dropout < 1.0:
            raise ValueError("action_residual_dropout must be in [0, 1)")
        if self.action_residual_dropout and not self.bounded_residual_action:
            raise ValueError("residual dropout requires bounded residual action")
        if self.action_residual_dropout and self.action_residual_dim == 0:
            raise ValueError("canonical-only actions cannot drop a residual")
        if self.mode_set_prior and self.flow_source_components < 2:
            raise ValueError("mode-set prior requires at least two components")
        if self.mode_set_geometry_weight < 0.0:
            raise ValueError("mode-set geometry weight must be non-negative")
        if self.mode_set_normalize_prototypes and not self.mode_set_prior:
            raise ValueError("prototype normalization requires a mode-set prior")
        if self.mode_set_normalize_prototypes and not self.normalize_posterior:
            raise ValueError(
                "prototype normalization requires a normalized posterior"
            )
        if self.mode_set_transformer and not self.mode_set_prior:
            raise ValueError("mode-set transformer requires a mode-set prior")
        if self.mode_set_transformer and self.flow_hidden_dim % self.heads:
            raise ValueError(
                "mode-set transformer hidden dimension must divide into heads"
            )
        if self.mode_set_global_codebook and not self.mode_set_prior:
            raise ValueError("global codebook requires a mode-set prior")
        if self.condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if self.rgb_short_side < 16:
            raise ValueError("rgb_short_side must be at least 16")
        if self.rgb_pad_multiple < 1:
            raise ValueError("rgb_pad_multiple must be positive")
        if self.rgb_render_chunk < 1:
            raise ValueError("rgb_render_chunk must be positive")
        if self.rgb_loss_weight < 0.0:
            raise ValueError("rgb_loss_weight must be non-negative")
        if self.rgb_ssim_weight < 0.0:
            raise ValueError("rgb_ssim_weight must be non-negative")
        if self.rgb_change_loss_weight < 0.0:
            raise ValueError("RGB change loss weight must be non-negative")
        if self.rgb_change_threshold <= 0.0:
            raise ValueError("RGB change threshold must be positive")
        if self.language_effect_weight < 0.0:
            raise ValueError("language_effect_weight must be non-negative")
        if self.zero_action_margin_weight < 0.0:
            raise ValueError("zero-action margin weight must be non-negative")
        if self.zero_action_relative_margin <= 0.0:
            raise ValueError("zero-action relative margin must be positive")
        if self.persistent_object_memory != self.relative_geometry:
            raise ValueError(
                "persistent object memory and relative geometry must be enabled together"
            )
        if self.architecture == "object_memory_v1":
            if not self.persistent_object_memory or not self.hard_token_gate:
                raise ValueError(
                    "object_memory_v1 requires persistent memory and hard token gates"
                )
            if not self.continuous_effect_action or not self.factorized_dynamics:
                raise ValueError(
                    "object_memory_v1 requires continuous effects and factorized Dynamics"
                )
            if (self.action_tokens, self.action_dim) != (4, 32):
                raise ValueError("object_memory_v1 action contract is [4,32]")
            if (self.min_active_tokens, self.max_micro_tokens) != (64, 256):
                raise ValueError("object_memory_v1 token gate contract is [64,256]")
            if self.condition_dim != 0 or self.rgb_supervision:
                raise ValueError("object_memory_v1 is language-free and feature-only")
            if any(
                (
                    self.canonical_center_action,
                    self.canonical_semantic_action,
                    self.rgb_semantic_action,
                    self.object_aligned_actions,
                )
            ):
                raise ValueError("object_memory_v1 forbids explicit action anchors")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def canonical_action_dim(self) -> int:
        if self.canonical_semantic_action:
            return 6
        if self.canonical_center_action:
            return min(3, self.action_dim)
        return 0

    @property
    def action_residual_dim(self) -> int:
        return self.action_dim - self.canonical_action_dim

    @classmethod
    def tiny(cls, feature_dim: int) -> "AdaptiveGaussianWMConfig":
        return cls(
            feature_dim=feature_dim,
            token_dim=48,
            object_dim=48,
            model_dim=64,
            max_micro_tokens=24,
            object_slots=4,
            slot_iterations=2,
            dynamics_layers=2,
            heads=4,
            action_tokens=2,
            action_dim=12,
            flow_hidden_dim=96,
            flow_steps=8,
            density_mode="adaptive",
            joint_flow=True,
            normalize_posterior=True,
            slot_auxiliary=True,
            structured_action=True,
            center_conditioned_posterior=True,
            temporal_prior_context=True,
        )

    @classmethod
    def full(cls, feature_dim: int) -> "AdaptiveGaussianWMConfig":
        return cls(
            feature_dim=feature_dim,
            token_dim=768,
            object_dim=1536,
            model_dim=1536,
            max_micro_tokens=256,
            object_slots=16,
            slot_iterations=3,
            dynamics_layers=28,
            heads=16,
            action_tokens=4,
            action_dim=64,
            flow_hidden_dim=2048,
            flow_steps=32,
            density_mode="adaptive",
            joint_flow=True,
            normalize_posterior=True,
            slot_auxiliary=True,
            structured_action=True,
            center_conditioned_posterior=True,
            temporal_prior_context=True,
        )

    @classmethod
    def object_memory_full(cls, feature_dim: int) -> "AdaptiveGaussianWMConfig":
        return cls(
            feature_dim=feature_dim,
            token_dim=768,
            object_dim=1536,
            model_dim=1536,
            max_micro_tokens=256,
            min_active_tokens=64,
            object_slots=16,
            slot_iterations=3,
            dynamics_layers=28,
            heads=16,
            action_tokens=4,
            action_dim=32,
            flow_hidden_dim=2048,
            flow_steps=32,
            density_mode="adaptive",
            joint_flow=True,
            normalize_posterior=True,
            slot_auxiliary=True,
            structured_action=True,
            decoupled_jepa_slots=True,
            temporal_prior_context=True,
            spatial_slot_attention=True,
            architecture="object_memory_v1",
            persistent_object_memory=True,
            relative_geometry=True,
            hard_token_gate=True,
            continuous_effect_action=True,
            factorized_dynamics=True,
            gaussian_feature_residual=True,
            gaussian_children=4,
            hierarchical_gaussian_carrier=True,
        )

    @classmethod
    def probe(cls, feature_dim: int) -> "AdaptiveGaussianWMConfig":
        return cls(
            feature_dim=feature_dim,
            token_dim=96,
            object_dim=96,
            model_dim=128,
            max_micro_tokens=64,
            object_slots=8,
            slot_iterations=3,
            dynamics_layers=3,
            heads=8,
            action_tokens=4,
            action_dim=24,
            flow_hidden_dim=256,
            flow_steps=12,
            density_mode="adaptive",
            joint_flow=True,
            normalize_posterior=True,
            slot_auxiliary=True,
            structured_action=True,
            center_conditioned_posterior=True,
            temporal_prior_context=True,
        )
