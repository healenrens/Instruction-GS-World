"""Configuration for the adaptive GPSToken object-latent world model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .config_validation import validate_action_and_prior_contracts


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
    dense_object_readout: bool = False
    dense_readout_dim: int = 256
    full_dino_features: bool = False
    explicit_background_state: bool = False
    change_residual_readout: bool = False
    change_readout_dim: int = 256
    dual_horizon_dynamics: bool = False
    goal_rollout_weight: float = 1.0
    path_consistency_weight: float = 0.25
    persistent_identity_key: bool = False
    identity_memory_update_rate: float = 0.1
    relative_transport_dynamics: bool = False
    transport_max_support_units: float = 4.0
    factorized_lifecycle: bool = False
    lifecycle_survival_prior: float = 0.98
    lifecycle_birth_prior: float = 0.02
    lifecycle_focal_gamma: float = 2.0
    causal_object_correspondence: bool = False
    track_presence_semantics: bool = False
    correspondence_temperature: float = 0.5
    correspondence_sinkhorn_iterations: int = 32
    correspondence_dustbin_logit: float = 0.0
    correspondence_residual_scale: float = 0.1

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
            "dense_readout_dim": self.dense_readout_dim,
            "change_readout_dim": self.change_readout_dim,
            "correspondence_sinkhorn_iterations": (
                self.correspondence_sinkhorn_iterations
            ),
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.architecture not in (
            "legacy",
            "object_memory_v1",
            "object_memory_v2",
            "object_memory_v3",
        ):
            raise ValueError(
                "architecture must be legacy or an object_memory_v1-v3 variant"
            )
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
        if self.dense_object_readout and self.hierarchical_gaussian_carrier:
            raise ValueError(
                "dense and hierarchical Gaussian readouts are mutually exclusive"
            )
        if self.change_residual_readout and (
            self.dense_object_readout or self.hierarchical_gaussian_carrier
        ):
            raise ValueError(
                "change residual, dense, and hierarchical readouts are mutually exclusive"
            )
        if self.memory_motion_scale <= 0.0:
            raise ValueError("memory_motion_scale must be positive")
        if not 0.0 < self.identity_memory_update_rate <= 1.0:
            raise ValueError("identity_memory_update_rate must be in (0, 1]")
        if self.transport_max_support_units <= 0.0:
            raise ValueError("transport_max_support_units must be positive")
        for name, value in (
            ("lifecycle_survival_prior", self.lifecycle_survival_prior),
            ("lifecycle_birth_prior", self.lifecycle_birth_prior),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.lifecycle_focal_gamma < 0.0:
            raise ValueError("lifecycle_focal_gamma must be non-negative")
        if self.correspondence_temperature <= 0.0:
            raise ValueError("correspondence_temperature must be positive")
        if self.correspondence_residual_scale < 0.0:
            raise ValueError("correspondence_residual_scale must be non-negative")
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
        validate_action_and_prior_contracts(self)
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
        if self.goal_rollout_weight < 0.0 or self.path_consistency_weight < 0.0:
            raise ValueError("dual-horizon loss weights must be non-negative")
        if self.persistent_object_memory != self.relative_geometry:
            raise ValueError(
                "persistent object memory and relative geometry must be enabled together"
            )
        if self.architecture in (
            "object_memory_v1",
            "object_memory_v2",
            "object_memory_v3",
        ):
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
            if not self.full_dino_features or self.feature_dim != 1024:
                raise ValueError("backbone-native DINOv2-L requires feature_dim=1024")
            if not self.change_residual_readout or not self.explicit_background_state:
                raise ValueError(
                    "object_memory_v1 requires change-only readout and background state"
                )
            if self.explicit_background_state and self.aggregation_mode != "competitive":
                raise ValueError(
                    "explicit background state requires competitive aggregation"
                )
            if any(
                (
                    self.canonical_center_action,
                    self.canonical_semantic_action,
                    self.rgb_semantic_action,
                    self.object_aligned_actions,
                )
            ):
                raise ValueError("object_memory_v1 forbids explicit action anchors")
        if self.architecture == "object_memory_v2" and not all(
            (
                self.persistent_identity_key,
                self.relative_transport_dynamics,
                self.factorized_lifecycle,
            )
        ):
            raise ValueError(
                "object_memory_v2 requires identity, relative transport, and lifecycle"
            )
        if self.architecture == "object_memory_v3" and not all(
            (
                self.persistent_identity_key,
                self.relative_transport_dynamics,
                self.factorized_lifecycle,
                self.causal_object_correspondence,
                self.track_presence_semantics,
            )
        ):
            raise ValueError(
                "object_memory_v3 requires identity, correspondence, relative "
                "transport, lifecycle, and track-presence semantics"
            )
        if self.dual_horizon_dynamics and self.architecture not in (
            "object_memory_v1",
            "object_memory_v2",
            "object_memory_v3",
        ):
            raise ValueError("dual-horizon Dynamics requires Object Memory")

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
            gaussian_children=1,
            hierarchical_gaussian_carrier=False,
            dense_object_readout=False,
            dense_readout_dim=256,
            full_dino_features=True,
            explicit_background_state=True,
            change_residual_readout=True,
            change_readout_dim=256,
        )

    @classmethod
    def object_memory_lifecycle_full(
        cls,
        feature_dim: int,
    ) -> "AdaptiveGaussianWMConfig":
        return replace(
            cls.object_memory_full(feature_dim),
            architecture="object_memory_v2",
            persistent_identity_key=True,
            relative_transport_dynamics=True,
            factorized_lifecycle=True,
        )

    @classmethod
    def object_memory_correspondence_full(
        cls,
        feature_dim: int,
    ) -> "AdaptiveGaussianWMConfig":
        return replace(
            cls.object_memory_lifecycle_full(feature_dim),
            architecture="object_memory_v3",
            causal_object_correspondence=True,
            track_presence_semantics=True,
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
