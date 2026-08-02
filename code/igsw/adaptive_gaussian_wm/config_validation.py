"""Validation helpers for action and prior configuration contracts."""

from __future__ import annotations


def validate_action_and_prior_contracts(config) -> None:
    if config.object_aligned_actions and config.action_tokens != config.object_slots:
        raise ValueError(
            "object-aligned actions require action_tokens == object_slots"
        )
    if config.object_aligned_actions and not config.structured_action:
        raise ValueError("object-aligned actions require structured_action")
    if config.prior_query_residual and not config.structured_action:
        raise ValueError("prior query residual requires structured_action")
    if config.decoupled_jepa_slots and (
        config.slot_geometry_fusion
        or config.slot_center_fusion
        or config.slot_feature_fusion
    ):
        raise ValueError("decoupled JEPA slots cannot use slot fusion")
    if config.kinematic_action_modulation and not config.decoupled_jepa_slots:
        raise ValueError("kinematic action modulation requires decoupled slots")
    if config.learned_velocity_baseline and not config.kinematic_action_modulation:
        raise ValueError("learned velocity baseline requires kinematic modulation")
    if config.canonical_center_action and not config.center_conditioned_posterior:
        raise ValueError(
            "canonical center action requires a center-conditioned posterior"
        )
    if not 0.0 < config.canonical_center_gate <= 1.0:
        raise ValueError("canonical center gate must be in (0, 1]")
    if not config.canonical_center_action and config.canonical_center_gate != 1.0:
        raise ValueError("canonical center gate requires center actions")
    if config.canonical_semantic_action and (
        not config.canonical_center_action or config.action_dim < 6
    ):
        raise ValueError(
            "canonical semantic action requires center action and action_dim >= 6"
        )
    if config.canonical_activity_gate and (
        not config.canonical_semantic_action or not config.object_aligned_actions
    ):
        raise ValueError(
            "canonical activity gate requires object-aligned semantic actions"
        )
    if config.canonical_activity_power <= 0.0:
        raise ValueError("canonical activity power must be positive")
    if not config.canonical_activity_gate and config.canonical_activity_power != 0.5:
        raise ValueError("canonical activity power requires its gate")
    if config.learned_semantic_action_basis and not config.canonical_semantic_action:
        raise ValueError("learned semantic basis requires semantic actions")
    if config.rgb_semantic_action and (
        not config.canonical_semantic_action
        or not config.object_aligned_actions
        or not config.rgb_supervision
    ):
        raise ValueError(
            "RGB semantic actions require object alignment and RGB supervision"
        )
    if config.rgb_semantic_action and config.learned_semantic_action_basis:
        raise ValueError("RGB semantic actions cannot use a slot basis")
    if config.semantic_action_basis_weight < 0.0:
        raise ValueError("semantic action basis weight must be non-negative")
    if config.learned_semantic_action_basis != (
        config.semantic_action_basis_weight > 0.0
    ):
        raise ValueError("learned semantic basis requires a positive weight")
    if config.bounded_residual_action and (
        not config.object_aligned_actions or not config.canonical_semantic_action
    ):
        raise ValueError(
            "bounded action embedding requires canonical Object-Slot actions"
        )
    if not 0.0 < config.action_residual_gate <= 1.0:
        raise ValueError("action_residual_gate must be in (0, 1]")
    if not config.bounded_residual_action and config.action_residual_gate != 1.0:
        raise ValueError("residual gate requires bounded residual action")
    if config.action_residual_dim == 0 and config.action_residual_gate != 1.0:
        raise ValueError("canonical-only actions cannot gate a residual")
    if not 0.0 <= config.action_residual_dropout < 1.0:
        raise ValueError("action_residual_dropout must be in [0, 1)")
    if config.action_residual_dropout and not config.bounded_residual_action:
        raise ValueError("residual dropout requires bounded residual action")
    if config.action_residual_dropout and config.action_residual_dim == 0:
        raise ValueError("canonical-only actions cannot drop a residual")
    if config.mode_set_prior and config.flow_source_components < 2:
        raise ValueError("mode-set prior requires at least two components")
    if config.mode_set_geometry_weight < 0.0:
        raise ValueError("mode-set geometry weight must be non-negative")
    if config.mode_set_normalize_prototypes and not config.mode_set_prior:
        raise ValueError("prototype normalization requires a mode-set prior")
    if config.mode_set_normalize_prototypes and not config.normalize_posterior:
        raise ValueError(
            "prototype normalization requires a normalized posterior"
        )
    if config.mode_set_transformer and not config.mode_set_prior:
        raise ValueError("mode-set transformer requires a mode-set prior")
    if config.mode_set_transformer and config.flow_hidden_dim % config.heads:
        raise ValueError(
            "mode-set transformer hidden dimension must divide into heads"
        )
    if config.mode_set_global_codebook and not config.mode_set_prior:
        raise ValueError("global codebook requires a mode-set prior")
