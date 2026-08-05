"""Named model profiles kept separate from config validation."""
from __future__ import annotations

from dataclasses import replace


def tiny_profile(config_type, feature_dim: int):
    return config_type(
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


def full_profile(config_type, feature_dim: int):
    return config_type(
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


def object_memory_profile(config_type, feature_dim: int):
    return config_type(
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


def lifecycle_profile(config_type, feature_dim: int):
    return replace(
        object_memory_profile(config_type, feature_dim),
        architecture="object_memory_v2",
        persistent_identity_key=True,
        relative_transport_dynamics=True,
        factorized_lifecycle=True,
    )


def correspondence_profile(config_type, feature_dim: int):
    return replace(
        lifecycle_profile(config_type, feature_dim),
        architecture="object_memory_v3",
        causal_object_correspondence=True,
        track_presence_semantics=True,
        correspondence_sinkhorn_iterations=256,
        correspondence_logit_clip=4.0,
        correspondence_mass_tolerance=5e-4,
    )


def object_region_profile(config_type, feature_dim: int):
    return replace(
        correspondence_profile(config_type, feature_dim),
        architecture="object_region_memory_v1",
        object_region_memory=True,
        gaussian_feature_residual=False,
        change_residual_readout=False,
        explicit_background_state=True,
        dual_horizon_dynamics=True,
    )


def dual_encoder_region_profile(config_type, feature_dim: int):
    return replace(
        object_region_profile(config_type, feature_dim),
        architecture="object_region_dual_encoder_v1",
        dual_visual_encoder=True,
        video_vae_latent_dim=48,
        video_vae_feature_dim=96,
        video_vae_clip_frames=5,
        video_vae_short_side=256,
        video_vae_batch=1,
        video_detail_loss_weight=0.5,
    )


def probe_profile(config_type, feature_dim: int):
    return config_type(
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
