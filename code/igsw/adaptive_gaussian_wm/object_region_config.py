"""Validation shared by single- and dual-encoder object-region profiles."""
from __future__ import annotations


OBJECT_REGION_ARCHITECTURES = (
    "object_region_memory_v1",
    "object_region_dual_encoder_v1",
)


def validate_object_region_config(config) -> None:
    if config.architecture not in OBJECT_REGION_ARCHITECTURES:
        return
    if not config.object_region_memory:
        raise ValueError("object-region architectures require persistent regions")
    state_shape = (
        config.feature_dim,
        config.token_dim,
        config.object_dim,
        config.model_dim,
        config.object_slots,
        config.region_dim,
        config.region_identity_dim,
        config.region_owners,
    )
    if state_shape != (1024, 768, 1536, 1536, 16, 768, 128, 18):
        raise ValueError("object-region state dimensions differ from the contract")
    if not 0.0 < config.region_scene_fraction <= 0.25:
        raise ValueError("region_scene_fraction must be in (0, 0.25]")
    if config.region_owners != config.object_slots + 2:
        raise ValueError("region owners must be object slots plus scene/transient")
    if config.dino_trainable_blocks != 12:
        raise ValueError("object-region JEPA requires 12 trainable DINO blocks")
    if config.dino_model_name != "vit_large_patch14_dinov2.lvd142m":
        raise ValueError("object-region JEPA requires DINOv2-L/14")
    if config.dino_image_size != 518:
        raise ValueError("object-region JEPA requires the 518px DINO patch grid")
    if config.dino_projector_dim != config.region_dim:
        raise ValueError("DINO projector output must equal region_dim")
    if config.curriculum_spatial_steps >= config.curriculum_posterior_steps:
        raise ValueError("object-region curriculum boundaries are not ordered")
    if config.goal_stability_threshold <= 0.0:
        raise ValueError("goal stability threshold must be positive")
    if config.change_residual_readout or config.dense_object_readout:
        raise ValueError("object-region JEPA excludes dense future core readout")
    if config.gaussian_feature_residual or config.rgb_supervision:
        raise ValueError("object-region JEPA excludes Gaussian/RGB core supervision")
    required = (
        config.persistent_identity_key,
        config.relative_transport_dynamics,
        config.factorized_lifecycle,
        config.causal_object_correspondence,
        config.track_presence_semantics,
        config.dual_horizon_dynamics,
    )
    if not all(required):
        raise ValueError("object-region JEPA requires v42 lifecycle and dual horizon")
    is_dual = config.architecture == "object_region_dual_encoder_v1"
    if config.dual_visual_encoder != is_dual:
        raise ValueError("dual_visual_encoder must match the v44 architecture")
    if not is_dual:
        if any(
            (
                config.video_vae_latent_dim,
                config.video_vae_feature_dim,
                config.video_vae_model,
                config.video_vae_contract,
            )
        ):
            raise ValueError("v43 cannot carry a video VAE contract")
        return
    if bool(config.video_vae_model) != bool(config.video_vae_contract):
        raise ValueError("v44 VAE model and contract paths must be set together")
    if config.video_vae_model and not config.video_vae_model.startswith("/"):
        raise ValueError("v44 video_vae_model must be an absolute path")
    if config.video_vae_contract and not config.video_vae_contract.startswith("/"):
        raise ValueError("v44 video_vae_contract must be an absolute path")
    if (config.video_vae_latent_dim, config.video_vae_feature_dim) != (48, 96):
        raise ValueError("v44 requires 48D appearance plus 48D motion features")
    if config.video_vae_clip_frames != 5:
        raise ValueError("v44 requires five-frame causal VAE clips")
    if config.video_vae_short_side < 128 or config.video_vae_short_side % 16:
        raise ValueError("v44 VAE short side must be a multiple of 16")
    if config.video_vae_batch < 1:
        raise ValueError("v44 VAE batch must be positive")
    if config.video_detail_loss_weight <= 0.0:
        raise ValueError("v44 requires a positive video detail objective")
