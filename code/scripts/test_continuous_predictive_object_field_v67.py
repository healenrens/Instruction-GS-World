#!/usr/bin/env python3
"""Numerical structural test for the v67 state and Dynamics paths."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_field_sampling_v67 import (  # noqa: E402
    context_coordinate_mask_v67,
    evenly_spaced_query_indices_v67,
    stratified_query_coordinates_v67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_object_field_v67 import (  # noqa: E402
    ContinuousPredictiveObjectFieldV67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_teacher_v67 import (  # noqa: E402
    ContinuousPredictiveTeacherBatchV67,
)
from igsw.adaptive_gaussian_wm.v67_config import (  # noqa: E402
    DYNAMICS_STAGE,
    STATE_STAGE,
    ContinuousPredictiveObjectFieldConfigV67,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compact_config() -> ContinuousPredictiveObjectFieldConfigV67:
    config = ContinuousPredictiveObjectFieldConfigV67(
        dino_dim=64,
        siglip_dim=48,
        semantic_dim=16,
        native_tile_size=64,
        native_tile_stride=48,
        local_radii_pixels=(4.0, 8.0, 16.0),
        local_tokens_per_scale=4,
        tracker_grid_side=4,
        query_count=4,
        crop_side=5,
        field_dim=64,
        local_channels=16,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ffn_multiplier=2,
        identity_dim=16,
        dynamic_dim=32,
        code_dim=48,
        effect_dim=32,
        operator_heads=4,
        operator_layers=2,
        fourier_bands=4,
    )
    config.validate()
    return config


def synthetic_batch(config, device: torch.device):
    torch.manual_seed(67)
    batch, frames, height, width = 2, config.clip_frames, 48, 64
    sequence = torch.tensor((11, 29), device=device)
    rgb = torch.randint(
        0, 256, (batch, frames, 3, height, width), device=device, dtype=torch.uint8
    )
    pixel_valid = torch.ones(
        batch, frames, height, width, device=device, dtype=torch.bool
    )
    frame_times = torch.arange(frames, device=device).float()[None].expand(batch, -1)
    frame_times = frame_times * torch.tensor((0.1, 0.2), device=device)[:, None]
    anchors = stratified_query_coordinates_v67(
        sequence, config.tracker_grid_side, config.coordinate_jitter_fraction
    )
    point = torch.arange(config.candidate_count, device=device).float()
    velocity = 0.025 * torch.stack((point.sin(), point.cos()), dim=-1)
    velocity = velocity[None].expand(batch, -1, -1).clone()
    velocity[1] = velocity[1].roll(3, dims=0) * 1.7
    elapsed = frame_times - frame_times[:, config.source_frame : config.source_frame + 1]
    tracks = anchors[:, None] + elapsed[:, :, None, None] * velocity[:, None]
    tracks = tracks.clamp(-0.96, 0.96)
    visibility = torch.ones(
        batch, frames, config.candidate_count, device=device, dtype=torch.bool
    )
    visibility[:, config.midpoint_frame, ::5] = False
    latent = F.normalize(
        torch.randn(batch, config.candidate_count, config.semantic_dim, device=device),
        dim=-1,
    )
    time_signal = elapsed[:, :, None, None] * velocity[:, None, :, :1]
    projection = torch.randn(1, 1, 1, config.semantic_dim, device=device)
    dino = F.normalize(latent[:, None] + time_signal * projection, dim=-1)
    siglip = F.normalize(
        latent[:, None].roll(2, dims=-1) - 0.7 * time_signal * projection,
        dim=-1,
    )
    scales = torch.full(
        (batch, config.candidate_count), config.base_query_scale, device=device
    )
    query_indices = evenly_spaced_query_indices_v67(
        config.candidate_count, config.query_count, device
    )
    target = ContinuousPredictiveTeacherBatchV67(
        anchor_coordinates=anchors,
        track_coordinates=tracks,
        scales=scales,
        dino=dino,
        siglip=siglip,
        visibility=visibility,
        reliability=torch.ones(batch, config.candidate_count, device=device),
        query_indices=query_indices,
        context_mask=context_coordinate_mask_v67(
            config.candidate_count, config.context_fraction, sequence
        ),
    )
    batch_payload = {
        "video_rgb": rgb,
        "video_pixel_valid": pixel_valid,
        "native_image_hw": torch.tensor(
            ((height, width), (height, width)), device=device
        ),
        "frame_times": frame_times,
        "sequence_index": sequence,
        "source_index": torch.tensor((0, 1), device=device),
        "temporal_step_seconds": torch.tensor((0.1, 0.2), device=device),
        "decode_replaced": torch.zeros(batch, device=device, dtype=torch.bool),
    }
    return batch_payload, target


def gradients_are_complete(model) -> tuple[bool, list[str]]:
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    finite = all(
        bool(parameter.grad.isfinite().all())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    return finite and not missing, missing


def future_swapped(batch, target, config):
    swapped_batch = dict(batch)
    swapped_rgb = batch["video_rgb"].clone()
    swapped_rgb[:, config.source_frame + 1 :] = swapped_rgb[
        :, config.source_frame + 1 :
    ].roll(1, dims=0)
    swapped_batch["video_rgb"] = swapped_rgb

    def swap_future(value):
        changed = value.clone()
        changed[:, config.source_frame + 1 :] = changed[
            :, config.source_frame + 1 :
        ].roll(1, dims=0)
        return changed

    swapped_target = replace(
        target,
        track_coordinates=swap_future(target.track_coordinates),
        dino=swap_future(target.dino),
        siglip=swap_future(target.siglip),
        visibility=swap_future(target.visibility),
    )
    return swapped_batch, swapped_target


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = compact_config()
    batch, target = synthetic_batch(config, device)

    state_model = ContinuousPredictiveObjectFieldV67(config, STATE_STAGE).to(device)
    state_model.train()
    state_output = state_model(batch, target)
    require(bool(state_output["loss"].isfinite()), "v67 E0 loss is non-finite")
    state_output["loss"].backward()
    state_gradients, state_missing = gradients_are_complete(state_model)
    require(state_gradients, f"v67 E0 has missing/non-finite gradients: {state_missing}")

    state_model.eval()
    swapped_batch, swapped_target = future_swapped(batch, target, config)
    with torch.no_grad():
        _, _, _, original_source = state_model.encode_source(batch, target, False)
        _, _, _, changed_source = state_model.encode_source(
            swapped_batch, swapped_target, False
        )
        _, original_target = state_model.encode_target(
            batch, target, config.target_frame
        )
        _, changed_target = state_model.encode_target(
            swapped_batch, swapped_target, config.target_frame
        )
    source_difference = float(
        (original_source.mean - changed_source.mean).abs().max()
    )
    target_difference = float(
        (original_target.mean - changed_target.mean).abs().max()
    )
    require(source_difference < 1e-7, "v67 source path reads future observations")
    require(target_difference > 1e-6, "v67 target path ignores changed future")

    dynamics_model = ContinuousPredictiveObjectFieldV67(config, DYNAMICS_STAGE).to(
        device
    )
    dynamics_model.load_state_dict(state_model.state_dict(), strict=True)
    dynamics_model.configure_stage(DYNAMICS_STAGE)
    dynamics_model.train()
    dynamics_output = dynamics_model(batch, target)
    require(bool(dynamics_output["loss"].isfinite()), "v67 E1 loss is non-finite")
    dynamics_output["loss"].backward()
    dynamics_gradients, dynamics_missing = gradients_are_complete(dynamics_model)
    require(
        dynamics_gradients,
        f"v67 E1 has missing/non-finite gradients: {dynamics_missing}",
    )
    effect_difference = float(
        (
            dynamics_output["goal_correct"].code.mean
            - dynamics_output["goal_shuffled"].code.mean
        )
        .abs()
        .mean()
    )
    report = {
        "status": "passed",
        "device": str(device),
        "state_loss": float(state_output["loss"].detach()),
        "dynamics_loss": float(dynamics_output["loss"].detach()),
        "source_future_swap_max_difference": source_difference,
        "target_future_swap_max_difference": target_difference,
        "state_trainable_tensors": sum(
            parameter.requires_grad for parameter in state_model.parameters()
        ),
        "dynamics_trainable_tensors": sum(
            parameter.requires_grad for parameter in dynamics_model.parameters()
        ),
        "synthetic_effect_intervention_difference": effect_difference,
        "fixed_object_count": False,
        "rgb_reconstruction": False,
        "tracker_is_dynamic_target": False,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
