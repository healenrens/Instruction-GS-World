"""Remote-only structural tests for RGB, language conditioning, and DDP stats."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.representation import (  # noqa: E402
    reconstruct_current,
)
from igsw.adaptive_gaussian_wm.readout_runtime import (  # noqa: E402
    residual_future_features,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    residual_future_rgb,
)
from igsw.adaptive_gaussian_wm.training_modes import (  # noqa: E402
    configure_posterior_dynamics_gate,
    staged_loss_weights,
)
from igsw.distributed import init_torchrun  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rgb_from_features(features: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, frames, tokens, _ = features.shape
    grid = round(tokens**0.5)
    _require(grid * grid == tokens, "synthetic feature grid must be square")
    rgb = torch.sigmoid(features[..., :3]).reshape(
        batch,
        frames,
        grid,
        grid,
        3,
    )
    rgb = rgb.permute(0, 1, 4, 2, 3).flatten(0, 1)
    rgb = F.interpolate(
        rgb,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return (rgb.reshape(batch, frames, 3, height, width) * 255.0).round().to(
        torch.uint8
    )


def make_supervised_batch(
    batch_size: int,
    feature_dim: int,
    condition_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    grid_size = 6
    batch = make_synthetic_batch(
        feature_dim=feature_dim,
        batch_size=batch_size,
        history_frames=1,
        future_steps=1,
        grid_size=grid_size,
        device=device,
        paired_futures=True,
        mode_count=2,
    )
    height, width = 24, 32
    batch["condition_feature"] = torch.randn(
        batch_size,
        condition_dim,
        device=device,
    )
    batch["feature_grid_hw"] = torch.tensor(
        [grid_size, grid_size],
        device=device,
    )[None].expand(batch_size, -1)
    batch["history_rgb"] = _rgb_from_features(
        batch["history_features"],
        height,
        width,
    )
    batch["history_rgb"][:, :, 0] = 255
    batch["history_rgb"][:, :, 1] = 0
    batch["future_rgb"] = _rgb_from_features(
        batch["future_features"],
        height,
        width,
    )
    batch["history_rgb_valid"] = torch.ones(
        batch_size,
        1,
        height,
        width,
        device=device,
        dtype=torch.bool,
    )
    batch["future_rgb_valid"] = batch["history_rgb_valid"].clone()
    return batch


def make_config(feature_dim: int, condition_dim: int) -> AdaptiveGaussianWMConfig:
    return replace(
        AdaptiveGaussianWMConfig.tiny(feature_dim),
        condition_dim=condition_dim,
        rgb_supervision=True,
        rgb_short_side=24,
        rgb_pad_multiple=8,
        rgb_render_chunk=256,
        rgb_loss_weight=0.5,
        rgb_ssim_weight=0.2,
        max_micro_tokens=12,
        action_query_modulation=True,
    )


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _paired_permutation(batch_size: int, device: torch.device) -> torch.Tensor:
    _require(batch_size % 2 == 0, "causal test batch must contain complete pairs")
    return torch.arange(batch_size, device=device).reshape(-1, 2).flip(1).flatten()


def _swap_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    swapped = _clone_batch(batch)
    permutation = _paired_permutation(
        batch["history_features"].shape[0],
        batch["history_features"].device,
    )
    for name in (
        "future_features",
        "future_coordinates",
        "future_valid",
        "future_times",
        "future_rgb",
        "future_rgb_valid",
    ):
        swapped[name] = swapped[name][permutation]
    return swapped


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach())


def _gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.float().square().sum())
    return total**0.5


def check_causal_contract(
    model: AdaptiveGaussianObjectWorldModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    swapped_future = _swap_future(batch)
    swapped_instruction = _clone_batch(batch)
    permutation = _paired_permutation(
        batch["history_features"].shape[0],
        batch["history_features"].device,
    )
    swapped_instruction["condition_feature"] = batch["condition_feature"][
        permutation
    ]
    mask = torch.zeros(
        batch["history_features"].shape[0],
        1,
        model.config.object_slots,
        device=batch["history_features"].device,
        dtype=torch.bool,
    )
    with torch.no_grad():
        history = model.encode_history(batch)
        future_history = model.encode_history(swapped_future)
        condition = model.encode_condition(batch)
        future_condition = model.encode_condition(swapped_future)
        instruction_condition = model.encode_condition(swapped_instruction)
        scale = batch["future_times"] / model.config.gap_reference
        history_scale = batch["history_times"] / model.config.gap_reference
        prior = model.prior_context(history, scale, history_scale, condition)
        prior_future = model.prior_context(
            future_history,
            scale,
            history_scale,
            future_condition,
        )
        prior_instruction = model.prior_context(
            history,
            scale,
            history_scale,
            instruction_condition,
        )
        output = model(batch, history_mask=mask)
        output_future = model(swapped_future, history_mask=mask)
        zero_actions = torch.zeros_like(output["posterior_actions"])
        dynamics = model(
            batch,
            history_mask=mask,
            actions_override=zero_actions,
        )
        dynamics_instruction = model(
            swapped_instruction,
            history_mask=mask,
            actions_override=zero_actions,
        )
    history_difference = _max_difference(
        history["slots"],
        future_history["slots"],
    )
    prior_future_difference = _max_difference(prior, prior_future)
    prior_instruction_difference = _max_difference(prior, prior_instruction)
    posterior_future_difference = _max_difference(
        output["posterior_actions"],
        output_future["posterior_actions"],
    )
    dynamics_instruction_difference = _max_difference(
        dynamics["predicted_future_slots"],
        dynamics_instruction["predicted_future_slots"],
    )
    _require(history_difference < 1e-6, "future swap changed history encoder")
    _require(prior_future_difference < 1e-6, "future swap changed prior context")
    _require(posterior_future_difference > 1e-6, "posterior ignored future")
    _require(prior_instruction_difference > 1e-6, "prior ignored instruction")
    _require(
        dynamics_instruction_difference > 1e-6,
        "dynamics ignored instruction",
    )
    return {
        "history_future_swap_max": history_difference,
        "prior_future_swap_max": prior_future_difference,
        "posterior_future_swap_max": posterior_future_difference,
        "prior_instruction_swap_max": prior_instruction_difference,
        "dynamics_instruction_swap_max": dynamics_instruction_difference,
    }


def check_backward(
    model_or_ddp,
    model: AdaptiveGaussianObjectWorldModel,
    batch: dict[str, torch.Tensor],
    world_size: int,
) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    representation = model_or_ddp(batch, phase="representation")
    _require(torch.isfinite(representation["loss"]).item(), "non-finite RGB loss")
    with torch.no_grad():
        current = reconstruct_current(model, batch)
        shuffled = reconstruct_current(
            model,
            batch,
            current["history"],
            current["slots"].slots.roll(1, dims=1),
        )
        covariance = current["readout"].covariance
        min_eigenvalue = float(
            torch.linalg.eigvalsh(covariance.float())[..., 0].min()
        )
        feature_slot_effect = _max_difference(
            current["feature"],
            shuffled["feature"],
        )
        rgb_slot_effect = _max_difference(current["rgb"], shuffled["rgb"])
    _require(
        min_eigenvalue >= model.config.covariance_floor * 0.999,
        "Gaussian readout violated covariance floor",
    )
    _require(feature_slot_effect > 1e-6, "feature readout ignored object slots")
    _require(rgb_slot_effect > 1e-6, "RGB readout ignored object slots")
    representation["loss"].backward()
    rgb_gradient = _gradient_norm(model.object_aggregator.rgb_head)
    assignment_gradient = float(
        model.allocator.query_projection.weight.grad.float().norm()
    )
    _require(rgb_gradient > 0.0, "RGB slot decoder received no gradient")
    _require(
        assignment_gradient > 0.0,
        "representation loss received no assignment gradient",
    )

    model.zero_grad(set_to_none=True)
    weights = AdaptiveGaussianLossWeights()
    output = model_or_ddp(batch, phase="joint_loss", loss_weights=weights)
    _require(torch.isfinite(output["loss"]).item(), "non-finite joint loss")
    output["loss"].backward()
    language_gradient = _gradient_norm(model.language_condition)
    dynamics_gradient = _gradient_norm(model.dynamics)
    _require(language_gradient > 0.0, "language projector received no gradient")
    _require(dynamics_gradient > 0.0, "dynamics received no gradient")
    rendered = output["rendered_future_rgb"]
    _require(
        rendered.shape == batch["future_rgb"].shape,
        "future RGB render shape mismatch",
    )
    expected_statistics = (
        batch["history_features"].shape[0]
        * batch["future_features"].shape[1]
        * model.config.action_tokens
        * world_size
    )
    statistical_samples = float(output["parts"]["action_statistical_samples"])
    effect_samples = float(output["parts"]["effect_statistical_samples"])
    _require(
        statistical_samples == expected_statistics,
        "cross-rank action statistics have the wrong sample count",
    )
    _require(
        effect_samples
        == batch["history_features"].shape[0]
        * batch["future_features"].shape[1]
        * world_size,
        "cross-rank effect relation matrix has the wrong sample count",
    )
    model.zero_grad(set_to_none=True)
    configure_posterior_dynamics_gate(model)
    _require(
        not any(parameter.requires_grad for parameter in model.allocator.parameters()),
        "posterior gate left the observation allocator trainable",
    )
    _require(
        all(parameter.requires_grad for parameter in model.gaussian_readout.parameters()),
        "posterior gate froze the future Gaussian readout",
    )
    posterior_gate = model_or_ddp(
        batch,
        phase="posterior_dynamics_loss",
        loss_weights=staged_loss_weights(True),
    )
    _require(
        posterior_gate["rendered_current_rgb"] is None,
        "posterior Dynamics phase supervised the frozen current RGB",
    )
    _require(
        posterior_gate["residual_reference_rgb"] is not None,
        "posterior Dynamics phase skipped the residual RGB reference",
    )
    _require(
        posterior_gate["rendered_future_rgb"] is not None,
        "posterior Dynamics phase skipped future RGB",
    )
    reference_features = posterior_gate["residual_reference_features"]
    feature_identity = residual_future_features(
        reference_features,
        reference_features,
        batch,
    )
    expected_features = batch["history_features"][:, -1:].expand_as(
        feature_identity
    )
    feature_identity_error = _max_difference(
        feature_identity,
        expected_features,
    )
    reference_rgb = posterior_gate["residual_reference_rgb"]
    rgb_identity = residual_future_rgb(reference_rgb, reference_rgb, batch)
    expected_rgb = (
        batch["history_rgb"][:, -1:].float() / 255.0
    ).expand_as(rgb_identity)
    rgb_identity_error = _max_difference(rgb_identity, expected_rgb)
    _require(
        feature_identity_error < 1e-6 and rgb_identity_error < 1e-6,
        "zero Gaussian residual did not reproduce the causal current input",
    )
    _require(
        torch.isfinite(posterior_gate["loss"]).item(),
        "non-finite posterior Dynamics loss",
    )
    specificity = float(
        posterior_gate["parts"]["action_specificity"].detach()
    )
    specificity_action_rms = float(
        posterior_gate["parts"]["action_specificity_action_rms"].detach()
    )
    specificity_sample_std = float(
        posterior_gate["parts"]["action_specificity_sample_std"].detach()
    )
    _require(specificity >= 0.0, "negative action specificity loss")
    _require(
        specificity_action_rms > 0.0,
        "cross-rank shuffled actions are identical",
    )
    posterior_gate["loss"].backward()
    future_readout_gradient = _gradient_norm(model.gaussian_readout)
    future_rgb_gradient = _gradient_norm(model.object_aggregator.rgb_head)
    _require(
        future_readout_gradient > 0.0 and future_rgb_gradient > 0.0,
        "future RGB supervision did not reach the trainable readout",
    )
    return {
        "representation_loss": float(representation["loss"].detach()),
        "readout_min_covariance_eigenvalue": min_eigenvalue,
        "feature_slot_effect": feature_slot_effect,
        "rgb_slot_effect": rgb_slot_effect,
        "joint_loss": float(output["loss"].detach()),
        "posterior_dynamics_loss": float(posterior_gate["loss"].detach()),
        "action_specificity_loss": specificity,
        "action_specificity_action_rms": specificity_action_rms,
        "action_specificity_sample_std": specificity_sample_std,
        "residual_feature_identity_max": feature_identity_error,
        "residual_rgb_identity_max": rgb_identity_error,
        "future_readout_gradient_norm": future_readout_gradient,
        "future_rgb_head_gradient_norm": future_rgb_gradient,
        "rgb_gradient_norm": rgb_gradient,
        "assignment_gradient_norm": assignment_gradient,
        "language_gradient_norm": language_gradient,
        "dynamics_gradient_norm": dynamics_gradient,
        "action_statistical_samples": statistical_samples,
        "effect_statistical_samples": effect_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    context = init_torchrun()
    device = torch.device(context.device)
    if not torch.cuda.is_available():
        raise RuntimeError("this structural test must run on the remote CUDA host")
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed_all(args.seed + context.rank)
    feature_dim, condition_dim = 12, 16
    model = AdaptiveGaussianObjectWorldModel(
        make_config(feature_dim, condition_dim)
    ).to(device)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if context.distributed
        else model
    )
    batch = make_supervised_batch(
        args.batch,
        feature_dim,
        condition_dim,
        device,
    )
    causal = check_causal_contract(model, batch)
    backward = check_backward(
        wrapped,
        model,
        batch,
        context.world_size,
    )
    if context.distributed:
        dist.barrier()
    if context.is_main:
        report = {"status": "ok", **causal, **backward}
        if args.output:
            output = os.path.abspath(args.output)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(json.dumps(report, sort_keys=True))
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
