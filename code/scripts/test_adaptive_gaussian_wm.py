"""End-to-end structural, gradient, and causality tests on the server GPU."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    adaptive_world_model_loss,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.scale import (  # noqa: E402
    inverse_signed_gap_scale,
    signed_gap_scale,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach())


def _swap_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    swapped = dict(batch)
    order = torch.arange(
        batch["future_features"].shape[0] - 1,
        -1,
        -1,
        device=batch["future_features"].device,
    )
    for key in ("future_features", "future_labels"):
        swapped[key] = batch[key][order]
    return swapped


def _fixed_history_mask(
    batch_size: int,
    history_frames: int,
    object_slots: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        batch_size,
        history_frames,
        object_slots,
        device=device,
        dtype=torch.bool,
    )
    if history_frames > 1:
        mask[:, 1:, ::2] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--decoupled_jepa_slots", action="store_true")
    parser.add_argument("--action_query_modulation", action="store_true")
    parser.add_argument("--action_film_modulation", action="store_true")
    parser.add_argument("--kinematic_action_modulation", action="store_true")
    parser.add_argument("--learned_velocity_baseline", action="store_true")
    parser.add_argument("--spatial_slot_attention", action="store_true")
    parser.add_argument("--token_spatial_precision_floor", type=float, default=0.0)
    parser.add_argument("--flow_endpoint_prediction", action="store_true")
    parser.add_argument("--correlated_flow_source", action="store_true")
    parser.add_argument("--flow_source_scale", type=float, default=1.0)
    parser.add_argument("--multi_query_prior_context", action="store_true")
    parser.add_argument("--prior_query_residual", action="store_true")
    parser.add_argument("--flow_source_components", type=int, default=1)
    parser.add_argument("--flow_lift_scale", type=float, default=1.0)
    parser.add_argument("--flow_responsibility_floor", type=float, default=0.05)
    parser.add_argument("--flow_source_fit_weight", type=float, default=0.1)
    parser.add_argument("--balanced_source_assignment", action="store_true")
    parser.add_argument("--flow_assignment_temperature", type=float, default=0.05)
    args = parser.parse_args()
    torch.manual_seed(31)
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is not available")

    config = AdaptiveGaussianWMConfig.tiny(feature_dim=16)
    config = replace(
        config,
        decoupled_jepa_slots=args.decoupled_jepa_slots,
        action_query_modulation=args.action_query_modulation,
        action_film_modulation=args.action_film_modulation,
        kinematic_action_modulation=args.kinematic_action_modulation,
        learned_velocity_baseline=args.learned_velocity_baseline,
        spatial_slot_attention=args.spatial_slot_attention,
        token_spatial_precision_floor=args.token_spatial_precision_floor,
        flow_endpoint_prediction=args.flow_endpoint_prediction,
        correlated_flow_source=args.correlated_flow_source,
        flow_source_scale=args.flow_source_scale,
        multi_query_prior_context=args.multi_query_prior_context,
        prior_query_residual=args.prior_query_residual,
        flow_source_components=args.flow_source_components,
        flow_lift_scale=args.flow_lift_scale,
        flow_responsibility_floor=args.flow_responsibility_floor,
        flow_source_fit_weight=args.flow_source_fit_weight,
        flow_balanced_source_assignment=args.balanced_source_assignment,
        flow_assignment_temperature=args.flow_assignment_temperature,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    free_action_config = replace(
        config,
        action_tokens=config.object_slots,
    )
    free_model = AdaptiveGaussianObjectWorldModel(
        free_action_config
    ).to(device)
    aligned_model = AdaptiveGaussianObjectWorldModel(
        replace(free_action_config, object_aligned_actions=True)
    ).to(device)
    free_parameter_count = sum(
        parameter.numel() for parameter in free_model.parameters()
    )
    aligned_parameter_count = sum(
        parameter.numel() for parameter in aligned_model.parameters()
    )
    _require(
        aligned_parameter_count == free_parameter_count,
        "aligned and free action models are not capacity matched",
    )
    del free_model, aligned_model
    model.train()
    batch_size = 6
    batch = make_synthetic_batch(
        feature_dim=config.feature_dim,
        batch_size=batch_size,
        history_frames=3,
        future_steps=2,
        grid_size=7,
        device=device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
    )
    history_mask = _fixed_history_mask(
        batch_size,
        3,
        config.object_slots,
        device,
    )
    output = model(batch, history_mask=history_mask)
    _require(
        output["predicted_future_slots"].shape
        == (batch_size, 2, config.object_slots, config.object_dim),
        "future slot shape mismatch",
    )
    _require(
        output["rendered_future_features"].shape
        == batch["future_features"].shape,
        "rendered feature shape mismatch",
    )
    _require(
        output["gaussian_readout"].center.shape
        == (batch_size, 2, config.max_micro_tokens, 2),
        "Gaussian readout shape mismatch",
    )
    _require(
        output["posterior_actions"].shape
        == (batch_size, 2, config.action_tokens, config.action_dim),
        "posterior shape mismatch",
    )
    prior_context_shape = (
        (batch_size, 2, config.action_tokens, config.model_dim)
        if config.multi_query_prior_context
        else (batch_size, 2, config.model_dim)
    )
    _require(output["prior_context"].shape == prior_context_shape,
             "prior context shape mismatch")

    loss, parts = adaptive_world_model_loss(
        model,
        batch,
        output,
        AdaptiveGaussianLossWeights(),
    )
    _require(bool(torch.isfinite(loss)), "loss is not finite")
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    _require(bool(gradients), "no gradients were produced")
    _require(
        all(bool(torch.isfinite(value).all()) for value in gradients.values()),
        "non-finite gradient detected",
    )
    required_gradient_prefixes = (
        "allocator.",
        "object_aggregator.",
        "latent_actions.posterior.",
        "dynamics.",
        "gaussian_readout.",
        "latent_actions.prior.",
    )
    for prefix in required_gradient_prefixes:
        _require(
            any(
                name.startswith(prefix) and float(value.abs().sum()) > 0.0
                for name, value in gradients.items()
            ),
            f"missing nonzero gradient for {prefix}",
        )
    _require(
        all(parameter.grad is None for parameter in model.target_allocator.parameters()),
        "EMA target allocator received gradients",
    )
    _require(
        all(
            parameter.grad is None
            for parameter in model.target_object_aggregator.parameters()
        ),
        "EMA target object aggregator received gradients",
    )

    model.eval()
    swapped = _swap_future(batch)
    with torch.no_grad():
        history_a = model.encode_history(batch)
        history_b = model.encode_history(swapped)
        history_difference = _max_difference(
            history_a["slots"],
            history_b["slots"],
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            config.gap_reference,
        )
        history_scale = signed_gap_scale(
            batch["history_times"],
            config.gap_reference,
        )
        prior_a = model.prior_context(history_a, future_scale, history_scale)
        prior_b = model.prior_context(history_b, future_scale, history_scale)
        prior_difference = _max_difference(prior_a, prior_b)
        prior_time_shift = model.prior_context(
            history_a,
            future_scale,
            history_scale.roll(1, dims=1),
        )
        prior_time_difference = _max_difference(prior_a, prior_time_shift)
        output_a = model(batch, history_mask=history_mask)
        output_b = model(swapped, history_mask=history_mask)
        posterior_difference = _max_difference(
            output_a["posterior_actions"],
            output_b["posterior_actions"],
        )
        history_prediction_difference = _max_difference(
            output_a["predicted_history_slots"],
            output_b["predicted_history_slots"],
        )
    _require(history_difference == 0.0, "future changed the online history encoder")
    _require(prior_difference == 0.0, "future changed Flow Prior context")
    _require(
        prior_time_difference > 1e-6,
        "history time scale did not modulate Flow Prior context",
    )
    _require(
        posterior_difference > 1e-6,
        "Action Posterior did not respond to changed future",
    )
    _require(
        history_prediction_difference == 0.0,
        "future-conditioned posterior leaked into history reconstruction",
    )

    model.zero_grad(set_to_none=True)
    history_for_prior = model.encode_history(batch)
    context_for_prior = model.prior_context(
        history_for_prior,
        future_scale,
        history_scale,
    )
    observed_flow_times = []
    flow_time_hook = model.latent_actions.prior.register_forward_hook(
        lambda _module, inputs, _output: observed_flow_times.append(
            inputs[1].detach()
        )
    )
    prior_only_loss = model.latent_actions.prior.loss(
        output_a["posterior_actions"].detach(),
        context_for_prior,
    )
    flow_time_hook.remove()
    _require(bool(observed_flow_times), "flow loss did not call the velocity field")
    shared_flow_time = all(
        bool((value == value[:, :1]).all())
        for value in observed_flow_times
    )
    _require(
        shared_flow_time,
        "future tokens within a trajectory used different flow times",
    )
    prior_only_loss.backward()
    disallowed_prior_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not name.startswith("latent_actions.prior.")
        and not name.startswith("latent_actions.prior_history_input.")
        and not name.startswith("latent_actions.prior_gap_input.")
        and not name.startswith("latent_actions.prior_slot_input.")
        and not name.startswith("latent_actions.prior_center_input.")
        and not name.startswith("latent_actions.prior_history_scale_input.")
        and not name.startswith("latent_actions.prior_attention.")
        and not name.startswith("latent_actions.prior_norm.")
        and name != "latent_actions.prior_query"
    ]
    _require(
        not disallowed_prior_gradients,
        f"flow loss escaped prior modules: {disallowed_prior_gradients}",
    )

    model.eval()
    with torch.no_grad():
        state = model.allocator(
            batch["history_features"][:, -1],
            batch["history_coordinates"][:, -1],
            batch["history_valid"][:, -1],
        )
        permutation = torch.randperm(
            batch["history_features"].shape[2],
            device=device,
        )
        permuted_state = model.allocator(
            batch["history_features"][:, -1, permutation],
            batch["history_coordinates"][:, -1, permutation],
            batch["history_valid"][:, -1, permutation],
        )
        permutation_difference = max(
            _max_difference(state.latent, permuted_state.latent),
            _max_difference(state.center, permuted_state.center),
            _max_difference(state.covariance, permuted_state.covariance),
        )
        activation_std = float(
            state.activation.sum(dim=1).squeeze(-1).std().detach()
        )
        assignment_sum_error = float(
            (state.assignment.sum(dim=1) - 1.0).abs().max().detach()
        )
        flow_samples = model.latent_actions.prior.sample(
            prior_a,
            sample_count=4,
            stochastic=True,
        )
        flow_diversity = float(flow_samples.std(dim=0).mean().detach())
    _require(permutation_difference < 2e-5, "grid permutation changed token state")
    _require(activation_std > 1e-7, "effective GPSToken count is sample-constant")
    _require(assignment_sum_error < 2e-5, "GPSToken partition does not sum to one")
    _require(flow_diversity > 1e-5, "Flow Prior samples are identical")

    one_frame_batch = make_synthetic_batch(
        feature_dim=config.feature_dim,
        batch_size=3,
        history_frames=1,
        future_steps=2,
        grid_size=6,
        device=device,
        paired_futures=False,
        irregular_gaps=True,
    )
    with torch.no_grad():
        one_frame_output = model(one_frame_batch)
    _require(
        not bool(one_frame_output["history_mask"].any()),
        "single-frame history must not mask its identity anchor",
    )
    scale_check = signed_gap_scale(
        torch.tensor([-2.0, 0.0, 2.0], device=device),
        reference=1.0,
    )
    _require(
        bool((scale_check == -scale_check.flip(0)).all()),
        "signed gap scale is not antisymmetric",
    )
    _require(
        torch.allclose(
            inverse_signed_gap_scale(scale_check, reference=1.0),
            torch.tensor([-2.0, 0.0, 2.0], device=device),
        ),
        "signed gap scale inverse is not exact",
    )

    report = {
        "status": "ok",
        "device": str(device),
        "decoupled_jepa_slots": args.decoupled_jepa_slots,
        "action_query_modulation": args.action_query_modulation,
        "action_film_modulation": args.action_film_modulation,
        "kinematic_action_modulation": args.kinematic_action_modulation,
        "learned_velocity_baseline": args.learned_velocity_baseline,
        "spatial_slot_attention": args.spatial_slot_attention,
        "token_spatial_precision_floor": args.token_spatial_precision_floor,
        "flow_endpoint_prediction": args.flow_endpoint_prediction,
        "correlated_flow_source": args.correlated_flow_source,
        "flow_source_scale": args.flow_source_scale,
        "multi_query_prior_context": args.multi_query_prior_context,
        "prior_query_residual": args.prior_query_residual,
        "flow_source_components": args.flow_source_components,
        "flow_lift_scale": args.flow_lift_scale,
        "flow_responsibility_floor": args.flow_responsibility_floor,
        "flow_source_fit_weight": args.flow_source_fit_weight,
        "flow_balanced_source_assignment": args.balanced_source_assignment,
        "flow_assignment_temperature": args.flow_assignment_temperature,
        "loss": float(loss.detach()),
        "loss_parts": {name: float(value.detach()) for name, value in parts.items()},
        "gradient_tensor_count": len(gradients),
        "aligned_action_parameter_count": aligned_parameter_count,
        "free_action_parameter_count": free_parameter_count,
        "history_future_swap_max_abs_difference": history_difference,
        "prior_future_swap_max_abs_difference": prior_difference,
        "prior_history_time_shift_max_abs_difference": prior_time_difference,
        "posterior_future_swap_max_abs_difference": posterior_difference,
        "history_mask_branch_future_swap_max_abs_difference": (
            history_prediction_difference
        ),
        "grid_permutation_max_abs_difference": permutation_difference,
        "effective_token_count_batch_std": activation_std,
        "assignment_sum_max_error": assignment_sum_error,
        "flow_sample_mean_std": flow_diversity,
        "shared_trajectory_flow_time": shared_flow_time,
        "single_frame_history_passed": True,
        "three_frame_history_passed": True,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
