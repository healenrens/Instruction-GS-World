"""Verify language-free sequence data and strict causal/time-conditioned model paths."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import warm_start_model  # noqa: E402
from igsw.adaptive_gaussian_wm.observed_action import posterior_from_targets  # noqa: E402
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402


def maximum_difference(left, right) -> float:
    if isinstance(left, dict):
        return max(
            maximum_difference(left[name], right[name])
            for name in ("slots", "activity", "center")
        )
    return float((left.float() - right.float()).abs().max())


def prediction_difference(left: dict, right: dict) -> float:
    names = (
        "predicted_future_slots",
        "predicted_future_centers",
        "predicted_future_object_features",
        "rendered_future_features",
        "rendered_future_rgb",
    )
    values = []
    for name in names:
        left_value = left[name]
        right_value = right[name]
        if left_value is None or right_value is None:
            if left_value is not right_value:
                raise ValueError(f"prediction field availability differs: {name}")
            continue
        values.append(float((left_value.float() - right_value.float()).abs().max()))
    if not values:
        raise ValueError("causal prediction comparison has no tensor fields")
    return max(values)


def swapped_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    fields = (
        "future_features",
        "future_coordinates",
        "future_valid",
        "future_rgb",
        "future_rgb_valid",
        "future_frame_indices",
        "future_control_indices",
    )
    for name in fields:
        if name in result:
            result[name] = torch.roll(result[name], shifts=1, dims=0)
    return result


def validate_data(
    dataset: CausalVisualSequenceDataset,
    sample_count: int,
) -> dict:
    history_gaps = []
    future_gaps = []
    layouts = set()
    for index in range(min(sample_count, len(dataset))):
        sample = dataset[index]
        forbidden = [
            name
            for name in sample
            if (
                name.startswith("condition")
                or "instruction" in name
                or name == "task_index"
            )
        ]
        if forbidden:
            raise ValueError(f"semantic fields leaked into model sample: {forbidden}")
        history = sample["history_times"]
        future = sample["future_times"]
        if (
            not bool((history[1:] > history[:-1]).all())
            or float(history[-1]) != 0.0
            or not bool((future > 0.0).all())
            or not bool((future[1:] > future[:-1]).all())
        ):
            raise ValueError("relative physical timestamps violate causal ordering")
        control_delta = (
            sample["future_control_indices"] - sample["history_control_indices"][-1]
        ).float()
        recovered = control_delta / (250.0 / 15.0)
        if float((recovered - future).abs().max()) > 1e-6:
            raise ValueError("future seconds do not match RoboTwin control indices")
        history_gaps.extend(history.tolist())
        future_gaps.extend(future.tolist())
        layouts.add(
            (
                tuple(sample["history_frame_indices"].tolist()),
                tuple(sample["future_frame_indices"].tolist()),
            )
        )
    return {
        "examples": len(dataset),
        "feature_dim": dataset.feature_dim,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "layout_count": len(layouts),
        "history_seconds_range": [min(history_gaps), max(history_gaps)],
        "future_seconds_range": [min(future_gaps), max(future_gaps)],
        "condition_dim": dataset.condition_dim,
        "semantic_fields_present": False,
    }


def language_free_config(checkpoint: dict) -> AdaptiveGaussianWMConfig:
    source = AdaptiveGaussianWMConfig(**checkpoint["config"])
    return replace(
        source,
        condition_dim=0,
        token_conditioned_prior=False,
        language_effect_weight=0.0,
    )


@torch.inference_mode()
def validate_model(
    dataset: CausalVisualSequenceDataset,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = language_free_config(checkpoint)
    model = AdaptiveGaussianObjectWorldModel(config).to(device).eval()
    warm_start = warm_start_model(model, checkpoint)
    second = min(len(dataset.anchors), len(dataset) - 1)
    batch = default_collate([dataset[0], dataset[second]])
    batch = move_to_device(batch, device)
    swapped = swapped_future(batch)
    amp = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )
    with amp:
        history = model.encode_history(batch)
        swapped_history = model.encode_history(swapped)
        _, future = model.encode_targets(batch)
        _, swapped_target = model.encode_targets(swapped)
        history_scale = signed_gap_scale(
            batch["history_times"],
            config.gap_reference,
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            config.gap_reference,
        )
        posterior = posterior_from_targets(
            model,
            batch,
            history,
            future,
            future_scale,
            None,
        )[0]
        swapped_posterior = posterior_from_targets(
            model,
            swapped,
            swapped_history,
            swapped_target,
            future_scale,
            None,
        )[0]
        prior = model.prior_context(
            history,
            future_scale,
            history_scale,
        )
        swapped_prior = model.prior_context(
            swapped_history,
            future_scale,
            history_scale,
        )
        longer_times = batch["future_times"] * 1.5
        longer_scale = signed_gap_scale(longer_times, config.gap_reference)
        longer_prior = model.prior_context(
            history,
            longer_scale,
            history_scale,
        )
        mask = torch.zeros(
            history["slots"].shape[:3],
            device=device,
            dtype=torch.bool,
        )
        fixed_action_output = model(
            batch,
            history_mask=mask,
            actions_override=posterior,
        )
        swapped_fixed_action_output = model(
            swapped,
            history_mask=mask,
            actions_override=posterior,
        )
        base_prediction = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            posterior,
            mask,
            history["center"],
            None,
        ).future_slots
        longer_prediction = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            longer_scale,
            posterior,
            mask,
            history["center"],
            None,
        ).future_slots

    metrics = {
        "history_future_swap_max_difference": maximum_difference(
            history,
            swapped_history,
        ),
        "prior_future_swap_max_difference": maximum_difference(
            prior,
            swapped_prior,
        ),
        "posterior_future_swap_mean_difference": float(
            (posterior.float() - swapped_posterior.float()).abs().mean()
        ),
        "fixed_action_future_swap_max_difference": prediction_difference(
            fixed_action_output,
            swapped_fixed_action_output,
        ),
        "fixed_action_override_max_difference": max(
            float(
                (
                    fixed_action_output["dynamics_actions"].float()
                    - posterior.float()
                ).abs().max()
            ),
            float(
                (
                    swapped_fixed_action_output["dynamics_actions"].float()
                    - posterior.float()
                ).abs().max()
            ),
        ),
        "prior_horizon_change_mean_difference": float(
            (prior.float() - longer_prior.float()).abs().mean()
        ),
        "dynamics_horizon_change_mean_difference": float(
            (base_prediction.float() - longer_prediction.float()).abs().mean()
        ),
        "condition_dim": config.condition_dim,
        "language_projector_absent": model.language_condition is None,
        "dynamics_language_path_absent": (
            model.dynamics.condition_input is None
            and model.dynamics.condition_modulations is None
        ),
        "warm_start": warm_start,
    }
    checks = {
        "history_invariant_to_future_swap": (
            metrics["history_future_swap_max_difference"] == 0.0
        ),
        "prior_invariant_to_future_images": (
            metrics["prior_future_swap_max_difference"] == 0.0
        ),
        "posterior_reads_future": (
            metrics["posterior_future_swap_mean_difference"] > 1e-6
        ),
        "fixed_action_blocks_future_observation_side_channels": (
            metrics["fixed_action_future_swap_max_difference"] <= 1e-6
            and metrics["fixed_action_override_max_difference"] <= 1e-6
        ),
        "prior_reads_requested_horizon": (
            metrics["prior_horizon_change_mean_difference"] > 1e-7
        ),
        "dynamics_reads_requested_horizon": (
            metrics["dynamics_horizon_change_mean_difference"] > 1e-7
        ),
        "language_path_absent": (
            metrics["language_projector_absent"]
            and metrics["dynamics_language_path_absent"]
        ),
    }
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values())
    if not metrics["passed"]:
        raise RuntimeError(f"visual sequence core contract failed: {checks}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        load_rgb=True,
    )
    report = {
        "status": "passed",
        "checkpoint": (
            os.path.abspath(args.checkpoint) if args.checkpoint else ""
        ),
        "data_root": os.path.abspath(args.data),
        "data": validate_data(dataset, args.samples),
        "model": None,
    }
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        report["model"] = validate_model(dataset, args.checkpoint, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
