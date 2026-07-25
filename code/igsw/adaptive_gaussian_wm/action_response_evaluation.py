"""Zero-training response scan for canonical posterior actions and Dynamics."""
from __future__ import annotations

import torch

from .action_embedding import bounded_action_embedding, gate_canonical_center
from .counterfactuals import predict_shuffled_action, render_state
from .goal_eval_predictions import prediction_errors
from .goal_eval_statistics import (
    clustered_paired_comparison,
    paired_comparison,
)


SCALES = (0.25, 0.5, 2.0, 4.0)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _action_variants(model, posterior: torch.Tensor) -> dict[str, torch.Tensor]:
    if posterior.shape[-1] != 6 or model.config.canonical_action_dim != 6:
        raise ValueError("canonical action response scan requires exactly 6D actions")
    if not model.config.bounded_residual_action:
        raise ValueError("action response scan requires bounded action projection")
    zero = torch.zeros_like(posterior)
    variants = {
        "zero_action": zero,
        "posterior": posterior,
        **{
            f"scale_{str(scale).replace('.', 'p')}": scale * posterior
            for scale in SCALES
        },
        "center_only": torch.cat((posterior[..., :3], zero[..., 3:]), dim=-1),
        "rgb_only": torch.cat((zero[..., :3], posterior[..., 3:]), dim=-1),
    }
    gate = model.config.canonical_center_gate
    variants["equivalent_center_gate_1"] = torch.cat(
        (posterior[..., :3] / gate, posterior[..., 3:]),
        dim=-1,
    )
    return variants


def _action_condition(model, action: torch.Tensor) -> torch.Tensor:
    canonical = gate_canonical_center(
        action,
        model.config.canonical_center_gate,
    )
    return bounded_action_embedding(model.dynamics.action_input, canonical)


def _weighted_object_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    weight = activity.float()
    return (error * weight).flatten(1).sum(dim=1) / (
        weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    )


def _weighted_center_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    return _weighted_object_mse(prediction, target, activity)


def _change_weighted_feature_mse(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_features"].float()
    current = batch["history_features"][:, -1:].float()
    error = (prediction.float() - target).square().mean(dim=-1)
    change = (target - current).square().mean(dim=-1).sqrt()
    weight = change * batch["future_valid"].float()
    return (error * weight).flatten(1).sum(dim=1) / (
        weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    )


def _change_weighted_rgb(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = batch["future_rgb"].float() / 255.0
    current = batch["history_rgb"][:, -1:].float() / 255.0
    error = torch.sqrt((prediction.float() - target).square() + 1e-6).mean(
        dim=2
    )
    change = (target - current).abs().mean(dim=2)
    weight = change * batch["future_rgb_valid"].float()
    return (error * weight).flatten(1).sum(dim=1) / (
        weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    )


def _prediction_metrics(
    model,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    values = prediction_errors(prediction, batch, output, model)
    values.update(
        object_feature_mse=_weighted_object_mse(
            prediction["object_feature"],
            output["target_future_object_features"],
            output["target_future_activity"],
        ),
        center_mse=_weighted_center_mse(
            prediction["center"],
            output["target_future_centers"],
            output["target_future_activity"],
        ),
        change_weighted_feature_mse=_change_weighted_feature_mse(
            prediction["feature"],
            batch,
        ),
        change_weighted_rgb=_change_weighted_rgb(
            prediction["rgb"],
            batch,
        ),
    )
    return values


@torch.no_grad()
def evaluate_action_response(model, loader, device: torch.device) -> dict:
    values: dict[str, dict[str, list[torch.Tensor]]] = {}
    effects: dict[str, dict[str, list[torch.Tensor]]] = {}
    sequence_ids = []
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        history_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, history_mask=history_mask)
            variants = _action_variants(model, output["posterior_actions"])
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
            if zero_rgb is None:
                raise ValueError("action response scan requires RGB predictions")
            predictions = {
                "zero_action": {
                    "latent": output["zero_action_future_slots"],
                    "center": output["zero_action_future_centers"],
                    "feature": zero_feature,
                    "rgb": zero_rgb,
                },
                "posterior": {
                    "latent": output["predicted_future_slots"],
                    "center": output["predicted_future_centers"],
                    "feature": output["rendered_future_features"],
                    "rgb": output["rendered_future_rgb"],
                },
            }
            for name, action in variants.items():
                if name in predictions:
                    continue
                slots, centers = predict_shuffled_action(
                    model,
                    batch,
                    output,
                    action,
                    use_dynamics_condition=False,
                )
                feature, rgb = render_state(
                    model,
                    batch,
                    output,
                    slots,
                    centers,
                )
                if rgb is None:
                    raise ValueError("action response scan requires RGB")
                predictions[name] = {
                    "latent": slots,
                    "center": centers,
                    "feature": feature,
                    "rgb": rgb,
                }
            for prediction in predictions.values():
                prediction["object_feature"] = (
                    model.object_aggregator.decode_feature(
                        prediction["latent"]
                    )
                )
            conditions = {
                name: _action_condition(model, action)
                for name, action in variants.items()
            }
        zero = predictions["zero_action"]
        zero_condition = conditions["zero_action"]
        for name, prediction in predictions.items():
            for metric, error in _prediction_metrics(
                model,
                prediction,
                batch,
                output,
            ).items():
                values.setdefault(metric, {}).setdefault(name, []).append(
                    error.cpu()
                )
            condition = conditions[name]
            for stage, difference in (
                ("projected_action", condition - zero_condition),
                ("slot", prediction["latent"] - zero["latent"]),
                ("center", prediction["center"] - zero["center"]),
                (
                    "object_feature",
                    prediction["object_feature"] - zero["object_feature"],
                ),
                ("dense_feature", prediction["feature"] - zero["feature"]),
                ("rgb", prediction["rgb"] - zero["rgb"]),
            ):
                effects.setdefault(stage, {}).setdefault(name, []).append(
                    difference.float()
                    .flatten(1)
                    .square()
                    .mean(dim=1)
                    .sqrt()
                    .cpu()
                )
        sequence_ids.append(batch["sequence_index"].cpu())

    tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in variants.items()
        }
        for metric, variants in values.items()
    }
    effect_tensors = {
        stage: {
            name: torch.cat(chunks)
            for name, chunks in variants.items()
        }
        for stage, variants in effects.items()
    }
    clusters = torch.cat(sequence_ids)
    comparisons = {
        metric: {
            name: paired_comparison(error, variants["zero_action"])
            for name, error in variants.items()
            if name != "zero_action"
        }
        for metric, variants in tensors.items()
    }
    clustered = {
        metric: {
            name: clustered_paired_comparison(
                error,
                variants["zero_action"],
                clusters,
            )
            for name, error in variants.items()
            if name != "zero_action"
        }
        for metric, variants in tensors.items()
    }
    feature = clustered["feature_mse"]
    best = max(feature, key=lambda name: feature[name]["relative_improvement"])
    return {
        "samples": len(clusters),
        "clusters": len(torch.unique(clusters)),
        "action_source": "future_conditioned_posterior_oracle",
        "deployable_prediction": False,
        "canonical_center_gate": model.config.canonical_center_gate,
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in variants.items()
            }
            for metric, variants in tensors.items()
        },
        "comparison_to_zero": comparisons,
        "clustered_comparison_to_zero": clustered,
        "effect_rms_from_zero": {
            stage: {
                name: float(value.mean())
                for name, value in variants.items()
            }
            for stage, variants in effect_tensors.items()
        },
        "best_feature_variant": best,
        "gate": {
            "posterior_feature_ci_positive": feature["posterior"][
                "positive_ci95_lower"
            ],
            "posterior_feature_relative_1pct": (
                feature["posterior"]["relative_improvement"] >= 0.01
            ),
            "some_variant_beats_posterior": (
                feature[best]["relative_improvement"]
                > feature["posterior"]["relative_improvement"]
            ),
        },
    }
