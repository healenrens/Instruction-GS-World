"""Evaluate deployable and oracle Prior predictions on visual sequences."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from evaluate_posterior_dynamics_gate import (  # noqa: E402
    comparison,
    masked_feature_mse,
    rgb_distance,
    weighted_latent_mse,
)
from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel, AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.counterfactuals import (  # noqa: E402
    predict_shuffled_action,
    render_state,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import CausalVisualSequenceDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_baselines import linear_history_prediction  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_eval_controls import (  # noqa: E402
    ablate_future_time,
    ablate_history,
    empty_history_mask,
    grouped_metric_means,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


PREDICTIONS = (
    "posterior", "deterministic_prior", "zero_action", "copy",
    "linear_history", "history_ablated_prior", "time_ablated_prior",
)

def prior_prediction(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = model.latent_actions.prior.sample(
        output["prior_context"],
        sample_count=1,
        stochastic=stochastic,
    )[0]
    slots, centers = predict_shuffled_action(
        model,
        batch,
        output,
        actions,
        use_dynamics_condition=False,
    )
    features, rgb = render_state(model, batch, output, slots, centers)
    if rgb is None:
        raise ValueError("visual sequence Prior evaluation requires RGB")
    return actions, slots, features, rgb


def errors(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    output: dict,
    model,
) -> dict[str, torch.Tensor]:
    return {
        "feature_mse": masked_feature_mse(
            prediction["feature"].float(),
            batch["future_features"].float(),
            batch["future_valid"],
        ),
        "latent_mse": weighted_latent_mse(
            prediction["latent"].float(),
            output["target_future_slots"].float(),
            output["target_future_activity"],
        ),
        "rgb_distance": rgb_distance(
            prediction["rgb"].float(),
            batch["future_rgb"],
            batch["future_rgb_valid"],
            model.config.rgb_ssim_weight,
        ),
    }


def per_query_feature_error(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    error = (
        prediction.float() - batch["future_features"].float()
    ).square().mean(dim=-1)
    weight = batch["future_valid"].float()
    return (error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    prior_samples: int,
) -> dict:
    values = {
        metric: {name: [] for name in PREDICTIONS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    oracle = {
        metric: []
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    action_errors = {"deterministic": [], "best_of_n": []}
    action_diversity = []
    feature_diversity = []
    history_context_change = []
    time_context_change = []
    query_times = []
    anchor_indices = []
    per_query = {name: [] for name in (
        "posterior", "deterministic_prior", "copy", "linear_history"
    )}
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        mask = empty_history_mask(batch, model.config.object_slots)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, history_mask=mask)
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
            deterministic = prior_prediction(
                model,
                batch,
                output,
                stochastic=False,
            )
            history_batch = ablate_history(batch)
            history_output = model(
                history_batch,
                history_mask=mask,
            )
            history_prior = prior_prediction(
                model,
                history_batch,
                history_output,
                stochastic=False,
            )
            time_batch = ablate_future_time(batch)
            time_output = model(
                time_batch,
                history_mask=mask,
            )
            time_prior = prior_prediction(
                model,
                time_batch,
                time_output,
                stochastic=False,
            )
        if zero_rgb is None:
            raise ValueError("visual sequence Prior evaluation requires RGB")
        future_count = batch["future_features"].shape[1]
        predictions = {
            "posterior": {
                "feature": output["rendered_future_features"],
                "latent": output["predicted_future_slots"],
                "rgb": output["rendered_future_rgb"],
            },
            "deterministic_prior": {
                "feature": deterministic[2],
                "latent": deterministic[1],
                "rgb": deterministic[3],
            },
            "zero_action": {
                "feature": zero_feature,
                "latent": output["zero_action_future_slots"],
                "rgb": zero_rgb,
            },
            "copy": {
                "feature": batch["history_features"][:, -1:].expand(
                    -1, future_count, -1, -1
                ),
                "latent": output["online_history_slots"][:, -1:].expand(
                    -1, future_count, -1, -1
                ),
                "rgb": batch["history_rgb"][:, -1:].expand(
                    -1, future_count, -1, -1, -1
                ).float()
                / 255.0,
            },
            "linear_history": linear_history_prediction(
                batch,
                output["online_history_slots"],
            ),
            "history_ablated_prior": {
                "feature": history_prior[2],
                "latent": history_prior[1],
                "rgb": history_prior[3],
            },
            "time_ablated_prior": {
                "feature": time_prior[2],
                "latent": time_prior[1],
                "rgb": time_prior[3],
            },
        }
        for name, prediction in predictions.items():
            for metric, value in errors(
                prediction,
                batch,
                output,
                model,
            ).items():
                values[metric][name].append(value.cpu())
        query_times.append(batch["future_times"].cpu())
        anchor_indices.append(batch["anchor_frame_index"].cpu())
        for name in per_query:
            per_query[name].append(
                per_query_feature_error(
                    predictions[name]["feature"],
                    batch,
                ).cpu()
            )
        history_context_change.append(
            (
                output["prior_context"].float()
                - history_output["prior_context"].float()
            ).abs().mean(dim=tuple(range(1, output["prior_context"].ndim))).cpu()
        )
        time_context_change.append(
            (
                output["prior_context"].float()
                - time_output["prior_context"].float()
            ).abs().mean(dim=tuple(range(1, output["prior_context"].ndim))).cpu()
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled_actions = model.latent_actions.prior.sample(
                output["prior_context"],
                sample_count=prior_samples,
                stochastic=True,
            )
        sampled_errors = {
            metric: []
            for metric in oracle
        }
        sampled_features = []
        for actions in sampled_actions:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                slots, centers = predict_shuffled_action(
                    model,
                    batch,
                    output,
                    actions,
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
                raise ValueError("sampled Prior did not produce RGB")
            sampled_features.append(feature.float().cpu())
            sample_prediction = {
                "feature": feature,
                "latent": slots,
                "rgb": rgb,
            }
            sample_values = errors(sample_prediction, batch, output, model)
            for metric, value in sample_values.items():
                sampled_errors[metric].append(value.cpu())
        for metric, chunks in sampled_errors.items():
            oracle[metric].append(torch.stack(chunks).amin(dim=0))
        sampled_features = torch.stack(sampled_features)
        feature_diversity.append(
            sampled_features.std(dim=0, unbiased=False).flatten(1).mean(dim=1)
        )
        action_diversity.append(
            sampled_actions.float().std(dim=0, unbiased=False).flatten(1).mean(dim=1).cpu()
        )
        posterior_action = output["posterior_actions"].float()
        deterministic_error = (
            deterministic[0].float() - posterior_action
        ).square().mean(dim=(1, 2, 3))
        sample_action_error = (
            sampled_actions.float() - posterior_action[None]
        ).square().mean(dim=(2, 3, 4))
        action_errors["deterministic"].append(deterministic_error.cpu())
        action_errors["best_of_n"].append(sample_action_error.amin(dim=0).cpu())

    tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in prediction.items()
        }
        for metric, prediction in values.items()
    }
    oracle_tensors = {
        metric: torch.cat(chunks)
        for metric, chunks in oracle.items()
    }
    comparisons = {
        metric: {
            reference: comparison(
                prediction["deterministic_prior"],
                prediction[reference],
            )
            for reference in (
                "zero_action",
                "copy",
                "linear_history",
                "history_ablated_prior",
                "time_ablated_prior",
            )
        }
        for metric, prediction in tensors.items()
    }
    times = torch.cat(query_times)
    anchors = torch.cat(anchor_indices)
    query_errors = {
        name: torch.cat(chunks)
        for name, chunks in per_query.items()
    }
    return {
        "samples": len(times),
        "prior_samples": prior_samples,
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in prediction.items()
            }
            for metric, prediction in tensors.items()
        },
        "deterministic_prior_comparison": comparisons,
        "oracle_best_of_n": {
            "label": "coverage diagnostic, not deployable ranking",
            "mean": {
                metric: float(value.mean())
                for metric, value in oracle_tensors.items()
            },
        },
        "action_code_mse": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in action_errors.items()
        },
        "prior_action_diversity": float(torch.cat(action_diversity).mean()),
        "prior_feature_diversity": float(torch.cat(feature_diversity).mean()),
        "history_context_change": float(torch.cat(history_context_change).mean()),
        "time_context_change": float(torch.cat(time_context_change).mean()),
        "per_future_query": [
            {
                "query": index,
                "mean_seconds": float(times[:, index].mean()),
                "feature_mse": {
                    name: float(value[:, index].mean())
                    for name, value in query_errors.items()
                },
            }
            for index in range(times.shape[1])
        ],
        "per_anchor": grouped_metric_means(tensors, anchors),
        "gate": {
            "history_improves_prior_feature": (
                comparisons["feature_mse"]["history_ablated_prior"][
                    "absolute_improvement"
                ]
                > 0.0
            ),
            "time_improves_prior_feature": (
                comparisons["feature_mse"]["time_ablated_prior"][
                    "absolute_improvement"
                ]
                > 0.0
            ),
            "deterministic_prior_beats_copy_feature": (
                comparisons["feature_mse"]["copy"]["absolute_improvement"] > 0.0
            ),
            "deterministic_prior_beats_linear_feature": (
                comparisons["feature_mse"]["linear_history"]["absolute_improvement"]
                > 0.0
            ),
        },
    }


def saved_or_override(saved: dict, name: str, override):
    return override if override not in (0, "") else saved[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "heldseed", "heldtask"),
                        required=True)
    parser.add_argument("--max_items", type=int, default=48)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prior_samples", type=int, default=4)
    parser.add_argument("--history_frames", type=int, default=0)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--sequence_anchors", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch, args.prior_samples) < 1 or args.max_items < args.batch:
        raise ValueError("invalid Prior evaluation size")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    saved = checkpoint.get("args", {})
    if saved.get("data_format") != "sequence":
        raise ValueError("checkpoint was not trained with visual sequences")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("visual sequence Prior evaluation requires RGB and no language")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved_or_override(saved, "history_frames", args.history_frames),
        future_frames=saved_or_override(saved, "future_frames", args.future_frames),
        anchors=saved_or_override(saved, "sequence_anchors", args.sequence_anchors),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, True)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "evaluation": evaluate(model, loader, device, args.prior_samples),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
