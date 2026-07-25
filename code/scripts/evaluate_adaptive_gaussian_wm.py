"""Held-split evaluation for a full-state adaptive world-model checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.counterfactuals import (  # noqa: E402
    predict_shuffled_action,
    render_state,
)
from igsw.adaptive_gaussian_wm.evaluation_runtime import (  # noqa: E402
    restore_rng_state,
    rng_state,
)
from igsw.adaptive_gaussian_wm.metrics import pearson_correlation  # noqa: E402
from igsw.adaptive_gaussian_wm.motion_evaluation import (  # noqa: E402
    slot_motion_scores,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    masked_rgb_mean,
    rgb_reconstruction_loss,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


def per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction - target).square().mean(dim=(1, 2, 3))


def per_sample_latent_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.nn.functional.normalize(prediction, dim=-1)
        - torch.nn.functional.normalize(target, dim=-1)
    ).square().mean(dim=(1, 2, 3))


def per_sample_rgb_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> torch.Tensor:
    return torch.stack(
        [
            rgb_reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                valid[index : index + 1],
                ssim_weight,
            )[0]
            for index in range(len(prediction))
        ]
    )


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    prior_samples: int,
) -> dict:
    feature_errors = {
        name: [] for name in ("posterior", "zero_action", "shuffled_action", "copy")
    }
    latent_errors = {
        name: [] for name in ("posterior", "zero_action", "shuffled_action", "copy")
    }
    rgb_errors = {
        name: []
        for name in (
            "current",
            "current_global_color",
            "posterior",
            "zero_action",
            "shuffled_action",
        )
    }
    center_errors = {"posterior": [], "copy": []}
    prior_errors = []
    wrong_instruction_prior_errors = []
    prior_diversity = []
    effective_counts = []
    complexities = []
    prior_future_differences = []
    posterior_future_differences = []
    prior_instruction_differences = []
    motion_scores = {
        "motion_r2": [],
        "dynamic_motion_r2": [],
        "shuffled_motion_r2": [],
    }
    path_offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        object_mask = torch.zeros(
            batch["history_features"].shape[0],
            1,
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        output = model(batch, history_mask=object_mask)
        zero_feature, zero_rgb = render_state(
            model,
            batch,
            output,
            output["zero_action_future_slots"],
            output["zero_action_future_centers"],
        )
        shuffled_slots, shuffled_centers = predict_shuffled_action(
            model,
            batch,
            output,
        )
        shuffled_feature, shuffled_rgb = render_state(
            model,
            batch,
            output,
            shuffled_slots,
            shuffled_centers,
        )
        current_copy = batch["history_features"][:, -1, None]
        for name, prediction in {
            "posterior": output["rendered_future_features"],
            "zero_action": zero_feature,
            "shuffled_action": shuffled_feature,
            "copy": current_copy,
        }.items():
            feature_errors[name].append(
                per_sample_mse(prediction, batch["future_features"])
            )
        latent_copy = output["online_history_slots"][:, -1, None]
        for name, prediction in {
            "posterior": output["predicted_future_slots"],
            "zero_action": output["zero_action_future_slots"],
            "shuffled_action": shuffled_slots,
            "copy": latent_copy,
        }.items():
            latent_errors[name].append(
                per_sample_latent_mse(
                    prediction,
                    output["target_future_slots"],
                )
            )
        center_errors["posterior"].append(
            per_sample_mse(
                output["predicted_future_centers"],
                output["target_future_centers"],
            )
        )
        center_errors["copy"].append(
            per_sample_mse(
                output["target_history_centers"][:, -1, None],
                output["target_future_centers"],
            )
        )
        if model.config.rgb_supervision:
            if zero_rgb is None or shuffled_rgb is None:
                raise ValueError("RGB-enabled evaluation requires auxiliary renders")
            current_target = batch["history_rgb"][:, -1:]
            current_valid = batch["history_rgb_valid"][:, -1:]
            background = masked_rgb_mean(current_target, current_valid)
            background = background[..., None, None].expand_as(current_target)
            predictions = {
                "current": output["rendered_current_rgb"],
                "current_global_color": background,
                "posterior": output["rendered_future_rgb"],
                "zero_action": zero_rgb,
                "shuffled_action": shuffled_rgb,
            }
            for name, prediction in predictions.items():
                target = current_target if name.startswith("current") else batch["future_rgb"]
                valid = current_valid if name.startswith("current") else batch["future_rgb_valid"]
                rgb_errors[name].append(
                    per_sample_rgb_distance(
                        prediction,
                        target,
                        valid,
                        model.config.rgb_ssim_weight,
                    )
                )

        state = rng_state(device)
        samples = model.predict_prior_features(batch, prior_samples)
        sample_error = (
            samples - batch["future_features"][None]
        ).square().mean(dim=(2, 3, 4))
        prior_errors.append(sample_error)
        prior_diversity.append(samples.std(dim=0, unbiased=False).mean())
        if model.config.condition_dim > 0:
            store = loader.dataset.condition_store
            if store is None:
                raise ValueError("language-conditioned evaluation requires a cache")
            wrong = dict(batch)
            wrong_index = (
                batch["condition_index"].detach().cpu() + 1
            ).remainder(len(store.features))
            wrong["condition_feature"] = store.features[wrong_index].to(device)
            restore_rng_state(state, device)
            wrong_samples = model.predict_prior_features(wrong, prior_samples)
            wrong_instruction_prior_errors.append(
                (
                    wrong_samples - batch["future_features"][None]
                ).square().mean(dim=(2, 3, 4))
            )
            wrong_condition = model.encode_condition(wrong)
            history = {
                "slots": output["online_history_slots"],
                "activity": torch.stack(
                    [item.activity for item in output["history_slot_states"]],
                    dim=1,
                ),
                "center": output["online_history_centers"],
            }
            wrong_context = model.prior_context(
                history,
                signed_gap_scale(
                    batch["future_times"],
                    model.config.gap_reference,
                ),
                signed_gap_scale(
                    batch["history_times"],
                    model.config.gap_reference,
                ),
                wrong_condition,
            )
            prior_instruction_differences.append(
                (output["prior_context"] - wrong_context).abs().mean()
            )

        token_state = output["history_token_states"][-1]
        effective_counts.append(token_state.activation.sum(dim=1).squeeze(-1))
        complexities.append(
            batch["history_features"][:, -1].var(dim=(1, 2))
        )
        order = torch.arange(
            len(batch["future_features"]) - 1,
            -1,
            -1,
            device=device,
        )
        swapped = dict(batch)
        swapped["future_features"] = batch["future_features"][order]
        swapped_output = model(swapped, history_mask=object_mask)
        prior_future_differences.append(
            (output["prior_context"] - swapped_output["prior_context"]).abs().max()
        )
        posterior_future_differences.append(
            (
                output["posterior_actions"]
                - swapped_output["posterior_actions"]
            )
            .abs()
            .mean()
        )
        batch_size = len(batch["history_features"])
        batch_motion = slot_motion_scores(
            output,
            loader.dataset.paths[path_offset : path_offset + batch_size],
            loader.dataset.grid_height,
            loader.dataset.grid_width,
        )
        path_offset += batch_size
        for name, values in batch_motion.items():
            motion_scores[name].extend(values)

    feature = {name: torch.cat(values) for name, values in feature_errors.items()}
    latent = {name: torch.cat(values) for name, values in latent_errors.items()}
    center = {name: torch.cat(values) for name, values in center_errors.items()}
    prior = torch.cat(prior_errors, dim=1)
    effective = torch.cat(effective_counts)
    complexity = torch.cat(complexities)
    result = {
        "samples": int(len(feature["posterior"])),
        "feature_mse": {
            name: float(value.mean()) for name, value in feature.items()
        },
        "latent_mse": {
            name: float(value.mean()) for name, value in latent.items()
        },
        "center_mse": {
            name: float(value.mean()) for name, value in center.items()
        },
        "posterior_improvement_vs_copy": float(
            (feature["copy"].mean() - feature["posterior"].mean())
            / feature["copy"].mean().clamp_min(1e-8)
        ),
        "prior_feature_mse": {
            str(count): float(prior[:count].min(dim=0).values.mean())
            for count in (1, prior_samples)
        },
        "prior_feature_diversity": float(torch.stack(prior_diversity).mean()),
        "effective_token_count_mean": float(effective.mean()),
        "effective_token_count_std": float(effective.std(unbiased=False)),
        "effective_token_feature_variance_pearson": pearson_correlation(
            effective,
            complexity,
        ),
        "prior_future_swap_max_abs_difference": float(
            torch.stack(prior_future_differences).max()
        ),
        "posterior_future_swap_mean_abs_difference": float(
            torch.stack(posterior_future_differences).mean()
        ),
        "slot_motion": {
            name: (
                float(torch.tensor(values).mean())
                if values
                else 0.0
            )
            for name, values in motion_scores.items()
        },
    }
    if model.config.rgb_supervision:
        result["rgb_distance"] = {
            name: float(torch.cat(values).mean())
            for name, values in rgb_errors.items()
        }
        current_rgb = result["rgb_distance"]["current"]
        baseline_rgb = result["rgb_distance"]["current_global_color"]
        result["current_rgb_improvement_vs_global_color"] = (
            baseline_rgb - current_rgb
        ) / max(baseline_rgb, 1e-8)
    if wrong_instruction_prior_errors:
        wrong_prior = torch.cat(wrong_instruction_prior_errors, dim=1)
        correct = prior[:prior_samples].min(dim=0).values.mean()
        wrong = wrong_prior[:prior_samples].min(dim=0).values.mean()
        result["language_prior"] = {
            "correct_instruction_best_of_n_feature_mse": float(correct),
            "wrong_instruction_best_of_n_feature_mse": float(wrong),
            "correct_improvement_vs_wrong": float(
                (wrong - correct) / wrong.clamp_min(1e-8)
            ),
            "prior_context_mean_abs_difference": float(
                torch.stack(prior_instruction_differences).mean()
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", default="")
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prior_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch <= 0 or args.prior_samples <= 0:
        raise ValueError("batch and prior_samples must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        condition_cache=args.condition_cache,
        load_rgb=config.rgb_supervision,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(
        config,
        dataset,
        config.condition_dim > 0,
        config.rgb_supervision,
    )
    expected_cache = checkpoint.get("args", {}).get("condition_feature_sha256", "")
    if expected_cache and expected_cache != dataset.condition_store.feature_sha256:
        raise ValueError("evaluation condition cache differs from checkpoint")
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
    metrics = evaluate(model, loader, device, args.prior_samples)
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "config": config.to_dict(),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
