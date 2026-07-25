"""Small diverse feasibility experiment for adaptive GPSToken Object-JEPA."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    adaptive_world_model_loss,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.baselines import FlatFeaturePredictor  # noqa: E402
from igsw.adaptive_gaussian_wm.metrics import (  # noqa: E402
    masked_feature_mse,
    paired_max_difference,
    pearson_correlation,
    slot_purity,
)
from igsw.adaptive_gaussian_wm.training import representation_pretrain_loss  # noqa: E402

def _model_batch(batch: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    if variant != "no_scale":
        return batch
    result = dict(batch)
    result["history_times"] = torch.zeros_like(batch["history_times"])
    result["future_times"] = torch.zeros_like(batch["future_times"])
    return result


def train_object_model(
    variant: str,
    config: AdaptiveGaussianWMConfig,
    steps: int,
    pretrain_steps: int,
    batch_size: int,
    grid_size: int,
    future_steps: int,
    device: torch.device,
) -> tuple[AdaptiveGaussianObjectWorldModel, list[dict[str, float]]]:
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.2,
        feature=0.5,
        allocator=0.2,
        slot=0.02,
        action=0.5,
    )
    trace = []
    model.train()
    for _ in range(pretrain_steps):
        history_frames = 1 if torch.rand(()) < 0.25 else 3
        batch = make_synthetic_batch(
            config.feature_dim,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            paired_futures=True,
            irregular_gaps=True,
        )
        loss = representation_pretrain_loss(model, batch)[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_target()
    for step in range(1, steps + 1):
        history_frames = 1 if torch.rand(()) < 0.25 else 3
        batch = make_synthetic_batch(
            config.feature_dim,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            paired_futures=True,
            irregular_gaps=True,
        )
        batch = _model_batch(batch, variant)
        history_mask = None
        if variant == "no_mask":
            history_mask = torch.zeros(
                batch_size,
                history_frames,
                config.object_slots,
                device=device,
                dtype=torch.bool,
            )
        output = model(batch, history_mask=history_mask)
        loss, parts = adaptive_world_model_loss(model, batch, output, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_target()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "future": float(parts["future"].detach()),
                    "feature": float(parts["feature"].detach()),
                    "flow": float(parts["flow"].detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return model, trace


def train_flat_model(
    feature_dim: int,
    steps: int,
    batch_size: int,
    grid_size: int,
    future_steps: int,
    device: torch.device,
) -> tuple[FlatFeaturePredictor, list[dict[str, float]]]:
    model = FlatFeaturePredictor(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trace = []
    model.train()
    for step in range(1, steps + 1):
        history_frames = 1 if torch.rand(()) < 0.25 else 3
        batch = make_synthetic_batch(
            feature_dim,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            paired_futures=True,
            irregular_gaps=True,
        )
        scale = signed_gap_scale(batch["future_times"], 1.0)
        prediction = model(
            batch["history_features"],
            batch["future_coordinates"],
            scale,
        )
        loss = masked_feature_mse(
            prediction,
            batch["future_features"],
            batch["future_valid"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return model, trace


@torch.no_grad()
def evaluate_object_model(
    model: AdaptiveGaussianObjectWorldModel,
    variant: str,
    batch_size: int,
    grid_size: int,
    future_steps: int,
    device: torch.device,
) -> dict:
    model.eval()
    per_history = {}
    all_effective = []
    all_complexity = []
    for history_frames in (1, 3):
        torch.manual_seed(700 + history_frames)
        batch = make_synthetic_batch(
            model.config.feature_dim,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            paired_futures=True,
            irregular_gaps=True,
        )
        batch = _model_batch(batch, variant)
        zero_mask = torch.zeros(
            batch_size,
            history_frames,
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        output = model(batch, history_mask=zero_mask)
        feature_mse = masked_feature_mse(
            output["rendered_future_features"],
            batch["future_features"],
            batch["future_valid"],
        )
        current_copy = batch["history_features"][:, -1, None].expand_as(
            batch["future_features"]
        )
        copy_mse = masked_feature_mse(
            current_copy,
            batch["future_features"],
            batch["future_valid"],
        )
        foreground = batch["future_labels"] >= 0
        foreground_mse = masked_feature_mse(
            output["rendered_future_features"], batch["future_features"], foreground
        )
        copy_foreground_mse = masked_feature_mse(
            current_copy, batch["future_features"], foreground
        )
        latent_mse = (
            F.normalize(output["predicted_future_slots"], dim=-1)
            - F.normalize(output["target_future_slots"], dim=-1)
        ).square().mean()
        copy_latent = output["online_history_slots"][:, -1, None].expand_as(
            output["target_future_slots"]
        )
        copy_latent_mse = (
            F.normalize(copy_latent, dim=-1)
            - F.normalize(output["target_future_slots"], dim=-1)
        ).square().mean()
        effective = output["history_token_states"][-1].activation.sum(dim=1).squeeze(-1)
        all_effective.append(effective)
        all_complexity.append(batch["complexity"])
        prior_pair_difference = paired_max_difference(output["prior_context"])
        posterior_pair_difference = float(
            (
                output["posterior_actions"][0::2]
                - output["posterior_actions"][1::2]
            )
            .abs()
            .mean()
        )

        prior_features = model.predict_prior_features(batch, sample_count=8)
        sample_error = (
            prior_features - batch["future_features"][None]
        ).square().mean(dim=(2, 3, 4))
        coverage = {
            str(count): float(sample_error[:count].min(dim=0).values.mean())
            for count in (1, 2, 4, 8)
        }
        prior_diversity = float(prior_features.std(dim=0).mean())

        scaled_batch = dict(batch)
        scaled_batch["history_times"] = batch["history_times"] * 1.7
        scaled_batch["future_times"] = batch["future_times"] * 1.7
        scaled_output = model(scaled_batch, history_mask=zero_mask)
        scale_sensitivity = float(
            (
                output["predicted_future_slots"]
                - scaled_output["predicted_future_slots"]
            )
            .abs()
            .mean()
        )
        per_history[str(history_frames)] = {
            "feature_mse": float(feature_mse),
            "current_copy_feature_mse": float(copy_mse),
            "foreground_feature_mse": float(foreground_mse),
            "current_copy_foreground_mse": float(copy_foreground_mse),
            "relative_improvement_vs_copy": float(
                (copy_mse - feature_mse) / copy_mse.clamp_min(1e-8)
            ),
            "future_latent_mse": float(latent_mse),
            "current_copy_future_latent_mse": float(copy_latent_mse),
            "slot_purity": slot_purity(
                output,
                batch["history_labels"][:, -1],
            ),
            "prior_pair_max_difference": prior_pair_difference,
            "posterior_pair_mean_difference": posterior_pair_difference,
            "prior_feature_best_of_n_mse": coverage,
            "prior_feature_sample_diversity": prior_diversity,
            "scale_sensitivity": scale_sensitivity,
        }
    effective = torch.cat(all_effective)
    complexity = torch.cat(all_complexity)
    return {
        "by_history_frames": per_history,
        "effective_token_count_mean": float(effective.mean()),
        "effective_token_count_std": float(effective.std()),
        "effective_token_complexity_pearson": pearson_correlation(
            effective,
            complexity,
        ),
    }


@torch.no_grad()
def evaluate_flat_model(
    model: FlatFeaturePredictor,
    feature_dim: int,
    batch_size: int,
    grid_size: int,
    future_steps: int,
    device: torch.device,
) -> dict:
    model.eval()
    per_history = {}
    for history_frames in (1, 3):
        torch.manual_seed(700 + history_frames)
        batch = make_synthetic_batch(
            feature_dim,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            paired_futures=True,
            irregular_gaps=True,
        )
        prediction = model(
            batch["history_features"],
            batch["future_coordinates"],
            signed_gap_scale(batch["future_times"], 1.0),
        )
        feature_mse = masked_feature_mse(
            prediction,
            batch["future_features"],
            batch["future_valid"],
        )
        copy = batch["history_features"][:, -1, None].expand_as(
            batch["future_features"]
        )
        copy_mse = masked_feature_mse(
            copy,
            batch["future_features"],
            batch["future_valid"],
        )
        per_history[str(history_frames)] = {
            "feature_mse": float(feature_mse),
            "current_copy_feature_mse": float(copy_mse),
            "relative_improvement_vs_copy": float(
                (copy_mse - feature_mse) / copy_mse.clamp_min(1e-8)
            ),
        }
    return {"by_history_frames": per_history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("full", "no_mask", "no_scale", "flat"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pretrain_steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    start = time.time()

    config = AdaptiveGaussianWMConfig.tiny(args.feature_dim)
    if args.variant == "flat":
        model, trace = train_flat_model(
            args.feature_dim,
            args.steps,
            args.batch_size,
            args.grid_size,
            args.future_steps,
            device,
        )
        metrics = evaluate_flat_model(
            model,
            args.feature_dim,
            args.batch_size * 2,
            args.grid_size,
            args.future_steps,
            device,
        )
    else:
        model, trace = train_object_model(
            args.variant,
            config,
            args.steps,
            args.pretrain_steps,
            args.batch_size,
            args.grid_size,
            args.future_steps,
            device,
        )
        metrics = evaluate_object_model(
            model,
            args.variant,
            args.batch_size * 2,
            args.grid_size,
            args.future_steps,
            device,
        )

    if args.checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save(
            {
                "variant": args.variant,
                "config": config.to_dict(),
                "model": model.state_dict(),
            },
            args.checkpoint,
        )
    report = {
        "status": "ok",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "pretrain_steps": 0 if args.variant == "flat" else args.pretrain_steps,
        "batch_size": args.batch_size,
        "grid_size": args.grid_size,
        "future_steps": args.future_steps,
        "device": str(device),
        "elapsed_seconds": time.time() - start,
        "training_trace": trace,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
