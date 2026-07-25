"""Evaluate deployable history-and-instruction Prior against posterior and zero."""
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
    predict_shuffled_action, render_state,
)
from igsw.adaptive_gaussian_wm.pair_dataset import CausalPairFeatureDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.instruction_groups import (  # noqa: E402
    build_condition_task_bank, select_different_task_condition,
)
from igsw.adaptive_gaussian_wm.prior_gate_metrics import (  # noqa: E402
    feature_error, latent_error, paired_comparison, rgb_error,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device, validate_data_model_contract,
)
VARIANTS = ("posterior", "prior", "wrong_instruction", "zero_action")
def _history_from_output(output: dict) -> dict[str, torch.Tensor]:
    return {
        "slots": output["online_history_slots"],
        "activity": torch.stack(
            [state.activity for state in output["history_slot_states"]],
            dim=1,
        ),
        "center": output["online_history_centers"],
    }


@torch.no_grad()
def evaluate(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    condition_bank: torch.Tensor,
    condition_token_bank: torch.Tensor | None,
    condition_token_valid_bank: torch.Tensor | None,
    condition_task_bank: torch.Tensor,
    wrong_task_rank: int,
    dynamics_condition_scope: str,
) -> dict:
    use_dynamics_condition = dynamics_condition_scope == "prior_and_dynamics"
    values = {
        metric: {variant: [] for variant in VARIANTS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    action_rms = []
    wrong_action_rms = []
    prior_slot_effect = []
    wrong_slot_effect = []
    canonical_mse = []
    canonical_zero_mse = []
    canonical_wrong_mse = []
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else torch.no_grad
    )
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        history_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with amp_context():
            output = model(batch, history_mask=history_mask)
            prior_actions = model.latent_actions.prior.sample(
                output["prior_context"],
                sample_count=1,
                stochastic=False,
            )[0]
            condition = output["language_condition"]
            if condition is None:
                raise ValueError("Prior gate requires language conditioning")
            wrong_index = select_different_task_condition(
                batch["condition_index"].detach().cpu(),
                batch["task_index"].detach().cpu(),
                condition_task_bank,
                condition_bank,
                wrong_task_rank,
            )
            wrong_condition = model.language_condition(
                condition_bank[wrong_index].to(device, non_blocking=True)
            )
            history_scale = signed_gap_scale(
                batch["history_times"],
                model.config.gap_reference,
            )
            future_scale = signed_gap_scale(
                batch["future_times"],
                model.config.gap_reference,
            )
            wrong_context = model.prior_context(
                _history_from_output(output),
                future_scale,
                history_scale,
                wrong_condition,
                (
                    condition_token_bank[wrong_index].to(
                        device,
                        non_blocking=True,
                    )
                    if condition_token_bank is not None
                    else None
                ),
                (
                    condition_token_valid_bank[wrong_index].to(
                        device,
                        non_blocking=True,
                    )
                    if condition_token_valid_bank is not None
                    else None
                ),
            )
            wrong_actions = model.latent_actions.prior.sample(
                wrong_context,
                sample_count=1,
                stochastic=False,
            )[0]
            posterior_slots, posterior_centers = predict_shuffled_action(
                model,
                batch,
                output,
                output["posterior_actions"],
                use_dynamics_condition=use_dynamics_condition,
            )
            prior_slots, prior_centers = predict_shuffled_action(
                model,
                batch,
                output,
                prior_actions,
                use_dynamics_condition=use_dynamics_condition,
            )
            wrong_slots, wrong_centers = predict_shuffled_action(
                model,
                batch,
                output,
                wrong_actions,
                wrong_condition,
                use_dynamics_condition=use_dynamics_condition,
            )
            zero_slots, zero_centers = predict_shuffled_action(
                model,
                batch,
                output,
                torch.zeros_like(prior_actions),
                use_dynamics_condition=use_dynamics_condition,
            )
            posterior_features, posterior_rgb = render_state(
                model,
                batch,
                output,
                posterior_slots,
                posterior_centers,
            )
            prior_features, prior_rgb = render_state(
                model,
                batch,
                output,
                prior_slots,
                prior_centers,
            )
            wrong_features, wrong_rgb = render_state(
                model,
                batch,
                output,
                wrong_slots,
                wrong_centers,
            )
            zero_features, zero_rgb = render_state(
                model,
                batch,
                output,
                zero_slots,
                zero_centers,
            )
        if any(
            value is None
            for value in (
                posterior_rgb,
                prior_rgb,
                wrong_rgb,
                zero_rgb,
            )
        ):
            raise ValueError("Prior gate requires RGB supervision")
        feature_predictions = {
            "posterior": posterior_features.float(),
            "prior": prior_features.float(),
            "wrong_instruction": wrong_features.float(),
            "zero_action": zero_features.float(),
        }
        latent_predictions = {
            "posterior": posterior_slots.float(),
            "prior": prior_slots.float(),
            "wrong_instruction": wrong_slots.float(),
            "zero_action": zero_slots.float(),
        }
        rgb_predictions = {
            "posterior": posterior_rgb.float(),
            "prior": prior_rgb.float(),
            "wrong_instruction": wrong_rgb.float(),
            "zero_action": zero_rgb.float(),
        }
        for variant in VARIANTS:
            values["feature_mse"][variant].append(
                feature_error(
                    feature_predictions[variant],
                    batch["future_features"].float(),
                    batch["future_valid"],
                ).cpu()
            )
            values["latent_mse"][variant].append(
                latent_error(
                    latent_predictions[variant],
                    output["target_future_slots"].float(),
                    output["target_future_activity"],
                ).cpu()
            )
            values["rgb_distance"][variant].append(
                rgb_error(
                    rgb_predictions[variant],
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    model.config.rgb_ssim_weight,
                ).cpu()
            )
        posterior_actions = output["posterior_actions"].float()
        action_rms.append(
            (prior_actions.float() - posterior_actions)
            .square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        wrong_action_rms.append(
            (wrong_actions.float() - prior_actions.float())
            .square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        prior_slot_effect.append(
            (prior_slots.float() - zero_slots.float())
            .square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        wrong_slot_effect.append(
            (wrong_slots.float() - prior_slots.float())
            .square().mean(dim=(1, 2, 3)).sqrt().cpu()
        )
        if model.config.canonical_semantic_action:
            canonical_mse.append(
                (prior_actions[..., :6].float() - posterior_actions[..., :6])
                .square().mean(dim=(1, 2, 3)).cpu()
            )
            canonical_zero_mse.append(
                posterior_actions[..., :6].float()
                .square().mean(dim=(1, 2, 3)).cpu()
            )
            canonical_wrong_mse.append(
                (wrong_actions[..., :6].float() - posterior_actions[..., :6])
                .square().mean(dim=(1, 2, 3)).cpu()
            )

    tensors = {
        metric: {
            variant: torch.cat(chunks)
            for variant, chunks in variants.items()
        }
        for metric, variants in values.items()
    }
    means = {
        metric: {
            variant: float(value.mean())
            for variant, value in variants.items()
        }
        for metric, variants in tensors.items()
    }
    comparisons = {
        metric: {
            "prior_vs_zero": paired_comparison(
                variants["prior"], variants["zero_action"]
            ),
            "prior_vs_wrong_instruction": paired_comparison(
                variants["prior"],
                variants["wrong_instruction"],
            ),
            "posterior_vs_prior": paired_comparison(
                variants["posterior"],
                variants["prior"],
            ),
        }
        for metric, variants in tensors.items()
    }
    gap_closure = {}
    for metric, variants in means.items():
        denominator = variants["zero_action"] - variants["posterior"]
        gap_closure[metric] = (
            (variants["zero_action"] - variants["prior"]) / denominator
            if denominator > 0.0
            else None
        )
    prior_zero = [
        item["prior_vs_zero"]
        for item in comparisons.values()
    ]
    instruction = [
        item["prior_vs_wrong_instruction"]
        for item in comparisons.values()
    ]
    return {
        "samples": int(next(iter(tensors.values()))["prior"].shape[0]),
        "mean": means,
        "comparison": comparisons,
        "posterior_gap_closure": gap_closure,
        "diagnostics": {
            "prior_posterior_action_rms": float(torch.cat(action_rms).mean()),
            "correct_wrong_instruction_action_rms": float(
                torch.cat(wrong_action_rms).mean()
            ),
            "prior_vs_zero_slot_rms": float(
                torch.cat(prior_slot_effect).mean()
            ),
            "correct_wrong_instruction_slot_rms": float(
                torch.cat(wrong_slot_effect).mean()
            ),
            "canonical_prior_posterior_mse": (
                float(torch.cat(canonical_mse).mean())
                if canonical_mse
                else None
            ),
            "canonical_zero_posterior_mse": (
                float(torch.cat(canonical_zero_mse).mean())
                if canonical_zero_mse
                else None
            ),
            "canonical_wrong_posterior_mse": (
                float(torch.cat(canonical_wrong_mse).mean())
                if canonical_wrong_mse
                else None
            ),
        },
        "gate": {
            "prior_beats_zero_all_metrics": all(
                item["absolute_improvement"] > 0.0
                for item in prior_zero
            ),
            "prior_beats_zero_all_metrics_2se": all(
                item["positive_2se_margin"]
                for item in prior_zero
            ),
            "correct_instruction_beats_wrong_all_metrics": all(
                item["absolute_improvement"] > 0.0
                for item in instruction
            ),
            "correct_instruction_beats_wrong_all_metrics_2se": all(
                item["positive_2se_margin"]
                for item in instruction
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--token_gate_scale", type=float, default=1.0)
    parser.add_argument("--wrong_task_rank", type=int, default=0)
    parser.add_argument("--dynamics_condition_scope",
                        choices=("prior_and_dynamics", "prior_only"),
                        default="prior_and_dynamics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items % args.batch:
        raise ValueError("max_items must be divisible by batch >= 2")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, True, True)
    if dataset.condition_store is None or len(dataset.condition_store.features) < 2:
        raise ValueError("Prior gate requires at least two cached instructions")
    condition_task_bank = build_condition_task_bank(
        dataset.all_paths, dataset.condition_store)
    expected_cache = checkpoint.get("args", {}).get(
        "condition_feature_sha256",
        "",
    )
    if expected_cache and expected_cache != dataset.condition_store.feature_sha256:
        raise ValueError("evaluation condition cache differs from checkpoint")
    expected_tokens = checkpoint.get("args", {}).get(
        "condition_token_sha256",
        "",
    )
    if (
        expected_tokens
        and expected_tokens != dataset.condition_store.token_feature_sha256
    ):
        raise ValueError("evaluation token condition cache differs from checkpoint")
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.token_gate_scale < 0.0:
        raise ValueError("token gate scale must be non-negative")
    if config.token_conditioned_prior:
        with torch.no_grad():
            model.latent_actions.prior_token_conditioner.gate.mul_(
                args.token_gate_scale
            )
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
        "split": args.split,
        "wrong_instruction_scope": args.dynamics_condition_scope,
        "action_anchor": (
            "object_slot" if config.object_aligned_actions else "global"
        ),
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "action_residual_gate": config.action_residual_gate,
        "action_residual_dropout": config.action_residual_dropout,
        "canonical_activity_gate": config.canonical_activity_gate,
        "canonical_activity_power": config.canonical_activity_power,
        "token_conditioned_prior": config.token_conditioned_prior,
        "token_gate_scale": args.token_gate_scale,
        "wrong_task_rank": args.wrong_task_rank,
        "learned_semantic_action_basis": config.learned_semantic_action_basis,
        "metrics": evaluate(
            model,
            loader,
            device,
            args.amp,
            dataset.condition_store.features,
            dataset.condition_store.token_features,
            dataset.condition_store.token_valid,
            condition_task_bank,
            args.wrong_task_rank,
            args.dynamics_condition_scope,
        ),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
