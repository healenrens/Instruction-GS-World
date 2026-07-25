"""Counterfactual evaluation of physical-time use by the image-goal Prior."""
from __future__ import annotations

import torch

from .counterfactuals import predict_shuffled_action, render_state
from .goal_conditioning import encode_explicit_goal
from .goal_eval_predictions import history_from_output, prediction_errors
from .goal_eval_statistics import (
    clustered_paired_comparison,
    clustered_relative_margin_test,
    goal_action_weight,
    paired_comparison,
    weighted_action_errors,
)
from .goal_prior_contract import goal_prior_context


TIME_VARIANTS = (
    "correct_time",
    "wrong_query_time",
    "wrong_goal_time",
    "wrong_joint_time",
)
WRONG_TIME_VARIANTS = TIME_VARIANTS[1:]


def _move_batch(
    batch: dict,
    device: torch.device,
) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _collect_time_bank(loader) -> tuple[torch.Tensor, torch.Tensor]:
    query_times = []
    goal_times = []
    for batch in loader:
        query_times.append(batch["future_times"].float())
        goal_times.append(batch["goal_time"].float())
    if not query_times:
        raise ValueError("wrong-time evaluation requires at least one batch")
    return torch.cat(query_times), torch.cat(goal_times)


def _farthest_indices(
    source: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    source_flat = source.float().reshape(source.shape[0], -1)
    bank_flat = bank.float().reshape(bank.shape[0], -1)
    if source_flat.shape[1] != bank_flat.shape[1]:
        raise ValueError("source time and time bank widths differ")
    distance = (
        source_flat[:, None] - bank_flat[None]
    ).square().mean(dim=-1)
    position = distance.argmax(dim=1)
    selected = bank[position]
    selected_distance = (
        selected.float().reshape(source.shape[0], -1) - source_flat
    ).square().mean(dim=-1)
    if bool((selected_distance <= 1e-12).any()):
        raise ValueError(
            "wrong-time evaluation requires at least two distinct horizons"
        )
    return position


def _condition_batches(
    batch: dict[str, torch.Tensor],
    query_bank: torch.Tensor,
    goal_bank: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    if not torch.allclose(
        goal_bank.float(),
        query_bank[:, -1].float(),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("time bank violates endpoint/query time identity")
    position = _farthest_indices(
        batch["future_times"].cpu(),
        query_bank,
    )
    wrong_query = query_bank[position].to(batch["future_times"])
    wrong_goal = goal_bank[position].to(batch["goal_time"])
    query_batch = dict(batch)
    query_batch["future_times"] = wrong_query
    goal_batch = dict(batch)
    goal_batch["goal_time"] = wrong_goal
    joint_batch = dict(query_batch)
    joint_batch["goal_time"] = wrong_goal
    return {
        "correct_time": batch,
        "wrong_query_time": query_batch,
        "wrong_goal_time": goal_batch,
        "wrong_joint_time": joint_batch,
    }, {
        "query_time_rms_seconds": (
            wrong_query.float() - batch["future_times"].float()
        ).square().mean(dim=1).sqrt(),
        "goal_time_abs_seconds": (
            wrong_goal.float() - batch["goal_time"].float()
        ).abs(),
    }


@torch.no_grad()
def evaluate_goal_time_counterfactual(
    model,
    conditioner,
    loader,
    device: torch.device,
    action_activity_floor: float = 0.25,
    query_time_bank: torch.Tensor | None = None,
    goal_time_bank: torch.Tensor | None = None,
) -> dict:
    """Hold the goal and Dynamics time fixed while replacing Prior time inputs."""
    if (query_time_bank is None) != (goal_time_bank is None):
        raise ValueError("query and goal time banks must be supplied together")
    if query_time_bank is None:
        query_bank, goal_bank = _collect_time_bank(loader)
    else:
        query_bank = query_time_bank.float().cpu()
        goal_bank = goal_time_bank.float().cpu()
    values = {
        metric: {name: [] for name in TIME_VARIANTS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    action_errors = {name: [] for name in TIME_VARIANTS}
    weighted_actions: dict[str, dict[str, list[torch.Tensor]]] = {}
    context_rms = {name: [] for name in WRONG_TIME_VARIANTS}
    action_rms = {name: [] for name in WRONG_TIME_VARIANTS}
    sequence_ids = []
    substitutions = {
        "query_time_rms_seconds": [],
        "goal_time_abs_seconds": [],
    }
    sample_offset = 0
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        batch_size = batch["future_times"].shape[0]
        condition_batches, time_delta = _condition_batches(
            batch,
            query_bank,
            goal_bank,
        )
        for name, value in time_delta.items():
            substitutions[name].append(value.cpu())
        history_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, history_mask=history_mask)
            history = history_from_output(output)
            goal = encode_explicit_goal(model, batch, history)
            contexts = {
                name: goal_prior_context(
                    model,
                    conditioner,
                    condition_batch,
                    history,
                    goal,
                )[0]
                for name, condition_batch in condition_batches.items()
            }
            actions = {
                name: model.latent_actions.prior.sample(
                    context,
                    sample_count=1,
                    stochastic=False,
                )[0]
                for name, context in contexts.items()
            }
            predictions = {}
            for name, action in actions.items():
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
                    raise ValueError(
                        "wrong-time evaluation requires RGB predictions"
                    )
                predictions[name] = {
                    "feature": feature,
                    "latent": slots,
                    "rgb": rgb,
                }
        target_action = output["posterior_actions"].float()
        action_weight = goal_action_weight(
            history["activity"],
            output["target_future_activity"],
            target_action.shape,
            action_activity_floor,
            model.config.canonical_activity_power,
            model.config.canonical_activity_gate,
        )
        for name, prediction in predictions.items():
            for metric, value in prediction_errors(
                prediction,
                batch,
                output,
                model,
            ).items():
                values[metric][name].append(value.cpu())
            action_errors[name].append(
                (actions[name].float() - target_action)
                .square()
                .mean(dim=(1, 2, 3))
                .cpu()
            )
            for channel, error in weighted_action_errors(
                actions[name],
                target_action,
                action_weight,
            ).items():
                weighted_actions.setdefault(
                    channel,
                    {item: [] for item in TIME_VARIANTS},
                )[name].append(error.cpu())
        correct_context = contexts["correct_time"].float()
        correct_action = actions["correct_time"].float()
        for name in WRONG_TIME_VARIANTS:
            context_rms[name].append(
                (correct_context - contexts[name].float())
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .cpu()
            )
            action_rms[name].append(
                (correct_action - actions[name].float())
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .cpu()
            )
        sequence_ids.append(batch["sequence_index"].cpu())
        sample_offset += batch_size
    if sample_offset != len(query_bank):
        raise RuntimeError("time bank and evaluation loader differ")

    tensors = {
        metric: {
            name: torch.cat(chunks)
            for name, chunks in predictions.items()
        }
        for metric, predictions in values.items()
    }
    action_tensors = {
        name: torch.cat(chunks) for name, chunks in action_errors.items()
    }
    weighted_action_tensors = {
        channel: {
            name: torch.cat(chunks)
            for name, chunks in predictions.items()
        }
        for channel, predictions in weighted_actions.items()
    }
    clusters = torch.cat(sequence_ids)
    comparisons = {
        metric: {
            name: paired_comparison(
                predictions["correct_time"],
                predictions[name],
            )
            for name in WRONG_TIME_VARIANTS
        }
        for metric, predictions in tensors.items()
    }
    clustered_comparisons = {
        metric: {
            name: clustered_paired_comparison(
                predictions["correct_time"],
                predictions[name],
                clusters,
            )
            for name in WRONG_TIME_VARIANTS
        }
        for metric, predictions in tensors.items()
    }
    action_comparisons = {
        name: paired_comparison(
            action_tensors["correct_time"],
            action_tensors[name],
        )
        for name in WRONG_TIME_VARIANTS
    }
    clustered_action_comparisons = {
        name: clustered_paired_comparison(
            action_tensors["correct_time"],
            action_tensors[name],
            clusters,
        )
        for name in WRONG_TIME_VARIANTS
    }
    weighted_action_comparisons = {
        channel: {
            name: paired_comparison(
                predictions["correct_time"],
                predictions[name],
            )
            for name in WRONG_TIME_VARIANTS
        }
        for channel, predictions in weighted_action_tensors.items()
    }
    clustered_weighted_action_comparisons = {
        channel: {
            name: clustered_paired_comparison(
                predictions["correct_time"],
                predictions[name],
                clusters,
            )
            for name in WRONG_TIME_VARIANTS
        }
        for channel, predictions in weighted_action_tensors.items()
    }
    margin_tests = {
        "all_5pct": {
            name: clustered_relative_margin_test(
                weighted_action_tensors["all"]["correct_time"],
                weighted_action_tensors["all"][name],
                clusters,
                0.05,
            )
            for name in WRONG_TIME_VARIANTS
        },
        "center_2pct": {
            name: clustered_relative_margin_test(
                weighted_action_tensors["center_0_3"]["correct_time"],
                weighted_action_tensors["center_0_3"][name],
                clusters,
                0.02,
            )
            for name in WRONG_TIME_VARIANTS
        },
    }
    action_gate = {
        name: (
            margin_tests["all_5pct"][name]["margin_ci95_passed"]
            and margin_tests["center_2pct"][name]["margin_ci95_passed"]
        )
        for name in WRONG_TIME_VARIANTS
    }
    prediction_gate = {
        metric: {
            name: comparison["positive_ci95_lower"]
            for name, comparison in variants.items()
        }
        for metric, variants in clustered_comparisons.items()
    }
    return {
        "samples": sample_offset,
        "clusters": len(torch.unique(clusters)),
        "protocol": {
            "joint_time": "paired physical-horizon counterfactual",
            "single_time_branch": "wiring ablation only",
            "history": "fixed",
            "goal_image": "fixed",
            "flow_source": "deterministic_zero",
            "dynamics_time": "correct",
        },
        "time_substitution": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in substitutions.items()
        },
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in predictions.items()
            }
            for metric, predictions in tensors.items()
        },
        "action_code_mse": {
            name: float(value.mean())
            for name, value in action_tensors.items()
        },
        "correct_time_comparison": comparisons,
        "clustered_correct_time_comparison": clustered_comparisons,
        "action_code_comparison": action_comparisons,
        "clustered_action_code_comparison": clustered_action_comparisons,
        "activity_weighted_action_code_mse": {
            channel: {
                name: float(value.mean())
                for name, value in predictions.items()
            }
            for channel, predictions in weighted_action_tensors.items()
        },
        "activity_weighted_action_code_comparison": (
            weighted_action_comparisons
        ),
        "clustered_activity_weighted_action_code_comparison": (
            clustered_weighted_action_comparisons
        ),
        "clustered_action_margin_tests": margin_tests,
        "context_rms_difference": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in context_rms.items()
        },
        "action_rms_difference": {
            name: float(torch.cat(chunks).mean())
            for name, chunks in action_rms.items()
        },
        "gate": {
            "query_branch_action_sensitivity": action_gate[
                "wrong_query_time"
            ],
            "goal_branch_action_sensitivity": action_gate[
                "wrong_goal_time"
            ],
            "physical_horizon_improves_action": action_gate[
                "wrong_joint_time"
            ],
            "physical_horizon_improves_world_prediction": (
                action_gate["wrong_joint_time"]
                and all(
                    prediction_gate[metric]["wrong_joint_time"]
                    for metric in (
                        "feature_mse",
                        "latent_mse",
                        "rgb_distance",
                    )
                )
            ),
        },
    }
