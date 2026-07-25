"""Held-split metrics for the deployable explicit image-goal action Prior."""
from __future__ import annotations

import torch

from .counterfactuals import predict_shuffled_action, render_state
from .goal_conditioning import encode_explicit_goal
from .goal_contrast import select_hard_wrong_goal
from .goal_eval_bank import build_goal_bank, select_goal_from_bank
from .goal_eval_predictions import history_from_output, prediction_errors
from .goal_eval_statistics import (
    clustered_paired_comparison,
    clustered_relative_margin_test,
    goal_action_weight,
    paired_comparison,
    weighted_action_errors,
)
from .goal_prior_contract import (
    causal_goal_prior_contract,
    goal_prior_context,
)


PREDICTIONS = ("posterior", "correct_goal", "wrong_goal", "zero_action", "copy")


@torch.no_grad()
def evaluate_goal_prior(
    model,
    conditioner,
    loader,
    device,
    wrong_goal_scope: str = "held",
    action_activity_floor: float = 0.25,
    goal_bank: dict[str, torch.Tensor] | None = None,
) -> dict:
    if wrong_goal_scope not in ("held", "batch"):
        raise ValueError("wrong goal scope must be held or batch")
    if wrong_goal_scope == "held" and goal_bank is None:
        goal_bank = build_goal_bank(model, loader, device)
    if wrong_goal_scope == "batch":
        goal_bank = None
    values = {
        metric: {name: [] for name in PREDICTIONS}
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    action_errors = {"correct_goal": [], "wrong_goal": [], "zero_action": []}
    weighted_actions: dict[str, dict[str, list[torch.Tensor]]] = {}
    context_change = []
    wrong_similarity = []
    wrong_different_source = []
    wrong_eligible_candidates = []
    sequence_ids = []
    query_times = []
    per_query = {name: [] for name in ("correct_goal", "wrong_goal", "copy")}
    contract = None
    for cpu_batch in loader:
        batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in cpu_batch.items()
        }
        if contract is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                contract = causal_goal_prior_contract(
                    model,
                    conditioner,
                    batch,
                )
        history_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, history_mask=history_mask)
            history = history_from_output(output)
            goal = encode_explicit_goal(model, batch, history)
            correct_context, _ = goal_prior_context(
                model,
                conditioner,
                batch,
                history,
                goal,
            )
            if goal_bank is None:
                wrong, selection = select_hard_wrong_goal(
                    history["slots"][:, -1],
                    history["activity"][:, -1],
                    goal,
                    batch["sequence_index"],
                )
                selection = {
                    "cosine_similarity": selection[
                        "wrong_goal_cosine_similarity"
                    ].expand(len(batch["sequence_index"])),
                    "different_source": selection[
                        "wrong_goal_unique_fraction"
                    ].expand(len(batch["sequence_index"])),
                    "eligible_candidates": (
                        batch["sequence_index"][:, None]
                        != batch["sequence_index"][None]
                    ).sum(dim=1),
                }
            else:
                wrong, selection = select_goal_from_bank(
                    history["slots"][:, -1],
                    history["activity"][:, -1],
                    goal,
                    batch["sequence_index"],
                    goal_bank,
                )
            wrong_context, _ = goal_prior_context(
                model,
                conditioner,
                batch,
                history,
                wrong,
            )
            correct_action = model.latent_actions.prior.sample(
                correct_context,
                sample_count=1,
                stochastic=False,
            )[0]
            wrong_action = model.latent_actions.prior.sample(
                wrong_context,
                sample_count=1,
                stochastic=False,
            )[0]
            correct_slots, correct_centers = predict_shuffled_action(
                model,
                batch,
                output,
                correct_action,
                use_dynamics_condition=False,
            )
            wrong_slots, wrong_centers = predict_shuffled_action(
                model,
                batch,
                output,
                wrong_action,
                use_dynamics_condition=False,
            )
            correct_feature, correct_rgb = render_state(
                model,
                batch,
                output,
                correct_slots,
                correct_centers,
            )
            wrong_feature, wrong_rgb = render_state(
                model,
                batch,
                output,
                wrong_slots,
                wrong_centers,
            )
            zero_feature, zero_rgb = render_state(
                model,
                batch,
                output,
                output["zero_action_future_slots"],
                output["zero_action_future_centers"],
            )
        if any(
            value is None
            for value in (
                output["rendered_future_rgb"],
                correct_rgb,
                wrong_rgb,
                zero_rgb,
            )
        ):
            raise ValueError("image-goal evaluation requires RGB predictions")
        future_count = batch["future_features"].shape[1]
        predictions = {
            "posterior": {
                "feature": output["rendered_future_features"],
                "latent": output["predicted_future_slots"],
                "rgb": output["rendered_future_rgb"],
            },
            "correct_goal": {
                "feature": correct_feature,
                "latent": correct_slots,
                "rgb": correct_rgb,
            },
            "wrong_goal": {
                "feature": wrong_feature,
                "latent": wrong_slots,
                "rgb": wrong_rgb,
            },
            "zero_action": {
                "feature": zero_feature,
                "latent": output["zero_action_future_slots"],
                "rgb": zero_rgb,
            },
            "copy": {
                "feature": batch["history_features"][:, -1:].expand(
                    -1,
                    future_count,
                    -1,
                    -1,
                ),
                "latent": output["online_history_slots"][:, -1:].expand(
                    -1,
                    future_count,
                    -1,
                    -1,
                ),
                "rgb": batch["history_rgb"][:, -1:].expand(
                    -1,
                    future_count,
                    -1,
                    -1,
                    -1,
                ).float()
                / 255.0,
            },
        }
        for name, prediction in predictions.items():
            for metric, value in prediction_errors(
                prediction,
                batch,
                output,
                model,
            ).items():
                values[metric][name].append(value.cpu())
        target_action = output["posterior_actions"].float()
        action_weight = goal_action_weight(
            history["activity"],
            output["target_future_activity"],
            target_action.shape,
            action_activity_floor,
            model.config.canonical_activity_power,
            model.config.canonical_activity_gate,
        )
        for name, action in (
            ("correct_goal", correct_action),
            ("wrong_goal", wrong_action),
            ("zero_action", torch.zeros_like(correct_action)),
        ):
            action_errors[name].append(
                (action.float() - target_action)
                .square()
                .mean(dim=(1, 2, 3))
                .cpu()
            )
            for channel, error in weighted_action_errors(
                action,
                target_action,
                action_weight,
            ).items():
                weighted_actions.setdefault(
                    channel,
                    {
                        item: []
                        for item in (
                            "correct_goal",
                            "wrong_goal",
                            "zero_action",
                        )
                    },
                )[name].append(error.cpu())
        context_change.append(
            (correct_context.float() - wrong_context.float())
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .cpu()
        )
        wrong_similarity.append(selection["cosine_similarity"].float().cpu())
        wrong_different_source.append(
            selection["different_source"].float().cpu()
        )
        wrong_eligible_candidates.append(
            selection["eligible_candidates"].float().cpu()
        )
        sequence_ids.append(batch["sequence_index"].cpu())
        query_times.append(batch["future_times"].cpu())
        for name in per_query:
            error = (
                predictions[name]["feature"].float()
                - batch["future_features"].float()
            ).square().mean(dim=-1)
            weight = batch["future_valid"].float()
            per_query[name].append(
                (
                    (error * weight).sum(dim=-1)
                    / weight.sum(dim=-1).clamp_min(1.0)
                ).cpu()
            )

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
            reference: paired_comparison(
                predictions["correct_goal"],
                predictions[reference],
            )
            for reference in ("wrong_goal", "zero_action", "copy")
        }
        for metric, predictions in tensors.items()
    }
    clustered_comparisons = {
        metric: {
            reference: clustered_paired_comparison(
                predictions["correct_goal"],
                predictions[reference],
                clusters,
            )
            for reference in ("wrong_goal", "zero_action", "copy")
        }
        for metric, predictions in tensors.items()
    }
    action_comparisons = {
        reference: paired_comparison(
            action_tensors["correct_goal"],
            action_tensors[reference],
        )
        for reference in ("wrong_goal", "zero_action")
    }
    clustered_action_comparisons = {
        reference: clustered_paired_comparison(
            action_tensors["correct_goal"],
            action_tensors[reference],
            clusters,
        )
        for reference in ("wrong_goal", "zero_action")
    }
    weighted_action_comparisons = {
        channel: {
            reference: paired_comparison(
                predictions["correct_goal"],
                predictions[reference],
            )
            for reference in ("wrong_goal", "zero_action")
        }
        for channel, predictions in weighted_action_tensors.items()
    }
    clustered_weighted_action_comparisons = {
        channel: {
            reference: clustered_paired_comparison(
                predictions["correct_goal"],
                predictions[reference],
                clusters,
            )
            for reference in ("wrong_goal", "zero_action")
        }
        for channel, predictions in weighted_action_tensors.items()
    }
    times = torch.cat(query_times)
    query_errors = {
        name: torch.cat(chunks) for name, chunks in per_query.items()
    }
    wrong_action = action_comparisons["wrong_goal"]
    wrong_feature = comparisons["feature_mse"]["wrong_goal"]
    paper_action = clustered_weighted_action_comparisons["all"]["wrong_goal"]
    paper_feature = clustered_comparisons["feature_mse"]["wrong_goal"]
    margin_tests = {
        "activity_weighted_action": clustered_relative_margin_test(
            weighted_action_tensors["all"]["correct_goal"],
            weighted_action_tensors["all"]["wrong_goal"],
            clusters,
            0.05,
        ),
        "feature_mse": clustered_relative_margin_test(
            tensors["feature_mse"]["correct_goal"],
            tensors["feature_mse"]["wrong_goal"],
            clusters,
            0.05,
        ),
    }
    return {
        "samples": len(times),
        "clusters": len(torch.unique(clusters)),
        "causal_contract": contract,
        "mean": {
            metric: {
                name: float(value.mean())
                for name, value in predictions.items()
            }
            for metric, predictions in tensors.items()
        },
        "correct_goal_comparison": comparisons,
        "clustered_correct_goal_comparison": clustered_comparisons,
        "action_code_mse": {
            name: float(value.mean())
            for name, value in action_tensors.items()
        },
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
        "clustered_margin_test_5pct": margin_tests,
        "no_op_goal_selection": {
            "scope": wrong_goal_scope,
            "selection_objective": (
                "closest other goal to current history"
            ),
            "pool_entries": (
                len(goal_bank["slots"])
                if goal_bank is not None
                else int(loader.batch_size)
            ),
            "pool_unique_sequences": (
                len(torch.unique(goal_bank["sequence_index"]))
                if goal_bank is not None
                else None
            ),
            "mean_eligible_entries": float(
                torch.cat(wrong_eligible_candidates).mean()
            ),
            "mean_cosine_similarity": float(
                torch.cat(wrong_similarity).mean()
            ),
            "different_source_fraction": float(
                torch.cat(wrong_different_source).mean()
            ),
        },
        "correct_wrong_context_rms": float(
            torch.cat(context_change).mean()
        ),
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
        "gate": {
            "causal_contract": bool(contract["gate"]["all_passed"]),
            "correct_goal_beats_wrong_action": (
                wrong_action["absolute_improvement"] > 0.0
            ),
            "correct_goal_beats_wrong_weighted_action": (
                paper_action["positive_ci95_lower"]
            ),
            "correct_goal_beats_wrong_feature": (
                wrong_feature["absolute_improvement"] > 0.0
            ),
            "paper_goal_margin_5pct": (
                paper_action["relative_improvement"] >= 0.05
                and margin_tests["activity_weighted_action"][
                    "margin_ci95_passed"
                ]
                and paper_feature["relative_improvement"] >= 0.05
                and margin_tests["feature_mse"]["margin_ci95_passed"]
            ),
        },
    }
