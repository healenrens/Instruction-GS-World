"""Diagnose whether ordered mode-set labels agree with posterior/effect modes."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.mode_set_prior import (  # noqa: E402
    ModeSetActionPrior,
    ordered_group_responsibility,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402


def rank_components(values: torch.Tensor) -> torch.Tensor:
    """Rank C alternatives along an oriented principal axis."""
    centered = values - values.mean(dim=1, keepdim=True)
    principal = torch.linalg.svd(
        centered.detach(),
        full_matrices=False,
    ).Vh[:, 0]
    anchor = principal.abs().argmax(dim=-1, keepdim=True)
    orientation = torch.gather(principal, 1, anchor).sign()
    orientation = torch.where(
        orientation == 0,
        torch.ones_like(orientation),
        orientation,
    )
    score = torch.einsum(
        "gcd,gd->gc",
        centered.detach(),
        principal * orientation,
    )
    return score.argsort(dim=-1).argsort(dim=-1)


def permutation_histogram(
    assignment: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int]:
    selected = assignment[mask].detach().cpu()
    histogram: dict[str, int] = {}
    for row in selected:
        key = "".join(str(int(value)) for value in row)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def optimal_assignment(
    cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-to-component assignment and mean matched cost."""
    components = cost.shape[-1]
    permutations = torch.tensor(
        tuple(itertools.permutations(range(components))),
        device=cost.device,
        dtype=torch.long,
    )
    expanded = cost[:, None].expand(-1, len(permutations), -1, -1)
    indices = permutations[None, :, :, None].expand(
        cost.shape[0],
        -1,
        -1,
        1,
    )
    permutation_cost = torch.gather(
        expanded,
        dim=3,
        index=indices,
    ).squeeze(-1).mean(dim=-1)
    best = permutation_cost.argmin(dim=-1)
    assignment = permutations[best]
    matched = torch.gather(
        cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1).mean(dim=-1)
    return assignment, matched


def matched_cost(
    cost: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(
        cost,
        dim=2,
        index=assignment[..., None],
    ).squeeze(-1).mean(dim=-1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value[mask].mean())


def pairwise_active_cost(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    if normalize:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
    error = (
        prediction[:, None] - target[:, :, None]
    ).square().mean(dim=-1)
    weight = activity[:, :, None].to(error.dtype)
    return (
        (error * weight).sum(dim=(-1, -2))
        / weight.sum(dim=(-1, -2)).clamp_min(1.0)
    )


def active_target_distance(
    target: torch.Tensor,
    activity: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    representation = F.normalize(target, dim=-1) if normalize else target
    error = (
        representation[:, :, None] - representation[:, None, :]
    ).square().mean(dim=-1)
    weight = torch.minimum(
        activity[:, :, None],
        activity[:, None, :],
    ).to(error.dtype)
    return (
        (error * weight).sum(dim=(-1, -2))
        / weight.sum(dim=(-1, -2)).clamp_min(1.0)
    )


def active_coverage(
    prediction_cost: torch.Tensor,
    target_distance: torch.Tensor,
    ambiguity: torch.Tensor,
) -> tuple[float, float]:
    off_diagonal = ~torch.eye(
        target_distance.shape[-1],
        device=target_distance.device,
        dtype=torch.bool,
    )[None]
    separation = target_distance.masked_fill(
        ~off_diagonal,
        torch.inf,
    ).amin(dim=(1, 2))
    valid = ambiguity & torch.isfinite(separation) & (separation > 1e-8)
    hit = prediction_cost <= 0.25 * separation[:, None, None]
    recall = hit.any(dim=-1).float().mean(dim=-1)
    precision = hit.any(dim=1).float().mean(dim=-1)
    return masked_mean(recall, valid), masked_mean(precision, valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_groups", type=int, default=128)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--semantic_branch_strength", type=float, default=2.0)
    parser.add_argument("--evaluation_seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.eval_groups, args.grid_size, args.future_steps) <= 0:
        raise ValueError("eval_groups, grid_size, and future_steps must be positive")

    torch.manual_seed(args.evaluation_seed)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**state["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    prior = model.latent_actions.prior
    if not isinstance(prior, ModeSetActionPrior):
        raise ValueError("checkpoint must contain ModeSetActionPrior")
    components = prior.components
    if components != 3:
        raise ValueError("diagnostic currently requires exactly three components")

    batch = make_synthetic_batch(
        config.feature_dim,
        args.eval_groups * components,
        3,
        args.future_steps,
        args.grid_size,
        device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=components,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        balanced_ambiguity=True,
        semantic_branch_strength=args.semantic_branch_strength,
    )
    with torch.no_grad():
        output = model(
            batch,
            history_mask=torch.zeros(
                args.eval_groups * components,
                3,
                config.object_slots,
                device=device,
                dtype=torch.bool,
            ),
        )
        logits, prototypes = prior._distribution(output["prior_context"])

    target_centers = output["target_future_centers"]
    current_centers = output["target_history_centers"][:, -1]
    activity = output["target_future_activity"]
    unweighted = ordered_group_responsibility(
        target_centers,
        current_centers,
        components,
        batch["group_id"],
    ).argmax(dim=-1).reshape(args.eval_groups, components)
    weighted_effect = (
        (target_centers - current_centers[:, None])
        * activity[..., None]
    ).reshape(args.eval_groups, components, -1)
    weighted = rank_components(weighted_effect)
    posterior = output["posterior_actions"].reshape(
        args.eval_groups,
        components,
        -1,
    )
    posterior_order = rank_components(posterior)

    representatives = torch.arange(
        0,
        args.eval_groups * components,
        components,
        device=device,
    )
    prototype_set = prototypes[representatives]
    posterior_set = output["posterior_actions"].reshape(
        args.eval_groups,
        components,
        *output["posterior_actions"].shape[1:],
    )
    action_cost = (
        prototype_set[:, None] - posterior_set[:, :, None]
    ).square().mean(dim=(-1, -2, -3))
    optimal_action, optimal_action_cost = optimal_assignment(action_cost)
    ordered_action_cost = matched_cost(action_cost, unweighted)
    weighted_action_cost = matched_cost(action_cost, weighted)

    representative_batch = {
        key: value[representatives] if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    history = model.encode_history(representative_batch)
    history_scale = signed_gap_scale(
        representative_batch["history_times"],
        config.gap_reference,
    )
    future_scale = signed_gap_scale(
        representative_batch["future_times"],
        config.gap_reference,
    )
    history_mask = torch.zeros(
        history["slots"].shape[:3],
        device=device,
        dtype=torch.bool,
    )
    component_slots = []
    component_centers = []
    with torch.no_grad():
        for component in range(components):
            prediction = model.dynamics(
                history["slots"],
                history["activity"],
                history_scale,
                future_scale,
                prototype_set[:, component],
                history_mask,
                history["center"],
            )
            component_slots.append(prediction.future_slots)
            component_centers.append(
                prediction.future_centers
                if prediction.future_centers is not None
                else model.object_aggregator.decode_center(
                    prediction.future_slots
                )
            )
    predicted_slots = torch.stack(component_slots, dim=1)
    predicted_centers = torch.stack(component_centers, dim=1)
    predicted_features = model.object_aggregator.decode_feature(
        predicted_slots
    )
    target_slots = output["target_future_slots"].reshape(
        args.eval_groups,
        components,
        *output["target_future_slots"].shape[1:],
    )
    target_center_set = target_centers.reshape(
        args.eval_groups,
        components,
        *target_centers.shape[1:],
    )
    target_features = output["target_future_object_features"].reshape(
        args.eval_groups,
        components,
        *output["target_future_object_features"].shape[1:],
    )
    target_activity = output["target_future_activity"].reshape(
        args.eval_groups,
        components,
        *output["target_future_activity"].shape[1:],
    )
    slot_cost = (
        F.normalize(predicted_slots[:, None], dim=-1)
        - F.normalize(target_slots[:, :, None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    center_cost = (
        predicted_centers[:, None] - target_center_set[:, :, None]
    ).square().mean(dim=(-1, -2, -3))
    effect_cost = slot_cost + 10.0 * center_cost
    optimal_effect, optimal_effect_cost = optimal_assignment(effect_cost)
    ordered_effect_cost = matched_cost(effect_cost, unweighted)
    weighted_effect_cost = matched_cost(effect_cost, weighted)

    ambiguity = batch["ambiguity"][representatives].bool()
    active_slot_coverage = active_coverage(
        pairwise_active_cost(
            predicted_slots,
            target_slots,
            target_activity,
            True,
        ),
        active_target_distance(
            target_slots,
            target_activity,
            True,
        ),
        ambiguity,
    )
    active_feature_coverage = active_coverage(
        pairwise_active_cost(
            predicted_features,
            target_features,
            target_activity,
            True,
        ),
        active_target_distance(
            target_features,
            target_activity,
            True,
        ),
        ambiguity,
    )
    active_center_coverage = active_coverage(
        pairwise_active_cost(
            predicted_centers,
            target_center_set,
            target_activity,
            False,
        ),
        active_target_distance(
            target_center_set,
            target_activity,
            False,
        ),
        ambiguity,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "evaluation_seed": args.evaluation_seed,
        "eval_groups": args.eval_groups,
        "ambiguous_groups": int(ambiguity.sum()),
        "deterministic_groups": int((~ambiguity).sum()),
        "mode_probability_mean": F.softmax(logits, dim=-1).mean(
            dim=0
        ).tolist(),
        "active_slot_recall": active_slot_coverage[0],
        "active_slot_precision": active_slot_coverage[1],
        "active_feature_recall": active_feature_coverage[0],
        "active_feature_precision": active_feature_coverage[1],
        "active_center_recall": active_center_coverage[0],
        "active_center_precision": active_center_coverage[1],
        "unweighted_permutations_ambiguous": permutation_histogram(
            unweighted,
            ambiguity,
        ),
        "weighted_permutations_ambiguous": permutation_histogram(
            weighted,
            ambiguity,
        ),
        "posterior_permutations_ambiguous": permutation_histogram(
            posterior_order,
            ambiguity,
        ),
        "optimal_action_permutations_ambiguous": permutation_histogram(
            optimal_action,
            ambiguity,
        ),
        "optimal_effect_permutations_ambiguous": permutation_histogram(
            optimal_effect,
            ambiguity,
        ),
        "weighted_vs_unweighted_agreement_ambiguous": masked_mean(
            (weighted == unweighted).all(dim=-1).float(),
            ambiguity,
        ),
        "posterior_vs_unweighted_agreement_ambiguous": masked_mean(
            (posterior_order == unweighted).all(dim=-1).float(),
            ambiguity,
        ),
        "optimal_action_vs_unweighted_agreement_ambiguous": masked_mean(
            (optimal_action == unweighted).all(dim=-1).float(),
            ambiguity,
        ),
        "optimal_effect_vs_unweighted_agreement_ambiguous": masked_mean(
            (optimal_effect == unweighted).all(dim=-1).float(),
            ambiguity,
        ),
        "ordered_action_fit_mse_ambiguous": masked_mean(
            ordered_action_cost,
            ambiguity,
        ),
        "weighted_action_fit_mse_ambiguous": masked_mean(
            weighted_action_cost,
            ambiguity,
        ),
        "optimal_action_fit_mse_ambiguous": masked_mean(
            optimal_action_cost,
            ambiguity,
        ),
        "ordered_effect_cost_ambiguous": masked_mean(
            ordered_effect_cost,
            ambiguity,
        ),
        "weighted_effect_cost_ambiguous": masked_mean(
            weighted_effect_cost,
            ambiguity,
        ),
        "optimal_effect_cost_ambiguous": masked_mean(
            optimal_effect_cost,
            ambiguity,
        ),
        "ordered_over_optimal_action_ratio_ambiguous": masked_mean(
            ordered_action_cost / optimal_action_cost.clamp_min(1e-8),
            ambiguity,
        ),
        "ordered_over_optimal_effect_ratio_ambiguous": masked_mean(
            ordered_effect_cost / optimal_effect_cost.clamp_min(1e-8),
            ambiguity,
        ),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
