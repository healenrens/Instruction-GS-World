"""Zero-action-normalized Dynamics effects for deployable Prior training."""
from __future__ import annotations

import torch


def _future_centers(model, prediction) -> torch.Tensor:
    if prediction.future_centers is not None:
        return prediction.future_centers
    return model.object_aggregator.decode_center(prediction.future_slots)


def _predict_dynamics(
    model,
    history: dict[str, torch.Tensor],
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    actions: torch.Tensor,
    condition: torch.Tensor | None,
):
    history_mask = torch.zeros(
        history["slots"].shape[:3],
        device=history["slots"].device,
        dtype=torch.bool,
    )
    return model.dynamics(
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        actions,
        history_mask,
        history["center"],
        condition,
    )


def _component_errors(
    model,
    prediction,
    target_slots: torch.Tensor,
    target_features: torch.Tensor,
    target_centers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_features = model.object_aggregator.decode_feature(
        prediction.future_slots
    )
    predicted_centers = _future_centers(model, prediction)
    reduce_slots = tuple(range(1, prediction.future_slots.ndim))
    reduce_features = tuple(range(1, predicted_features.ndim))
    reduce_centers = tuple(range(1, predicted_centers.ndim))
    return {
        "slot": (
            prediction.future_slots.float() - target_slots.float()
        ).square().mean(dim=reduce_slots),
        "feature": (
            predicted_features.float() - target_features.float()
        ).square().mean(dim=reduce_features),
        "center": (
            predicted_centers.float() - target_centers.float()
        ).square().mean(dim=reduce_centers),
    }


def relative_dynamics_effects(
    model,
    history: dict[str, torch.Tensor],
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    action_variants: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    condition: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Measure each action effect relative to the zero-action error."""
    if not action_variants:
        raise ValueError("relative Dynamics effect requires an action variant")
    with torch.no_grad():
        target = _predict_dynamics(
            model,
            history,
            history_scale,
            future_scale,
            target_actions,
            condition,
        )
        target_slots = target.future_slots
        target_features = model.object_aggregator.decode_feature(target_slots)
        target_centers = _future_centers(model, target)
        zero = _predict_dynamics(
            model,
            history,
            history_scale,
            future_scale,
            torch.zeros_like(target_actions),
            condition,
        )
        zero_errors = _component_errors(
            model,
            zero,
            target_slots,
            target_features,
            target_centers,
        )
        denominators = {
            name: 1.05 * value.mean().detach().clamp_min(1e-8)
            for name, value in zero_errors.items()
        }

    results: dict[str, dict[str, torch.Tensor]] = {}
    zero_relative = {
        name: zero_errors[name] / denominators[name]
        for name in zero_errors
    }
    zero_total = torch.stack(tuple(zero_relative.values()), dim=0).mean(dim=0)
    results["_zero"] = {
        "per_sample": zero_total,
        "total": zero_total.mean(),
        **{name: value.mean() for name, value in zero_relative.items()},
    }
    for variant, actions in action_variants.items():
        prediction = _predict_dynamics(
            model,
            history,
            history_scale,
            future_scale,
            actions,
            condition,
        )
        errors = _component_errors(
            model,
            prediction,
            target_slots,
            target_features,
            target_centers,
        )
        relative = {
            name: errors[name] / denominators[name]
            for name in errors
        }
        total = torch.stack(tuple(relative.values()), dim=0).mean(dim=0)
        results[variant] = {
            "per_sample": total,
            "total": total.mean(),
            **{name: value.mean() for name, value in relative.items()},
        }
    return results
