"""Causal-boundary checks for the v39 dual-horizon model."""

from __future__ import annotations

import torch

from .dual_horizon_runtime import prepare_transition_effects
from .scale import signed_gap_scale


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _paths(model, batch: dict) -> tuple[dict, dict, dict, torch.Tensor, torch.Tensor]:
    history = model.encode_history(batch)
    target_history, target = model.encode_targets(batch)
    history_scale = signed_gap_scale(
        batch["history_times"], model.config.gap_reference
    )
    future_scale = signed_gap_scale(
        batch["future_times"], model.config.gap_reference
    )
    effects = prepare_transition_effects(
        model,
        batch,
        history,
        target,
        history_scale,
        future_scale,
        None,
        True,
        None,
        False,
    )
    return history, target_history, effects, history_scale, future_scale


def _require_zero(value: float, message: str) -> None:
    if value != 0.0:
        raise RuntimeError(message)


@torch.no_grad()
def verify_v39_causal_paths(model, batch: dict) -> dict[str, float]:
    history, target_history, effects, _, _ = _paths(model, batch)

    future_changed = dict(batch)
    future_changed["future_features"] = batch["future_features"].roll(1, dims=-1)
    changed_history, changed_target_history, changed_effects, _, _ = _paths(
        model, future_changed
    )
    history_difference = _difference(history["slots"], changed_history["slots"])
    target_history_difference = _difference(
        target_history["slots"], changed_target_history["slots"]
    )
    prior_difference = _difference(
        effects.prior_context, changed_effects.prior_context
    )
    posterior_difference = _difference(effects.posterior, changed_effects.posterior)
    _require_zero(history_difference, "future content reached history encoder")
    _require_zero(
        target_history_difference, "future content reached target history encoder"
    )
    _require_zero(prior_difference, "future content reached prior context")
    if posterior_difference <= 1e-6:
        raise RuntimeError("Posterior ignored changed future content")
    if effects.posterior.shape[-3:] != (2, 4, 32):
        raise RuntimeError("effect shape differs from [B,2,4,32]")

    time_changed = dict(batch)
    time_changed["future_observation_times"] = (
        batch["future_observation_times"] + 123.0
    )
    time_history, time_target_history, time_effects, _, _ = _paths(
        model, time_changed
    )
    time_difference = max(
        _difference(history["slots"], time_history["slots"]),
        _difference(target_history["slots"], time_target_history["slots"]),
        _difference(effects.prior_context, time_effects.prior_context),
        _difference(effects.posterior, time_effects.posterior),
    )
    _require_zero(time_difference, "true terminal duration reached a model path")

    sidecar_changed = dict(batch)
    for name, value in batch.items():
        if name.startswith("teacher_") and torch.is_tensor(value):
            sidecar_changed[name] = torch.zeros_like(value)
    sidecar_history, _, sidecar_effects, _, _ = _paths(model, sidecar_changed)
    sidecar_difference = max(
        _difference(history["slots"], sidecar_history["slots"]),
        _difference(effects.prior_context, sidecar_effects.prior_context),
        _difference(effects.posterior, sidecar_effects.posterior),
    )
    _require_zero(sidecar_difference, "teacher sidecar reached model inputs")

    zero = torch.zeros_like(effects.selected[:, 0])
    zero_composed = model.effect_composer(zero, zero)
    zero_difference = float(zero_composed.abs().max())
    _require_zero(zero_difference, "zero effects did not compose to zero")
    return {
        "history_future_swap_max_difference": history_difference,
        "target_history_future_swap_max_difference": target_history_difference,
        "prior_future_swap_max_difference": prior_difference,
        "posterior_future_swap_max_difference": posterior_difference,
        "terminal_duration_model_path_max_difference": time_difference,
        "sidecar_model_path_max_difference": sidecar_difference,
        "zero_composed_effect_max": zero_difference,
    }
