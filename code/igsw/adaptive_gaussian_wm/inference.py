"""Prior-sampled inference paths kept separate from the training forward."""
from __future__ import annotations

import torch

from .dynamics_runtime import run_object_dynamics
from .dual_horizon_runtime import (
    compose_dynamics_effects,
    transition_effect_scales,
)
from .feature_readout_runtime import decode_inference_features
from .scale import signed_gap_scale


@torch.no_grad()
def predict_prior_states(
    model,
    batch: dict[str, torch.Tensor],
    sample_count: int,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = model.encode_history(batch)
    condition = model.encode_condition(batch)
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    effect_scale = transition_effect_scales(model, batch)
    context = model.prior_context(
        history,
        effect_scale,
        history_scale,
        condition,
        batch.get("condition_tokens"),
        batch.get("condition_token_valid"),
    )
    action_samples = model.latent_actions.prior.sample(
        context,
        sample_count=sample_count,
        stochastic=stochastic,
    )
    dynamics_condition = (
        None if model.config.token_conditioned_prior else condition
    )
    predictions = []
    centers = []
    empty_mask = torch.zeros(
        history["slots"].shape[:3],
        device=history["slots"].device,
        dtype=torch.bool,
    )
    for actions in action_samples:
        dynamics_actions = compose_dynamics_effects(model, actions)
        prediction = run_object_dynamics(
            model,
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            dynamics_actions,
            empty_mask,
            history["center"],
            dynamics_condition,
            history_relative_scale=history.get("relative_scale"),
            history_relative_disparity=history.get("relative_disparity"),
            history_relations=history.get("relations"),
            history_existence=history.get("existence"),
        )
        predictions.append(prediction.future_slots)
        centers.append(
            prediction.future_centers
            if prediction.future_centers is not None
            else model.object_aggregator.decode_center(prediction.future_slots)
        )
    return torch.stack(predictions), torch.stack(centers)


@torch.no_grad()
def predict_prior_features(
    model,
    batch: dict[str, torch.Tensor],
    sample_count: int,
    stochastic: bool,
) -> torch.Tensor:
    history = model.encode_history(batch)
    condition = model.encode_condition(batch)
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    effect_scale = transition_effect_scales(model, batch)
    context = model.prior_context(
        history,
        effect_scale,
        history_scale,
        condition,
        batch.get("condition_tokens"),
        batch.get("condition_token_valid"),
    )
    action_samples = model.latent_actions.prior.sample(
        context,
        sample_count=sample_count,
        stochastic=stochastic,
    )
    dynamics_condition = (
        None if model.config.token_conditioned_prior else condition
    )
    current_tokens = history["token_states"][-1]
    current_slots = history["slot_states"][-1]
    empty_mask = torch.zeros(
        history["slots"].shape[:3],
        device=history["slots"].device,
        dtype=torch.bool,
    )
    features = []
    for actions in action_samples:
        dynamics_actions = compose_dynamics_effects(model, actions)
        prediction = run_object_dynamics(
            model,
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            dynamics_actions,
            empty_mask,
            history["center"],
            dynamics_condition,
            history_relative_scale=history.get("relative_scale"),
            history_relative_disparity=history.get("relative_disparity"),
            history_relations=history.get("relations"),
            history_existence=history.get("existence"),
        )
        features.append(
            decode_inference_features(
                model,
                batch,
                current_tokens,
                current_slots,
                prediction,
            )
        )
    return torch.stack(features)
