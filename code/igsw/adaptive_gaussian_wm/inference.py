"""Prior-sampled inference paths kept separate from the training forward."""
from __future__ import annotations

import torch

from .readout_runtime import decode_gaussian_readout
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
    context = model.prior_context(
        history,
        future_scale,
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
        prediction = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions,
            empty_mask,
            history["center"],
            dynamics_condition,
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
    context = model.prior_context(
        history,
        future_scale,
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
        prediction = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions,
            empty_mask,
            history["center"],
            dynamics_condition,
        )
        predicted_centers = (
            prediction.future_centers
            if prediction.future_centers is not None
            else model.object_aggregator.decode_center(prediction.future_slots)
        )
        readout, _ = decode_gaussian_readout(
            model,
            batch,
            current_tokens,
            current_slots,
            prediction.future_slots,
            predicted_centers,
        )
        rendered = model.gaussian_readout.splat_features(
            readout,
            batch["future_coordinates"],
        )[0]
        features.append(rendered)
    return torch.stack(features)
