"""Current-frame anchor and diagnostics for the dense object readout."""

from __future__ import annotations

import torch

from .change_objectives import dense_feature_loss


def _assignment_js(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.float().clamp_min(1e-7)
    target = target.float().clamp_min(1e-7)
    midpoint = 0.5 * (prediction + target)
    divergence = 0.5 * (
        prediction * (prediction.log() - midpoint.log())
        + target * (target.log() - midpoint.log())
    ).sum(dim=2)
    weight = valid.float()
    return (divergence * weight).sum() / weight.sum().clamp_min(1.0)


def dense_readout_objective(
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    state = output.get("current_dense_readout")
    if state is None:
        raise ValueError("dense readout objective requires current_dense_readout")
    tokens = output["history_token_states"][-1]
    target = batch["history_features"][:, -1:]
    valid = batch["history_valid"][:, -1:]
    coverage_weight = torch.ones_like(state.coverage)
    reconstruction = dense_feature_loss(
        state.feature,
        target,
        valid,
        coverage_weight,
    )
    token_reconstruction = dense_feature_loss(
        tokens.reconstructed_features[:, None],
        target,
        valid,
        coverage_weight,
    )
    assignment_js = _assignment_js(
        state.assignment,
        tokens.assignment[:, None],
        valid,
    )
    feature_energy = state.feature_residual.square().mean()
    assignment_energy = state.assignment_residual.square().mean()
    background_energy = state.background_residual.square().mean()
    activation_energy = (
        (state.activation.float() - tokens.activation[:, None].float()).square().mean()
    )
    regularization = (
        0.1 * assignment_js
        + 0.1 * assignment_energy
        + feature_energy
        + background_energy
        + 0.1 * activation_energy
    )
    valid_weight = valid.float()
    denominator = valid_weight.sum().clamp_min(1.0)
    coverage_mean = (state.coverage.float() * valid_weight).sum() / denominator
    coverage_fraction = ((state.coverage > 1e-4) & valid).float().sum() / denominator
    return (
        reconstruction,
        regularization,
        {
            "dense_readout_current_feature": reconstruction.detach(),
            "dense_readout_token_feature": token_reconstruction.detach(),
            "dense_readout_gain_over_token": (
                token_reconstruction.detach() - reconstruction.detach()
            ),
            "dense_readout_assignment_js": assignment_js.detach(),
            "dense_readout_assignment_residual_energy": assignment_energy.detach(),
            "dense_readout_feature_residual_energy": feature_energy.detach(),
            "dense_readout_background_residual_energy": background_energy.detach(),
            "dense_readout_activation_residual_energy": activation_energy.detach(),
            "dense_readout_coverage_mean": coverage_mean.detach(),
            "dense_readout_coverage_fraction": coverage_fraction.detach(),
        },
    )
