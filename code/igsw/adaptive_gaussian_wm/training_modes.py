"""Explicit parameter and objective boundaries for staged validation."""
from __future__ import annotations

from .loss_weights import AdaptiveGaussianLossWeights


def configure_posterior_dynamics_gate(
    model,
    update_scope: str = "full",
) -> None:
    """Freeze the learned observation space and exclude the deploy-time Prior."""
    if update_scope not in ("full", "action_projection"):
        raise ValueError(f"unknown posterior update scope: {update_scope}")
    if update_scope == "action_projection":
        if model.config.action_residual_dim != 0:
            raise ValueError("action projection tuning requires canonical-only actions")
        if not model.config.bounded_residual_action:
            raise ValueError("action projection tuning requires bounded actions")
        model.requires_grad_(False)
        projection = model.dynamics.action_input
        if projection.bias is not None or projection.in_features != 6:
            raise ValueError("canonical action projection must be an unbiased 6D map")
        projection.weight.requires_grad_(True)
        return
    frozen_modules = (
        model.allocator,
        model.object_aggregator,
        model.target_allocator,
        model.target_object_aggregator,
        model.latent_actions.prior,
    )
    for module in frozen_modules:
        module.requires_grad_(False)
    for parameter in model.latent_actions.prior_condition_parameters():
        parameter.requires_grad_(False)
    if model.object_aggregator.rgb_head is None:
        raise ValueError("posterior Dynamics RGB gate requires an RGB head")
    model.object_aggregator.rgb_head.requires_grad_(True)


def configure_posterior_core_training(model) -> None:
    """Train representation, Posterior, Dynamics, and readout without a blind Prior."""
    model.latent_actions.prior.requires_grad_(False)
    for parameter in model.latent_actions.prior_condition_parameters():
        parameter.requires_grad_(False)


def update_target_for_training(
    model,
    posterior_dynamics_gate: bool,
    freeze_target: bool = False,
) -> None:
    """Keep the EMA teacher fixed while validating a frozen observation space."""
    if not posterior_dynamics_gate and not freeze_target:
        model.update_target()


def staged_loss_weights(
    posterior_dynamics_gate: bool,
    posterior_core_training: bool = False,
) -> AdaptiveGaussianLossWeights:
    if posterior_dynamics_gate and posterior_core_training:
        raise ValueError("posterior training modes are mutually exclusive")
    if posterior_dynamics_gate:
        return AdaptiveGaussianLossWeights(
            future=1.0,
            history=0.5,
            flow=0.0,
            feature=0.5,
            allocator=0.0,
            slot=0.0,
            action=0.5,
            action_specificity=1.0,
            rgb=1.0,
        )
    if posterior_core_training:
        return AdaptiveGaussianLossWeights(
            future=1.0,
            history=0.5,
            flow=0.0,
            feature=0.5,
            allocator=0.2,
            slot=0.2,
            action=0.5,
            action_specificity=1.0,
            rgb=1.0,
        )
    return AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.2,
        feature=0.5,
        allocator=0.2,
        slot=0.2,
        action=0.5,
        action_specificity=0.0,
        rgb=1.0,
    )
