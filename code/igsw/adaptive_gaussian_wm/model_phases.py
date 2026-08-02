"""Training-phase decisions kept outside the model data path."""
from __future__ import annotations

import torch


def joint_phase_flags(phase: str, architecture: str) -> tuple[bool, bool, bool]:
    allowed = {
        "joint",
        "joint_loss",
        "posterior_dynamics_loss",
        "object_memory_representation_loss",
        "history_prior_loss",
    }
    if phase not in allowed:
        raise ValueError(f"unknown training phase: {phase}")
    representation = phase == "object_memory_representation_loss"
    if representation and architecture not in (
        "object_memory_v1",
        "object_memory_v2",
        "object_memory_v3",
    ):
        raise ValueError("object-memory representation phase requires staged architecture")
    return phase != "joint", phase == "posterior_dynamics_loss", representation


def select_dynamics_actions(
    model,
    posterior: torch.Tensor,
    prior_context: torch.Tensor,
    use_posterior: bool,
    actions_override: torch.Tensor | None,
    action_free: bool,
) -> torch.Tensor:
    if action_free:
        return torch.zeros_like(posterior)
    if actions_override is not None:
        if actions_override.shape != posterior.shape:
            raise ValueError("actions_override must match posterior actions")
        return actions_override
    if use_posterior:
        return posterior
    return model.latent_actions.prior.sample(
        prior_context,
        sample_count=1,
        stochastic=True,
    )[0]
