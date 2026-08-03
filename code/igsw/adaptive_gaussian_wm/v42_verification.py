"""Server-only stress checks for stable correspondence and presence objectives."""

from __future__ import annotations

import math

import torch

from .object_correspondence import _augmented_optimal_transport


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mass_errors(
    transport: torch.Tensor,
    unmatched: torch.Tensor,
    discovery: torch.Tensor,
) -> tuple[float, float]:
    row = (transport.sum(dim=-1) + unmatched - 1.0).abs().amax()
    column = (transport.sum(dim=-2) + discovery - 1.0).abs().amax()
    return float(row), float(column)


@torch.no_grad()
def verify_v42_stability_contracts(model) -> dict[str, float]:
    config = model.config
    _require(config.architecture == "object_memory_v3", "model is not v42")
    _require(
        config.correspondence_sinkhorn_iterations >= 256,
        "v42 requires at least 256 Sinkhorn iterations",
    )
    _require(
        0.0 < config.correspondence_logit_clip <= 4.0,
        "v42 correspondence logits are not safely bounded",
    )
    module = model.object_memory.correspondence
    _require(module is not None, "v42 correspondence module is missing")
    device = next(module.parameters()).device
    slots = config.object_slots
    count = slots * slots
    ascending = torch.linspace(
        -config.correspondence_logit_clip,
        config.correspondence_logit_clip,
        count,
        device=device,
        dtype=torch.float32,
    ).reshape(1, slots, slots)
    checkerboard = torch.where(
        (torch.arange(count, device=device).reshape(slots, slots) % 2) == 0,
        torch.full((), config.correspondence_logit_clip, device=device),
        torch.full((), -config.correspondence_logit_clip, device=device),
    )[None]
    stress_scores = torch.cat((ascending, checkerboard), dim=0)
    dustbin = torch.zeros(
        stress_scores.shape[0], 1, 1, device=device, dtype=torch.float32
    )
    transport, unmatched, discovery = _augmented_optimal_transport(
        stress_scores,
        dustbin,
        config.correspondence_sinkhorn_iterations,
    )
    row_error, column_error = _mass_errors(transport, unmatched, discovery)
    tolerance = float(config.correspondence_mass_tolerance)
    _require(row_error < tolerance, "stress Sinkhorn row mass is not conserved")
    _require(
        column_error < tolerance,
        "stress Sinkhorn column mass is not conserved",
    )
    return {
        "correspondence_stress_row_mass_max_difference": row_error,
        "correspondence_stress_column_mass_max_difference": column_error,
        "correspondence_logit_clip": float(config.correspondence_logit_clip),
        "correspondence_mass_tolerance": tolerance,
    }


def verify_v42_objective_parts(parts: dict[str, float]) -> dict[str, float]:
    required = (
        "geometry_track_presence",
        "lifecycle_history_presence_anchor",
        "lifecycle_presence_mass_calibration",
    )
    missing = [name for name in required if name not in parts]
    _require(not missing, f"v42 lifecycle objectives are missing: {missing}")
    nonfinite = [name for name in required if not math.isfinite(float(parts[name]))]
    _require(not nonfinite, f"v42 lifecycle objectives are non-finite: {nonfinite}")
    return {f"{name}_verified": float(parts[name]) for name in required}
