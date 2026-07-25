"""Deployable history-only baselines for visual-sequence evaluation."""
from __future__ import annotations

import torch


def _linear_extrapolation(
    history: torch.Tensor,
    history_times: torch.Tensor,
    future_times: torch.Tensor,
) -> torch.Tensor:
    """Fit a slope around the current value and query it at future times."""
    if history.ndim < 3:
        raise ValueError("history values must have shape [B,T,...]")
    if history_times.shape != history.shape[:2]:
        raise ValueError("history times must match the history batch and time axes")
    if future_times.ndim != 2 or future_times.shape[0] != history.shape[0]:
        raise ValueError("future times must have shape [B,Q]")
    extra_dims = history.ndim - 2
    history_time_shape = (
        *history_times.shape,
        *(1 for _ in range(extra_dims)),
    )
    future_time_shape = (
        *future_times.shape,
        *(1 for _ in range(extra_dims)),
    )
    history_time = history_times.float().reshape(history_time_shape)
    current = history[:, -1].float()
    displacement = history.float() - current[:, None]
    denominator = history_times.float().square().sum(dim=1).reshape(
        history.shape[0],
        *(1 for _ in range(extra_dims)),
    ).clamp_min(1e-6)
    slope = (history_time * displacement).sum(dim=1) / denominator
    return (
        current[:, None]
        + future_times.float().reshape(future_time_shape) * slope[:, None]
    )


def linear_history_prediction(
    batch: dict[str, torch.Tensor],
    history_slots: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extrapolate dense features, object latents, and RGB from observed history."""
    feature = _linear_extrapolation(
        batch["history_features"],
        batch["history_times"],
        batch["future_times"],
    )
    latent = _linear_extrapolation(
        history_slots,
        batch["history_times"],
        batch["future_times"],
    )
    rgb = _linear_extrapolation(
        batch["history_rgb"].float() / 255.0,
        batch["history_times"],
        batch["future_times"],
    ).clamp(0.0, 1.0)
    return {
        "feature": feature,
        "latent": latent,
        "rgb": rgb,
    }
