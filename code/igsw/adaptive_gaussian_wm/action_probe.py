"""Deterministic ridge probes for latent-action information content."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RidgeProbe:
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    weight: torch.Tensor
    penalty: float

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.x_mean.shape[1]:
            raise ValueError("ridge probe input width differs from training")
        normalized = (inputs.float() - self.x_mean) / self.x_scale
        return normalized @ self.weight + self.y_mean


def fit_ridge(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    penalty: float,
) -> RidgeProbe:
    if inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets):
        raise ValueError("ridge inputs and targets must be aligned matrices")
    if len(inputs) <= inputs.shape[1]:
        raise ValueError("ridge probe requires more rows than input dimensions")
    if penalty <= 0.0:
        raise ValueError("ridge penalty must be positive")
    inputs = inputs.float()
    targets = targets.float()
    x_mean = inputs.mean(dim=0, keepdim=True)
    x_scale = inputs.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    y_mean = targets.mean(dim=0, keepdim=True)
    normalized = (inputs - x_mean) / x_scale
    centered_target = targets - y_mean
    gram = normalized.transpose(0, 1) @ normalized
    regularizer = penalty * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    weight = torch.linalg.solve(
        gram + regularizer,
        normalized.transpose(0, 1) @ centered_target,
    )
    return RidgeProbe(x_mean, x_scale, y_mean, weight, penalty)


def state_groups(feature_dim: int, object_count: int) -> dict[str, torch.Tensor]:
    if feature_dim <= 0 or object_count <= 0:
        raise ValueError("state group dimensions must be positive")
    width = feature_dim + 3
    index = torch.arange(object_count * width).reshape(object_count, width)
    return {
        "all": index.flatten(),
        "feature": index[:, :feature_dim].flatten(),
        "center": index[:, feature_dim : feature_dim + 2].flatten(),
        "activity": index[:, feature_dim + 2 :].flatten(),
    }


def per_row_group_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("probe predictions and targets must align")
    error = (prediction.float() - target.float()).square()
    return {
        name: error[:, indices.to(error.device)].mean(dim=1)
        for name, indices in groups.items()
    }


def explained_fraction(error: torch.Tensor, baseline: torch.Tensor) -> float:
    if error.shape != baseline.shape or error.ndim != 1:
        raise ValueError("explained fraction requires aligned error vectors")
    return float(1.0 - error.mean() / baseline.mean().clamp_min(1e-8))


def gain_fraction(
    action_error: torch.Tensor,
    joint_error: torch.Tensor,
    copy_error: torch.Tensor,
) -> float:
    if not (
        action_error.shape == joint_error.shape == copy_error.shape
        and action_error.ndim == 1
    ):
        raise ValueError("gain fraction requires aligned error vectors")
    action_gain = copy_error.mean() - action_error.mean()
    joint_gain = copy_error.mean() - joint_error.mean()
    return float(action_gain / joint_gain.clamp_min(1e-8))
