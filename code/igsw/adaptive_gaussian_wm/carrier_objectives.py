"""Current-only support distillation for the hierarchical Gaussian carrier."""
from __future__ import annotations

import torch

from .dense_readout_objective import dense_readout_objective
from .gaussian_math import mahalanobis_squared_from_precision, precision_2d
from .readout_repair import current_readout_objective, first_query


def carrier_responsibility(
    state,
    coordinates: torch.Tensor,
    parent_count: int,
    children: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = first_query(state)
    if current.center.shape[2] != parent_count * children:
        raise ValueError("hierarchical state does not match parent/child contract")
    difference = coordinates[:, :, None].float() - current.center[..., None, :].float()
    distance = mahalanobis_squared_from_precision(
        precision_2d(current.covariance.float()), difference
    )
    weight = torch.exp(-0.5 * distance)
    weight = weight * current.opacity.squeeze(-1)[..., None].float()
    weight = weight * current.activation.squeeze(-1)[..., None].float()
    order = torch.softmax(current.depth_order.squeeze(-1).float(), dim=2)
    weight = weight * order[..., None] * current.center.shape[2]
    grouped = weight.unflatten(2, (parent_count, children)).sum(dim=3)
    coverage = grouped.sum(dim=2)
    responsibility = grouped / coverage[:, :, None].clamp_min(1e-6)
    return responsibility, coverage


def support_distillation_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tokens = output["history_token_states"][-1]
    children = model.config.gaussian_children
    responsibility, coverage = carrier_responsibility(
        output["current_gaussian_readout"],
        batch["history_coordinates"][:, -1:],
        tokens.latent.shape[1],
        children,
    )
    target = (
        tokens.assignment.detach().float()
        * tokens.activation.detach().float()
    )[:, None]
    target_coverage = target.sum(dim=2).clamp(0.0, 1.0)
    target = target / target_coverage[:, :, None].clamp_min(1e-6)
    prediction = responsibility.clamp_min(1e-7)
    target = target.clamp_min(1e-7)
    midpoint = 0.5 * (prediction + target)
    divergence = 0.5 * (
        target * (target.log() - midpoint.log())
        + prediction * (prediction.log() - midpoint.log())
    ).sum(dim=2)
    valid = batch["history_valid"][:, -1:].float()
    foreground = valid * target_coverage
    support_js = (divergence * foreground).sum() / foreground.sum().clamp_min(1.0)
    coverage_loss = (
        (coverage - target_coverage).square() * valid
    ).sum() / valid.sum().clamp_min(1.0)
    support = support_js + coverage_loss

    current = first_query(output["current_gaussian_readout"])
    child_activation = current.activation.unflatten(
        2, (tokens.latent.shape[1], children)
    )
    parent_activation = tokens.activation[:, None]
    child_total = child_activation.sum(dim=3)
    compact = (
        (child_total - parent_activation).square()
        / parent_activation.clamp_min(1e-4)
    ).mean()
    coverage_fraction = (
        ((coverage > 1e-4) & batch["history_valid"][:, -1:]).float().sum()
        / valid.sum().clamp_min(1.0)
    )
    return support, compact, {
        "carrier_support_js": support_js.detach(),
        "carrier_support_coverage": coverage_loss.detach(),
        "carrier_child_activation": child_activation.mean().detach(),
        "carrier_child_activation_sum": child_total.mean().detach(),
        "carrier_coverage_fraction": coverage_fraction.detach(),
        "carrier_children": support.new_tensor(float(children)),
    }


def carrier_loss_bundle(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    reference: torch.Tensor,
    weights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    current = reference * 0.0
    regularization = reference * 0.0
    support = reference * 0.0
    compact = reference * 0.0
    parts = {}
    if weights.current_readout > 0.0:
        if model.config.dense_object_readout:
            current, regularization, parts = dense_readout_objective(
                batch, output
            )
        else:
            current, regularization, parts = current_readout_objective(
                model, batch, output
            )
    if weights.carrier_support > 0.0 or weights.carrier_compact > 0.0:
        if not model.config.hierarchical_gaussian_carrier:
            raise ValueError("carrier losses require hierarchical configuration")
        support, compact, carrier_parts = support_distillation_loss(
            model, batch, output
        )
        parts.update(carrier_parts)
    return current, regularization, support, compact, parts
