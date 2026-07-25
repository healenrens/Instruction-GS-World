"""Grouped optimal-transport coupling for conditional flow matching."""
from __future__ import annotations

import torch
import torch.nn.functional as F


_PERMUTATIONS = {
    1: ((0,),),
    2: ((0, 1), (1, 0)),
    3: (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ),
}


def grouped_optimal_transport_target(
    source: torch.Tensor,
    target: torch.Tensor,
    group_id: torch.Tensor | None,
) -> torch.Tensor:
    """Pair same-context targets with source noise at minimum assignment cost."""
    if group_id is None:
        return target
    ids, counts = torch.unique_consecutive(
        group_id.flatten(),
        return_counts=True,
    )
    if source.shape != target.shape or group_id.numel() != source.shape[0]:
        raise ValueError("grouped flow source, target, and group_id must align")
    if ids.numel() != torch.unique(ids).numel() or not bool(
        (counts == counts[0]).all()
    ):
        raise ValueError("grouped flow requires complete contiguous equal-size groups")
    group_size = int(counts[0])
    if group_size not in _PERMUTATIONS:
        raise ValueError("grouped flow supports at most three futures per context")
    groups = ids.numel()
    source_group = source.flatten(1).reshape(groups, group_size, -1)
    target_group = target.detach().flatten(1).reshape(groups, group_size, -1)
    cost = (
        source_group[:, :, None] - target_group[:, None, :]
    ).square().mean(dim=-1)
    permutation = torch.tensor(
        _PERMUTATIONS[group_size],
        device=source.device,
        dtype=torch.long,
    )
    row = torch.arange(group_size, device=source.device)
    candidate_cost = torch.stack(
        [cost[:, row, order].sum(dim=-1) for order in permutation],
        dim=-1,
    )
    best = candidate_cost.argmin(dim=-1)
    chosen = permutation[best]
    coupled = torch.gather(
        target_group,
        1,
        chosen[..., None].expand(-1, -1, target_group.shape[-1]),
    )
    return coupled.reshape_as(target)


def shared_flow_time(target: torch.Tensor) -> torch.Tensor:
    sample_time = torch.rand(
        target.shape[0],
        1,
        device=target.device,
        dtype=target.dtype,
    )
    return sample_time.expand(-1, target.shape[1])


def sample_flow_source(
    prior,
    batch: int,
    future_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if prior.correlated_source:
        return prior.source_scale * torch.randn(
            batch,
            1,
            1,
            prior.action_dim,
            device=device,
            dtype=dtype,
        ).expand(
            -1,
            future_count,
            prior.action_tokens,
            -1,
        ).clone()
    return prior.source_scale * torch.randn(
        batch,
        future_count,
        prior.action_tokens,
        prior.action_dim,
        device=device,
        dtype=dtype,
    )


def flow_training_objective(
    prior,
    target: torch.Tensor,
    context: torch.Tensor,
    group_id: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lifted_objective = getattr(prior, "training_objective", None)
    if lifted_objective is not None:
        if weight is not None:
            raise ValueError("lifted flow does not support action weights")
        return lifted_objective(target, context, group_id)
    source = sample_flow_source(
        prior,
        target.shape[0],
        target.shape[1],
        target.device,
        target.dtype,
    )
    target = grouped_optimal_transport_target(source, target, group_id).detach()
    flow_time = shared_flow_time(target)
    interpolation = flow_time[..., None, None]
    latent = (1.0 - interpolation) * source + interpolation * target
    target_velocity = target - source
    endpoint_prediction = bool(prior.endpoint_prediction)
    prediction = prior(latent, flow_time, context, endpoint_prediction)
    training_target = target if endpoint_prediction else target_velocity
    endpoint = (
        prediction
        if endpoint_prediction
        else latent + (1.0 - interpolation) * prediction
    )
    if weight is None:
        loss = F.mse_loss(prediction, training_target)
    else:
        if weight.shape != prediction.shape[:-1]:
            raise ValueError("flow weight must have shape [B,Q,A]")
        error = (prediction - training_target).square().mean(dim=-1)
        loss = (error * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss, endpoint, target
