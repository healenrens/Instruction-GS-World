"""Small differentiable collectives for batch-statistic losses."""
from __future__ import annotations

import torch
import torch.distributed as dist


def gather_batch_with_grad(value: torch.Tensor) -> torch.Tensor:
    """Gather the leading batch axis while preserving gradients on every rank."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    if value.ndim == 0:
        raise ValueError("distributed batch gather requires a batch axis")
    from torch.distributed.nn.functional import all_gather

    value = value.contiguous()
    gathered = all_gather(value)
    if len(gathered) != dist.get_world_size():
        raise RuntimeError("differentiable all_gather returned the wrong world size")
    return torch.cat(tuple(gathered), dim=0)


def gather_batch_without_grad(value: torch.Tensor) -> torch.Tensor:
    """Gather detached batch values using contiguous collective buffers."""
    value = value.detach().contiguous()
    if not dist.is_available() or not dist.is_initialized():
        return value
    if value.ndim == 0:
        raise ValueError("distributed batch gather requires a batch axis")
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


def statistical_batch_size(local_count: int, device: torch.device) -> torch.Tensor:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    return torch.tensor(float(local_count * world_size), device=device)
