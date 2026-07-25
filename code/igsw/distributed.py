"""Torchrun environment validation and deterministic rank sharding."""

from __future__ import annotations

import hashlib
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence, TypeVar


_T = TypeVar("_T")
_TORCHRUN_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@dataclass(frozen=True)
class TorchrunContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_rank: int
    device: str
    hostname: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def read_torchrun_context() -> TorchrunContext:
    present = [name for name in _TORCHRUN_ENV if name in os.environ]
    if not present:
        return TorchrunContext(False, 0, 1, 0, 1, 0, "cuda", socket.gethostname())

    missing = [name for name in _TORCHRUN_ENV if name not in os.environ]
    if missing:
        raise RuntimeError(
            "partial torchrun environment; missing "
            + ", ".join(missing)
            + ". Launch with torchrun instead of python."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid global rank topology: rank={rank}, world_size={world_size}")
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"invalid local rank topology: local_rank={local_rank}, "
            f"local_world_size={local_world_size}"
        )
    if world_size % local_world_size:
        raise ValueError(
            f"world_size={world_size} is not divisible by local_world_size={local_world_size}"
        )

    node_rank = int(os.environ.get("GROUP_RANK", str(rank // local_world_size)))
    node_count = world_size // local_world_size
    if not 0 <= node_rank < node_count:
        raise ValueError(f"invalid node rank topology: node_rank={node_rank}, node_count={node_count}")

    return TorchrunContext(
        True,
        rank,
        world_size,
        local_rank,
        local_world_size,
        node_rank,
        f"cuda:{local_rank}",
        socket.gethostname(),
    )


def init_torchrun(backend: str = "nccl") -> TorchrunContext:
    context = read_torchrun_context()
    if not context.distributed:
        return context

    import torch
    import torch.distributed as dist

    timeout_minutes = int(os.environ.get("TORCH_DIST_TIMEOUT_MINUTES", "30"))
    if timeout_minutes < 1:
        raise ValueError(f"invalid TORCH_DIST_TIMEOUT_MINUTES={timeout_minutes}")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL torchrun requires CUDA")
        device_count = torch.cuda.device_count()
        if context.local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={context.local_rank} but only {device_count} CUDA devices are visible"
            )
        torch.cuda.set_device(context.local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),
        device_id=(
            torch.device(context.device)
            if backend == "nccl"
            else None
        ),
    )
    if dist.get_rank() != context.rank or dist.get_world_size() != context.world_size:
        raise RuntimeError("torch.distributed topology differs from the torchrun environment")

    print(
        "[distributed] "
        f"host={context.hostname} node_rank={context.node_rank} "
        f"rank={context.rank}/{context.world_size} "
        f"local_rank={context.local_rank}/{context.local_world_size}",
        flush=True,
    )
    return context


def assert_same_paths(paths: Sequence[str], context: TorchrunContext, label: str) -> None:
    if not context.distributed:
        return

    import torch.distributed as dist

    digest = hashlib.sha256("\0".join(paths).encode("utf-8")).hexdigest()
    signature = (len(paths), digest)
    signatures: list[tuple[int, str] | None] = [None] * context.world_size
    dist.all_gather_object(signatures, signature)
    if any(item != signature for item in signatures):
        raise RuntimeError(f"{label} differs across ranks: {signatures}")


def shard_for_rank(
    items: Sequence[_T],
    rank: int,
    world_size: int,
    label: str,
) -> list[_T]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid shard topology: rank={rank}, world_size={world_size}")
    shard = list(items[rank::world_size])
    if not shard:
        raise RuntimeError(
            f"{label} has {len(items)} items, leaving rank {rank}/{world_size} with an empty shard"
        )
    return shard
