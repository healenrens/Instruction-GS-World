"""Distributed execution helpers for the v62 held-data audits."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedAuditContextV62:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def initialize_distributed_audit_v62():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return DistributedAuditContextV62(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )


def shard_indices_v62(indices, context):
    return indices[context.rank :: context.world_size]


def gather_rank_payloads_v62(payload, context):
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, payload)
    return gathered


def finish_distributed_audit_v62():
    dist.barrier()
    dist.destroy_process_group()
