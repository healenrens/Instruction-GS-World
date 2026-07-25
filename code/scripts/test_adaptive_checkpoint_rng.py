"""CPU contract test for checkpoint RNG capture and restoration."""
from __future__ import annotations

import json
import os
import random
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    collect_rng_states,
    restore_rng_state,
)


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if distributed else 0
    context = SimpleNamespace(
        device="cpu",
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )
    random.seed(17 + rank)
    torch.manual_seed(17 + rank)
    states = collect_rng_states(context)
    expected_python = random.random()
    expected_torch = torch.rand(4)

    random.seed(999)
    torch.manual_seed(999)
    restore_rng_state({"rng_states": states}, context)
    if random.random() != expected_python:
        raise AssertionError("Python RNG state did not restore")
    if not torch.equal(torch.rand(4), expected_torch):
        raise AssertionError("Torch RNG state did not restore")
    if len(states) != world_size:
        raise AssertionError("RNG state gather did not include every rank")
    if distributed:
        dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "world_size": len(states),
                    "python_rng": True,
                    "torch_rng": True,
                },
                sort_keys=True,
            )
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
