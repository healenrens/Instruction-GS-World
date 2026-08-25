"""Deterministic unseen-window sampling for v59 evaluation."""

from __future__ import annotations

from .group_balanced_sampler import build_training_sampler


def training_base_indices_v59(dataset, checkpoint: dict) -> frozenset[int]:
    """Reconstruct every base index consumed by the saved DDP training run."""
    args = checkpoint["args"]
    batch = int(args["batch"])
    grad_accum = int(args["grad_accum"])
    seed = int(args["seed"])
    world_size = int(checkpoint["world_size"])
    remaining_updates = int(checkpoint["global_step"])
    excluded: set[int] = set()

    for rank in range(world_size):
        sampler = build_training_sampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            batch_size=batch,
            grad_accum=grad_accum,
        )
        updates_per_epoch = len(sampler) // batch // grad_accum
        rank_remaining = remaining_updates
        epoch = 0
        while rank_remaining:
            updates = min(rank_remaining, updates_per_epoch)
            sampler.set_epoch(seed + epoch)
            base_sampler = sampler.sampler
            count = updates * batch * grad_accum
            indices = base_sampler.sample_indices(0, count)
            excluded.update(int(value) for value in indices.tolist())
            rank_remaining -= updates
            epoch += 1
    return frozenset(excluded)


def unseen_source_evaluation_indices_v59(
    dataset,
    source_index: int,
    count: int,
    excluded: frozenset[int],
) -> tuple[int, ...]:
    """Select task-spread source indices that were absent from training."""
    candidate_count = max(count * 2, count + 32)
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        candidates = dataset.balanced_source_evaluation_indices(
            source_index, candidate_count
        )
        for index in candidates:
            if index not in excluded and index not in seen:
                selected.append(index)
                seen.add(index)
                if len(selected) == count:
                    break
        if len(selected) < count:
            candidate_count *= 2
            if candidate_count > len(dataset):
                raise ValueError(
                    f"source {source_index} cannot supply {count} unseen windows"
                )
    return tuple(selected)


def training_exclusion_contract_v59(
    checkpoint: dict, excluded: frozenset[int], selected: set[int]
) -> dict[str, int | str]:
    return {
        "scope": "train_split_sampler_unseen_windows",
        "checkpoint_step": int(checkpoint["global_step"]),
        "checkpoint_world_size": int(checkpoint["world_size"]),
        "training_base_index_count": len(excluded),
        "evaluation_base_index_count": len(selected),
        "training_evaluation_overlap_count": len(excluded.intersection(selected)),
    }
