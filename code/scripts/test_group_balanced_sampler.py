"""CPU contract test for deterministic distributed task balancing."""
from __future__ import annotations

from collections import Counter
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    DistributedGroupBalancedSampler,
    sqrt_coverage_targets,
)


class ToyDataset:
    sampling_group_spans = ((0, 5), (5, 17), (17, 53), (53, 97))

    def __len__(self) -> int:
        return 97


def group_for_index(index: int) -> int:
    for group, (start, end) in enumerate(ToyDataset.sampling_group_spans):
        if start <= index < end:
            return group
    raise AssertionError(f"sample index is outside groups: {index}")


def epoch_indices(epoch: int) -> list[list[int]]:
    lengths = tuple(
        end - start for start, end in ToyDataset.sampling_group_spans
    )
    targets = sqrt_coverage_targets(lengths)
    samples_per_rank = (
        sum(targets) + 15
    ) // 16
    outputs = []
    for rank in range(16):
        sampler = DistributedGroupBalancedSampler(
            ToyDataset(),
            ToyDataset.sampling_group_spans,
            targets,
            num_replicas=16,
            rank=rank,
            seed=17,
            samples_per_rank=samples_per_rank,
        )
        sampler.set_epoch(epoch)
        values = list(sampler)
        if len(values) != len(sampler):
            raise AssertionError("sampler length differs from emitted samples")
        outputs.append(values)
    return outputs


def main() -> None:
    first = epoch_indices(3)
    repeated = epoch_indices(3)
    changed = epoch_indices(4)
    if first != repeated:
        raise AssertionError("sampler is not deterministic within an epoch")
    if first == changed:
        raise AssertionError("sampler order did not change across epochs")

    counts = Counter(
        group_for_index(index)
        for rank_values in first
        for index in rank_values
    )
    covered = {
        group: {
            index
            for rank_values in first
            for index in rank_values
            if group_for_index(index) == group
        }
        for group in range(4)
    }
    for group, (start, end) in enumerate(ToyDataset.sampling_group_spans):
        if covered[group] != set(range(start, end)):
            raise AssertionError(f"group {group} was not fully covered")
    synchronized_diversity = [
        len(
            {
                group_for_index(first[rank][position])
                for rank in range(len(first))
            }
        )
        for position in range(len(first[0]))
    ]
    if min(synchronized_diversity) < 2:
        raise AssertionError("synchronized ranks collapse to one sampling group")

    print(
        json.dumps(
            {
                "status": "ok",
                "world_size": 16,
                "samples_per_rank": len(first[0]),
                "global_group_counts": dict(sorted(counts.items())),
                "full_source_coverage": True,
                "min_synchronized_group_diversity": min(
                    synchronized_diversity
                ),
                "deterministic": True,
                "epoch_changes": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
