"""Deterministic task-balanced sampling for distributed episode training."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


def _stable_integer(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _coprime_step(length: int, token: int) -> int:
    if length == 1:
        return 0
    candidate = token % length or 1
    while math.gcd(candidate, length) != 1:
        candidate = (candidate + 1) % length or 1
    return candidate


def sqrt_coverage_targets(lengths: tuple[int, ...]) -> tuple[int, ...]:
    if not lengths or min(lengths) <= 0:
        raise ValueError("sampling group lengths must be positive")
    maximum = max(lengths)
    targets = []
    for length in lengths:
        product = length * maximum
        root = math.isqrt(product)
        targets.append(root if root * root == product else root + 1)
    return tuple(targets)


class DistributedGroupBalancedSampler(Sampler[int]):
    """Cover every group, then apply deterministic square-root rebalancing."""

    def __init__(
        self,
        dataset,
        group_spans: tuple[tuple[int, int], ...],
        group_targets: tuple[int, ...],
        num_replicas: int,
        rank: int,
        seed: int,
        samples_per_rank: int,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank is outside the distributed world")
        spans = tuple((int(start), int(end)) for start, end in group_spans)
        if not spans:
            raise ValueError("group-balanced sampling requires at least one group")
        cursor = 0
        for start, end in spans:
            if start != cursor or end <= start or end > len(dataset):
                raise ValueError("sampling groups must partition the dataset")
            cursor = end
        if cursor != len(dataset):
            raise ValueError("sampling groups do not cover the dataset")

        targets = tuple(int(value) for value in group_targets)
        if len(targets) != len(spans) or any(
            target < end - start
            for target, (start, end) in zip(targets, spans)
        ):
            raise ValueError("group targets must cover every source index")
        if samples_per_rank <= 0:
            raise ValueError("samples_per_rank must be positive")
        total_size = samples_per_rank * num_replicas
        if total_size < sum(targets):
            raise ValueError("distributed epoch is smaller than coverage targets")
        targets = list(targets)
        largest = max(range(len(targets)), key=targets.__getitem__)
        targets[largest] += total_size - sum(targets)

        self.dataset = dataset
        self.group_spans = spans
        self.group_targets = tuple(targets)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self.num_samples = int(samples_per_rank)
        self.total_size = total_size

    def __iter__(self) -> Iterator[int]:
        if self.num_samples == 0:
            return iter(())
        local_positions = torch.arange(
            self.start_index,
            self.num_samples,
            dtype=torch.long,
        )
        global_positions = local_positions * self.num_replicas + self.rank
        order_offset = _stable_integer(
            "global-offset", self.seed, self.epoch
        ) % self.total_size
        order_step = _coprime_step(
            self.total_size,
            _stable_integer("global-step", self.seed, self.epoch),
        )
        virtual = (
            order_offset + order_step * global_positions
        ).remainder(self.total_size)

        target_ends = torch.tensor(self.group_targets).cumsum(0)
        target_starts = torch.cat(
            (torch.zeros(1, dtype=torch.long), target_ends[:-1])
        )
        group_ids = torch.bucketize(virtual, target_ends, right=True)
        ordinals = virtual - target_starts[group_ids]
        source_starts = torch.tensor(
            [start for start, _ in self.group_spans],
            dtype=torch.long,
        )
        source_lengths = torch.tensor(
            [end - start for start, end in self.group_spans],
            dtype=torch.long,
        )
        source_offsets = torch.tensor(
            [
                _stable_integer("source-offset", self.seed, self.epoch, group)
                % length
                for group, length in enumerate(source_lengths.tolist())
            ],
            dtype=torch.long,
        )
        source_steps = torch.tensor(
            [
                _coprime_step(
                    length,
                    _stable_integer("source-step", self.seed, self.epoch, group),
                )
                for group, length in enumerate(source_lengths.tolist())
            ],
            dtype=torch.long,
        )
        indices = source_starts[group_ids] + (
            source_offsets[group_ids]
            + source_steps[group_ids] * ordinals
        ).remainder(source_lengths[group_ids])
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError("sampler start index is outside the local epoch")
        self.start_index = int(start_index)


def build_training_sampler(
    dataset,
    num_replicas: int,
    rank: int,
    seed: int,
    batch_size: int,
    grad_accum: int,
):
    if getattr(dataset, "balance_sampling", False):
        global_batch = batch_size * grad_accum * num_replicas
        minimum = sum(dataset.sampling_group_targets)
        optimizer_steps = math.ceil(minimum / global_batch)
        return DistributedGroupBalancedSampler(
            dataset,
            dataset.sampling_group_spans,
            dataset.sampling_group_targets,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            samples_per_rank=optimizer_steps * batch_size * grad_accum,
        )
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
