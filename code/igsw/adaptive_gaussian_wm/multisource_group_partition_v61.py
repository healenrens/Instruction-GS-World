"""Deterministic source-local held-group partition for v61 experiments."""

from __future__ import annotations


def select_group_partition_v61(episodes, partition: str, stride: int):
    if partition not in ("all", "train", "held"):
        raise ValueError(f"unsupported multisource group partition: {partition}")
    if stride < 2:
        raise ValueError("held group stride must be at least two")
    if partition == "all":
        return list(episodes)
    groups_by_source = {}
    for episode in episodes:
        groups_by_source.setdefault(episode.source_index, set()).add(episode.group)
    held_groups = {
        (source_index, group)
        for source_index, groups in groups_by_source.items()
        for position, group in enumerate(sorted(groups))
        if position % stride == 0
    }
    return [
        episode
        for episode in episodes
        if ((episode.source_index, episode.group) in held_groups)
        == (partition == "held")
    ]
