"""Deterministic episode-uniform sampling, with separate window choices per visit."""

from collections import defaultdict
import math

import torch
from torch.utils.data import Sampler


def episode_key(entry):
    return (entry["source"], entry["group"], int(entry["episode_index"]))


class EpisodeUniformSamplerV69(Sampler):
    def __init__(self, dataset, rank=0, world=1, batch=1, seed=17, start=0):
        groups = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            groups[episode_key(entry)].append(index)
        self.groups = list(groups.values())
        self.rank, self.world, self.batch, self.seed = rank, world, batch, seed
        self.epoch, self.start = 0, start

    def __len__(self):
        global_batch = self.world * self.batch
        return math.ceil(len(self.groups) / global_batch) * self.batch - self.start

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=rng).tolist()
        total = math.ceil(len(order) / (self.world * self.batch)) * self.world * self.batch
        padded = (order * math.ceil(total / len(order)))[:total]
        choices = []
        for occurrence, group in enumerate(padded):
            windows = self.groups[group]
            index = windows[int(torch.randint(len(windows), (), generator=rng))]
            choices.append((index, self.epoch, occurrence))
        return iter(choices[self.rank::self.world][self.start:])
