"""Shared full-field-of-view export and paired scene selection."""

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def native_rgb(frames, hw):
    h, w = map(int, hw)
    size = frames.shape[-1]
    rh, rw = round(h*size/max(h, w)), round(w*size/max(h, w))
    top, left = (size-rh)//2, (size-rw)//2
    cropped = frames[:, :, top:top+rh, left:left+rw]
    value = F.interpolate(cropped.float(), (h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    return value.permute(0, 2, 3, 1).cpu().numpy()


def native_field_labels(masks, hw, image_size):
    h, w = map(int, hw)
    rh, rw = round(h*image_size/max(h, w)), round(w*image_size/max(h, w))
    top, left = (image_size-rh)//2, (image_size-rw)//2
    side = round(masks.shape[-1]**.5)
    labels = []
    for frame in masks:
        raster = F.interpolate(frame[None].reshape(1, len(frame), side, side).float(),
                               (image_size, image_size), mode="bilinear", align_corners=False)
        raster = F.interpolate(raster[:, :, top:top+rh, left:left+rw], (h, w), mode="bilinear", align_corners=False)
        labels.append(raster.argmax(1)[0].cpu().numpy().astype(np.int16))
    return np.stack(labels)


def paired_indices(dataset, items):
    scenes = defaultdict(list)
    for index, entries in enumerate(dataset.episodes):
        scenes[entries[0]["scene_seed"]].append(index)
    rng = np.random.default_rng(dataset.seed)
    indices = []
    for scene in rng.permutation(sorted(scenes)):
        if len(indices)+len(scenes[scene]) > items:
            break
        indices.extend(scenes[scene])
    return indices
