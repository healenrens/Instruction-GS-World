"""Multi-scale proposal coverage and nonduplicating points, not instance labels."""

import cv2
import numpy as np


def select_scale_coverage(regions, limit, shape):
    if not regions:
        return []
    ordered = sorted(range(len(regions)), key=lambda i: regions[i]["area"])
    bands = [list(part) for part in np.array_split(ordered, 3)]
    covered = np.zeros(shape, bool)
    selected = []
    while len(selected) < limit and any(bands):
        for band, candidates in enumerate(bands):
            if not candidates or len(selected) == limit:
                continue

            def utility(index):
                region = regions[index]
                left, top, right, bottom = region["box"]
                new = region["mask"] & ~covered[top:bottom, left:right]
                return float(new.sum() / region["area"]), region["sam_score"]

            index = max(candidates, key=utility)
            fraction, _ = utility(index)
            item = {
                **regions[index],
                "scale_band": band,
                "novel_support_fraction": fraction,
            }
            selected.append(item)
            left, top, right, bottom = item["box"]
            covered[top:bottom, left:right] |= item["mask"]
            candidates.remove(index)
    return selected


def spread_inside_points(mask, count, occupied):
    if count == 0:
        return np.empty((0, 2), np.float32)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    interior = mask & (distance >= min(2.0, float(distance.max()) * 0.5)) & ~occupied
    yy, xx = np.where(interior)
    if not len(xx):
        return np.empty((0, 2), np.float32)
    xy = np.stack([xx, yy], -1).astype(np.float32)
    if occupied.any():
        nearest_image = cv2.distanceTransform(
            (~occupied).astype(np.uint8), cv2.DIST_L2, 5
        )
        nearest = nearest_image[yy, xx] ** 2
        next_index = int(nearest.argmax())
    else:
        nearest = np.full(len(xy), np.inf, np.float32)
        next_index = int(distance[yy, xx].argmax())
    selected = []
    for _ in range(min(count, len(xy))):
        selected.append(next_index)
        nearest = np.minimum(nearest, ((xy - xy[next_index]) ** 2).sum(-1))
        nearest[selected] = -1
        next_index = int(nearest.argmax())
    occupied[yy[selected], xx[selected]] = True
    return xy[selected]
