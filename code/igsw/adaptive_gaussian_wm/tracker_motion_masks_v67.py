"""Motion-region proposals for visual review, not instance/object ground truth."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


def residual_motion(gray, other, seed):
    flow = cv2.calcOpticalFlowFarneback(gray, other, None, 0.5, 4, 15, 3, 5, 1.1, 0)
    height, width = gray.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    coordinates = np.stack((xx, yy), -1)
    # This grid fits camera motion only; it is NOT the tracker query sampler.
    step = max(1, round(np.sqrt(height * width / 8000)))
    fit_xy = coordinates[::step, ::step].reshape(-1, 2)
    fit_to = (coordinates + flow)[::step, ::step].reshape(-1, 2)
    cv2.setRNGSeed(seed)
    transform, inliers = cv2.estimateAffinePartial2D(
        fit_xy,
        fit_to,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None:
        return None, {"status": "camera_fit_unavailable", "accepted": False}
    global_flow = coordinates @ transform[:, :2].T + transform[:, 2] - coordinates
    residual = np.linalg.norm(flow - global_flow, axis=-1).astype(np.float32)
    return residual, {
        "status": "estimated",
        "accepted": True,
        "affine": transform.tolist(),
        "ransac_inlier_fraction": float(inliers.mean()),
    }


def segment_motion(video, frame, fps, args):
    gray = cv2.cvtColor(video[frame].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
    offset = max(1, round(fps * args.motion_pair_seconds))
    neighbors = sorted(
        {max(0, frame - offset), min(len(video) - 1, frame + offset)} - {frame}
    )
    residual = np.zeros(gray.shape, np.float32)
    comparisons = []
    for neighbor in neighbors:
        other = cv2.cvtColor(
            video[neighbor].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY
        )
        change, fit = residual_motion(gray, other, args.seed)
        fit.update({"local_frame": neighbor, "delta_seconds": (neighbor - frame) / fps})
        comparisons.append(fit)
        if change is not None:
            residual = np.maximum(residual, change)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = max(args.motion_min_px, median + 3.0 * 1.4826 * mad)
    mask = (residual > threshold).astype(np.uint8)
    # Closing fills small holes but does not erase tiny components with an opening.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    number, components, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    retained = [
        i for i in range(1, number) if stats[i, cv2.CC_STAT_AREA] >= args.mask_min_area
    ]
    mask = np.isin(components, retained)
    components = np.where(mask, components, 0).astype(np.int32)
    return components, {
        "local_frame": frame,
        "comparisons": comparisons,
        "threshold_residual_px": threshold,
        "residual_median_px": median,
        "residual_mad_px": mad,
        "mask_fraction": float(mask.mean()),
        "component_areas_px": {
            str(i): int(stats[i, cv2.CC_STAT_AREA]) for i in retained
        },
    }


def component_points(components, budget):
    labels, areas = np.unique(components[components > 0], return_counts=True)
    selected, names = [], []
    if len(labels) == 0 or budget == 0:
        return np.empty((0, 2), np.float32), names
    order = np.argsort(-areas, kind="stable")[:budget]
    labels, areas = labels[order], areas[order]
    allocation = np.ones(len(labels), np.int64)
    for _ in range(budget - len(labels)):
        utility = np.sqrt(areas) / (allocation + 1)
        utility[allocation >= areas] = -1
        index = int(utility.argmax())
        if utility[index] < 0:
            break
        allocation[index] += 1
    for label, count in zip(labels, allocation):
        yy, xx = np.where(components == label)
        xy = np.stack((xx, yy), -1).astype(np.float32)
        # Deterministic farthest-point coverage inside each component, not full-image sampling.
        start = int(np.sum((xy - xy.mean(0)) ** 2, -1).argmin())
        distances = np.full(len(xy), np.inf, np.float32)
        indices = []
        for _ in range(int(count)):
            indices.append(start)
            distances = np.minimum(distances, np.sum((xy - xy[start]) ** 2, -1))
            distances[indices] = -1
            start = int(distances.argmax())
        selected.extend(xy[indices])
        names.extend([int(label)] * len(indices))
    return np.asarray(selected, np.float32).reshape(-1, 2), names


def build_motion_queries(video, case, args, directory):
    fps = case["record"]["fps"]
    interval = max(1, round(args.query_every_seconds * fps))
    anchors = list(range(0, len(video), interval))
    if anchors[-1] != len(video) - 1:
        anchors.append(len(video) - 1)
    segmented = [segment_motion(video, frame, fps, args) for frame in anchors]
    nonempty = [
        i for i, (components, _) in enumerate(segmented) if (components > 0).any()
    ]
    budgets = {
        i: args.point_budget // len(nonempty) + (j < args.point_budget % len(nonempty))
        for j, i in enumerate(nonempty)
    }
    xy_rows, frame_rows, labels, views = [], [], [], []
    first = case["first_frame"]
    mask_dir = directory / "motion_masks"
    mask_dir.mkdir(exist_ok=True)
    for anchor_index, (frame, (components, info)) in enumerate(zip(anchors, segmented)):
        xy, component_ids = component_points(components, budgets.get(anchor_index, 0))
        frame_rows.extend([first + frame] * len(xy))
        xy_rows.extend(xy.tolist())
        labels.extend(
            f"motion_t{first + frame}_component{label}" for label in component_ids
        )
        image = video[frame].permute(1, 2, 0).numpy().copy()
        mask = components > 0
        image[mask] = (image[mask] * 0.5 + np.array([255, 64, 32]) * 0.5).astype(
            np.uint8
        )
        overlay = Image.fromarray(image)
        draw = ImageDraw.Draw(overlay)
        for x, y in xy.tolist():
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 255))
        draw.rectangle((0, 0, overlay.width, 20), fill="black")
        draw.text(
            (4, 4),
            f"frame={first + frame} mask=motion proposal, points={len(xy)}",
            fill="white",
        )
        overlay_path = f"motion_masks/frame_{first + frame}_queries.png"
        overlay.save(directory / overlay_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            mask_dir / f"frame_{first + frame}_mask.png"
        )
        np.savez_compressed(
            mask_dir / f"frame_{first + frame}_components.npz", components=components
        )
        info.update(
            {"frame": first + frame, "point_count": len(xy), "overlay": overlay_path}
        )
        views.append(info)
    queries = {
        "xy": torch.tensor(xy_rows, dtype=torch.float32).reshape(-1, 2),
        "frames": torch.tensor(frame_rows, dtype=torch.long),
        "labels": labels,
    }
    report = {
        "kind": "camera_compensated_motion_region_proposals_not_instance_masks",
        "future_used_for_offline_query_selection": True,
        "image_resolution": "decoded native RGB; no proposal downscale",
        "point_budget": args.point_budget,
        "actual_points": len(labels),
        "sampling": "sqrt-area component allocation, deterministic within-mask farthest points",
        "empty_policy": "keep case with zero points; never substitute grid or another video",
        "views": views,
    }
    return queries, report
