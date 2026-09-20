"""Region-balanced queries: object candidates first, robot/unknown as context."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .grounded_tracker_masks_v67 import inside_points
from .tracker_visual_review_v67 import read_json, write_json


ROLE_COLORS = {
    "object_candidate": (70, 230, 120),
    "robot_context": (255, 155, 45),
    "unknown": (210, 140, 255),
    "scene_context": (130, 160, 180),
}


def save_tensor(path, payload):
    temporary = Path(path).with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def allocate_regions(records, budget, minimum, maximum):
    counts = np.zeros(len(records), np.int64)
    capacity = np.array([min(item["area"], maximum) for item in records], np.int64)
    # Each candidate gets a turn; large arms cannot consume small-object capacity.
    for _ in range(minimum):
        for index in np.argsort([item["area"] for item in records], kind="stable"):
            if budget > 0 and counts[index] < capacity[index]:
                counts[index] += 1
                budget -= 1
    while budget > 0 and np.any(counts < capacity):
        utility = np.sqrt([item["area"] for item in records]) / (counts + 1)
        utility[counts >= capacity] = -1
        index = int(utility.argmax())
        counts[index] += 1
        budget -= 1
    return counts.tolist()


def render_proposals(rgb, masks, records, points, frame, directory):
    overlay = rgb.astype(np.float32).copy()
    # Draw large masks first so that small objects remain inspectable.
    for index in sorted(range(len(masks)), key=lambda i: -records[i]["area"]):
        mask = masks[index]
        tint = np.array(ROLE_COLORS[records[index]["role"]])
        overlay[mask] = overlay[mask] * 0.72 + tint * 0.28
    image = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for row in points:
        x, y = row["xy"]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=ROLE_COLORS[row["role"]])
    draw.rectangle((0, 0, image.width, 20), fill="black")
    draw.text(
        (4, 4),
        "green=object candidate orange=robot purple=unknown blue=scene",
        fill="white",
    )
    name = f"grounded_masks/frame_{frame}_queries.png"
    image.save(directory / name)
    return name


def build_grounded_queries(video, case, args, segmenter, directory, configuration):
    directory = Path(directory)
    mask_dir = directory / "grounded_masks"
    mask_dir.mkdir(exist_ok=True)
    fps, first = case["record"]["fps"], case["first_frame"]
    interval = max(1, round(fps * args.query_every_seconds))
    anchors = sorted(set(range(0, len(video), interval)) | {len(video) - 1})
    all_records, frame_records = [], []
    for local in anchors:
        frame = first + local
        info_path = mask_dir / f"frame_{frame}.json"
        masks_path = mask_dir / f"frame_{frame}.npz"
        if (
            args.reuse_completed
            and info_path.is_file()
            and read_json(info_path)["configuration"] == configuration
        ):
            info = read_json(info_path)
        else:
            masks, regions, detection = segmenter.frame(
                video[local].permute(1, 2, 0).numpy(),
                f"case={case['case_id']} frame={frame}",
            )
            for record in regions:
                record.update(
                    {
                        "frame": frame,
                        "local_frame": local,
                        "region_id": f"f{frame}_r{record['mask_index']}",
                    }
                )
            np.savez_compressed(
                masks_path, **{f"m{index}": mask for index, mask in enumerate(masks)}
            )
            info = {
                "configuration": configuration,
                "regions": regions,
                "detection": detection,
                "frame": frame,
                "mask_archive": str(masks_path.relative_to(directory)),
            }
            write_json(info_path, info)
        frame_records.append(info)
        all_records.extend(info["regions"])

    robot_budget = round(args.point_budget * args.robot_point_fraction)
    other_budget = round(args.point_budget * args.other_context_fraction)
    budgets = {
        "object": args.point_budget - robot_budget - other_budget,
        "robot": robot_budget,
        "other": other_budget,
    }
    groups = {
        "object": [r for r in all_records if r["role"] == "object_candidate"],
        "robot": [r for r in all_records if r["role"] == "robot_context"],
        "other": [r for r in all_records if r["role"] in ("unknown", "scene_context")],
    }
    for name in ("robot", "other"):
        if not groups[name]:
            budgets["object"] += budgets[name]
            budgets[name] = 0
    counts = {}
    for name, records in groups.items():
        allocations = allocate_regions(
            records,
            budgets[name],
            args.minimum_region_points,
            args.maximum_region_points,
        )
        counts.update(
            {record["region_id"]: count for record, count in zip(records, allocations)}
        )

    point_rows, views = [], []
    for info in frame_records:
        with np.load(directory / info["mask_archive"]) as archive:
            masks = [archive[f"m{record['mask_index']}"] for record in info["regions"]]
        per_frame = []
        for mask, record in zip(masks, info["regions"]):
            xy = inside_points(mask, counts[record["region_id"]])
            left, top, right, bottom = record["box"]
            for coordinate in xy.tolist():
                row = {
                    "point_id": len(point_rows),
                    "xy": coordinate,
                    "frame": record["frame"],
                    "region_id": record["region_id"],
                    "role": record["role"],
                    "region_area_px": record["area"],
                    "region_diagonal_px": float(np.hypot(right - left, bottom - top)),
                    "sam_score": record["sam_score"],
                    "robot_overlap": record["robot_overlap"],
                }
                point_rows.append(row)
                per_frame.append(row)
            record["query_count"] = len(xy)
        local = info["frame"] - first
        overlay = render_proposals(
            video[local].permute(1, 2, 0).numpy(),
            masks,
            info["regions"],
            per_frame,
            info["frame"],
            directory,
        )
        union = np.zeros(case["height"] * case["width"], bool).reshape(
            case["height"], case["width"]
        )
        for mask in masks:
            union |= mask
        views.append(
            {
                "frame": info["frame"],
                "point_count": len(per_frame),
                "mask_fraction": float(union.mean()),
                "overlay": overlay,
                "roles": dict(Counter(row["role"] for row in per_frame)),
                "mask_archive": info["mask_archive"],
                "detection": info["detection"],
            }
        )
    queries = {
        "xy": torch.tensor(
            [row["xy"] for row in point_rows], dtype=torch.float32
        ).reshape(-1, 2),
        "frames": torch.tensor([row["frame"] for row in point_rows], dtype=torch.long),
        "labels": [row["region_id"] for row in point_rows],
        "metadata": point_rows,
    }
    report = {
        "kind": "grounding_dino_sam2_region_queries_v1",
        "configuration": configuration,
        "future_used_for_offline_query_selection": True,
        "actual_points": len(point_rows),
        "point_budget": args.point_budget,
        "role_budgets": budgets,
        "role_counts": dict(Counter(row["role"] for row in point_rows)),
        "views": views,
        "regions": all_records,
        "unselected_region_count": sum(count == 0 for count in counts.values()),
        "semantics": "pseudo roles at query frame; region IDs are not persistent object IDs",
        "sampling": "native mask interiors; region-balanced deterministic farthest points; no motion prefilter",
        "sam_grid_role": "segmentation prompts only; these are not CoTracker queries",
    }
    save_tensor(directory / "queries.pt", queries)
    write_json(directory / "sampling.json", report)
    return queries, report
