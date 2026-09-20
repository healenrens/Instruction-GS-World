"""Motion-supported SAM prompts followed by region-balanced native queries."""

from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
import torch

from .grounded_region_coverage_v67 import spread_inside_points
from .grounded_tracker_sampling_v67 import allocate_regions, render_proposals, save_tensor
from .tracker_visual_review_v67 import write_json


def spaced_indices(xy, ids, count):
    remaining = list(ids)
    chosen = []
    if remaining:
        chosen.append(remaining.pop(0))
    while remaining and len(chosen) < count:
        distances = ((xy[remaining, None] - xy[chosen][None]) ** 2).sum(-1).min(1)
        chosen.append(remaining.pop(int(distances.argmax())))
    return chosen


@torch.no_grad()
def refine_mask(segmenter, image, positives, negatives, embeddings):
    xy = np.concatenate((positives, negatives)).tolist()
    labels = [1] * len(positives) + [0] * len(negatives)
    inputs = segmenter.sam_processor(images=image, input_points=[[xy]], input_labels=[[labels]], return_tensors="pt").to(segmenter.device)
    sizes = inputs.pop("original_sizes")
    if embeddings is not None:
        inputs.pop("pixel_values")
        inputs["image_embeddings"] = embeddings
    output = segmenter.sam_model(**inputs, multimask_output=True)
    masks = segmenter.sam_processor.post_process_masks(output.pred_masks.float().cpu(), sizes.cpu())[0][0].bool().numpy()
    scores = output.iou_scores[0, 0].float().cpu().numpy()
    xy = np.rint(np.array(xy)).astype(int)
    measured = []
    for mask, score in zip(masks, scores):
        response = mask[xy[:, 1], xy[:, 0]]
        positive = float(response[:len(positives)].mean())
        negative = float(response[len(positives):].mean()) if len(negatives) else 0.0
        measured.append((positive - negative + .1 * float(score), positive, negative))
    best = max(range(len(masks)), key=lambda i: measured[i][0])
    return masks[best], {"positive_prompt_coverage": measured[best][1], "negative_prompt_hit": measured[best][2],
                         "sam_score": float(scores[best]), "positive_points": positives.tolist(), "negative_points": negatives.tolist()}, output.image_embeddings


def refine_and_densify(rgb, case, pilot_queries, pilot, evidence, background, sampling, segmenter, directory, args, *, render=True):
    result = deepcopy(pilot_queries)
    for row in result["metadata"]:
        row["refined_support"] = False
    height, width = case["height"], case["width"]
    region_results, mask_records, full_masks = [], [], {}
    reference = background["reference"]
    reference_static = background["inlier"].float().mean(0).numpy() >= .8
    for view in sampling["views"]:
        frame = view["frame"]
        local = frame - case["first_frame"]
        image = Image.fromarray(rgb[local].permute(1, 2, 0).numpy())
        embeddings = None
        xy = pilot["tracks"][local].numpy()
        visible = (pilot["visibility"][local] & pilot["in_bounds"][local]).numpy()
        with np.load(directory / view["mask_archive"]) as archive:
            for region in [r for r in sampling["regions"] if r["frame"] == frame and r["role"] == "object_candidate"]:
                ids = [i for i, row in enumerate(result["metadata"]) if row["region_id"] == region["region_id"] and row["role"] == "object_candidate" and evidence["moving"][i] and visible[i]]
                if not ids:
                    continue
                positive_ids = spaced_indices(xy, ids, 8)
                positives = xy[positive_ids]
                original = archive[f"m{region['mask_index']}"]
                neighborhood = cv2.dilate(original.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                ref_xy = reference["tracks"][local].numpy()
                ref_good = (reference["visibility"][local] & reference["in_bounds"][local]).numpy() & reference_static
                reference_ids = np.flatnonzero(ref_good)
                pixels = np.rint(ref_xy[reference_ids]).astype(int)
                reference_ids = reference_ids[neighborhood[pixels[:, 1], pixels[:, 0]]]
                # Do not place a negative directly on top of an observed moving surface point.
                if len(reference_ids):
                    far = np.linalg.norm(ref_xy[reference_ids, None] - positives[None], axis=-1).min(1) > 5.0
                    reference_ids = reference_ids[far]
                negative_ids = spaced_indices(ref_xy, reference_ids.tolist(), 8)
                negatives = ref_xy[negative_ids]
                mask, measurement, embeddings = refine_mask(segmenter, image, positives, negatives, embeddings)
                accepted = measurement["positive_prompt_coverage"] >= .75 and measurement["negative_prompt_hit"] <= .25 and mask.sum() >= args.sam_min_area
                region_results.append({"region_id": region["region_id"], "accepted": bool(accepted), **measurement})
                if not accepted:
                    continue
                record = {**region, "area": int(mask.sum()), "sam_score": measurement["sam_score"]}
                mask_records.append(record)
                full_masks[region["region_id"]] = mask
                for i, row in enumerate(result["metadata"]):
                    if row["region_id"] == region["region_id"]:
                        x, y = np.rint(row["xy"]).astype(int)
                        row["refined_support"] = bool(mask[y, x])
        embeddings = None
    budget = max(0, args.point_budget - len(result["xy"]))
    allocations = allocate_regions(mask_records, budget, args.minimum_region_points, args.maximum_region_points)
    for record, count in zip(mask_records, allocations):
        frame, region = record["frame"], record["region_id"]
        occupied = np.zeros((height, width), bool)
        existing = result["xy"][result["frames"] == frame].round().long().numpy()
        occupied[existing[:, 1], existing[:, 0]] = True
        extra = spread_inside_points(full_masks[region], count, occupied)
        for coordinate in extra.tolist():
            result["metadata"].append({"point_id": len(result["metadata"]), "xy": coordinate, "frame": frame,
                "region_id": region, "role": "object_candidate", "region_area_px": record["area"],
                "region_diagonal_px": float(np.hypot(record["box"][2] - record["box"][0], record["box"][3] - record["box"][1])),
                "sam_score": record["sam_score"], "robot_overlap": record["robot_overlap"], "refined_support": True})
        result["xy"] = torch.cat((result["xy"], torch.from_numpy(extra).float()))
        result["frames"] = torch.cat((result["frames"], torch.full((len(extra),), frame, dtype=torch.long)))
        result["labels"].extend([region] * len(extra))
    mask_path = directory / "refined_masks.npz"
    np.savez_compressed(mask_path, **full_masks)
    views = []
    for frame in sorted({r["frame"] for r in mask_records}) if render else []:
        records = [r for r in mask_records if r["frame"] == frame]
        points = [r for r in result["metadata"] if r["frame"] == frame]
        # Keep the original proposal image intact; refinement has a separate location.
        render_dir = directory / "refinement"
        (render_dir / "grounded_masks").mkdir(parents=True, exist_ok=True)
        path = render_proposals(rgb[frame-case["first_frame"]].permute(1, 2, 0).numpy(),
                               [full_masks[r["region_id"]] for r in records], records, points, frame, render_dir)
        views.append({"frame": frame, "overlay": f"refinement/{path}"})
    report = {"regions": region_results, "views": views, "raw_pilot_points": len(pilot_queries["xy"]),
              "dense_points": len(result["xy"]), "role_counts": dict(Counter(r["role"] for r in result["metadata"])),
              "teacher_only": True, "future_used": True, "mask_is_pseudo": True}
    save_tensor(directory / "dense_queries.pt", result)
    write_json(directory / "refinement.json", report)
    return result, report
