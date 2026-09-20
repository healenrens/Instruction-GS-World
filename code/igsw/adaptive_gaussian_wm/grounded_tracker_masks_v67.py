"""Frozen GroundingDINO + SAM2 proposals at query frames, not object ground truth."""

import numpy as np
from PIL import Image
import torch

from .grounded_region_coverage_v67 import select_scale_coverage


def mask_box(mask):
    yy, xx = np.where(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def mask_iou(first, second):
    ax, ay, ar, ab = first["box"]
    bx, by, br, bb = second["box"]
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ar, br), min(ab, bb)
    if left >= right or top >= bottom:
        return 0.0
    intersection = np.logical_and(
        first["mask"][top - ay : bottom - ay, left - ax : right - ax],
        second["mask"][top - by : bottom - by, left - bx : right - bx],
    ).sum()
    return float(intersection / (first["area"] + second["area"] - intersection))


def mask_region(mask, left=0, top=0):
    x, y, right, bottom = mask_box(mask)
    return {
        "mask": mask[y:bottom, x:right].copy(),
        "box": [x + left, y + top, right + left, bottom + top],
        "area": int(mask.sum()),
    }


def crop_windows(width, height, divisions):
    boxes = [[0, 0, width, height]]
    if divisions > 1:
        # Overlap prevents a small object on a tile boundary from always being cut.
        crop_w, crop_h = (
            round(width / divisions * 1.25),
            round(height / divisions * 1.25),
        )
        for top in np.linspace(0, height - crop_h, divisions).round().astype(int):
            for left in np.linspace(0, width - crop_w, divisions).round().astype(int):
                boxes.append(
                    [int(left), int(top), int(left + crop_w), int(top + crop_h)]
                )
    return boxes


class GroundedTrackerMasks:
    def __init__(self, args, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers import Sam2Model, Sam2Processor

        self.args, self.device = args, device
        self.ground_processor = AutoProcessor.from_pretrained(
            args.grounding_model, local_files_only=True
        )
        self.ground_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                args.grounding_model, local_files_only=True, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.sam_processor = Sam2Processor.from_pretrained(
            args.sam_model, local_files_only=True
        )
        self.sam_model = (
            Sam2Model.from_pretrained(
                args.sam_model, local_files_only=True, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    @torch.no_grad()
    def robot_boxes(self, image):
        inputs = self.ground_processor(
            images=image,
            text="robot arm. robot gripper. robot hand.",
            return_tensors="pt",
        ).to(self.device)
        outputs = self.ground_model(**inputs)
        result = self.ground_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.args.robot_box_threshold,
            text_threshold=self.args.robot_text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        return [
            {"box": box.cpu().tolist(), "score": float(score), "phrase": str(label)}
            for box, score, label in zip(result["boxes"], result["scores"], labels)
        ]

    @torch.no_grad()
    def masks(self, image, points=None, boxes=None, keep_alternatives=False):
        prompts = points if points is not None else boxes
        embeddings = None
        for first in range(0, len(prompts), self.args.sam_prompt_batch):
            part = prompts[first : first + self.args.sam_prompt_batch]
            prompt = (
                {
                    "input_points": [[[[float(x), float(y)]] for x, y in part]],
                    "input_labels": [[[1] for _ in part]],
                }
                if points is not None
                else {"input_boxes": [part]}
            )
            inputs = self.sam_processor(images=image, **prompt, return_tensors="pt").to(
                self.device
            )
            original_sizes = inputs.pop("original_sizes")
            if embeddings is not None:
                inputs.pop("pixel_values")
                inputs["image_embeddings"] = embeddings
            outputs = self.sam_model(**inputs, multimask_output=True)
            embeddings = outputs.image_embeddings
            scores = outputs.iou_scores[0].float()
            if keep_alternatives:
                # Point prompts can describe a part or its enclosing object. Keep both.
                low_res = outputs.pred_masks[0].flatten(0, 1)
                selected_scores = scores.flatten()
            else:
                best = scores.argmax(-1)
                low_res = outputs.pred_masks[
                    0, torch.arange(len(part), device=self.device), best
                ]
                selected_scores = scores[
                    torch.arange(len(part), device=self.device), best
                ]
            stable = (low_res > 1.0).sum((-1, -2)).float() / (low_res > -1.0).sum(
                (-1, -2)
            ).clamp_min(1)
            masks = (
                self.sam_processor.post_process_masks(
                    low_res[None, :, None].float().cpu(), original_sizes.cpu()
                )[0][:, 0]
                .bool()
                .numpy()
            )
            for index, mask in enumerate(masks):
                yield mask, float(selected_scores[index]), float(stable[index])

    def frame(self, rgb, progress):
        args = self.args
        image = Image.fromarray(rgb)
        height, width = rgb.shape[:2]
        detections = self.robot_boxes(image)
        robot_regions = []
        for detection, (mask, quality, stability) in zip(
            detections, self.masks(image, boxes=[item["box"] for item in detections])
        ):
            if mask.sum() < args.sam_min_area:
                continue
            region = mask_region(mask)
            if any(
                mask_iou(region, other) >= args.mask_dedup_iou
                for other in robot_regions
            ):
                continue
            robot_regions.append(
                {
                    **detection,
                    **region,
                    "role": "robot_context",
                    "sam_score": quality,
                    "stability": stability,
                    "robot_overlap": 1.0,
                    "crop_box": [0, 0, width, height],
                }
            )
        robot_union = np.zeros((height, width), bool)
        for region in robot_regions:
            left, top, right, bottom = region["box"]
            robot_union[top:bottom, left:right] |= region["mask"]

        proposals = []
        for crop_index, box in enumerate(
            crop_windows(width, height, args.sam_crop_divisions)
        ):
            left, top, right, bottom = box
            crop = image.crop(box)
            yy, xx = np.meshgrid(
                (np.arange(args.sam_grid_side) + 0.5)
                * crop.height
                / args.sam_grid_side,
                (np.arange(args.sam_grid_side) + 0.5) * crop.width / args.sam_grid_side,
                indexing="ij",
            )
            seeds = np.stack([xx.ravel(), yy.ravel()], -1).tolist()
            print(
                f"[grounded-tracker] {progress} crop={crop_index} sam_prompts={len(seeds)}",
                flush=True,
            )
            for local, quality, stability in self.masks(
                crop, points=seeds, keep_alternatives=True
            ):
                area = int(local.sum())
                if (
                    area < args.sam_min_area
                    or quality < args.sam_score_threshold
                    or stability < args.sam_stability_threshold
                ):
                    continue
                # A tile-truncated mask is not a new object; the full view is retained.
                local_box = mask_box(local)
                cut = (
                    (left > 0 and local_box[0] == 0)
                    or (top > 0 and local_box[1] == 0)
                    or (right < width and local_box[2] == crop.width)
                    or (bottom < height and local_box[3] == crop.height)
                )
                if cut:
                    continue
                region = mask_region(local, left, top)
                x, y, xr, yb = region["box"]
                overlap = float((region["mask"] & robot_union[y:yb, x:xr]).sum() / area)
                # Robot overlap is recorded, never subtracted from object proposals.
                role = (
                    "scene_context"
                    if area / (height * width) > args.scene_area_fraction
                    else "object_candidate"
                )
                proposals.append(
                    {
                        **region,
                        "role": role,
                        "sam_score": quality,
                        "stability": stability,
                        "robot_overlap": overlap,
                        "crop_box": box,
                    }
                )

        kept = []
        # Deduplicate near-identical masks without removing part/whole alternatives.
        for item in sorted(proposals, key=lambda value: -value["sam_score"]):
            if not any(mask_iou(item, other) >= args.mask_dedup_iou for other in kept):
                kept.append(item)
        objects = select_scale_coverage(
            [item for item in kept if item["role"] == "object_candidate"],
            args.max_masks_per_frame,
            (height, width),
        )
        context = sorted(
            [item for item in kept if item["role"] != "object_candidate"],
            key=lambda item: -item["sam_score"],
        )
        selected = objects + context[: args.max_context_masks]
        masks, records = [], []
        for item in robot_regions + selected:
            mask = np.zeros((height, width), bool)
            left, top, right, bottom = item["box"]
            mask[top:bottom, left:right] = item["mask"]
            masks.append(mask)
            records.append({key: value for key, value in item.items() if key != "mask"})
        for index, record in enumerate(records):
            record["mask_index"] = index
        return (
            masks,
            records,
            {
                "robot_detections": detections,
                "robot_mask_count": len(robot_regions),
                "robot_detection_status": "detected"
                if robot_regions
                else "not_detected_not_proven_absent",
                "proposal_count_before_capacity": len(kept),
                "retained_region_count": len(masks),
                "unselected_region_count": len(kept) - len(selected),
                "robot_role_status": "single_frame_hypotheses_only",
                "selection": "all_multimask_scales_then_scale_balanced_new_support",
            },
        )
