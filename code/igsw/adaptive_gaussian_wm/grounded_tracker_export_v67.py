"""One selection contract shared by the training-candidate export and display."""

from collections import Counter
import math

import numpy as np
import torch

from .grounded_tracker_sampling_v67 import ROLE_COLORS, save_tensor
from .tracker_motion_display_v67 import subset_tracks
from .tracker_visual_review_v67 import write_json


def select_training_points(native, queries, args):
    coordinates = native["tracks"].float().numpy()
    valid = (native["visibility"] & native["in_bounds"]).numpy() & np.isfinite(
        coordinates
    ).all(-1)
    valid[native["query_local_frames"].numpy(), np.arange(coordinates.shape[1])] = False
    rows, target, uncertain, context = [], [], [], []
    for index, metadata in enumerate(queries["metadata"]):
        positions = coordinates[:, index][valid[:, index]]
        span, jitter = None, None
        enough = len(positions) >= args.minimum_visible_frames
        triples = valid[2:, index] & valid[1:-1, index] & valid[:-2, index]
        # Second differences estimate high-frequency track variation, not confidence/GT noise.
        if triples.any():
            second = (
                coordinates[2:, index]
                - 2 * coordinates[1:-1, index]
                + coordinates[:-2, index]
            )
            jitter = float(
                np.median(np.linalg.norm(second[triples], axis=-1)) / np.sqrt(6)
            )
        threshold = max(
            args.motion_floor_pixels,
            args.motion_region_fraction * metadata["region_diagonal_px"],
            args.motion_noise_multiplier * (jitter or 0.0),
        )
        if enough:
            low, high = np.quantile(positions, [0.05, 0.95], axis=0)
            span = float(np.linalg.norm(high - low))
        moving = enough and span >= threshold
        use_target = metadata["role"] == "object_candidate" and moving
        use_uncertain = metadata["role"] == "unknown" and moving
        use_context = metadata["role"] != "object_candidate"
        target.append(use_target)
        uncertain.append(use_uncertain)
        context.append(use_context)
        rows.append(
            {
                **metadata,
                "valid_nonquery_frames": len(positions),
                "span_px": span,
                "second_difference_scale_px": jitter,
                "motion_threshold_px": threshold,
                "moving": bool(moving),
                "object_motion_target": bool(use_target),
                "uncertain_motion_candidate": bool(use_uncertain),
                "context": bool(use_context),
                "displayed": bool(use_target or use_context),
            }
        )
    candidate = torch.tensor(target, dtype=torch.bool)
    ranked_ids = sorted(
        torch.where(candidate)[0].tolist(),
        key=lambda index: (-rows[index]["span_px"], index),
    )
    keep = min(
        len(ranked_ids), max(0, math.ceil(len(ranked_ids) * args.motion_top_fraction))
    )
    retained_ids = ranked_ids[:keep]
    target = torch.zeros_like(candidate)
    target[retained_ids] = True
    for rank, index in enumerate(ranked_ids, start=1):
        rows[index]["motion_rank_in_clip"] = rank
    for index, row in enumerate(rows):
        row["motion_rank_in_clip"] = row.get("motion_rank_in_clip")
        row["object_motion_candidate_before_topk"] = bool(candidate[index])
        row["object_motion_target"] = bool(target[index])
        row["topk_rejected"] = bool(candidate[index] and not target[index])
        row["displayed"] = bool(target[index] or context[index])
    uncertain = torch.tensor(uncertain, dtype=torch.bool)
    context = torch.tensor(context, dtype=torch.bool)
    display_ids = torch.where(target | context)[0]
    report = {
        "contract": "grounded_object_motion_selection_v3",
        "raw_point_count": len(rows),
        "shown_point_count": len(display_ids),
        "object_motion_target_count": int(target.sum()),
        "object_motion_candidate_count_before_topk": int(candidate.sum()),
        "object_motion_candidate_ids_before_topk": torch.where(candidate)[0].tolist(),
        "motion_top_fraction": args.motion_top_fraction,
        "topk_scope": "per clip, moving object candidates only; robot/unknown/scene never compete",
        "topk_score": "native nonquery visible in-bounds xy 5%-95% span in pixels",
        "topk_tie_break": "original point ID ascending",
        "topk_cutoff_span_px": rows[retained_ids[-1]]["span_px"]
        if retained_ids
        else None,
        "motion_ranked_object_ids": ranked_ids,
        "region_candidates_before_topk": dict(
            Counter(rows[i]["region_id"] for i in ranked_ids)
        ),
        "region_targets_after_topk": dict(
            Counter(rows[i]["region_id"] for i in retained_ids)
        ),
        "topk_is_not_tracking_confidence": True,
        "uncertain_motion_candidate_count": int(uncertain.sum()),
        "context_count": int(context.sum()),
        "role_counts": dict(Counter(row["role"] for row in rows)),
        "minimum_visible_frames": args.minimum_visible_frames,
        "motion_floor_pixels": args.motion_floor_pixels,
        "motion_region_fraction": args.motion_region_fraction,
        "motion_noise_multiplier": args.motion_noise_multiplier,
        "selected_point_ids": display_ids.tolist(),
        "object_target_ids": torch.where(target)[0].tolist(),
        "points": rows,
        "motion_coordinate_system": "image pixels, not camera compensated or metric motion",
        "comparison": "same native-selected IDs in both panels and training candidate export",
        "role_scope": "cross-anchor robot evidence, not a per-frame segmentation or verified identity",
        "uncertain_motion_candidate_ids": torch.where(uncertain)[0].tolist(),
        "uncertain_is_not_robot_or_background_gt": True,
    }
    return target, context, display_ids, report


def styled_tracks(prediction, queries, ids):
    result = subset_tracks(prediction, ids)
    result["point_colors"] = [
        ROLE_COLORS[queries["metadata"][i]["role"]] for i in ids.tolist()
    ]
    result["legend"] = "green=object motion; orange=robot; purple=unknown; blue=scene"
    return result


def export_training_candidates(
    directory, case, queries, native, sampled, args, parameters, role_evidence
):
    target, context, ids, report = select_training_points(native, queries, args)
    write_json(directory / "motion_filter.json", report)
    payload = {
        "contract": "grounded_object_motion_teacher_v3",
        "source_revision": args.source_revision,
        "case": case,
        "queries": queries,
        "native": native,
        "sampled": sampled,
        "object_motion_target_mask": target,
        "object_motion_candidate_mask_before_topk": torch.tensor(
            [row["object_motion_candidate_before_topk"] for row in report["points"]],
            dtype=torch.bool,
        ),
        "uncertain_motion_candidate_mask": torch.tensor(
            [row["uncertain_motion_candidate"] for row in report["points"]],
            dtype=torch.bool,
        ),
        "role_evidence": role_evidence,
        "role_evidence_codes": {
            "query_anchor_excluded": -2,
            "unobserved": -1,
            "outside_robot_mask": 0,
            "robot_boundary": 1,
            "robot_core": 2,
        },
        "context_mask": context,
        "robot_context_mask": torch.tensor(
            [row["role"] == "robot_context" for row in queries["metadata"]]
        ),
        "display_point_ids": ids,
        "selection": report,
        "parameters": parameters,
        "teacher_only": True,
        "future_used_for_selection": True,
        "role_labels_are_pseudo": True,
        "student_may_read_future_tracks": False,
        "visibility_is_tracker_prediction": True,
        "region_identity_scope": "anchor-local, no object-ID equivalence across query frames",
    }
    save_tensor(directory / "training_candidates.pt", payload)
    return ids, report
