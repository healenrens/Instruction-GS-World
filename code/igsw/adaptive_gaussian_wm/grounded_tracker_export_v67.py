"""One selection contract shared by the training-candidate export and display."""

from collections import Counter

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
    rows, target, context = [], [], []
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
        use_context = metadata["role"] != "object_candidate"
        target.append(use_target)
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
                "context": bool(use_context),
                "displayed": bool(use_target or use_context),
            }
        )
    target = torch.tensor(target, dtype=torch.bool)
    context = torch.tensor(context, dtype=torch.bool)
    display_ids = torch.where(target | context)[0]
    report = {
        "contract": "grounded_object_motion_selection_v1",
        "raw_point_count": len(rows),
        "shown_point_count": len(display_ids),
        "object_motion_target_count": int(target.sum()),
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
        "role_scope": "query-frame pseudo role, not a per-frame segmentation or verified identity",
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
    directory, case, queries, native, sampled, args, parameters
):
    target, context, ids, report = select_training_points(native, queries, args)
    write_json(directory / "motion_filter.json", report)
    payload = {
        "contract": "grounded_object_motion_teacher_v1",
        "source_revision": args.source_revision,
        "case": case,
        "queries": queries,
        "native": native,
        "sampled": sampled,
        "object_motion_target_mask": target,
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
