"""Reusable offline supervision; future-derived selection never becomes student input."""

from collections import Counter, defaultdict
import math

import numpy as np
import torch

from .grounded_tracker_sampling_v67 import save_tensor
from .tracker_visual_review_v67 import write_json


CONTRACT = "grounded_object_motion_teacher_v4"


def region_balanced_selection(queries, evidence, relay, fraction):
    candidates = torch.tensor([row["role"] == "object_candidate" and row["refined_support"] for row in queries["metadata"]])
    candidates &= evidence["moving"] & relay["consistent"]
    groups = defaultdict(list)
    for point in torch.where(candidates)[0].tolist():
        groups[queries["metadata"][point]["region_id"]].append(point)
    selected, ranks = [], {}
    for ids in groups.values():
        ranked = sorted(ids, key=lambda i: (-float(evidence["span_px"][i]), i))
        ranks.update({point: rank + 1 for rank, point in enumerate(ranked)})
        xy = queries["xy"][ranked].numpy()
        low, high = xy.min(0), xy.max(0)
        cells = np.minimum(3, ((xy - low) / np.maximum(1, high - low) * 4).astype(int))
        queues = defaultdict(list)
        for point, cell in zip(ranked, cells):
            queues[tuple(cell)].append(point)
        wanted = min(len(ids), math.ceil(len(ids) * fraction))
        # Round-robin spatial cells, highest motion first within each cell.
        keys = sorted(queues, key=lambda cell: (-float(evidence["span_px"][queues[cell][0]]), cell))
        chosen = []
        while len(chosen) < wanted:
            for cell in keys:
                if queues[cell] and len(chosen) < wanted:
                    chosen.append(queues[cell].pop(0))
        selected.extend(chosen)
    target = torch.zeros_like(candidates)
    target[selected] = True
    return candidates, target, ranks


def export_motion_teacher(directory, case, queries, native, evidence, relay, background, sampling, roles, args):
    candidates, target, ranks = region_balanced_selection(queries, evidence, relay, args.motion_top_fraction)
    valid = evidence["valid"] & relay["valid"]
    target &= valid.sum(0) >= args.minimum_visible_frames
    context = ~target
    rows = []
    for i, row in enumerate(queries["metadata"]):
        reasons = []
        if row["role"] != "object_candidate":
            reasons.append("role_context")
        if not row["refined_support"]:
            reasons.append("no_motion_supported_mask")
        if not evidence["moving"][i]:
            reasons.append("no_resolved_background_relative_motion")
        if not relay["consistent"][i]:
            reasons.append("temporal_requery_disagreement_or_unknown")
        if candidates[i] and not target[i]:
            reasons.append("spatial_motion_budget")
        rows.append({**row, "span_px": float(evidence["span_px"][i]), "raw_span_px": float(evidence["raw_span_px"][i]),
                     "motion_threshold_px": float(evidence["threshold_px"][i]), "motion_rank_in_region": ranks.get(i),
                     "object_motion_target": bool(target[i]), "reasons": reasons})
    report = {"contract": CONTRACT, "raw_point_count": len(rows), "shown_point_count": len(rows),
              "object_motion_target_count": int(target.sum()), "object_motion_candidate_count_before_topk": int(candidates.sum()),
              "object_motion_candidate_ids_before_topk": torch.where(candidates)[0].tolist(),
              "object_target_ids": torch.where(target)[0].tolist(), "context_count": int(context.sum()),
              "role_counts": dict(Counter(row["role"] for row in rows)), "motion_top_fraction": args.motion_top_fraction,
              "points": rows, "selection": "per-anchor-region spatial-cell round-robin, residual-motion ranking within cells",
              "background_usable_frame_fraction": float(background["valid"].float().mean()),
              "uncertain_motion_candidate_count": sum(row["role"] == "unknown" and bool(evidence["moving"][i]) for i, row in enumerate(rows)),
              "topk_is_not_tracking_confidence": True}
    payload = {"contract": CONTRACT, "case": case, "queries": queries, "native": native,
               "object_motion_candidate_mask_before_topk": candidates, "object_motion_target_mask": target,
               "context_mask": context, "motion_evidence": evidence, "relay_evidence": relay,
               "background_motion": background, "target_valid": valid & target[None],
               "role_evidence": roles, "sampling": sampling, "selection": report,
               "source_revision": args.source_revision, "teacher_only": True, "future_used_for_selection": True,
               "student_input": "observed RGB and requested horizon only; no teacher masks, selection or future coordinates",
               "visibility_is_tracker_prediction": True, "region_ids_are_not_object_ids": True,
               "configuration": {k: v for k, v in vars(args).items() if not k.startswith("wandb_")}}
    save_tensor(directory / "teacher.pt", payload)
    write_json(directory / "motion_filter.json", report)
    return report
