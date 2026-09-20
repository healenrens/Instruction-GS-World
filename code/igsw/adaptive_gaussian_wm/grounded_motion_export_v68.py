"""Reusable offline supervision; future-derived selection never becomes student input."""

from collections import Counter
import math

import torch

from .grounded_tracker_sampling_v67 import save_tensor
from .tracker_visual_review_v67 import write_json


CONTRACT = "all_point_motion_teacher_v5"
SELECTION_POLICY = "global_all_roles_motion_top_fraction"


def all_point_motion_selection(evidence, fraction):
    # A displacement needs two measured locations; semantic roles never enter ranking.
    candidates = (evidence["valid"].sum(0) >= 2) & torch.isfinite(evidence["span_px"])
    ranked = sorted(torch.where(candidates)[0].tolist(), key=lambda i: (-float(evidence["span_px"][i]), i))
    ranks = {point: rank + 1 for rank, point in enumerate(ranked)}
    selected = ranked[:math.ceil(len(ranked) * fraction)]
    target = torch.zeros_like(candidates)
    target[selected] = True
    return candidates, target, ranks


def export_motion_teacher(directory, case, queries, native, evidence, relay, background, sampling, roles, args):
    candidates, target, ranks = all_point_motion_selection(evidence, args.motion_top_fraction)
    valid = evidence["valid"] & relay["valid"]
    context = ~target
    rows = []
    for i, row in enumerate(queries["metadata"]):
        reasons = []
        if not candidates[i]:
            reasons.append("fewer_than_two_measured_locations_or_nonfinite_span")
        if candidates[i] and not target[i]:
            reasons.append("below_global_motion_rank_cutoff")
        rows.append({**row, "span_px": float(evidence["span_px"][i]), "raw_span_px": float(evidence["raw_span_px"][i]),
                     "motion_rank_global": ranks.get(i), "selected_for_motion": bool(target[i]),
                     "valid_target_frames": int(valid[:, i].sum()) if target[i] else 0, "reasons": reasons})
    report = {"contract": CONTRACT, "raw_point_count": len(rows), "shown_point_count": len(rows),
              "object_motion_target_count": int(target.sum()), "object_motion_candidate_count_before_topk": int(candidates.sum()),
              "object_motion_candidate_ids_before_topk": torch.where(candidates)[0].tolist(),
              "object_target_ids": torch.where(target)[0].tolist(), "context_count": int(context.sum()),
              "role_counts": dict(Counter(row["role"] for row in rows)), "motion_top_fraction": args.motion_top_fraction,
              "points": rows, "selection": SELECTION_POLICY, "roles_used_for_selection": False,
              "minimum_motion_threshold_used": False, "per_region_quota_used": False,
              "rankable_point_count": int(candidates.sum()), "selected_point_count": int(target.sum()),
              "selected_with_valid_frames": int((target & valid.any(0)).sum()),
              "background_usable_frame_fraction": float(background["valid"].float().mean()),
              "uncertain_motion_candidate_count": sum(row["role"] == "unknown" and bool(evidence["moving"][i]) for i, row in enumerate(rows)),
              "topk_is_not_tracking_confidence": True}
    payload = {"contract": CONTRACT, "case": case, "queries": queries, "native": native,
               "object_motion_candidate_mask_before_topk": candidates, "object_motion_target_mask": target,
               "motion_target_mask": target, "selection_policy": SELECTION_POLICY,
               "legacy_object_keys_are_motion_aliases": True,
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
