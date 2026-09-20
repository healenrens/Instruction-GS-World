"""Cross-anchor robot evidence on tracks; semantic hypotheses, not calibrated labels."""

from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
import torch

from .tracker_visual_review_v67 import write_json
from .grounded_tracker_sampling_v67 import save_tensor


def resolve_track_roles(native, queries, sampling, directory, args):
    frames = native["frame_indices"].numpy()
    tracks = native["tracks"].float().numpy()
    valid = (native["visibility"] & native["in_bounds"]).numpy()
    valid &= np.isfinite(tracks).all(-1)
    anchors, evidence_rows = [], []
    for view in sampling["views"]:
        frame = int(view["frame"])
        at = int(np.searchsorted(frames, frame))
        rows = [row for row in sampling["regions"] if row["frame"] == frame]
        height, width = sampling["native_hw"]
        robot = np.zeros((height, width), bool)
        with np.load(directory / view["mask_archive"]) as archive:
            for row in rows:
                if row["role"] == "robot_context":
                    robot |= archive[f"m{row['mask_index']}"]
        distance = cv2.distanceTransform(robot.astype(np.uint8), cv2.DIST_L2, 3)
        core = robot & (distance > args.robot_core_margin_px)
        evidence = np.full(tracks.shape[1], -1, np.int8)
        ids = np.flatnonzero(valid[at])
        xy = np.rint(tracks[at, ids]).astype(np.int64)
        x, y = xy[:, 0], xy[:, 1]
        evidence[ids] = np.where(core[y, x], 2, np.where(robot[y, x], 1, 0))
        # Supplied query coordinates are not an independent cross-time confirmation.
        evidence[native["query_local_frames"].numpy() == at] = -2
        anchors.append(frame)
        evidence_rows.append(evidence)
    evidence = np.stack(evidence_rows)
    result = deepcopy(queries)
    details = []
    for index, row in enumerate(result["metadata"]):
        observed = int((evidence[:, index] >= 0).sum())
        positive = int((evidence[:, index] == 2).sum())
        boundary = int((evidence[:, index] == 1).sum())
        outside = int((evidence[:, index] == 0).sum())
        core_fraction = positive / observed if observed else None
        robot_fraction = (positive + boundary) / observed if observed else None
        if (
            positive >= args.robot_min_anchor_votes
            and core_fraction >= args.robot_confirmation_fraction
        ):
            role, reason = "robot_context", "repeated_robot_core_evidence"
        elif (
            outside >= args.robot_min_anchor_votes
            and robot_fraction <= args.robot_rejection_fraction
        ):
            role = (
                "scene_context"
                if row["role"] == "scene_context"
                else "object_candidate"
            )
            reason = "low_robot_support_across_visible_views"
        else:
            role, reason = "unknown", "insufficient_or_conflicting_robot_evidence"
        measurement = {
            "point_id": index,
            "proposal_role": row["role"],
            "resolved_role": role,
            "reason": reason,
            "observed_anchors": observed,
            "robot_core_votes": positive,
            "robot_boundary_votes": boundary,
            "outside_robot_votes": outside,
            "robot_core_fraction": core_fraction,
            "robot_mask_fraction": robot_fraction,
        }
        row.update(
            {"proposal_role": row["role"], "role": role, "role_evidence": measurement}
        )
        details.append(measurement)
    report = {
        "method": "native_tracks_cross_anchor_robot_mask_evidence_v1",
        "anchor_frames": anchors,
        "role_counts": dict(Counter(row["role"] for row in result["metadata"])),
        "changed_role_count": sum(
            row["proposal_role"] != row["resolved_role"] for row in details
        ),
        "minimum_anchor_votes": args.robot_min_anchor_votes,
        "confirmation_fraction": args.robot_confirmation_fraction,
        "rejection_fraction": args.robot_rejection_fraction,
        "robot_core_margin_px": args.robot_core_margin_px,
        "points": details,
        "scope": "uncalibrated teacher role evidence; persistent detector errors can survive",
        "future_used": True,
        "query_anchor_excluded_from_votes": True,
    }
    payload = {
        "anchor_frames": torch.tensor(anchors),
        "evidence": torch.from_numpy(evidence),
    }
    save_tensor(
        directory / "resolved_queries.pt", {"queries": result, "evidence": payload}
    )
    write_json(directory / "role_resolution.json", report)
    return result, payload, report
