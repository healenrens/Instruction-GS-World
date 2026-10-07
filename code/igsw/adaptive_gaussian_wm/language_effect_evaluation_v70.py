"""Per-trajectory measurements before any cross-case aggregation."""

import math

import torch


def trajectory_records(prediction, teacher, native_hw, history_frames, times, motion_floor_px=1.0):
    scale = (native_hw.float().flip(-1)-1).clamp_min(1) / 2
    truth = teacher["xy"][:, history_frames:].float()
    prediction = prediction.float()
    reference = teacher["reference_xy"].float()
    error = ((prediction-truth)*scale[:, None, None]).norm(dim=-1)
    motion = ((truth-reference[:, None])*scale[:, None, None]).norm(dim=-1)
    drift = ((prediction-reference[:, None])*scale[:, None, None]).norm(dim=-1)
    # Copy once; per-point scalar reads from CUDA would synchronize thousands of times.
    error, motion, drift = [value.detach().cpu() for value in (error, motion, drift)]
    valid = teacher["valid"][:, history_frames:].bool().cpu()
    times = times.float().cpu()
    point_present, point_ids, transport_weight, reference_index = [teacher[name].cpu() for name in
        ("point_present", "point_ids", "transport_weight", "reference_index")]
    rows = []
    for item in range(len(prediction)):
        tolerance = float((times[item, 1:] - times[item, :-1]).abs().median()) / 2 + 1e-4
        horizon_frames = {seconds: int((times[item] - seconds).abs().argmin()) for seconds in (1, 3, 5)}
        for point in torch.where(point_present[item])[0].tolist():
            known = valid[item, :, point]
            count = int(known.sum())
            if not count:
                continue
            amplitude = float(motion[item, known, point].mean())
            ade = float(error[item, known, point].mean())
            endpoint = bool(known[-1])
            endpoint_motion = float(motion[item, -1, point]) if endpoint else None
            fde = float(error[item, -1, point]) if endpoint else None
            horizons = {}
            for seconds, frame in horizon_frames.items():
                # Accept only a nearby requested frame, never an earlier visible substitute.
                available = abs(float(times[item, frame]) - seconds) <= tolerance and bool(known[frame])
                distance = float(motion[item, frame, point]) if available else None
                horizons[f"error_{seconds}s_px"] = float(error[item, frame, point]) if available else None
                horizons[f"relative_error_{seconds}s"] = (float(error[item, frame, point]) / distance
                                                          if available and distance >= motion_floor_px else None)
                horizons[f"actual_time_{seconds}s"] = float(times[item, frame])
            rows.append({**horizons, "item": item, "point_id": int(point_ids[item, point]),
                         "motion_selected": bool(transport_weight[item, point] > 0),
                         "reference_at_current": int(reference_index[item, point]) == history_frames - 1,
                         "valid_frames": count, "endpoint_visible": endpoint,
                         "ade_px": ade, "fde_px": fde, "motion_amplitude_px": amplitude,
                         "relative_ade": ade/amplitude if amplitude >= motion_floor_px else None,
                         "relative_fde": fde/endpoint_motion if endpoint and endpoint_motion >= motion_floor_px else None,
                         "static_drift_px": float(drift[item, known, point].mean()) if amplitude < motion_floor_px else None,
                         "times_seconds": times[item].float().cpu().tolist(),
                         "error_px": [float(error[item, t, point]) if known[t] else None for t in range(len(known))],
                         "motion_px": [float(motion[item, t, point]) if known[t] else None for t in range(len(known))]})
    return rows


def summarize_records(rows):
    summary = {"trajectories": len(rows), "visible_endpoints": sum(row["endpoint_visible"] for row in rows)}
    for key in ("ade_px", "fde_px", "relative_ade", "relative_fde", "static_drift_px", "motion_amplitude_px",
                "error_1s_px", "error_3s_px", "error_5s_px",
                "relative_error_1s", "relative_error_3s", "relative_error_5s"):
        values = torch.tensor([row[key] for row in rows if row[key] is not None and math.isfinite(row[key])])
        summary[key] = ({"count": len(values), "mean": float(values.mean()),
                         "p50": float(values.quantile(.5)), "p90": float(values.quantile(.9))}
                        if len(values) else {"count": 0, "mean": None, "p50": None, "p90": None})
    for low, high in ((0, 1), (1, 5), (5, 20), (20, 50), (50, float("inf"))):
        subset = [row["ade_px"] for row in rows if low <= row["motion_amplitude_px"] < high]
        summary[f"motion_{low}_{high}_px"] = {"count": len(subset), "ade_px": sum(subset)/len(subset) if subset else None}
    return summary


def conflict_target_measurements(prediction, teacher, native_hw, specification):
    """Conflicting language is scored only against its own manually specified endpoints."""
    xy = (prediction[0, -1].float()+1)*(native_hw[0].float().flip(-1)-1)/2
    ids = teacher["point_ids"][0].tolist()
    rows = []
    for target in specification.get("targets", []):
        if target["point_id"] not in ids:
            rows.append({**target, "error_px": None, "reason": "point_not_in_measurement_set"})
            continue
        point = ids.index(target["point_id"])
        error = float((xy[point]-xy.new_tensor(target["target_xy_px"])).norm())
        rows.append({**target, "error_px": error, "within_tolerance": error <= target["tolerance_px"]})
    return rows
