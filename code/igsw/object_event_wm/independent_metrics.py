"""External-instance measurements and paired real-outcome preference, not self-defined truth."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def label_statistics(labels):
    labels = np.asarray(labels, dtype=np.int64)
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    area = np.bincount(labels.ravel())
    x = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=len(area))
    y = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=len(area))
    centers = np.stack((x, y), -1)/np.maximum(area[:, None], 1)
    return area, centers


def anchor_assignment(predicted, truth, bodies):
    candidates = np.unique(predicted)
    stride = int(max(candidates.max(), truth.max(), max(bodies)))+1
    confusion = np.bincount(truth.astype(np.int64).ravel()*stride+predicted.astype(np.int64).ravel(),
                            minlength=stride*stride).reshape(stride, stride)
    gt_area, pred_area = confusion.sum(1), confusion.sum(0)
    seen = [int(body) for body in bodies if gt_area[int(body)] > 0]
    intersection = confusion[np.ix_(seen, candidates)]
    union = gt_area[seen, None]+pred_area[None, candidates]-intersection
    overlap = intersection/np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {seen[row]: int(candidates[col]) for row, col in zip(rows, cols)}
    scores = {seen[row]: float(overlap[row, col]) for row, col in zip(rows, cols)}
    return mapping, scores, seen


def _center(stats, label):
    area, centers = stats
    return centers[label] if label is not None and label < len(area) and area[label] > 0 else None


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _image_float(images):
    return images.astype(np.float32)/255 if images.dtype == np.uint8 else images.astype(np.float32).clip(0, 1)


def physical_motion_context(anchor_xyz, future_xyz, bodies):
    displacement = np.linalg.norm(future_xyz-anchor_xyz[None], axis=-1)
    path = np.linalg.norm(np.diff(np.concatenate((anchor_xyz[None], future_xyz)), axis=0), axis=-1)
    return [{"body_id": int(body), "rms_world_center_displacement_m": float(np.sqrt(np.mean(displacement[:, index]**2))),
             "world_center_path_length_m": float(path[:, index].sum())} for index, body in enumerate(bodies)]


def score_entities(predicted_anchor, predicted_future, truth_anchor, truth_future, entities, times):
    bodies = [int(entity["body_id"]) for entity in entities]
    mapping, initial_iou, seen = anchor_assignment(predicted_anchor, truth_anchor, bodies)
    source_gt, source_pred = label_statistics(truth_anchor), label_statistics(predicted_anchor)
    diagonal = float(np.hypot(*truth_anchor.shape))
    output = {body: {**entity, "matched_slot": mapping.get(body), "anchor_iou": initial_iou.get(body, 0.),
                     "anchor_visible": body in seen, "frames": []}
              for body, entity in zip(bodies, entities)}
    for frame, (pred, gt, time) in enumerate(zip(predicted_future, truth_future, times)):
        p_stats, g_stats = label_statistics(pred), label_statistics(gt)
        for body in bodies:
            slot = mapping.get(body)
            p_center, g_center = _center(p_stats, slot), _center(g_stats, body)
            g0, p0 = _center(source_gt, body), _center(source_pred, slot)
            visible = g_center is not None
            current = {"frame": frame, "seconds": float(time), "gt_visible": visible,
                       "prediction_missing": p_center is None if body in seen else None,
                       "gt_centroid_xy": g_center.tolist() if visible else None,
                       "predicted_centroid_xy": p_center.tolist() if p_center is not None else None,
                       "iou": None, "centroid_error_px": None, "relative_centroid_error_px": None,
                       "gt_displacement_px": None, "predicted_displacement_px": None}
            if visible and body in seen:
                matched = pred == slot if slot is not None else np.zeros_like(gt, dtype=bool)
                actual = gt == body
                current["iou"] = float(np.count_nonzero(matched & actual)/np.count_nonzero(matched | actual))
                current["centroid_error_px"] = float(np.linalg.norm(p_center-g_center)) if p_center is not None else diagonal
                if g0 is not None:
                    delta = g_center-g0
                    current["gt_displacement_px"] = float(np.linalg.norm(delta))
                    if p_center is not None and p0 is not None:
                        predicted_delta = p_center-p0
                        current["predicted_displacement_px"] = float(np.linalg.norm(predicted_delta))
                        current["relative_centroid_error_px"] = float(np.linalg.norm(predicted_delta-delta))
                    else:
                        current["predicted_displacement_px"] = diagonal
                        current["relative_centroid_error_px"] = diagonal
            output[body]["frames"].append(current)
    for row in output.values():
        frames = [f for f in row["frames"] if f["iou"] is not None]
        relative = [f for f in frames if f["relative_centroid_error_px"] is not None]
        amplitude = _rms([f["gt_displacement_px"] for f in relative])
        error = _rms([f["relative_centroid_error_px"] for f in relative])
        row["visible_frames"] = sum(f["gt_visible"] for f in row["frames"])
        row["scored_visible_frames"] = len(frames)
        row["missing_prediction_frames"] = sum(f["prediction_missing"] for f in frames)
        row["mean_visible_iou"] = float(np.mean([f["iou"] for f in frames])) if frames else None
        row["centroid_rmse_px"] = _rms([f["centroid_error_px"] for f in frames])
        row["relative_centroid_rmse_px"] = error
        row["gt_visible_centroid_rms_displacement_px"] = amplitude
        row["motion_normalized_relative_centroid_rmse"] = error/amplitude if amplitude is not None and amplitude >= 1. else None
        row["near_static_visible_centroid_drift_rms_px"] = _rms([f["predicted_displacement_px"] for f in relative]) if amplitude is not None and amplitude < 1. else None
    return {"assignment_uses": "last_observed_frame_only; no future rematching",
            "newly_visible_entities": "reported but not identity-scored without an observed anchor match",
            "trajectory_quantity": "visible_silhouette_centroid; not projected 3D body center",
            "missing_prediction_penalty_px": diagonal, "ratio_resolution_floor_px": 1.,
            "entities": list(output.values())}


def score_rgb_outcomes(prediction, truth, anchor_labels, future_labels, actor_ids, robot_ids,
                       times, alternative_rgb=None, alternative_labels=None):
    prediction, truth = _image_float(prediction), _image_float(truth)
    other = _image_float(alternative_rgb) if alternative_rgb is not None else None
    rows = []
    for frame, time in enumerate(times):
        mask = future_labels[frame]
        excluded = np.isin(mask, robot_ids)
        if alternative_labels is not None:
            excluded |= np.isin(alternative_labels[frame], robot_ids)
        regions, actor_rows = [], []
        for actor in actor_ids:
            region = (mask == actor) | (anchor_labels == actor)
            if alternative_labels is not None:
                region |= alternative_labels[frame] == actor
            region &= ~excluded
            regions.append(region)
            count = int(region.sum())
            own_error = float(abs(prediction[frame]-truth[frame])[region].mean()) if count else None
            alt_error = float(abs(prediction[frame]-other[frame])[region].mean()) if count and other is not None else None
            actor_rows.append({"body_id": int(actor), "pixels": count, "rgb_l1": own_error,
                               "opposite_order_rgb_l1": alt_error})
        region = np.logical_or.reduce(regions)
        count = int(region.sum())
        own_error = float(abs(prediction[frame]-truth[frame])[region].mean()) if count else None
        alt_error = float(abs(prediction[frame]-other[frame])[region].mean()) if count and other is not None else None
        rows.append({"frame": frame, "seconds": float(time), "actor_region_pixels": count,
                     "full_rgb_l1": float(abs(prediction[frame]-truth[frame]).mean()),
                     "actor_region_rgb_l1": own_error, "opposite_order_rgb_l1": alt_error,
                     "order_preference_margin": alt_error-own_error if alt_error is not None else None,
                     "actors": actor_rows})
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None
    return {"region_source": "union of simulator actor masks at anchor and both real futures; robot/tool pixels excluded",
            "order_measurement": "preference between two real same-scene outcomes at identical requested times",
            "not_measured": "generic task success or exact physical motion-onset classification",
            "full_rgb_l1": mean("full_rgb_l1"), "actor_region_rgb_l1": mean("actor_region_rgb_l1"),
            "order_preference_margin": mean("order_preference_margin"), "frames": rows}
