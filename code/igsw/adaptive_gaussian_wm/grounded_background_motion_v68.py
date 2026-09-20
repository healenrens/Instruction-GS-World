"""Image-plane background registration and independent relay measurements."""

import cv2
import numpy as np
import torch

from .tracker_visual_review_v67 import predict


def reference_queries(case, side):
    height, width = case["height"], case["width"]
    yy, xx = np.meshgrid((np.arange(side) + .5) * height / side,
                         (np.arange(side) + .5) * width / side, indexing="ij")
    xy = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), -1)).float()
    return {"xy": xy, "frames": torch.full((len(xy),), case["first_frame"], dtype=torch.long),
            "labels": ["background_reference" for _ in range(len(xy))]}


def project(xy, matrix):
    homogeneous = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), -1) @ matrix.T
    denominator = homogeneous[..., 2:]
    return np.divide(homogeneous[..., :2], denominator,
                     out=np.full_like(homogeneous[..., :2], np.nan), where=np.abs(denominator) > 1e-8)


def fit_background(reference, case, args):
    coordinates = reference["tracks"].numpy().astype(np.float64)
    valid = (reference["visibility"] & reference["in_bounds"]).numpy() & np.isfinite(coordinates).all(-1)
    source = coordinates[0]
    frame_count, point_count = coordinates.shape[:2]
    inverse = np.tile(np.eye(3), (frame_count, 1, 1))
    usable = np.zeros(frame_count, bool)
    residual = np.zeros((frame_count, point_count), np.float32)
    inlier = np.zeros((frame_count, point_count), bool)
    rows = []
    cv2.setRNGSeed(args.seed)
    for frame in range(frame_count):
        ids = np.flatnonzero(valid[0] & valid[frame])
        matrix, flags = (None, None)
        if len(ids) >= 8:
            matrix, flags = cv2.findHomography(source[ids], coordinates[frame, ids], cv2.RANSAC,
                                              args.background_ransac_px, maxIters=1000, confidence=.995)
        accepted = False
        cells = 0
        fraction = 0.0
        if matrix is not None and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-10:
            kept = ids[flags[:, 0].astype(bool)]
            fraction = len(kept) / len(ids)
            grid = np.floor(source[kept] / [case["width"], case["height"]] * 4).astype(int)
            cells = len(set(map(tuple, grid.tolist())))
            accepted = fraction >= .55 and cells >= 6
            if accepted:
                inverse[frame] = np.linalg.inv(matrix)
                usable[frame] = True
                error = np.linalg.norm(project(coordinates[frame], inverse[frame]) - source, axis=-1)
                residual[frame] = np.where(valid[frame] & valid[0], error, 0)
                inlier[frame, kept] = True
        rows.append({"frame": frame, "usable": accepted, "inlier_fraction": fraction, "covered_cells_of_16": cells})
    noise = np.zeros(frame_count, np.float32)
    for frame in range(frame_count):
        if inlier[frame].any():
            noise[frame] = np.quantile(residual[frame, inlier[frame]], .9)
    return {"inverse_homography": torch.from_numpy(inverse).float(), "valid": torch.from_numpy(usable),
            "inlier": torch.from_numpy(inlier), "fit_error_p90_px": torch.from_numpy(noise),
            "frames": rows, "reference": reference,
            "meaning": "dominant image-plane motion, not camera pose or certified static background"}


def motion_evidence(prediction, background, args):
    coordinates = prediction["tracks"].numpy().astype(np.float64)
    valid = (prediction["visibility"] & prediction["in_bounds"]).numpy() & np.isfinite(coordinates).all(-1)
    valid &= background["valid"].numpy()[:, None]
    compensated = np.stack([project(xy, matrix) for xy, matrix in zip(coordinates, background["inverse_homography"].numpy())])
    valid &= np.isfinite(compensated).all(-1)
    valid[prediction["query_local_frames"].numpy(), np.arange(coordinates.shape[1])] = False
    spans, moving, thresholds, jitter, raw_span = [], [], [], [], []
    for point in range(coordinates.shape[1]):
        good = valid[:, point]
        triple = good[2:] & good[1:-1] & good[:-2]
        second = np.diff(compensated[:, point], n=2, axis=0)
        scale = float(np.median(np.linalg.norm(second[triple], axis=-1)) / np.sqrt(6)) if triple.any() else 0.0
        noise = float(np.quantile(background["fit_error_p90_px"].numpy()[good], .9)) if good.any() else 0.0
        threshold = max(args.motion_floor_pixels, args.motion_noise_multiplier * scale, 2 * noise)
        span, raw = 0.0, 0.0
        if good.sum() >= args.minimum_visible_frames:
            span = float(np.linalg.norm(np.diff(np.quantile(compensated[good, point], [.05, .95], axis=0), axis=0)))
            raw = float(np.linalg.norm(np.diff(np.quantile(coordinates[good, point], [.05, .95], axis=0), axis=0)))
        spans.append(span)
        raw_span.append(raw)
        thresholds.append(threshold)
        jitter.append(scale)
        moving.append(bool(good.sum() >= args.minimum_visible_frames and span >= threshold))
    return {"compensated_coordinates": torch.from_numpy(np.nan_to_num(compensated)).float(),
            "valid": torch.from_numpy(valid), "moving": torch.tensor(moving),
            "span_px": torch.tensor(spans), "raw_span_px": torch.tensor(raw_span),
            "threshold_px": torch.tensor(thresholds), "second_difference_px": torch.tensor(jitter)}


def relay_evidence(model, rgb, indices, queries, primary, device, args):
    valid = primary["visibility"] & primary["in_bounds"] & torch.isfinite(primary["tracks"]).all(-1)
    frame_count, point_count = valid.shape
    distance = (torch.arange(frame_count)[:, None] - frame_count // 2).abs().expand(-1, point_count).clone()
    allowed = valid.clone()
    allowed[primary["query_local_frames"], torch.arange(point_count)] = False
    distance[~allowed] = frame_count + 1
    frames = distance.argmin(0)
    has_anchor = allowed.any(0)
    # Missing observations remain unknown; their supplied original query is only a placeholder.
    frames = torch.where(has_anchor, frames, primary["query_local_frames"])
    xy = primary["tracks"][frames, torch.arange(point_count)].clone()
    xy[~has_anchor] = queries["xy"][~has_anchor]
    relay = predict(model, rgb, indices, {"xy": xy, "frames": indices[frames]}, device, args.points_per_pass)
    joint = valid & relay["visibility"] & relay["in_bounds"] & has_anchor[None]
    joint[frames, torch.arange(point_count)] = False
    joint[primary["query_local_frames"], torch.arange(point_count)] = False
    error = (primary["tracks"] - relay["tracks"]).norm(dim=-1)
    joint &= torch.isfinite(error)
    reliable = joint & (error <= args.relay_max_error_px)
    enough = joint.sum(0) >= args.minimum_visible_frames
    consistent = enough & (reliable.sum(0).float() / joint.sum(0).clamp_min(1) >= .7)
    return {"prediction": relay, "anchor_frames": frames, "joint": joint, "error_px": error,
            "valid": reliable, "consistent": consistent,
            "meaning": "agreement with independent temporal requery, not ground-truth accuracy"}
