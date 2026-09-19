"""Display-only motion selection on saved tracks; no claim of tracking accuracy."""

import math

import torch


def select_moving_tracks(
    native, height, width, minimum_pixels, minimum_fraction, minimum_frames
):
    threshold = max(minimum_pixels, minimum_fraction * min(height, width))
    coordinates = native["tracks"].float()
    valid = (
        native["visibility"] & native["in_bounds"] & torch.isfinite(coordinates).all(-1)
    )
    valid = valid.clone()
    valid[native["query_local_frames"], torch.arange(coordinates.shape[1])] = False
    kept, measurements = [], []
    for point in range(coordinates.shape[1]):
        positions = coordinates[:, point][valid[:, point]]
        span = None
        if len(positions) >= minimum_frames:
            low, high = torch.quantile(positions, torch.tensor([0.05, 0.95]), dim=0)
            span = float((high - low).norm())
        selected = span is not None and math.isfinite(span) and span >= threshold
        if selected:
            kept.append(point)
        measurements.append(
            {
                "point_id": point,
                "visible_in_bounds_nonquery_frames": len(positions),
                "position_q05_q95_box_diagonal_px": span,
                "shown": selected,
            }
        )
    return torch.tensor(kept, dtype=torch.long), {
        "method": "native_visible_position_q05_q95_box_diagonal",
        "threshold_px": threshold,
        "minimum_pixels": minimum_pixels,
        "minimum_short_side_fraction": minimum_fraction,
        "minimum_valid_frames": minimum_frames,
        "raw_point_count": coordinates.shape[1],
        "shown_point_count": len(kept),
        "selected_point_ids": kept,
        "points": measurements,
        "scope": "display only; image-space motion, not camera-compensated object motion or accuracy",
        "comparison": "same native-selected IDs shown in both native and sampled panels",
    }


def subset_tracks(prediction, point_ids):
    return {
        "tracks": prediction["tracks"][:, point_ids],
        "visibility": prediction["visibility"][:, point_ids],
        "in_bounds": prediction["in_bounds"][:, point_ids],
        "query_local_frames": prediction["query_local_frames"][point_ids],
        "frame_indices": prediction["frame_indices"],
        "point_ids": point_ids,
    }
