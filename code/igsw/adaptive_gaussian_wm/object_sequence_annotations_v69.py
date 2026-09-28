"""Independent sparse pixel observations; no instance label is inferred from a tracker."""

import torch


def annotation_template_v69(sample):
    return {"contract": "object_video_sparse_truth_v1", "case_id": sample["case_id"],
            "provenance": "human_annotation", "uses_training_tracker": False,
            "sample_index": sample["sample_index"], "epoch": sample["epoch"], "occurrence": sample["occurrence"],
            "frame_indices": sample["frame_indices"].tolist(), "native_hw": sample["native_hw"].tolist(),
            "case_tags": [], "suggested_case_tags": ["rigid_translation", "rigid_rotation", "similar_appearance", "separated_different_motion", "contact", "static"],
            "queries": [], "tracks": [],
            "query_fields": {"object_id": "human entity name; multiple queries may share one entity",
                             "region_id": "optional independently named visible region, e.g. cap or body",
                             "frame_index": "observed absolute frame index", "xy_px": "native x,y"},
            "track_fields": {"object_id": "human entity name or background", "observations":
                             "list of {frame_index, xy_px:[x,y] or null, visible:true/false/null}",
                             "object_extent_px": "optional human-measured longest object extent in current image; null when unknown",
                             "region_id": "optional region name within the human-labeled entity",
                             "measurement_set": "primary or additional; neither enters Object Memory or Posterior",
                             "tracker_point_id": "optional original point ID, only for tracker calibration; never copy its predicted future coordinates as GT"},
            "unfilled_is_not_a_negative_label": True}


def independent_measurements_v69(batch, annotation):
    """One annotated clip. External points replace teacher measurements, never Student image inputs."""
    native_hw = batch["native_hw"][0]
    scale = (native_hw[[1, 0]].float()-1).clamp_min(1)
    frame_indices = batch["frame_indices"][0].tolist()
    lookup = {frame: i for i, frame in enumerate(frame_indices)}
    device = native_hw.device
    th, total, p = batch["history_frames"], len(frame_indices), len(annotation["tracks"])
    xy = torch.zeros((1, total, p, 2), device=device)
    valid = torch.zeros((1, total, p), device=device, dtype=torch.bool)
    observation = torch.full((1, total, p), -1, device=device, dtype=torch.int8)
    for point, track in enumerate(annotation["tracks"]):
        for row in track["observations"]:
            if row["frame_index"] not in lookup:
                continue
            frame = lookup[row["frame_index"]]
            if row["visible"] is not None:
                observation[0, frame, point] = int(row["visible"])
            if row["visible"] is True and row["xy_px"] is not None:
                xy[0, frame, point] = torch.tensor(row["xy_px"], device=device)/scale*2-1
                valid[0, frame, point] = True
    history_index = torch.arange(th, device=device)[:, None].expand(th, p)
    references = torch.where(valid[0, :th], history_index, -1).max(0).values
    present = references >= 0
    reference_xy = xy[:, references.clamp_min(0), torch.arange(p, device=device)]
    teacher = {"xy": xy, "valid": valid, "observation": observation,
               "reference_index": references.clamp_min(0)[None], "reference_xy": reference_xy,
               "point_present": present[None], "point_ids": torch.arange(p, device=device)[None],
               "transport_weight": present[None].float(), "relative_xy": torch.zeros_like(xy), "relative_valid": torch.zeros_like(valid)}
    historical_frames = set(frame_indices[:th])
    query_rows = [row for row in annotation["queries"] if row["frame_index"] in historical_frames]
    queries = {"xy": torch.tensor([row["xy_px"] for row in query_rows], device=device, dtype=torch.float32)[None]/scale*2-1,
               "frame_index": torch.tensor([[lookup[row["frame_index"]] for row in query_rows]], device=device),
               "valid": torch.ones((1, len(query_rows)), device=device, dtype=torch.bool)}
    labels = {"query_objects": [row["object_id"] for row in query_rows],
              "point_objects": [row["object_id"] for row in annotation["tracks"]],
              "query_regions": [row.get("region_id") or "unspecified" for row in query_rows],
              "point_regions": [row.get("region_id") or "unspecified" for row in annotation["tracks"]],
              "measurement_sets": [row.get("measurement_set") or "primary" for row in annotation["tracks"]],
              "case_tags": annotation.get("case_tags") or [],
              "object_extent_px": [row.get("object_extent_px") for row in annotation["tracks"]]}
    return {**batch, "teacher": teacher}, queries, labels
