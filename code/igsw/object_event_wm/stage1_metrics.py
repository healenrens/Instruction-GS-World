"""Case-first reconstruction and change scores on simulator-defined actor regions."""

from collections import defaultdict

import numpy as np


def average(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
            "p90": float(np.quantile(values, .9))}


def float_rgb(value):
    return value.astype(np.float32)/255 if value.dtype == np.uint8 else value.astype(np.float32)


def actor_change_records(prediction, predicted_anchor, truth, truth_anchor, labels,
                         anchor_labels, actors, robots, times, valid, anchor_valid=True):
    """Remove each path's own anchor bias; masks never come from the model."""
    prediction, predicted_anchor = float_rgb(prediction), float_rgb(predicted_anchor)
    truth, truth_anchor = float_rgb(truth), float_rgb(truth_anchor)
    rows = []
    for frame, time in enumerate(times):
        if not valid[frame]:
            continue
        previous_valid = bool(valid[frame-1]) if frame else bool(anchor_valid)
        previous_labels = labels[frame-1] if frame else anchor_labels
        excluded = np.isin(labels[frame], robots) | np.isin(anchor_labels, robots)
        actors_at_frame = []
        for actor in actors:
            region = ((labels[frame] == actor) | (anchor_labels == actor)) & ~excluded
            delta_region = (region | (previous_labels == actor)) & ~excluded
            delta_region &= ~np.isin(previous_labels, robots)
            p_delta = prediction[frame]-predicted_anchor
            t_delta = truth[frame]-truth_anchor
            p_step = prediction[frame]-(prediction[frame-1] if frame else predicted_anchor)
            t_step = truth[frame]-(truth[frame-1] if frame else truth_anchor)
            measured = bool(region.any())
            step_measured = previous_valid and bool(delta_region.any())
            actors_at_frame.append({
                "body_id": int(actor), "pixels": int(region.sum()),
                "visible_pixels": int((labels[frame] == actor).sum()),
                "rgb_l1": float(abs(prediction[frame]-truth[frame])[region].mean()) if measured else None,
                "anchor_change_l1": float(abs(p_delta-t_delta)[region].mean()) if measured and anchor_valid else None,
                "gt_change_l1": float(abs(t_delta)[region].mean()) if measured and anchor_valid else None,
                "predicted_change_l1": float(abs(p_delta)[region].mean()) if measured and anchor_valid else None,
                "step_change_l1": float(abs(p_step-t_step)[delta_region].mean()) if step_measured else None,
                "gt_step_change_l1": float(abs(t_step)[delta_region].mean()) if step_measured else None})
        rows.append({"query_frame_index": frame, "seconds": float(time), "actors": actors_at_frame})
    summaries = []
    for actor in actors:
        frames = [row for frame in rows for row in frame["actors"] if row["body_id"] == actor]
        amplitude = average([row["gt_change_l1"] for row in frames])
        error = average([row["anchor_change_l1"] for row in frames])
        end = next((row for frame in rows if frame["query_frame_index"] == len(times)-1
                    for row in frame["actors"] if row["body_id"] == actor), None)
        summaries.append({"body_id": int(actor),
            **{key: average([row[key] for row in frames]) for key in
               ("rgb_l1", "anchor_change_l1", "gt_change_l1", "predicted_change_l1",
                "step_change_l1", "gt_step_change_l1")},
            "change_normalized_error": error/amplitude if amplitude is not None and amplitude >= 1/255 else None,
            "terminal_visible": bool(end and end["visible_pixels"]),
            "terminal_rgb_l1": end["rgb_l1"] if end and end["visible_pixels"] else None})
    return {"region": "per actor: anchor/current simulator masks; previous mask also used for step change; robot excluded",
            "quantity": "RGB change, not flow or 3D trajectory; each path subtracts its own reconstructed anchor",
            "normalization_floor": 1/255, "normalization_floor_reason": "one uint8 RGB intensity level",
            "actors": summaries, "frames": rows}


def condition_scores(result):
    rows = result["rgb"]["frames"]
    changes = result["change"]["actors"]
    scores = {key: result["rgb"][key] for key in ("full_rgb_l1", "actor_region_rgb_l1")}
    scores.update({"actor_"+key: average([row[key] for row in changes]) for key in
                   ("anchor_change_l1", "step_change_l1", "change_normalized_error")})
    for low, high in ((0, 2), (2, 5), (5, 10)):
        scores[f"actor_rgb_l1_{low}_{high}s"] = average(
            [row["actor_region_rgb_l1"] for row in rows if low < row["seconds"] <= high])
    # A missing or occluded terminal target stays missing, never replaced by an earlier frame.
    scores["terminal_visible_actor_rgb_l1"] = average([row["terminal_rgb_l1"] for row in changes])
    if result.get("feature"):
        scores.update({key: average([row[key] for row in result["feature"]])
                       for key in ("feature_mse", "feature_change_l1")})
    return scores


def summarize_stage1(cases):
    """Pair within a case, then average paired programs within each scene."""
    conditions = defaultdict(lambda: defaultdict(list))
    scene_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pairs = defaultdict(list)
    for case in cases:
        values = {name: condition_scores(result) for name, result in case["conditions"].items()}
        for name, scores in values.items():
            for key, value in scores.items():
                conditions[name][key].append(value)
                scene_scores[name][key][case["scene_seed"]].append(value)
        for baseline in ("zero_effect", "reversed_effect_order", "source_state_copy", "rgb_copy"):
            for metric, posterior in values["posterior"].items():
                reference = values[baseline].get(metric)
                if posterior is not None and reference is not None:
                    pairs[(baseline, metric)].append({"case_id": case["case_id"], "scene_seed": case["scene_seed"],
                        "posterior": posterior, "baseline": reference, "gain": reference-posterior})
    comparisons = {}
    for (baseline, metric), rows in pairs.items():
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row["scene_seed"]].append(row["gain"])
        gains = [average(values) for values in by_scene.values()]
        comparisons.setdefault(baseline, {})[metric] = {
            "meaning": "baseline error minus posterior error; positive favors posterior",
            "case_gain": distribution([row["gain"] for row in rows]),
            "scene_gain": distribution(gains),
            "case_wins": sum(row["gain"] > 0 for row in rows),
            "case_ties": sum(row["gain"] == 0 for row in rows),
            "scene_wins": sum(gain > 0 for gain in gains), "cases": rows}
    return {"by_case": {name: {key: distribution(values) for key, values in metrics.items()}
                        for name, metrics in conditions.items()},
            "by_scene": {name: {key: distribution([average(v) for v in scenes.values()])
                               for key, scenes in metrics.items()} for name, metrics in scene_scores.items()},
            "paired_posterior_gain": comparisons}
