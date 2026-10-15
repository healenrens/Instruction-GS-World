"""Score native-view prediction files against the same lossless simulator truth."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from .data import SequenceConfig, write_json
from .evaluation import image_records, summarize_cases
from .independent_metrics import score_rgb_outcomes
from .paired_event_summary import summarize_paired_events
from .simulation_data import load_episode, load_truth, nearest_indices, requested_times


FUTURE_INPUT = {"posterior": True, "zero_effect": False, "source_state_copy": False,
                "correct_language": False, "no_language": False, "opposite_order_language": False,
                "current_image_copy": False, "oracle": True, "opposite_order": True,
                "current_copy": False, "identity_swap": True, "two_outcome_rgb_mean": True,
                "resize_reference": True}


def common_view(rgb, max_edge):
    values = rgb.astype(np.float32)
    if rgb.dtype == np.uint8:
        values /= 255.
    tensor = torch.from_numpy(values).permute(0, 3, 1, 2)
    h, w = tensor.shape[-2:]
    size = (round(h*max_edge/max(h, w)), round(w*max_edge/max(h, w)))
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)


def recorded_case(entry, config):
    meta, raw, recorded = load_episode(entry["episode"])
    desired = requested_times(config)
    indices = nearest_indices(recorded, desired)
    return {"entry": entry, "meta": meta, "rgb": raw[indices],
            "truth": load_truth(entry["episode"], indices), "times": desired,
            "valid": np.isclose(recorded[indices], desired, atol=.5/meta["physics_hz"], rtol=0)}


def sample_error_summary(cases):
    output = []
    for case in cases:
        modes = defaultdict(list)
        for name, result in case["conditions"].items():
            if result["sample_index"] is not None and result["rgb"]["actor_region_rgb_l1"] is not None:
                modes[result["mode"]].append({"condition": name, "sample_index": result["sample_index"],
                    "actor_rgb_l1": result["rgb"]["actor_region_rgb_l1"]})
        for mode, values in modes.items():
            output.append({"case_id": case["case_id"], "mode": mode, "samples": values,
                "sample_0_actor_rgb_l1": next((row["actor_rgb_l1"] for row in values if row["sample_index"] == 0), None),
                "expected_actor_rgb_l1": mean(row["actor_rgb_l1"] for row in values),
                "oracle_best_of_n_actor_rgb_l1": min(row["actor_rgb_l1"] for row in values),
                "oracle_is_not_deployment_precision": True})
    return output


@torch.no_grad()
def score_prediction_directory(manifest_path, prediction_root, output, max_edge=128, model_image_size=None):
    manifest = json.loads(Path(manifest_path).read_text())
    config = SequenceConfig(**manifest["sequence"])
    entries = {entry["case_id"]: entry for entry in manifest["entries"]}
    groups = defaultdict(list)
    for entry in entries.values():
        groups[(entry["partition"], entry["scene_seed"])].append(entry)
    prediction_root, output = Path(prediction_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [json.loads(line) for line in (prediction_root/"cases.jsonl").read_text().splitlines() if line.strip()]
    records, quality_cases, unscored = [], [], []
    th = config.history_frames
    with (output/"cases.jsonl").open("w") as stream:
        for source in inputs:
            entry = entries[source["case_id"]]
            original = recorded_case(entry, config)
            meta, desired = original["meta"], original["times"][th:]
            alternative_entry = next((other for other in groups[(entry["partition"], entry["scene_seed"])]
                if other["case_id"] != entry["case_id"] and other["status"] == entry["status"] == "completed"), None)
            alternative = recorded_case(alternative_entry, config) if alternative_entry is not None else None
            current = common_view(original["rgb"][th-1:th], max_edge)[0]
            actors = sorted({event["actor_body_id"] for event in meta["program"]})
            case = {"case_id": entry["case_id"], "scene_seed": entry["scene_seed"], "partition": entry["partition"],
                    "instruction": meta["instruction"], "combination": meta["combination"], "oracle_status": meta["status"],
                    "model_image_size": source.get("model_image_size", model_image_size),
                    "native_hw": list(original["rgb"].shape[1:3]), "common_max_edge": max_edge,
                    "conditions": {}, "unscored_conditions": []}
            quality = {"case_id": entry["case_id"], "conditions": {}}
            for name, source_result in source["conditions"].items():
                mode = source_result.get("mode", name)
                reads_future = FUTURE_INPUT[mode]
                opposite = mode == "opposite_order_language"
                reference, other = (alternative, original) if opposite else (original, alternative)
                if reference is None:
                    missing = {"case_id": entry["case_id"], "condition": name, "reason": "no realized opposite-order target"}
                    case["unscored_conditions"].append(missing)
                    unscored.append(missing)
                    continue
                if mode in ("current_copy", "current_image_copy"):
                    prediction = np.broadcast_to(original["rgb"][th-1], original["rgb"][th:].shape)
                    prediction_times = desired
                else:
                    path = prediction_root/source_result.get("file", f"{entry['case_id']}/{name}.npz")
                    with np.load(path, allow_pickle=False) as saved:
                        prediction, prediction_times = saved["rgb"], saved["times_seconds"]
                time_error = float(np.max(abs(prediction_times-desired)))
                if not np.allclose(prediction_times, desired, atol=.5/meta["physics_hz"], rtol=0):
                    missing = {"case_id": entry["case_id"], "condition": name,
                               "reason": "prediction times differ from the requested grid", "max_time_error_seconds": time_error}
                    case["unscored_conditions"].append(missing)
                    unscored.append(missing)
                    continue
                valid = reference["valid"][th:]
                reference_rgb = reference["rgb"][th:]
                other_rgb = other["rgb"][th:][valid] if other is not None else None
                other_labels = other["truth"]["segmentation"][th:][valid] if other is not None else None
                other_valid = other["valid"][th:][valid] if other is not None else None
                native = score_rgb_outcomes(prediction[valid], reference_rgb[valid], original["truth"]["segmentation"][th-1],
                    reference["truth"]["segmentation"][th:][valid], actors, meta["robot_and_tool_body_ids"], desired[valid],
                    other_rgb, other_labels, other_valid)
                for index, row in zip(np.flatnonzero(valid), native["frames"]):
                    row["query_frame_index"] = int(index)
                target_common, predicted_common = common_view(reference_rgb, max_edge), common_view(prediction, max_edge)
                mask = torch.from_numpy(valid)[:, None, None, None].expand(-1, 1, *target_common.shape[-2:])
                frames = image_records(predicted_common, target_common, mask, current, torch.from_numpy(desired))
                stochastic = mode in ("correct_language", "no_language", "opposite_order_language")
                case["conditions"][name] = {"mode": mode, "sample_index": source_result.get("sample_index", 0 if stochastic else None),
                    "seed": source_result.get("seed"), "uses_future_rgb": reads_future,
                    "model_image_size": source_result.get("model_image_size", case["model_image_size"]),
                    "condition_is_ambiguous": mode == "no_language", "reference_case_id": reference["entry"]["case_id"],
                    "reference_instruction": reference["meta"]["instruction"], "reference_combination": reference["meta"]["combination"],
                    "requested_frames": len(desired), "scored_reference_frames": int(valid.sum()),
                    "terminal_target_available": bool(valid[-1]), "prediction_time_max_difference_seconds": time_error,
                    "rgb": native, "common_image_quality": frames}
                quality["conditions"][name] = frames
            records.append(case)
            quality_cases.append(quality)
            stream.write(json.dumps(case, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps({"event": "common_scored", "case_id": case["case_id"], "conditions": len(case["conditions"])}), flush=True)
    report = {"contract": "cliport_common_native_view_v1", "manifest": str(Path(manifest_path).resolve()),
              "predictions": str(prediction_root.resolve()), "cases": len(records),
              "case_ids": [case["case_id"] for case in records], "sequence": manifest["sequence"],
              "common_max_edge": max_edge, "resampling": "native-view RGB to common size; bilinear antialias, align_corners=false",
              "ground_truth": "same original lossless RGB, not each model's resized training target",
              "actor_regions": "native simulator masks, no model or tracker mask",
              "common_image_quality": summarize_cases(quality_cases), "paired_outcome_preference": summarize_paired_events(records),
              "sample_errors": sample_error_summary(records), "unscored_conditions": unscored,
              "scope": "common-view image quality and paired real-outcome preference; not policy success",
              "not_equalized": ["model input resolution", "pretrained assets", "training compute"]}
    write_json(output/"report.json", report)
    return report
