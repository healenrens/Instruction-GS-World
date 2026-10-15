#!/usr/bin/env python3
"""Measure V71 outputs against simulator entities and paired real event orders."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.codec import FrozenVisualCodec
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import write_video
from igsw.object_event_wm.independent_metrics import score_entities, score_rgb_outcomes, physical_motion_context
from igsw.object_event_wm.paired_event_summary import summarize_paired_events
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.prediction import generate_from_history, history_batch, observe_history, make_history_request
from igsw.object_event_wm.prior_training import load_prior_model
from igsw.object_event_wm.simulation_data import load_episode, load_truth, nearest_indices
from igsw.object_event_wm.state import coordinates
from igsw.object_event_wm.text import TextFeatureStore
from igsw.object_event_wm.training import device_batch, read_checkpoint


def native_rgb(frames, hw):
    h, w = map(int, hw)
    size = frames.shape[-1]
    rh, rw = round(h*size/max(h, w)), round(w*size/max(h, w))
    top, left = (size-rh)//2, (size-rw)//2
    cropped = frames[:, :, top:top+rh, left:left+rw]
    value = F.interpolate(cropped.float(), (h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    return value.permute(0, 2, 3, 1).cpu().numpy()


def native_field_labels(masks, hw, image_size):
    h, w = map(int, hw)
    rh, rw = round(h*image_size/max(h, w)), round(w*image_size/max(h, w))
    top, left = (image_size-rh)//2, (image_size-rw)//2
    side = round(masks.shape[-1]**.5)
    labels = []
    for frame in masks:
        raster = F.interpolate(frame[None].reshape(1, len(frame), side, side).float(),
                               (image_size, image_size), mode="bilinear", align_corners=False)
        raster = F.interpolate(raster[:, :, top:top+rh, left:left+rw], (h, w), mode="bilinear", align_corners=False)
        labels.append(raster.argmax(1)[0].cpu().numpy().astype(np.int16))
    return np.stack(labels)


def paired_indices(dataset, items):
    scenes = defaultdict(list)
    for index, entries in enumerate(dataset.episodes):
        scenes[entries[0]["scene_seed"]].append(index)
    rng = np.random.default_rng(dataset.seed)
    indices = []
    for scene in rng.permutation(sorted(scenes)):
        if len(indices)+len(scenes[scene]) > items:
            break
        indices.extend(scenes[scene])
    return indices


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prior-checkpoint", default="")
    parser.add_argument("--text-cache", default="")
    parser.add_argument("--partition", default="engineering")
    parser.add_argument("--items", type=int, default=16, help="Maximum cases, keeping each selected scene's programs together")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device, prior, text_store = torch.device("cuda"), None, None
    if args.prior_checkpoint:
        model, prior, saved = load_prior_model(args.prior_checkpoint, device)
        text_store = TextFeatureStore(args.text_cache or saved["args"]["text_cache"])
    else:
        saved = read_checkpoint(args.checkpoint)
        model = ObjectEventWorldModel(ModelConfig(**saved["model_config"])).to(device).eval()
        model.load_state_dict(saved["model"])
    codec = FrozenVisualCodec(args.assets).to(device).eval()
    data = build_video_dataset(args.manifest, partition=args.partition)
    groups = defaultdict(list)
    for entry in data.manifest["entries"]:
        if entry["partition"] == args.partition:
            groups[entry["scene_seed"]].append(entry)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, paired_cases = [], []
    with (out/"cases.jsonl").open("w") as stream:
        for index in paired_indices(data, args.items):
            sample = data[index]
            th, hw = sample["history_frames"], sample["native_hw"].tolist()
            batch = device_batch(collate_object_event_video([sample]), device)
            meta, raw_rgb, recorded = load_episode(sample["episode_path"])
            truth = load_truth(sample["episode_path"], sample["frame_indices"].numpy())
            desired = sample["times"].numpy().astype(np.float64)
            actual_rgb = raw_rgb[sample["frame_indices"].numpy()]
            alternative = next((entry for entry in groups[meta["seed"]] if entry["case_id"] != sample["case_id"]
                                and entry["status"] == meta["status"] == "completed"), None)
            alternative_rgb, alternative_truth = None, None
            valid = sample["frame_valid"].numpy()[th:].copy()
            other_meta = None
            if alternative is not None:
                other_meta, other_rgb, other_times = load_episode(alternative["episode"])
                other_indices = nearest_indices(other_times, desired)
                valid &= np.isclose(other_times[other_indices][th:], desired[th:], atol=.5/meta["physics_hz"], rtol=0)
                alternative_rgb = other_rgb[other_indices][th:][valid]
                alternative_truth = load_truth(alternative["episode"], other_indices)["segmentation"][th:][valid]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch, codec, deterministic=True, render_all=True)
                xy = coordinates(codec.grid_size, device)
                effects = torch.stack([effect["value"] for effect in result["effects"]], 1)
                future_times = batch["times"][:, th:]
                zero_states = model.rollout(result["source"], torch.zeros_like(effects), future_times)
                zero_features, zero_masks = model.render_features(zero_states, xy)
                zero_rgb = codec.decode(zero_features)
                source_features, source_masks = model.render_features([result["source"]], xy)
                source_rgb = codec.decode(source_features)
                conditions = {"posterior": (result["posterior_rgb"][0], result["predicted_masks"][0]),
                              "zero_effect": (zero_rgb[0], zero_masks[0]),
                              "source_state_copy": (source_rgb[0].expand(len(effects[0]), -1, -1, -1),
                                                    source_masks[0].expand(len(effects[0]), -1, -1))}
                if prior is not None and meta["status"] == "completed":
                    history = observe_history(model, codec, history_batch(batch))
                    instruction_options = {"correct_language": sample["instruction"], "no_language": ""}
                    if other_meta is not None:
                        instruction_options["opposite_order_language"] = other_meta["instruction"]
                    for name, instruction in instruction_options.items():
                        text, text_valid = text_store.batch([instruction], device) if instruction else (None, None)
                        generated = generate_from_history(model, codec, prior, history, future_times, text, text_valid,
                                    generator=torch.Generator(device=device).manual_seed(data.seed+meta["seed"]))
                        conditions[name] = (generated["rgb"][0], generated["masks"][0])
            anchor_labels = native_field_labels(source_masks[0], hw, codec.image_size)[0]
            gt_anchor = truth["segmentation"][th-1]
            labels = truth["segmentation"][th:][valid]
            target = actual_rgb[th:][valid]
            times = desired[th:][valid]
            actors = sorted({event["actor_body_id"] for event in meta["program"]})
            case = {"case_id": sample["case_id"], "scene_seed": meta["seed"], "partition": args.partition,
                    "combination": meta["combination"],
                    "instruction": sample["instruction"], "events": meta["events"], "program": meta["program"],
                    "oracle_status": meta["status"],
                    "physics_hz": meta["physics_hz"], "history_seconds": meta["history_seconds"],
                    "alternative_instruction": other_meta["instruction"] if other_meta else None,
                    "prediction_horizon_seconds": data.config.future_seconds, "scored_frames": int(valid.sum()),
                    "terminal_target_available": bool(valid[-1]),
                    "state_layout": model.config.state_layout,
                    "field_labels_meaning": ("upsampled feature-field ownership, not an RGB segmentation head"
                        if model.config.state_layout == "object" else "single global field; no entity labels"),
                    "gt_physical_motion": physical_motion_context(truth["poses_xyz_xyzw"][th-1, :, :3],
                        truth["poses_xyz_xyzw"][th:, :, :3][valid], truth["body_ids"]),
                    "conditions": {}}
            case_dir = out/sample["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(case_dir/"history_request.json", make_history_request(sample))
            for name, (predicted, masks) in conditions.items():
                rgb = native_rgb(predicted, hw)
                predicted_labels = native_field_labels(masks, hw, codec.image_size)
                reversed_instruction = name == "opposite_order_language"
                reference_rgb = alternative_rgb if reversed_instruction else target
                reference_labels = alternative_truth if reversed_instruction else labels
                other_rgb = target if reversed_instruction else alternative_rgb
                other_labels = labels if reversed_instruction else alternative_truth
                case["conditions"][name] = {
                    "reference_instruction": other_meta["instruction"] if reversed_instruction else sample["instruction"],
                    "reference_combination": other_meta["combination"] if reversed_instruction else meta["combination"],
                    "condition_is_ambiguous": name == "no_language",
                    "rgb": score_rgb_outcomes(rgb[valid], reference_rgb, gt_anchor, reference_labels, actors,
                        meta["robot_and_tool_body_ids"], times, other_rgb, other_labels),
                    "field_ownership": (score_entities(anchor_labels, predicted_labels[valid], gt_anchor,
                                                       reference_labels, meta["entities"], times)
                        if model.config.state_layout == "object" else
                        {"available": False, "reason": "scene control has no independently composited entity fields"})}
                np.savez_compressed(case_dir/f"{name}.npz", rgb=rgb, field_labels=predicted_labels,
                                    anchor_field_labels=anchor_labels, times_seconds=desired[th:])
                preview = torch.from_numpy(np.concatenate((actual_rgb[:th].astype(np.float32)/255, rgb))).permute(0, 3, 1, 2)
                write_video(case_dir/f"{name}.mp4", preview, sample["times"])
            case["conditions"]["current_image_copy"] = {"reference_combination": meta["combination"], "rgb": score_rgb_outcomes(
                np.broadcast_to(actual_rgb[th-1], target.shape), target, gt_anchor, labels,
                actors, meta["robot_and_tool_body_ids"], times, alternative_rgb, alternative_truth)}
            stream.write(json.dumps(case, allow_nan=False)+"\n")
            stream.flush()
            paired_cases.append({"case_id": case["case_id"], "scene_seed": case["scene_seed"],
                                 "instruction": case["instruction"], "combination": case["combination"],
                                 "oracle_status": case["oracle_status"],
                                 "conditions": {name: {"rgb": result["rgb"],
                                     "reference_combination": result["reference_combination"],
                                     "reference_instruction": result.get("reference_instruction", case["instruction"])}
                                     for name, result in case["conditions"].items()}})
            summaries.append({"case_id": case["case_id"], "scene_seed": meta["seed"],
                              "combination": meta["combination"], "oracle_status": meta["status"],
                              "conditions": {name: {key: result["rgb"][key] for key in
                                  ("full_rgb_l1", "actor_region_rgb_l1", "order_preference_margin")}
                                  for name, result in case["conditions"].items()}})
            print(json.dumps(summaries[-1]), flush=True)
    write_json(out/"report.json", {"manifest": args.manifest, "checkpoint": args.checkpoint,
                "prior_checkpoint": args.prior_checkpoint, "partition": args.partition, "cases": summaries,
                "state_layout": model.config.state_layout,
                "paired_outcome_preference": summarize_paired_events(paired_cases),
                "truth_origin": "CLIPort simulator; never model mask or tracker",
                "posterior_uses_future": True, "prior_uses_future_rgb": False,
                "motion_metric_scope": "visible silhouette centroid, not 3D physical pose",
                "order_metric_scope": "RGB preference between two actual same-scene outcomes, not task success"})


if __name__ == "__main__":
    main()
