"""Held stage-one diagnosis: observed state, posterior, interventions and copies."""

from html import escape
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .codec import letterbox_video
from .data import collate_object_event_video, write_json
from .evaluation import write_video
from .independent_metrics import score_entities, score_rgb_outcomes, physical_motion_context
from .native_view import native_rgb, native_field_labels, paired_indices
from .simulation_data import load_truth
from .stage1_metrics import actor_change_records, condition_scores, summarize_stage1
from .state import coordinates
from .training import device_batch


def feature_records(features, source, target, target_source, valid, source_valid, times):
    predicted_delta = torch.diff(torch.cat((source, features), 1), dim=1)
    target_delta = torch.diff(torch.cat((target_source, target), 1), dim=1)
    paired = valid & torch.cat((source_valid, valid[:, :-1]), 1)
    rows = []
    for index in range(features.shape[1]):
        active, consecutive = valid[0, index], paired[0, index]
        if active.any():
            rows.append({"query_frame_index": index, "seconds": float(times[0, index]),
                "feature_mse": float((features[0, index]-target[0, index]).float().square()[active].mean()),
                "feature_change_l1": float((predicted_delta[0, index]-target_delta[0, index]).float()
                    .abs()[consecutive].mean()) if consecutive.any() else None})
    return rows


@torch.no_grad()
def predict_stage1(model, codec, batch, precision):
    device = next(model.parameters()).device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        result = model(batch, codec, deterministic=True, render_all=True)
        images, pixel_valid = letterbox_video(batch, model.view_size(codec))
        th = batch["history_frames"]
        times = batch["times"][:, th:]
        xy = coordinates(codec.grid_size, device)
        source_features, source_masks = model.render_features([result["source"]], xy)
        source_rgb = model.decode_features(codec, source_features)
        target_source = codec.encode(images[:, th-1:th])
        valid = codec.patch_valid(pixel_valid)
        effects = torch.stack([row["value"] for row in result["effects"]], 1)
        observed_features, _ = model.render_features(result["observed_states"][th:], xy)
        paths = {
            "posterior": (result["predicted_features"], result["posterior_rgb"], result["predicted_masks"], source_features),
            "observed_state": (observed_features, result["observed_rgb"], result["observed_masks"][:, th:], source_features),
            "codec_reference": (result["target_features"], model.decode_features(codec, result["target_features"]), None, target_source),
            "source_state_copy": (source_features.expand_as(result["target_features"]),
                source_rgb.expand_as(result["target_rgb"]),
                source_masks.expand_as(result["predicted_masks"]), source_features)}
        for name, effect in (("zero_effect", torch.zeros_like(effects)), ("reversed_effect_order", effects.flip(1))):
            states = model.rollout(result["source"], effect, times)
            features, masks = model.render_features(states, xy)
            paths[name] = features, model.decode_features(codec, features), masks, source_features
        outputs = {}
        for name, (features, rgb, masks, anchor) in paths.items():
            anchor_rgb = model.decode_features(codec, target_source) if name == "codec_reference" else source_rgb
            outputs[name] = {"rgb": rgb[0].float().clamp(0, 1).cpu(),
                "anchor": anchor_rgb[0, 0].float().clamp(0, 1).cpu(),
                "masks": masks[0].float().cpu() if masks is not None else None,
                "feature": feature_records(features, anchor, result["target_features"], target_source,
                                           valid[:, th:], valid[:, th-1:th], times)}
        outputs["rgb_copy"] = {"rgb": images[0, th-1:th].expand_as(result["target_rgb"][0]).cpu(),
                              "anchor": images[0, th-1].cpu(), "masks": None, "feature": []}
        outputs["resize_reference"] = {"rgb": result["target_rgb"][0].float().cpu(),
                                       "anchor": images[0, th-1].cpu(), "masks": None, "feature": []}
    return outputs, source_masks[0].float().cpu(), images[0].float().cpu()


def comparison_media(path, original, predictions, times, th):
    """A fixed panel order shows the same real history before every future."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    names = ("target", "posterior", "zero_effect", "observed_state", "source_state_copy",
             "rgb_copy", "reversed_effect_order", "codec_reference", "resize_reference")
    grid = []
    for frame in range(len(original)):
        canvas = Image.new("RGB", (3*256, 3*280), "white")
        draw = ImageDraw.Draw(canvas)
        for number, name in enumerate(names):
            if name == "target" or frame < th:
                picture = Image.fromarray(original[frame])
                picture.thumbnail((256, 256))
                padded = Image.new("RGB", (256, 256))
                padded.paste(picture, ((256-picture.width)//2, (256-picture.height)//2))
                picture = padded
            else:
                value = predictions[name]["rgb"][frame-th].permute(1, 2, 0).numpy()
                picture = Image.fromarray((value.clip(0, 1)*255).round().astype(np.uint8)).resize((256, 256))
            x, y = (number % 3)*256, (number//3)*280
            canvas.paste(picture, (x, y+24))
            phase = "history" if frame < th else "future"
            draw.text((x+4, y+4), f"{name}  {float(times[frame]):.1f}s {phase}", fill="black")
        grid.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()/255)
    write_video(path/"comparison.mp4", torch.stack(grid), times)
    Image.fromarray((grid[-1].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)).save(path/"terminal.jpg")


def write_review(out, cases, report):
    sections = []
    for case in cases:
        summary = {name: condition_scores(row) for name, row in case["conditions"].items()}
        lines = "".join(f"<tr><td>{escape(name)}</td><td>{row['actor_region_rgb_l1']:.5f}</td>"
                        f"<td>{row['actor_anchor_change_l1']:.5f}</td></tr>" for name, row in summary.items()
                        if row["actor_region_rgb_l1"] is not None and row["actor_anchor_change_l1"] is not None)
        media = (f'<video controls preload="none" src="cases/{case["case_id"]}/comparison.mp4"></video>'
                 if case["video_saved"] else "")
        sections.append(f'<section><h2>{escape(case["case_id"])}</h2><p>{escape(case["instruction"])}</p>'
                        f'<p>{escape(case["combination"])} / oracle {escape(case["oracle_status"])}</p>'
                        f'{media}<table><tr><th>Condition</th><th>Actor RGB L1</th><th>Actor change L1</th></tr>'
                        f'{lines}</table><a href="cases/{case["case_id"]}/metrics.json">Per-frame / per-actor results</a></section>')
    (out/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>V71 Stage 1 Held Review</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;letter-spacing:0}'
        'section{padding:20px 0;border-top:1px solid #bbb}video{width:768px;max-width:100%}'
        'table{border-collapse:collapse}td,th{padding:5px 16px;text-align:left}h2{overflow-wrap:anywhere}</style>'
        f'<h1>V71 Stage 1 / step {report["checkpoint_step"]}</h1>'
        '<p>Posterior reconstruction uses future observations. This is not history-only generation. '
        'Actor regions come from simulator truth, not model masks. RGB change is not physical flow.</p>'
        '<p><a href="report.json">Aggregate and paired case scores</a></p>'+''.join(sections)+'</html>')


@torch.no_grad()
def evaluate_stage1(model, codec, dataset, out, checkpoint_info, items=200, video_cases=12,
                    precision="bf16", resume=False):
    model.eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    selected = paired_indices(dataset, items)
    records = []
    device = next(model.parameters()).device
    with (out/"cases.jsonl").open("w") as stream:
        for number, index in enumerate(selected):
            entry = dataset.episodes[index][0]
            completed = out/"cases"/entry["case_id"]/"metrics.json"
            if resume and completed.is_file():
                case = json.loads(completed.read_text())
                records.append(case)
                stream.write(json.dumps(case, allow_nan=False)+"\n")
                stream.flush()
                print(json.dumps({"event": "stage1_held_reused", "completed": len(records),
                                  "planned": len(selected), "case_id": case["case_id"]}), flush=True)
                continue
            sample = dataset[index]
            th, hw = sample["history_frames"], sample["native_hw"].tolist()
            batch = device_batch(collate_object_event_video([sample]), device)
            meta = json.loads(Path(sample["episode_path"]).read_text())
            truth = load_truth(sample["episode_path"], sample["frame_indices"].numpy())
            rgb = sample["rgb"].permute(0, 2, 3, 1).numpy()
            valid = sample["frame_valid"].numpy()[th:]
            times = sample["times"].numpy()[th:]
            started = time.perf_counter()
            predictions, anchor_masks, _ = predict_stage1(model, codec, batch, precision)
            anchor_labels = native_field_labels(anchor_masks, hw, model.view_size(codec))[0]
            actors = sorted({event["actor_body_id"] for event in meta["program"]})
            case = {**checkpoint_info, "case_id": sample["case_id"], "scene_seed": meta["seed"], "partition": sample["partition"],
                "instruction": sample["instruction"], "combination": meta["combination"], "oracle_status": meta["status"],
                "history_frames": th, "frame_indices": sample["frame_indices"].tolist(), "times": sample["times"].tolist(),
                "scored_frames": int(valid.sum()), "terminal_frame_available": bool(valid[-1]),
                "model_image_size": model.view_size(codec), "native_hw": hw,
                "gt_physical_motion": physical_motion_context(truth["poses_xyz_xyzw"][th-1, :, :3],
                    truth["poses_xyz_xyzw"][th:, :, :3][valid], truth["body_ids"]), "conditions": {}}
            for name, output in predictions.items():
                predicted = native_rgb(output["rgb"], hw)
                anchor = native_rgb(output["anchor"][None], hw)[0]
                if name == "rgb_copy":
                    # The copy baseline is the original camera image, not a resized model input.
                    predicted = np.broadcast_to(rgb[th-1], rgb[th:].shape)
                    anchor = rgb[th-1]
                measured = {"uses_future_rgb": name in ("posterior", "observed_state", "reversed_effect_order",
                                                       "codec_reference", "resize_reference"),
                    "rgb": score_rgb_outcomes(predicted[valid], rgb[th:][valid], truth["segmentation"][th-1],
                        truth["segmentation"][th:][valid], actors, meta["robot_and_tool_body_ids"], times[valid]),
                    "change": actor_change_records(predicted, anchor, rgb[th:], rgb[th-1], truth["segmentation"][th:],
                        truth["segmentation"][th-1], actors, meta["robot_and_tool_body_ids"], times, valid,
                        bool(sample["frame_valid"][th-1])),
                    "feature": output["feature"]}
                for frame, row in zip(np.flatnonzero(valid), measured["rgb"]["frames"]):
                    row["query_frame_index"] = int(frame)
                if output["masks"] is not None and model.config.state_layout == "object":
                    fields = native_field_labels(output["masks"], hw, model.view_size(codec))
                    measured["field_ownership"] = score_entities(anchor_labels, fields[valid],
                        truth["segmentation"][th-1], truth["segmentation"][th:][valid], meta["entities"], times[valid])
                case["conditions"][name] = measured
            case["video_saved"] = number < video_cases
            case_dir = out/"cases"/sample["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            if case["video_saved"]:
                comparison_media(case_dir, rgb, predictions, sample["times"], th)
            case["seconds"] = time.perf_counter()-started
            write_json(case_dir/"metrics.json", case)
            stream.write(json.dumps(case, allow_nan=False)+"\n")
            stream.flush()
            records.append(case)
            print(json.dumps({"event": "stage1_held_case", "completed": len(records), "planned": len(selected),
                "case_id": case["case_id"], "seconds": case["seconds"],
                "posterior_actor_l1": case["conditions"]["posterior"]["rgb"]["actor_region_rgb_l1"],
                "zero_actor_l1": case["conditions"]["zero_effect"]["rgb"]["actor_region_rgb_l1"]}), flush=True)
    report = {"contract": "v71_stage1_held_change_v1", **checkpoint_info,
        "manifest": str(dataset.path), "sequence": vars(dataset.config),
        "cases": len(records), "scenes": len({row["scene_seed"] for row in records}),
        "selection": "fixed-seed scene permutation, all paired programs retained", "seed": dataset.seed,
        "truth": "lossless simulator RGB and independent body masks/poses",
        "rgb_scoring": "predictions clamped to [0,1], restored to native camera view; not the unclamped training loss",
        "feature_scoring": "normalized DINO features, valid patches, all future frames; not object ground truth",
        "field_ownership_scope": "feature-field ownership, not RGB segmentation or 3D object position",
        "stage": "posterior-conditioned reconstruction; no language prior",
        "video_cases": sum(row["video_saved"] for row in records),
        "rgb_copy_preview": "preview uses resized anchor; numerical baseline uses original camera image",
        "summary": summarize_stage1(records),
        "by_combination": {key: summarize_stage1([row for row in records if row["combination"] == key])
                           for key in sorted({row["combination"] for row in records})}}
    write_json(out/"report.json", report)
    write_review(out, records, report)
    return report
