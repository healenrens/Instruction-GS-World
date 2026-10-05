#!/usr/bin/env python3
"""Local paired evaluation through one frozen Dynamics; no remote media uploads."""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from igsw.adaptive_gaussian_wm.frozen_object_teacher_v70 import (
    history_context, perception_from_checkpoint, query_from_batch,
)
from igsw.adaptive_gaussian_wm.language_effect_dataset_v70 import LanguageEffectDatasetV70, collate_language_effect_v70
from igsw.adaptive_gaussian_wm.language_effect_evaluation_v70 import (
    conflict_target_measurements, summarize_records, trajectory_records,
)
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import (
    ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69,
)
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import sample_perception_v69
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69


def diagnostic_indices(entries, limit, seed):
    rng = random.Random(seed)
    sources = defaultdict(lambda: defaultdict(list))
    for index, entry in enumerate(entries):
        sources[entry["source"]][(entry["group"], entry["episode_index"])].append(index)
    queues = {}
    for source, episodes in sorted(sources.items()):
        queue = [rng.choice(windows) for windows in episodes.values()]
        rng.shuffle(queue)
        queues[source] = queue
    selected = []
    while len(selected) < limit and any(queues.values()):
        for queue in queues.values():
            if queue and len(selected) < limit:
                selected.append(queue.pop())
    return selected


def save_video(path, rgb, prediction, truth, valid, history_frames):
    h, w = rgb.shape[-2:]
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=5)
    stream.width, stream.height, stream.pix_fmt = (w+1)//2*2, (h+1)//2*2, "yuv420p"
    scale = torch.tensor([w-1, h-1])/2
    predicted, actual = (prediction.float().cpu()+1)*scale, (truth.float().cpu()+1)*scale
    for frame in range(len(rgb)):
        image = Image.fromarray(rgb[frame].permute(1, 2, 0).cpu().numpy())
        draw = ImageDraw.Draw(image)
        if frame >= history_frames:
            local = frame-history_frames
            for point in torch.where(valid[local].cpu())[0].tolist():
                x, y = predicted[local, point].tolist()
                gx, gy = actual[local, point].tolist()
                draw.ellipse((x-2, y-2, x+2, y+2), fill="red")
                draw.ellipse((gx-2, gy-2, gx+2, gy+2), fill="lime")
        draw.text((8, 8), "Prediction: red; tracker measurement: green", fill="white", stroke_width=1, stroke_fill="black")
        padded = Image.new("RGB", (stream.width, stream.height))
        padded.paste(image)
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(padded), format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", default="diagnostic")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--videos", type=int, default=24)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--frame_batch", type=int, default=8)
    parser.add_argument("--motion_floor_px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conflicts", default="")
    args = parser.parse_args()
    from igsw.adaptive_gaussian_wm.v70_checkpoint import load_inference_model_v70
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LanguageEffectDatasetV70(args.manifest, partition=args.partition)
    manifest = json.loads(Path(args.manifest).read_text())
    saved = torch.load(manifest["teacher_checkpoint"], map_location="cpu", weights_only=False)
    config = ObjectVideoConfigV69(**saved["config"])
    teacher = ObjectVideoWorldModelV69(config, stage="dynamics").requires_grad_(False).eval()
    teacher.load_state_dict(saved["model"])
    perception = perception_from_checkpoint(saved, config, args.frame_batch).to(device)
    teacher.to(device)
    del saved
    model, metadata = load_inference_model_v70(args.checkpoint, device)
    full_datasets = {}
    for split in ("held", "train"):
        full_datasets[split] = ObjectVideoSequenceDatasetV69(manifest["stage2_manifest"], config, args.seed, split)
    entry_lookup = {entry["case_id"]: (data, index) for data in full_datasets.values() for index, entry in enumerate(data.entries)}
    conflicts = {row["window_id"]: row for row in map(json.loads, Path(args.conflicts).read_text().splitlines())} if args.conflicts else {}
    all_rows, cases = [], []
    for index, dataset_index in enumerate(diagnostic_indices(dataset.entries, args.limit, args.seed)):
        entry = dataset.entries[dataset_index]
        batch = move_batch_v69(collate_language_effect_v70([dataset[dataset_index]]), device)
        full_data, full_index = entry_lookup[entry["case_id"]]
        full_data.annotations[entry["case_id"]] = {"frame_indices": entry["frame_indices"]}
        observed_cpu = full_data[full_index]
        full_batch = move_batch_v69(collate_object_video_v69([observed_cpu]), device)
        th = config.history_frames
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
            queries, history = teacher.encode_history(encoded, query_from_batch(batch))
            state_condition = history_context(history)
            observation = full_batch["teacher"]
            sampled, _ = sample_perception_v69(encoded, observation["xy"][:, :th], torch.arange(th, device=device)[None])
            columns = torch.arange(sampled.shape[2], device=device)[None]
            reference_features = sampled[torch.arange(len(sampled), device=device)[:, None], observation["reference_index"], columns]
            ownership, offset, local = teacher.reference_readout(history, observation["reference_xy"], observation["reference_index"], reference_features)
            times = full_batch["times"][:, th:]
            torch.cuda.synchronize()
            history_ms = (time.perf_counter()-began)*1000

            def decode(value):
                future = teacher.dynamics(history[-1], value, times, rollout=True)
                return teacher.render_sequence(future, observation["reference_xy"], ownership, offset, local)["positions"]

            predictions = {"posterior_mean": decode(batch["target_mean"].tanh()),
                           "persistence": observation["reference_xy"][:, None].expand(-1, len(times[0]), -1, -1)}
            conditions = {"correct_language": entry["instruction"], "no_language": ""}
            if entry["window_id"] in conflicts:
                conditions["conflict_language"] = conflicts[entry["window_id"]]["instruction"]
            latency = {}
            for name, instruction in conditions.items():
                start = time.perf_counter()
                inputs = model.prepare_inputs({**batch, "instruction": [instruction]})
                condition = model.encode_condition(inputs, state_condition)
                torch.cuda.synchronize()
                conditioner_ms = (time.perf_counter()-start)*1000
                for sample in range(args.samples):
                    torch.manual_seed(args.seed+index*args.samples+sample)
                    torch.cuda.manual_seed_all(args.seed+index*args.samples+sample)
                    start = time.perf_counter()
                    noise = torch.randn_like(batch["target_mean"])
                    u = model.sample(condition, steps=args.flow_steps, noise=noise)
                    positions = decode(u.tanh())
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    key = f"{name}/sample_{sample}"
                    predictions[key] = positions
                    latency[key] = {"history_and_reference_ms": history_ms, "vlm_context_ms": conditioner_ms,
                                    "sampling_and_dynamics_ms": elapsed, "full_ms": history_ms+conditioner_ms+elapsed}
        for name, positions in predictions.items():
            if name.startswith("conflict_language"):
                continue
            rows = trajectory_records(positions, observation, batch["native_hw"], th, times, args.motion_floor_px)
            all_rows.extend({**row, "condition": name, "source": entry["source"], "window_id": entry["window_id"]} for row in rows)
        case = {"window_id": entry["window_id"], "source": entry["source"], "instruction": entry["instruction"], "latency": latency}
        if entry["window_id"] in conflicts:
            case["conflict"] = {"specification": conflicts[entry["window_id"]],
                                "measurements": conflict_target_measurements(predictions["conflict_language/sample_0"], observation, batch["native_hw"], conflicts[entry["window_id"]]),
                                "original_video_is_not_conflict_ground_truth": True}
        if index < args.videos:
            name = f"case_{index:04d}.mp4"
            save_video(output/name, observed_cpu["rgb"], predictions["correct_language/sample_0"][0], observation["xy"][0, th:], observation["valid"][0, th:], th)
            case["video"] = name
        torch.save({"predictions": {name: value.cpu() for name, value in predictions.items()},
                    "teacher": {name: value.cpu() for name, value in observation.items()}, "times": times.cpu()}, output/f"case_{index:04d}.pt")
        cases.append(case)
        print(json.dumps({"event": "v70_evaluation", "completed": index+1, "window_id": entry["window_id"]}), flush=True)
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["condition"]].append(row)
        grouped[f'{row["condition"]}/source/{row["source"]}'].append(row)
    summaries = {name: summarize_records(rows) for name, rows in grouped.items()}
    for name, rows in grouped.items():
        clips = defaultdict(list)
        for row in rows:
            clips[row["window_id"]].append(row["ade_px"])
        values = [sum(errors)/len(errors) for errors in clips.values()]
        summaries[name]["episode_balanced_ade_px"] = sum(values)/len(values) if values else None
    for condition in ("correct_language", "no_language"):
        samples = [row for row in all_rows if row["condition"].startswith(condition+"/")]
        summaries[condition+"/expected_over_samples"] = summarize_records(samples)
        by_case = defaultdict(lambda: defaultdict(list))
        for row in samples:
            by_case[row["window_id"]][row["condition"]].append(row)
        oracle = []
        for choices in by_case.values():
            oracle.extend(min(choices.values(), key=lambda rows: sum(r["ade_px"] for r in rows)/len(rows)))
        summaries[condition+"/oracle_best_sample_per_clip"] = summarize_records(oracle)
    (output/"trajectories.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_rows))
    (output/"report.json").write_text(json.dumps({"args": vars(args), "checkpoint": metadata, "cases": cases, "metrics": summaries,
                                                "primary": "correct_language/sample_0", "teacher_is_pseudo_measurement": True,
                                                "selection": "seeded source-interleaved episodes, one labeled window per episode"}, indent=2))
    body = "".join(f'<h2>{html.escape(case["window_id"])}</h2><p>{html.escape(case["instruction"])}</p><video controls width="720" src="{case["video"]}"></video>' for case in cases if "video" in case)
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>V70 paired evaluation</title>'+body)


if __name__ == "__main__":
    main()
