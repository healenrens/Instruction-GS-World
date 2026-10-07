#!/usr/bin/env python3
"""Local paired evaluation through one frozen Dynamics; no remote media uploads."""

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist

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
from igsw.adaptive_gaussian_wm.v70_evaluation_protocol import (
    effect_diagnostics, evaluation_plan, evaluation_summaries, paired_noise,
)


def diagnostic_indices(entries, limit, seed):
    rng = random.Random(seed)
    sources = defaultdict(lambda: defaultdict(list))
    for index, entry in enumerate(entries):
        sources[entry["source"]][(entry["group"], entry["episode_index"])].append(index)
    queues = {}
    for source, episodes in sorted(sources.items()):
        queue = [rng.choice(sorted(windows, key=lambda i: entries[i]["window_id"]))
                 for _, windows in sorted(episodes.items())]
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
    parser.add_argument("--plan", default="", help="Reuse an exact diagnostic plan across checkpoints.")
    args = parser.parse_args()
    from igsw.adaptive_gaussian_wm.v70_checkpoint import load_inference_model_v70
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LanguageEffectDatasetV70(args.manifest, partition=args.partition)
    plan = (json.loads(Path(args.plan).read_text()) if args.plan else
            evaluation_plan(dataset.entries, diagnostic_indices(dataset.entries, args.limit, args.seed), args.seed))
    if rank == 0:
        (output / "selection.json").write_text(json.dumps(plan, indent=2))
    index_by_window = {entry["window_id"]: index for index, entry in enumerate(dataset.entries)}
    manifest = json.loads(Path(args.manifest).read_text())
    # Use the exact dependency recorded by the evaluated checkpoint.
    from igsw.adaptive_gaussian_wm.v70_checkpoint import read_checkpoint_metadata_v70
    metadata = read_checkpoint_metadata_v70(args.checkpoint)
    saved = torch.load(metadata["fixed_teacher"]["path"], map_location="cpu", weights_only=False)
    config = ObjectVideoConfigV69(**saved["config"])
    teacher = ObjectVideoWorldModelV69(config, stage="dynamics").requires_grad_(False).eval()
    teacher.load_state_dict(saved["model"])
    perception = perception_from_checkpoint(saved, config, args.frame_batch).to(device)
    teacher.to(device)
    del saved
    model, metadata = load_inference_model_v70(args.checkpoint, device)
    full_datasets = {}
    for split in ("held", "train"):
        full_datasets[split] = ObjectVideoSequenceDatasetV69(
            str((Path(args.manifest).parent / manifest["stage2_manifest"]).resolve()), config, args.seed, split)
    entry_lookup = {entry["case_id"]: (data, index) for data in full_datasets.values() for index, entry in enumerate(data.entries)}
    conflicts = {row["window_id"]: row for row in map(json.loads, Path(args.conflicts).read_text().splitlines())} if args.conflicts else {}
    all_rows, cases = [], []
    for index in range(rank, len(plan), world):
        planned = plan[index]
        dataset_index = index_by_window[planned["window_id"]]
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
            if planned["shuffled_instruction"] is not None:
                conditions["shuffled_language"] = planned["shuffled_instruction"]
            if entry["window_id"] in conflicts:
                conditions["conflict_language"] = conflicts[entry["window_id"]]["instruction"]
            latency, auxiliary = {}, {}
            for name, instruction in conditions.items():
                start = time.perf_counter()
                inputs = model.prepare_inputs({**batch, "instruction": [instruction]})
                condition = model.encode_condition(inputs, state_condition)
                torch.cuda.synchronize()
                conditioner_ms = (time.perf_counter()-start)*1000
                for sample in range(args.samples):
                    start = time.perf_counter()
                    noise = paired_noise(batch["target_mean"], args.seed, index, sample)
                    u = model.sample(condition, steps=args.flow_steps, noise=noise)
                    positions = decode(u.tanh())
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    key = f"{name}/sample_{sample}"
                    predictions[key] = positions
                    latency[key] = {"history_and_reference_ms": history_ms, "vlm_context_ms": conditioner_ms,
                                    "sampling_and_dynamics_ms": elapsed, "full_ms": history_ms+conditioner_ms+elapsed}
                    if sample == 0 and name != "conflict_language":
                        auxiliary[name] = effect_diagnostics(model, condition, batch["target_mean"], noise, u)
        case_rows = []
        for name, positions in predictions.items():
            if name.startswith("conflict_language"):
                continue
            rows = trajectory_records(positions, observation, batch["native_hw"], th, times, args.motion_floor_px)
            case_rows.extend({**row, "condition": name, "source": entry["source"], "window_id": entry["window_id"]} for row in rows)
        all_rows.extend(case_rows)
        case = {**planned, "case_index": index, "latency": latency, "effect_diagnostics": auxiliary,
                "trajectory_metrics": {name: summarize_records([row for row in case_rows if row["condition"] == name])
                                       for name in predictions if not name.startswith("conflict_language")},
                "shuffled_language_is_condition_ablation_not_conflict_ground_truth": True}
        if entry["window_id"] in conflicts:
            case["conflict"] = {"specification": conflicts[entry["window_id"]],
                                "measurements": conflict_target_measurements(predictions["conflict_language/sample_0"], observation, batch["native_hw"], conflicts[entry["window_id"]]),
                                "original_video_is_not_conflict_ground_truth": True}
        if index < args.videos:
            case["videos"] = {}
            for condition_name in ("correct_language/sample_0", "no_language/sample_0", "shuffled_language/sample_0", "posterior_mean", "persistence"):
                if condition_name not in predictions:
                    continue
                name = f'case_{index:04d}_{condition_name.replace("/", "_")}.mp4'
                save_video(output/name, observed_cpu["rgb"], predictions[condition_name][0], observation["xy"][0, th:], observation["valid"][0, th:], th)
                case["videos"][condition_name] = name
        torch.save({"predictions": {name: value.cpu() for name, value in predictions.items()},
                    "teacher": {name: value.cpu() for name, value in observation.items()}, "times": times.cpu()}, output/f"case_{index:04d}.pt")
        cases.append(case)
        print(json.dumps({"event": "v70_evaluation", "rank": rank, "completed_on_rank": len(cases),
                          "case_index": index, "window_id": entry["window_id"]}), flush=True)
    shard = output / f"rank_{rank:04d}"
    shard.mkdir(exist_ok=True)
    (shard / "trajectories.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_rows))
    (shard / "cases.json").write_text(json.dumps(cases))
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_rows = [json.loads(line) for r in range(world)
                    for line in (output / f"rank_{r:04d}" / "trajectories.jsonl").read_text().splitlines()]
        cases = sorted([case for r in range(world)
                        for case in json.loads((output / f"rank_{r:04d}" / "cases.json").read_text())],
                       key=lambda case: case["case_index"])
        write_report(output, args, metadata, all_rows, cases)
    if world > 1:
        dist.destroy_process_group()


def write_report(output, args, metadata, all_rows, cases):
    summaries = evaluation_summaries(all_rows)
    (output/"trajectories.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_rows))
    (output/"report.json").write_text(json.dumps({"args": vars(args), "checkpoint": metadata, "cases": cases, "metrics": summaries,
                                                "primary": "correct_language/sample_0", "teacher_is_pseudo_measurement": True,
                                                "reference": "last reliable historical observation, not necessarily t0",
                                                "static_definition": "mean displacement from that reference below motion_floor_px",
                                                "selection": "seeded source-interleaved episodes, one labeled window per episode",
                                                "evaluator_revision": os.environ.get("SOURCE_REVISION", "")}, indent=2))
    body = "".join(f'<h2>{html.escape(case["window_id"])}</h2><p>{html.escape(case["instruction"])}</p>' +
                   "".join(f'<h3>{html.escape(label)}</h3><video controls preload="none" width="720" src="{name}"></video>'
                           for label, name in case.get("videos", {}).items()) for case in cases if "videos" in case)
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>V70 paired evaluation</title>'+body)


if __name__ == "__main__":
    main()
