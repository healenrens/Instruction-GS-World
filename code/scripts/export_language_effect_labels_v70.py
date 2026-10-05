#!/usr/bin/env python3
"""Export small fixed-teacher targets, without copying RGB or dense features."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from igsw.distributed import init_torchrun
from igsw.adaptive_gaussian_wm.frozen_object_teacher_v70 import FixedEffectTeacherV70
from igsw.adaptive_gaussian_wm.object_video_rgb_v69 import read_object_video_frames_v69
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import (
    collate_object_video_v69, distinct_sequence_frames, move_batch_v69,
)
from igsw.adaptive_gaussian_wm.video_file_decoder import VideoDecodeError


def full_window(entry):
    indices = torch.tensor(entry["frame_indices"], dtype=torch.long)
    rgb, times, time_source = read_object_video_frames_v69(entry["case"], indices)
    if isinstance(rgb, VideoDecodeError):
        return {"decode_error": str(rgb), "window_id": entry["window_id"]}
    th = entry["history_frames"]
    return {"rgb": rgb, "times": (times-times[th-1]).float(),
            "frame_valid": distinct_sequence_frames(indices, th), "history_frames": th,
            "native_hw": torch.tensor(rgb.shape[-2:]), "time_source": time_source}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame_batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probe_output", default="")
    args = parser.parse_args()
    context = init_torchrun()
    device = torch.device(context.device)
    payload = json.loads(Path(args.manifest).read_text())
    teacher = FixedEffectTeacherV70(payload["teacher_checkpoint"], context.device, args.frame_batch)
    entries = payload["entries"][:args.limit] if args.limit else payload["entries"]
    if args.probe_output:
        rows = []
        for entry in entries[context.rank::context.world_size]:
            sample = full_window(entry)
            if "decode_error" in sample:
                rows.append(sample)
                continue
            batch = move_batch_v69(collate_object_video_v69([sample]), context.device)
            changed = {**batch, "rgb": batch["rgb"].clone()}
            th = entry["history_frames"]
            changed["rgb"][:, th:] = changed["rgb"][:, th:].flip(1)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                original, swapped = teacher(batch), teacher(changed)
            rows.append({"window_id": entry["window_id"],
                         "history_rgb_max_difference": float((batch["rgb"][:, :th].float()-changed["rgb"][:, :th].float()).abs().max()),
                         "future_rgb_max_difference": float((batch["rgb"][:, th:].float()-changed["rgb"][:, th:].float()).abs().max()),
                         "history_query_max_difference": max(float((original[key].float()-swapped[key].float()).abs().max()) for key in
                                                              ("query_xy", "query_frame_index", "query_features", "query_valid")),
                         "posterior_mean_max_difference": float((original["mean"]-swapped["mean"]).abs().max())})
        if context.distributed:
            gathered = [None] * context.world_size
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for shard in gathered for row in shard]
        if context.is_main:
            Path(args.probe_output).write_text(json.dumps({"cases": rows}, indent=2))
        if context.distributed:
            torch.distributed.destroy_process_group()
        return
    completed, missing = [], []
    for entry in entries[context.rank::context.world_size]:
        destination = Path(entry["label_path"])
        if destination.is_file():
            completed.append(entry["window_id"])
            continue
        sample = full_window(entry)
        if "decode_error" in sample:
            missing.append(sample)
            print(json.dumps(sample), flush=True)
            continue
        batch = move_batch_v69(collate_object_video_v69([sample]), context.device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            labels = teacher(batch)
        record = {name: tensor[0].detach().cpu() for name, tensor in labels.items()}
        record.update(window_id=entry["window_id"], frame_indices=entry["frame_indices"],
                      history_times=sample["times"][:entry["history_frames"]],
                      teacher=teacher.teacher, language_provenance=entry["language_provenance"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".rank{context.rank}.tmp")
        torch.save(record, temporary)
        os.replace(temporary, destination)
        completed.append(entry["window_id"])
        print(json.dumps({"event": "v70_label", "rank": context.rank,
                          "window_id": entry["window_id"], "saved": str(destination)}), flush=True)
    report = Path(args.manifest).with_suffix(f".labels.rank{context.rank}.json")
    report.write_text(json.dumps({"completed": completed, "decode_failures": missing}, indent=2))
    if context.distributed:
        torch.distributed.barrier()
    if context.is_main:
        available = [entry for entry in payload["entries"] if Path(entry["label_path"]).is_file()]
        labeled = {**payload, "entries": available, "unlabeled_windows": len(payload["entries"])-len(available)}
        labeled_path = Path(args.manifest).with_name("labeled_manifest.json")
        labeled_path.write_text(json.dumps(labeled, indent=2, ensure_ascii=False))
        print(json.dumps({"labeled_manifest": str(labeled_path), "windows": len(available)}), flush=True)
    if context.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
