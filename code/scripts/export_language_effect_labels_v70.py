#!/usr/bin/env python3
"""Export small fixed-teacher targets, without copying RGB or dense features."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from igsw.distributed import init_torchrun
from igsw.adaptive_gaussian_wm.frozen_object_teacher_v70 import FixedEffectTeacherV70
from igsw.adaptive_gaussian_wm.label_export_data_v70 import (
    LabelWindowDatasetV70, collate_label_windows_v70, full_window,
)
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import (
    collate_object_video_v69, move_batch_v69,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame_batch", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probe_output", default="")
    args = parser.parse_args()
    context = init_torchrun()
    device = torch.device(context.device)
    payload = json.loads(Path(args.manifest).read_text())
    teacher = FixedEffectTeacherV70(payload["teacher_checkpoint"], context.device, args.frame_batch)
    teacher.perception.batch_across_samples = True
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
    completed, missing, pending = [], [], []
    for entry in entries[context.rank::context.world_size]:
        if Path(entry["label_path"]).is_file():
            completed.append(entry["window_id"])
        else:
            pending.append(entry)
    worker_options = ({"multiprocessing_context": "spawn", "prefetch_factor": args.prefetch}
                      if args.workers else {})
    loader = DataLoader(LabelWindowDatasetV70(pending), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        collate_fn=collate_label_windows_v70, **worker_options)
    print(json.dumps({"event": "v70_export_start", "rank": context.rank,
                      "world_size": context.world_size, "pending": len(pending),
                      "reused": len(completed), "batch": args.batch, "workers": args.workers,
                      "prefetch": args.prefetch, "frame_batch": args.frame_batch}), flush=True)
    ready = time.perf_counter()
    for packed in loader:
        loaded = time.perf_counter()
        wait_seconds = loaded - ready
        missing.extend(packed["errors"])
        for error in packed["errors"]:
            print(json.dumps(error), flush=True)
        for indices, cpu_batch in packed["groups"]:
            started = time.perf_counter()
            batch = move_batch_v69(cpu_batch, context.device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                labels = teacher(batch)
            labels = {name: tensor.detach().cpu() for name, tensor in labels.items()}
            inference_seconds = time.perf_counter() - started
            saved_paths = []
            for item, index in enumerate(indices):
                entry = pending[index]
                record = {name: tensor[item].clone() for name, tensor in labels.items()}
                record.update(window_id=entry["window_id"], frame_indices=entry["frame_indices"],
                              history_times=cpu_batch["times"][item, :entry["history_frames"]].clone(),
                              teacher=teacher.teacher, language_provenance=entry["language_provenance"])
                destination = Path(entry["label_path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(f".rank{context.rank}.tmp")
                torch.save(record, temporary)
                os.replace(temporary, destination)
                completed.append(entry["window_id"])
                saved_paths.append(str(destination))
            del batch, labels
            memory = ({"peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3}
                      if device.type == "cuda" else {})
            print(json.dumps({"event": "v70_label_batch", "rank": context.rank,
                              "actual_batch": len(indices), "native_hw": cpu_batch["native_hw"][0].tolist(),
                              "loader_wait_seconds": wait_seconds, "inference_seconds": inference_seconds,
                              "write_seconds": time.perf_counter()-started-inference_seconds,
                              "rank_completed": len(completed), "saved": saved_paths, **memory}), flush=True)
            wait_seconds = 0.0
        ready = time.perf_counter()
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
