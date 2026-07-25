"""Build strict-causal arbitrary-frame pairs from RoboTwin video windows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import os
import random
import sys
import time

import torch

torch.set_num_threads(int(os.environ.get("RT2_PAIR_CPU_THREADS", "2")))
torch.set_num_interop_threads(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.causal_geometry import causal_geometry_from_prediction  # noqa: E402
from igsw.latent_particle_wm.pair_targets import (  # noqa: E402
    CAUSAL_PAIR_VERSION,
    tracker_pair_targets_on_grid,
)


def deterministic_pairs(
    filename: str,
    frame_count: int,
    horizons: tuple[int, ...],
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    candidates = [
        (start, start + horizon)
        for horizon in horizons
        for start in range(frame_count - horizon)
    ]
    if not candidates:
        raise ValueError("no valid frame pairs")
    digest = hashlib.sha256(f"{seed}:{filename}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    buckets = {
        "short": [pair for pair in candidates if pair[1] - pair[0] <= 3],
        "medium": [pair for pair in candidates if 4 <= pair[1] - pair[0] <= 7],
        "long": [pair for pair in candidates if pair[1] - pair[0] >= 8],
    }
    if count <= 0 or count >= len(candidates):
        return sorted(candidates)
    selected: list[tuple[int, int]] = []
    by_start: dict[int, list[tuple[int, int]]] = {}
    for pair in candidates:
        by_start.setdefault(pair[0], []).append(pair)
    comparable_starts = [start for start, pairs in by_start.items() if len(pairs) >= 2]
    generator.shuffle(comparable_starts)
    if count >= 2 and comparable_starts:
        anchor_pairs = by_start[comparable_starts[0]]
        generator.shuffle(anchor_pairs)
        selected.extend(anchor_pairs[:2])

    selected_set = set(selected)
    for name, pairs in buckets.items():
        buckets[name] = [pair for pair in pairs if pair not in selected_set]
        generator.shuffle(buckets[name])
    names = [name for name in ("short", "medium", "long") if buckets[name]]
    cursor = 0
    while len(selected) < count and names:
        name = names[cursor % len(names)]
        selected.append(buckets[name].pop())
        if not buckets[name]:
            names.remove(name)
            cursor = 0
        else:
            cursor += 1
    return sorted(selected)


def parse_horizons(text: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in text.split(",")}))
    if not horizons or horizons[0] < 1 or horizons[-1] > 12:
        raise ValueError("horizons must be between 1 and 12")
    return horizons


def parse_split_limits(text: str) -> dict[str, int]:
    if not text:
        return {}
    limits = {}
    for item in text.split(","):
        split, count = item.split(":", 1)
        if split not in {"train", "heldseed", "heldtask"}:
            raise ValueError(f"unknown split in --split_limits: {split}")
        limits[split] = int(count)
        if limits[split] < 1:
            raise ValueError("split limits must be positive")
    return limits


def filename_split(path: str) -> str:
    name = os.path.basename(path)
    for split in ("heldseed", "heldtask", "train"):
        if name.endswith(f"_{split}.pt"):
            return split
    raise ValueError(f"cannot infer split from {name}")


def stratified_clip_subset(files: list[str], limit: int, seed: int, split: str) -> list[str]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for path in files:
        by_task[os.path.basename(path).split("_ep", 1)[0]].append(path)
    for task, paths in by_task.items():
        digest = hashlib.sha256(f"{seed}:{split}:{task}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(paths)
    tasks = sorted(by_task)
    selected = []
    cursor = 0
    while len(selected) < limit and tasks:
        task = tasks[cursor % len(tasks)]
        selected.append(by_task[task].pop())
        if not by_task[task]:
            tasks.remove(task)
            cursor = 0
        else:
            cursor += 1
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/rt2_joint_src")
    parser.add_argument("--tracked", default="data/rt2_joint")
    parser.add_argument("--out", default="data/rt2_causal_pairs_v1")
    parser.add_argument("--spatrack_root", default="/mnt/pfs/public/xuhaoming/SpaTrackerV2")
    parser.add_argument("--horizons", default="1,2,3,4,6,8,10,12")
    parser.add_argument("--pairs_per_clip", type=int, default=6)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--max_match_px", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--split_limits",
        default="",
        help="task-stratified clip counts, e.g. train:64,heldseed:16,heldtask:18",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.grid != 48:
        raise ValueError(f"{CAUSAL_PAIR_VERSION} requires --grid 48")
    if not (0 <= args.shard < args.nshard):
        raise ValueError("invalid shard")
    if args.limit and args.split_limits:
        raise ValueError("--limit and --split_limits are mutually exclusive")
    horizons = parse_horizons(args.horizons)
    split_limits = parse_split_limits(args.split_limits)

    for key, value in {
        "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
    }.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, args.spatrack_root)
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    tracked_names = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(args.tracked, "*.pt"))
    }
    all_source_files = [
        path
        for path in sorted(glob.glob(os.path.join(args.source, "*.pt")))
        if os.path.basename(path) in tracked_names
    ]
    if split_limits:
        selected = []
        for split, limit in split_limits.items():
            available = [path for path in all_source_files if filename_split(path) == split]
            selected.extend(stratified_clip_subset(available, min(limit, len(available)), args.seed, split))
        source_files = sorted(selected)[args.shard :: args.nshard]
    else:
        source_files = all_source_files[args.shard :: args.nshard]
        if args.limit:
            source_files = source_files[: args.limit]
    if not source_files:
        raise ValueError("no source clips have matching tracker files")
    os.makedirs(args.out, exist_ok=True)
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()

    written = skipped = frames_encoded = matched_total = 0
    started_at = time.time()
    for clip_index, source_path in enumerate(source_files, 1):
        filename = os.path.basename(source_path)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        tracker = torch.load(
            os.path.join(args.tracked, filename),
            map_location="cpu",
            weights_only=False,
        )
        raw_frames = source["gt_rgb"].permute(0, 3, 1, 2).float()
        processed = preprocess_image(raw_frames)
        frame_count = len(processed)
        if tracker["traj"].shape[0] != frame_count:
            raise ValueError(
                f"{filename}: source has {frame_count} frames, "
                f"tracker has {tracker['traj'].shape[0]}"
            )
        if source["task"] != tracker["task"] or source["split"] != tracker["split"]:
            raise ValueError(f"{filename}: source/tracker metadata mismatch")
        pairs = deterministic_pairs(
            filename,
            frame_count,
            horizons,
            args.pairs_per_clip,
            args.seed,
        )
        pending_pairs = []
        for start, end in pairs:
            stem = os.path.splitext(filename)[0]
            out_path = os.path.join(args.out, f"{stem}_t{start:02d}_u{end:02d}.pt")
            if os.path.exists(out_path) and not args.overwrite:
                skipped += 1
            else:
                pending_pairs.append((start, end, out_path))
        starts = sorted({start for start, _, _ in pending_pairs})
        geometry = {}
        for start in starts:
            current = processed[start : start + 1][None]
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(current.cuda() / 255.0)
            geometry[start] = causal_geometry_from_prediction(
                prediction["points_map"],
                prediction["intrs"],
                args.grid,
            )
            frames_encoded += 1

        for start, end, out_path in pending_pairs:
            geom = geometry[start]
            labels = tracker_pair_targets_on_grid(
                geom["means"],
                geom["uv"],
                geom["K_intr"],
                tracker["traj"],
                tracker["K_intr"],
                tracker.get("vis"),
                start,
                end,
                args.max_match_px,
            )
            rgb_path = (
                processed[start : end + 1]
                .permute(0, 2, 3, 1)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
            )
            pair = {
                "means": geom["means"].cpu(),
                "uv": geom["uv"].cpu(),
                "K_intr": geom["K_intr"].cpu(),
                "viewmat": torch.eye(4),
                "H": int(geom["H"]),
                "W": int(geom["W"]),
                "traj": labels["traj"].cpu(),
                "vis": labels["vis"].cpu(),
                "geom_valid": labels["geom_valid"].cpu(),
                "geom_match_distance": labels["match_distance"].cpu(),
                "geometry_valid_count": int(labels["matched_tracks"]),
                "rgb_path": rgb_path,
                "instruction": source.get("instruction", tracker.get("instruction", "")),
                "task": source["task"],
                "split": source["split"],
                "source_name": filename,
                "start": start,
                "end": end,
                "horizon": end - start,
                "pair_version": CAUSAL_PAIR_VERSION,
                "geometry_input_source": f"source_rgb_frame_{start}_only",
                "geometry_target_source": "spatrack_full_video_relative_motion",
                "input_fields": [
                    "means",
                    "uv",
                    "K_intr",
                    "rgb_path[0]",
                    "instruction",
                    "horizon",
                ],
                "target_fields": ["traj", "vis", "geom_valid", "rgb_path[1:]"],
            }
            temporary = f"{out_path}.tmp.{os.getpid()}"
            torch.save(pair, temporary)
            os.replace(temporary, out_path)
            written += 1
            matched_total += int(labels["matched_tracks"])
        if clip_index == 1 or clip_index % 20 == 0 or clip_index == len(source_files):
            rate = written / max(time.time() - started_at, 1e-6)
            print(
                f"[causal-pairs] shard={args.shard}/{args.nshard} "
                f"clips={clip_index}/{len(source_files)} pairs={written} skipped={skipped} "
                f"encoded={frames_encoded} mean_match={matched_total / max(written, 1):.1f} "
                f"rate={rate:.2f}/s",
                flush=True,
            )
    print(
        f"[causal-pairs] DONE written={written} skipped={skipped} "
        f"frames_encoded={frames_encoded} out={os.path.abspath(args.out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
