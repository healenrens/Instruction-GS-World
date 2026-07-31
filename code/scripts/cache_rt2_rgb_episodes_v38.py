#!/usr/bin/env python3
"""Build a restartable RGB-only RoboTwin episode cache for JIT DINO."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    JIT_DINO_IMAGE_SIZE,
    PENDING_MANIFEST_NAME,
    RGB_EPISODE_CACHE_VERSION,
    SOURCE_INDEX_NAME,
    file_sha256,
    manifest_payload,
    validate_episode_payload,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_ROOT,
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    assign_episode_filenames,
    discover_lerobot_episodes,
    iter_episode_batches,
    parse_source_variants,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    DEFAULT_EPISODE_WINDOWS,
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
    parse_control_windows,
    preprocess_vggt_rgb,
)


def atomic_json(payload, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def discover(args) -> list[dict]:
    return assign_episode_filenames(
        discover_lerobot_episodes(
            args.source_root,
            args.source_variants,
            args.expected_source_fps,
        )
    )


def expected_manifest(args, episodes: list[dict], complete: bool = False) -> dict:
    return manifest_payload(
        episodes,
        args.source_root,
        parse_source_variants(args.source_variants),
        args.expected_source_fps,
        parse_control_windows(args.window_lengths),
        args.sample_stride,
        args.jpeg_quality,
        complete,
    )


def load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_prepared(args) -> tuple[list[dict], dict]:
    source_path = os.path.join(args.out, SOURCE_INDEX_NAME)
    pending_path = os.path.join(args.out, PENDING_MANIFEST_NAME)
    if not os.path.isfile(source_path) or not os.path.isfile(pending_path):
        raise ValueError("RGB cache resume requires source index and pending manifest")
    episodes = load_json(source_path)
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("RGB cache source index is invalid")
    pending = load_json(pending_path)
    expected = expected_manifest(args, episodes)
    if pending != expected:
        raise ValueError("pending RGB manifest differs from current arguments")
    return episodes, pending


def prepare(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    artifacts = [
        os.path.join(args.out, name)
        for name in (
            SOURCE_INDEX_NAME,
            PENDING_MANIFEST_NAME,
            EPISODE_MANIFEST_NAME,
            EPISODE_VERIFIED_NAME,
        )
    ]
    cached = [
        os.path.join(args.out, name)
        for name in os.listdir(args.out)
        if name.endswith(".pt")
    ]
    if args.prepare_mode == "fresh":
        if any(os.path.lexists(path) for path in artifacts) or cached:
            raise ValueError(
                "fresh RGB cache refuses existing state; use prepare_mode=resume"
            )
        episodes = discover(args)
        atomic_json(episodes, os.path.join(args.out, SOURCE_INDEX_NAME))
        atomic_json(
            expected_manifest(args, episodes),
            os.path.join(args.out, PENDING_MANIFEST_NAME),
        )
        print(f"[v38-rgb] prepared fresh episodes={len(episodes)}", flush=True)
        return
    saved, _ = load_prepared(args)
    current = discover(args)
    if current != saved:
        raise ValueError("RoboTwin source metadata changed since cache preparation")
    print(f"[v38-rgb] resumed manifest episodes={len(saved)}", flush=True)


def square_dino_input(frames: torch.Tensor) -> torch.Tensor:
    processed = preprocess_vggt_rgb(frames, JIT_DINO_IMAGE_SIZE)
    square = F.interpolate(
        processed.permute(0, 3, 1, 2).float(),
        size=(JIT_DINO_IMAGE_SIZE, JIT_DINO_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return square.round().clamp(0, 255).to(torch.uint8)


def resolve_encoded(values: list[torch.Tensor | Future[torch.Tensor]]) -> dict:
    encoded = [
        value.result() if isinstance(value, Future) else value for value in values
    ]
    if not encoded:
        raise ValueError("cannot store an empty RGB episode")
    lengths = torch.tensor([len(value) for value in encoded], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    return {
        "jpeg_bytes": torch.cat(encoded),
        "jpeg_offsets": offsets,
        "height": JIT_DINO_IMAGE_SIZE,
        "width": JIT_DINO_IMAGE_SIZE,
    }


def cache_episode(
    args,
    episode: dict,
    entry: dict,
    manifest: dict,
    executor: ThreadPoolExecutor,
) -> int:
    output = os.path.join(args.out, episode["filename"])
    if os.path.isfile(output) and not args.overwrite:
        payload = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
        validate_episode_payload(payload, output, entry, manifest)
        return 0
    encoded: list[torch.Tensor | Future[torch.Tensor]] = []
    for frames in iter_episode_batches(episode, args.frame_batch):
        for frame in square_dino_input(frames):
            encoded.append(executor.submit(encode_jpeg, frame, args.jpeg_quality))
    rgb = resolve_encoded(encoded)
    rgb["jpeg_quality"] = args.jpeg_quality
    payload = {
        "episode_version": RGB_EPISODE_CACHE_VERSION,
        "source_name": episode["filename"],
        "split": episode["split"],
        "control_hz": float(episode["control_hz"]),
        "frame_control_indices": torch.arange(
            int(episode["frame_count"]), dtype=torch.long
        ),
        "visual_preprocess": manifest["rgb_cache"]["preprocess"],
        "rgb": rgb,
    }
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    validate_episode_payload(payload, output, entry, manifest)
    return os.path.getsize(output)


def run_shard(args) -> None:
    episodes, pending = load_prepared(args)
    selected = list(zip(episodes, pending["episodes"], strict=True))[
        args.shard :: args.nshard
    ]
    if not selected:
        raise ValueError("RGB cache shard is empty")
    started = time.time()
    written = 0
    with ThreadPoolExecutor(max_workers=args.jpeg_workers) as executor:
        for index, (episode, entry) in enumerate(selected, 1):
            written += cache_episode(args, episode, entry, pending, executor)
            if index == 1 or index % 10 == 0 or index == len(selected):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[v38-rgb] shard={args.shard}/{args.nshard} "
                    f"episodes={index}/{len(selected)} rate={index / elapsed:.3f}/s "
                    f"written_gib={written / 2**30:.2f}",
                    flush=True,
                )


def finalize(args) -> None:
    episodes, pending = load_prepared(args)
    final_entries = []
    for episode, entry in zip(episodes, pending["episodes"], strict=True):
        path = os.path.join(args.out, episode["filename"])
        if not os.path.isfile(path):
            raise ValueError(f"RGB episode cache is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        validate_episode_payload(payload, path, entry, pending)
        final_entry = dict(entry)
        final_entry["cache_bytes"] = os.path.getsize(path)
        final_entry["cache_sha256"] = file_sha256(path)
        final_entries.append(final_entry)
    final = dict(pending)
    final["complete"] = True
    final["episodes"] = final_entries
    atomic_json(final, os.path.join(args.out, EPISODE_MANIFEST_NAME))
    print(f"[v38-rgb] finalized episodes={len(final_entries)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=LEROBOT_DEFAULT_ROOT)
    parser.add_argument("--source_variants", default=",".join(LEROBOT_DEFAULT_VARIANTS))
    parser.add_argument(
        "--expected_source_fps", type=float, default=LEROBOT_EXPECTED_FPS
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--window_lengths", default=",".join(map(str, DEFAULT_EPISODE_WINDOWS))
    )
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--frame_batch", type=int, default=32)
    parser.add_argument("--jpeg_workers", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare_mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    modes = int(args.prepare) + int(args.finalize)
    if modes > 1 or not 0 <= args.shard < args.nshard:
        raise ValueError("invalid RGB cache execution mode")
    if (
        not os.path.isabs(args.source_root)
        or not os.path.isabs(args.out)
        or args.frame_batch < 1
        or args.jpeg_workers < 1
        or args.sample_stride < 1
        or not 1 <= args.jpeg_quality <= 100
    ):
        raise ValueError("invalid RGB cache arguments")
    parse_source_variants(args.source_variants)
    parse_control_windows(args.window_lengths)
    return args


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare(args)
    elif args.finalize:
        finalize(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
