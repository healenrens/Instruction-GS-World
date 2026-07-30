"""Cache full RoboTwin episodes once for dense multi-duration WM sampling."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.episode_cache_contract import (  # noqa: E402
    file_sha256,
    validate_episode_cache_header,
)
from igsw.adaptive_gaussian_wm.episode_cache_encoding import (  # noqa: E402
    encoded_feature_sequence,
    packed_rgb_batch,
    pack_jpegs,
    projection_matrix,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    LEROBOT_SOURCE_KIND,
    HDF5_ROOT_SOURCE_KIND,
    assign_episode_filenames,
    discover_hdf5_episodes,
    discover_lerobot_episodes,
    hdf5_plan_episode,
    iter_episode_batches,
    parse_source_variants,
    source_index_sha256,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    DEFAULT_EPISODE_WINDOWS,
    EPISODE_CACHE_VERSION,
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
    EXPECTED_FRAME_COUNT,
    GROUP_SAMPLER_VERSION,
    RT2_HELDSEED_FRACTION,
    RT2_HELDTASKS,
    parse_control_windows,
    preprocess_vggt_rgb,
)
from igsw.gpstoken_wm.dino_features import DinoFeatures  # noqa: E402


PENDING_MANIFEST_NAME = "episode_manifest.pending.json"
SOURCE_INDEX_NAME = "episode_source_index.json"


def episode_index_from_plan(plan_path: str) -> list[dict]:
    with open(plan_path, encoding="utf-8") as handle:
        windows = json.load(handle)
    if not isinstance(windows, list) or not windows:
        raise ValueError("window plan must be a non-empty list")
    indexed = {}
    for window in windows:
        key = (str(window["task"]), int(window["ep"]))
        value = hdf5_plan_episode(window)
        previous = indexed.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"inconsistent episode metadata in plan: {key}")
        indexed[key] = value
    return [indexed[key] for key in sorted(indexed)]


def episode_index(args, use_prepared: bool = True) -> list[dict]:
    prepared = os.path.join(args.out, SOURCE_INDEX_NAME)
    if use_prepared and os.path.isfile(prepared):
        with open(prepared, encoding="utf-8") as handle:
            episodes = json.load(handle)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("prepared episode source index is invalid")
        return episodes
    episodes = (
        episode_index_from_plan(args.plan)
        if args.plan
        else (
            discover_lerobot_episodes(
                args.source_root,
                args.source_variants,
                args.expected_source_fps,
            )
            if args.source_format == "lerobot"
            else discover_hdf5_episodes(args.source_root)
        )
    )
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError("episode cache source is empty")
    return assign_episode_filenames(episodes)


def source_descriptor(args, episodes: list[dict]) -> dict:
    descriptor = {
        "kind": (
            "window_plan"
            if args.plan
            else (
                LEROBOT_SOURCE_KIND
                if args.source_format == "lerobot"
                else HDF5_ROOT_SOURCE_KIND
            )
        ),
        "path": os.path.abspath(args.plan or args.source_root),
        "index_sha256": source_index_sha256(episodes),
    }
    if not args.plan and args.source_format == "lerobot":
        descriptor.update(
            variants=list(parse_source_variants(args.source_variants)),
            expected_source_fps=float(args.expected_source_fps),
            source_frame_stride=1,
        )
    return descriptor


def episode_control_hz(episodes: list[dict]) -> float:
    values = {round(float(episode["control_hz"]), 9) for episode in episodes}
    if len(values) != 1:
        raise ValueError(f"episode sources use mixed control frequencies: {values}")
    value = values.pop()
    if value <= 0:
        raise ValueError("episode control frequency must be positive")
    return value


def manifest_payload(args, episodes: list[dict], complete: bool) -> dict:
    return {
        "episode_cache_version": EPISODE_CACHE_VERSION,
        "complete": complete,
        "source": source_descriptor(args, episodes),
        "split_contract": {
            "name": ("plan_provided" if args.plan else "rt2_task_md5_v1"),
            "heldseed_fraction": RT2_HELDSEED_FRACTION,
            "heldtasks": list(RT2_HELDTASKS),
        },
        "control_hz": episode_control_hz(episodes),
        "sample_frame_count": EXPECTED_FRAME_COUNT,
        "sampling": {
            "window_lengths": list(parse_control_windows(args.window_lengths)),
            "sample_stride": args.sample_stride,
            "group_balance": "task_sqrt_coverage",
            "group_sampling_temperature": 0.5,
            "group_sampler_version": GROUP_SAMPLER_VERSION,
        },
        "cache": {
            "model": args.model,
            "image_size": args.image_size,
            "feature_dim": args.feature_dim,
            "feature_contract": args.feature_contract,
            "projection_seed": args.projection_seed,
            "rgb_short_side": args.rgb_short_side,
            "rgb_pad_multiple": args.rgb_pad_multiple,
            "jpeg_quality": args.jpeg_quality,
        },
        "episodes": [
            {
                "filename": episode["filename"],
                "frame_count": episode["frame_count"],
                "split": episode["split"],
                "control_hz": float(episode["control_hz"]),
                "sampling_group": hashlib.sha256(episode["task"].encode()).hexdigest()[
                    :16
                ],
            }
            for episode in episodes
        ],
    }


def atomic_json(payload, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def prepare_manifest(args, episodes: list[dict]) -> None:
    os.makedirs(args.out, exist_ok=True)
    for name in (EPISODE_MANIFEST_NAME, EPISODE_VERIFIED_NAME):
        path = os.path.join(args.out, name)
        if os.path.lexists(path):
            os.unlink(path)
    source_index = os.path.join(args.out, SOURCE_INDEX_NAME)
    atomic_json(episodes, source_index)
    pending = os.path.join(args.out, PENDING_MANIFEST_NAME)
    atomic_json(manifest_payload(args, episodes, complete=False), pending)
    print(
        f"[visual-episode] prepared {pending} "
        f"source_index={source_index} episodes={len(episodes)}"
    )


def load_pending(args, episodes: list[dict]) -> dict:
    path = os.path.join(args.out, PENDING_MANIFEST_NAME)
    with open(path, encoding="utf-8") as handle:
        pending = json.load(handle)
    expected = manifest_payload(args, episodes, complete=False)
    if pending != expected:
        raise ValueError("pending episode manifest differs from cache arguments")
    return pending


def finalize_manifest(args, episodes: list[dict]) -> None:
    pending = load_pending(args, episodes)
    projection_hash = None
    final_entries = []
    for episode, entry in zip(episodes, pending["episodes"]):
        path = os.path.join(args.out, episode["filename"])
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise ValueError(f"episode cache is missing or empty: {path}")
        cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        current_hash = cache.get("projection_sha256")
        if not isinstance(current_hash, str) or len(current_hash) != 64:
            raise ValueError(f"episode projection hash is invalid: {path}")
        if projection_hash is None:
            projection_hash = current_hash
        elif projection_hash != current_hash:
            raise ValueError("episode caches use different DINO projections")
        validate_episode_cache_header(
            cache,
            path,
            episode,
            pending["cache"],
            projection_hash,
        )
        final_entry = dict(entry)
        final_entry["cache_bytes"] = os.path.getsize(path)
        final_entry["cache_sha256"] = file_sha256(path)
        final_entries.append(final_entry)
    final = dict(pending)
    final["complete"] = True
    final["projection_sha256"] = projection_hash
    final["episodes"] = final_entries
    path = os.path.join(args.out, EPISODE_MANIFEST_NAME)
    atomic_json(final, path)
    print(f"[visual-episode] finalized {path} episodes={len(episodes)}")


def cache_episode(
    args,
    episode: dict,
    extractor: DinoFeatures,
    projection: torch.Tensor | None,
    projection_hash: str,
    jpeg_executor: ThreadPoolExecutor | None,
) -> int:
    output = os.path.join(args.out, episode["filename"])
    if os.path.exists(output) and not args.overwrite:
        existing = torch.load(
            output,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        validate_episode_cache_header(
            existing,
            output,
            episode,
            {
                "model": args.model,
                "image_size": args.image_size,
                "feature_dim": args.feature_dim,
                "feature_contract": args.feature_contract,
                "projection_seed": args.projection_seed,
                "rgb_short_side": args.rgb_short_side,
                "rgb_pad_multiple": args.rgb_pad_multiple,
                "jpeg_quality": args.jpeg_quality,
            },
            projection_hash,
        )
        return 0
    feature_batches = []
    encoded_rgb = []
    rgb_metadata = None
    for frames in iter_episode_batches(episode, args.frame_batch):
        processed = preprocess_vggt_rgb(frames, args.image_size)
        feature_batches.append(
            encoded_feature_sequence(extractor, projection, processed)
        )
        packed, metadata = packed_rgb_batch(
            processed,
            args.rgb_short_side,
            args.rgb_pad_multiple,
            args.jpeg_quality,
            jpeg_executor,
        )
        if rgb_metadata is None:
            rgb_metadata = metadata
        elif rgb_metadata != metadata:
            raise ValueError("episode RGB dimensions changed between batches")
        encoded_rgb.extend(packed)
    if rgb_metadata is None:
        raise ValueError("episode contains no RGB frames")
    cache = {
        "episode_version": EPISODE_CACHE_VERSION,
        "source_name": episode["filename"],
        "split": episode["split"],
        "control_hz": float(episode["control_hz"]),
        "frame_control_indices": torch.arange(
            episode["frame_count"],
            dtype=torch.long,
        ),
        "dino": torch.cat(feature_batches),
        "model": args.model,
        "image_size": args.image_size,
        "feature_dim": args.feature_dim,
        "feature_contract": args.feature_contract,
        "projection_seed": args.projection_seed,
        "projection_sha256": projection_hash,
        "visual_preprocess": "spatracker_vggt_crop_width518",
        "rgb": pack_jpegs(encoded_rgb, rgb_metadata),
    }
    forbidden = {
        "instruction",
        "condition_feature",
        "condition_tokens",
        "task",
        "task_index",
        "language",
    }
    if forbidden.intersection(cache):
        raise ValueError("semantic fields are forbidden in visual episode caches")
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(cache, temporary)
    os.replace(temporary, output)
    return os.path.getsize(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan")
    source.add_argument("--source_root")
    parser.add_argument(
        "--source_format",
        choices=("hdf5", "lerobot"),
        default="hdf5",
    )
    parser.add_argument(
        "--source_variants",
        default=",".join(LEROBOT_DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--expected_source_fps",
        type=float,
        default=LEROBOT_EXPECTED_FPS,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument(
        "--feature_contract",
        choices=("backbone_native", "random_orthonormal_projection"),
        default="backbone_native",
    )
    parser.add_argument("--projection_seed", type=int, default=0)
    parser.add_argument("--frame_batch", type=int, default=13)
    parser.add_argument("--cpu_threads", type=int, default=16)
    parser.add_argument("--jpeg_workers", type=int, default=4)
    parser.add_argument("--rgb_short_side", type=int, default=256)
    parser.add_argument("--rgb_pad_multiple", type=int, default=16)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--window_lengths",
        default=",".join(map(str, DEFAULT_EPISODE_WINDOWS)),
    )
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--finalize_manifest", action="store_true")
    args = parser.parse_args()
    if args.prepare_manifest and args.finalize_manifest:
        raise ValueError("manifest modes are mutually exclusive")
    if not 0 <= args.shard < args.nshard:
        raise ValueError("invalid shard")
    if (
        args.frame_batch < 1
        or args.cpu_threads < 1
        or args.jpeg_workers < 0
        or args.sample_stride < 1
        or args.expected_source_fps <= 0
        or not 1 <= args.jpeg_quality <= 100
    ):
        raise ValueError("invalid episode cache configuration")
    if args.source_root and args.limit:
        raise ValueError("full source-root caches forbid --limit")
    parse_source_variants(args.source_variants)
    if args.feature_contract == "backbone_native" and args.projection_seed != 0:
        raise ValueError("backbone-native caches require projection_seed=0")
    if (
        args.feature_contract == "random_orthonormal_projection"
        and args.projection_seed < 0
    ):
        raise ValueError("projection seed must be non-negative")
    parse_control_windows(args.window_lengths)
    return args


def main() -> None:
    args = parse_args()
    episodes = episode_index(args, use_prepared=not args.prepare_manifest)
    if args.prepare_manifest:
        prepare_manifest(args, episodes)
        return
    if args.finalize_manifest:
        finalize_manifest(args, episodes)
        return
    load_pending(args, episodes)
    selected = episodes[args.shard :: args.nshard]
    if not selected:
        raise ValueError("episode cache shard is empty")
    for key, value in {
        "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
    }.items():
        os.environ.setdefault(key, value)
    torch.set_float32_matmul_precision("high")
    extractor = DinoFeatures(args.model, args.image_size).cuda().bfloat16().eval()
    if args.feature_contract == "backbone_native":
        if args.feature_dim != extractor.embed_dim:
            raise ValueError(
                "backbone-native feature_dim must equal extractor embed_dim: "
                f"{args.feature_dim} != {extractor.embed_dim}"
            )
        projection = None
        transform = (
            f"backbone_native_v1:{args.model}:{args.image_size}:{extractor.embed_dim}"
        ).encode()
        projection_hash = hashlib.sha256(transform).hexdigest()
    else:
        projection = projection_matrix(
            extractor.embed_dim,
            args.feature_dim,
            args.projection_seed,
        ).cuda()
        projection_hash = hashlib.sha256(projection.cpu().numpy().tobytes()).hexdigest()
    torch.set_num_threads(args.cpu_threads)
    started = time.time()
    written = 0
    executor_context = (
        ThreadPoolExecutor(max_workers=args.jpeg_workers)
        if args.jpeg_workers > 0
        else nullcontext(None)
    )
    with executor_context as jpeg_executor:
        for index, episode in enumerate(selected, 1):
            written += cache_episode(
                args,
                episode,
                extractor,
                projection,
                projection_hash,
                jpeg_executor,
            )
            if index == 1 or index % 10 == 0 or index == len(selected):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[visual-episode] shard={args.shard}/{args.nshard} "
                    f"{index}/{len(selected)} episodes={index / elapsed:.3f}/s "
                    f"written_gib={written / 2**30:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
