"""Long, non-wrist held clips; selection never consults tracker predictions."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

from .multisource_group_partition_v61 import select_group_partition_v61
from .multisource_video_index import load_multisource_index


def camera_view(episode, source, cache_kinds):
    parts = Path(episode.path).parts
    if "videos" in parts:
        camera = parts[parts.index("videos") + 1]
        if any(word in camera.lower() for word in ("wrist", "eye_in_hand", "gripper")):
            return camera, "wrist"
        external = ("head", "cam_high", "exterior", "camera_front", "camera_top")
        if any(word in camera.lower() for word in external):
            return camera, "external"
        if source["name"] == "bridge" and camera == "observation.images.image_0":
            return camera, "external_source_convention"
        return camera, "unknown"
    if source["adapter"] == "rgb_episode_cache":
        root = source["root"]
        if root not in cache_kinds:
            manifest = json.loads((Path(root) / "episode_manifest.json").read_text())
            cache_kinds[root] = manifest["source"]["kind"]
        if cache_kinds[root] == "robotwin2_lerobot_v3":
            return "observation.images.cam_high", "external_cache_builder_contract"
    return "unidentified", "unknown"


def select_long_cases(args):
    sources, episodes, payload = load_multisource_index(
        args.data_index, skip_missing_payloads=True
    )
    held = select_group_partition_v61(
        [item for item in episodes if item.split == "train"],
        "held",
        args.held_group_stride,
    )
    pools, counts, cache_kinds = (
        defaultdict(lambda: defaultdict(list)),
        defaultdict(Counter),
        {},
    )
    duration = max(10.0, args.clip_seconds)
    for episode in held:
        source = payload["sources"][episode.source_index]
        name = source["name"]
        counts[name]["held_episodes"] += 1
        camera, view = camera_view(episode, source, cache_kinds)
        if view in ("wrist", "unknown"):
            counts[name][f"excluded_{view}_camera"] += 1
            continue
        if episode.frame_count - 1 < math.ceil(duration * episode.fps):
            counts[name]["excluded_short_episode"] += 1
            continue
        counts[name]["eligible_episodes"] += 1
        pools[name][episode.group].append((episode, camera, view))
    cases = []
    for source in sources:
        groups = sorted(pools[source.name])
        if not groups:
            continue
        offset = args.seed % len(groups)
        groups = groups[offset:] + groups[:offset]
        queues = []
        for group in groups:
            entries = sorted(
                pools[source.name][group], key=lambda entry: entry[0].episode_index
            )
            offset = args.seed % len(entries)
            queues.append(entries[offset:] + entries[:offset])
        chosen = []
        # Round-robin across held groups, then episodes; no duplicate clip padding.
        for ordinal in range(max(map(len, queues))):
            for queue in queues:
                if ordinal < len(queue) and len(chosen) < args.cases_per_source:
                    chosen.append(queue[ordinal])
            if len(chosen) == args.cases_per_source:
                break
        for position, (episode, camera, view) in enumerate(chosen):
            span = math.ceil(duration * episode.fps)
            available = episode.frame_count - 1 - span
            first = round(available * (position + 1) / (len(chosen) + 1))
            record = asdict(episode)
            record["adapter"] = source.adapter
            record["sequence_index"] = episode.episode_index
            cases.append(
                {
                    "case_id": f"{source.name}_ep{episode.episode_index}_f{first}",
                    "source": source.name,
                    "group": episode.group,
                    "record": record,
                    "camera": camera,
                    "camera_evidence": view,
                    "first_frame": first,
                    "last_frame": first + span,
                    "anchor_frame": first + span // 2,
                    "clip_seconds": span / episode.fps,
                    "episode_seconds": (episode.frame_count - 1) / episode.fps,
                    "decode_replaced": False,
                }
            )
        counts[source.name]["selected"] = len(chosen)
    report = {
        "sources": dict(counts),
        "missing_payload_files": payload["runtime_missing_video_count"],
        "minimum_clip_seconds": duration,
        "camera_policy": "external/head only; wrist and unidentified cameras excluded",
        "selection": "deterministic held-group round-robin, not selected by motion success",
        "replacement": "none; a decode failure reports the original file",
    }
    return cases, report
