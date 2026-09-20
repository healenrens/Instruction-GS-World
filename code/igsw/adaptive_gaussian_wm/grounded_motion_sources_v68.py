"""One source/view policy for fresh cases, reused manifests and offline shards."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import fnmatch
import json
import math
from pathlib import Path

from .multisource_group_partition_v61 import select_group_partition_v61
from .multisource_video_index import load_multisource_index


SOURCES = ("robotwin", "agibot", "robomind", "bridge", "hy_embodied")
CAMERAS = {
    "robotwin": ("cam_high",),
    "agibot": ("head",),
    "robomind": ("camera_front_external", "camera_front", "camera_top"),
    "bridge": ("image_0",),
    "hy_embodied": ("cam_high",),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def camera_key(path):
    parts = Path(path).parts
    return parts[parts.index("videos") + 1] if "videos" in parts else ""


def permitted_camera(source, key):
    name = key.lower()
    wrist = ("wrist", "handeye", "hand_eye", "eye_in_hand", "gripper")
    return not any(word in name for word in wrist) and key.split(".")[-1] in CAMERAS.get(source, ())


def task_root(path):
    parts = Path(path).parts
    return Path(*parts[:parts.index("videos")])


@lru_cache(maxsize=16)
def camera_metadata(root):
    root = Path(root)
    info = read_json(root / "meta/info.json")
    keys = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    if str(info.get("codebase_version", "")).startswith("v3"):
        import pyarrow.dataset as ds
        paths = sorted(str(p) for p in (root / "meta/episodes").rglob("*.parquet"))
        dataset = ds.dataset(paths, format="parquet")
        columns = [n for n in dataset.schema.names if n in ("episode_index", "length") or n.startswith("videos/")]
        rows = dataset.to_table(columns=columns).to_pylist()
    else:
        rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    return info, keys, rows


def video_record(root, info, row, camera):
    values = {"video_key": camera, "episode_index": int(row["episode_index"])}
    if str(info.get("codebase_version", "")).startswith("v3"):
        values.update(chunk_index=int(row[f"videos/{camera}/chunk_index"]), file_index=int(row[f"videos/{camera}/file_index"]))
        offset = round(float(row[f"videos/{camera}/from_timestamp"]) * float(info["fps"]))
        length = round((float(row[f"videos/{camera}/to_timestamp"]) - float(row[f"videos/{camera}/from_timestamp"])) * float(info["fps"]))
    else:
        values["episode_chunk"] = values["episode_index"] // int(info["chunks_size"])
        offset, length = 0, int(row["length"])
    return {"path": str(root / info["video_path"].format(**values)), "frame_offset": offset,
            "frame_count": length, "fps": float(info["fps"])}


@lru_cache(maxsize=16)
def episode_lookup(root, camera):
    info, _, rows = camera_metadata(root)
    result = {}
    for row in rows:
        if str(info.get("codebase_version", "")).startswith("v3") and row.get(f"videos/{camera}/file_index") is None:
            continue
        record = video_record(Path(root), info, row, camera)
        result[(record["path"], record["frame_offset"])] = row
    return result


def alternate_views(case):
    record = case["record"]
    if record["adapter"] == "rgb_episode_cache":
        return [deepcopy(case)]
    root, old = task_root(record["path"]), camera_key(record["path"])
    info, keys, _ = camera_metadata(str(root))
    row = episode_lookup(str(root), old).get((record["path"], record["frame_offset"]))
    if row is None:
        return []
    views = []
    for key in keys:
        if str(info.get("codebase_version", "")).startswith("v3") and row.get(f"videos/{key}/file_index") is None:
            continue
        mapping = video_record(root, info, row, key)
        if Path(mapping["path"]).is_file() and mapping["frame_count"] > case["last_frame"]:
            view = deepcopy(case)
            view["record"].update(mapping)
            view["camera"] = key
            views.append(view)
    return views


def apply_view_policy(case, overrides):
    source, original = case["source"], case["record"]["path"]
    if source not in SOURCES:
        return None, "source_excluded"
    record = case["record"]
    if record["adapter"] == "rgb_episode_cache":
        # This adapter's builder is the already established RoboTwin cam_high cache.
        key = case.get("camera", "observation.images.cam_high")
        return (case, "external_cache_contract") if source == "robotwin" and permitted_camera(source, key) else (None, "unknown_cache_view")
    key = camera_key(original)
    requested = None
    for rule in overrides:
        if source == rule["source"] and fnmatch.fnmatch(original, rule["path_glob"]):
            requested = rule["camera"]
    if requested == "exclude":
        return None, "camera_override_excluded"
    if (requested is None and permitted_camera(source, key)) or (requested == key and permitted_camera(source, key)):
        result = deepcopy(case)
        result.update(camera=key, camera_evidence="external_source_key_policy")
        return result, "kept_external"
    views = alternate_views(case)
    order = CAMERAS[source]
    eligible = [v for v in views if permitted_camera(source, v["camera"]) and (requested is None or v["camera"] == requested)]
    eligible.sort(key=lambda v: order.index(v["camera"].split(".")[-1]))
    if not eligible:
        return None, "no_confirmed_external_mapping"
    result = eligible[0]
    result.update(camera_evidence="external_camera_metadata_remap", original_indexed_path=original,
                  original_indexed_camera=key, camera_remapped=True)
    return result, "remapped_external"


def select_data_cases(args):
    overrides = read_json(args.camera_overrides)["rules"] if args.camera_overrides else []
    counts, excluded = Counter(), []
    sources, episodes, _ = load_multisource_index(args.data_index, skip_missing_payloads=True)
    episodes = select_group_partition_v61([e for e in episodes if e.split == "train"], args.partition, args.held_group_stride)
    if args.case_manifest:
        allowed = {(sources[e.source_index].name, e.group, e.episode_index) for e in episodes}
        candidates = []
        for case in read_json(args.case_manifest)["cases"]:
            if (case["source"], case["group"], case["record"]["episode_index"]) in allowed:
                candidates.append(case)
            else:
                excluded.append({"case_id": case["case_id"], "path": case["record"]["path"], "reason": "outside_requested_partition"})
                counts[f"{case['source']}/outside_requested_partition"] += 1
        selection_mode = "reused cases with current partition/source/view policy reapplied"
    else:
        pools = defaultdict(lambda: defaultdict(list))
        for episode in episodes:
            name = sources[episode.source_index].name
            if name in SOURCES and episode.frame_count > math.ceil(args.clip_seconds * episode.fps):
                pools[name][episode.group].append(episode)
        candidates = []
        for name in SOURCES:
            groups = sorted(pools[name])
            if not groups:
                continue
            offset = args.seed % len(groups)
            groups = groups[offset:] + groups[:offset]
            queues = []
            for group in groups:
                entries = sorted(pools[name][group], key=lambda e: e.episode_index)
                offset = args.seed % len(entries)
                queues.append(entries[offset:] + entries[:offset])
            selected = 0
            for ordinal in range(max(map(len, queues))):
                for queue in queues:
                    if ordinal >= len(queue):
                        continue
                    episode = queue[ordinal]
                    record = asdict(episode)
                    record["adapter"] = sources[episode.source_index].adapter
                    span = math.ceil(args.clip_seconds * episode.fps)
                    available = episode.frame_count - span - 1
                    starts = range(0, available + 1, span) if args.all_episode_windows else [available // 2]
                    for first in starts:
                        case = {"case_id": f"{name}_ep{episode.episode_index}_f{first}", "source": name,
                                "group": episode.group, "record": record, "first_frame": first,
                                "last_frame": first + span, "anchor_frame": first + span // 2,
                                "clip_seconds": span / episode.fps, "episode_seconds": (episode.frame_count - 1) / episode.fps,
                                "camera": camera_key(record["path"]) or "observation.images.cam_high", "decode_replaced": False}
                        candidates.append(case)
                    selected += 1
                    if args.cases_per_source and selected >= args.cases_per_source:
                        break
                if args.cases_per_source and selected >= args.cases_per_source:
                    break
        selection_mode = f"group-round-robin {args.partition}; no tracker-success selection"
    cases = []
    for case in candidates:
        if case["last_frame"] - case["first_frame"] < math.ceil(max(10., args.clip_seconds) * case["record"]["fps"]):
            excluded.append({"case_id": case["case_id"], "path": case["record"]["path"], "reason": "clip_shorter_than_requested"})
            counts[f"{case['source']}/clip_shorter_than_requested"] += 1
            continue
        resolved, reason = apply_view_policy(case, overrides)
        counts[f"{case['source']}/{reason}"] += 1
        if resolved is not None:
            resolved["view_policy"] = "external_motion_data_v68"
            resolved["partition"] = args.partition
            cases.append(resolved)
        else:
            excluded.append({"case_id": case["case_id"], "path": case["record"]["path"], "reason": reason})
    return cases, {"mode": selection_mode, "counts": dict(counts), "excluded": excluded,
                   "included_sources": list(SOURCES), "partition": args.partition,
                   "camera_key_is_not_visual_confirmation": True}
