#!/usr/bin/env python3
"""Prepare episode-uniform candidate windows without running a teacher or changing old data."""

import argparse
from dataclasses import asdict
import math
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.adaptive_gaussian_wm.multisource_video_index import load_multisource_index
from igsw.adaptive_gaussian_wm.grounded_motion_sources_v68 import SOURCES, apply_view_policy, camera_key
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--clip_seconds", type=float, default=10.)
    p.add_argument("--exclude_sources", default="robotwin")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    sources, episodes, _ = load_multisource_index(args.data_index, skip_missing_payloads=True)
    excluded = set(args.exclude_sources.split(","))
    pool = [(i, e) for i, e in enumerate(episodes) if e.split == "train"
            and sources[e.source_index].name in SOURCES and sources[e.source_index].name not in excluded
            and e.frame_count > math.ceil(args.clip_seconds*e.fps)]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    cases, rejected = [], []
    for ordinal, episode in pool:
        source = sources[episode.source_index]
        span = math.ceil(args.clip_seconds * episode.fps)
        start = rng.randrange(episode.frame_count - span)
        record = {**asdict(episode), "adapter": source.adapter}
        case = {"case_id": f"{source.name}_entry{ordinal}_ep{episode.episode_index}_f{start}", "source": source.name,
                "group": episode.group, "record": record, "first_frame": start, "last_frame": start+span,
                "anchor_frame": start+span//2, "clip_seconds": span/episode.fps,
                "episode_seconds": (episode.frame_count-1)/episode.fps,
                "camera": camera_key(record["path"]) or "observation.images.cam_high", "partition": "train"}
        resolved, reason = apply_view_policy(case, [])
        if resolved is None:
            rejected.append({"case_id": case["case_id"], "reason": reason})
            continue
        cases.append(resolved)
        if len(cases) == args.count:
            break
    write_json(args.output, {"cases": cases, "rejected": rejected, "seed": args.seed,
                            "sampling": "episode_uniform_then_uniform_time", "motion_richness_used": False})
    print(f"[object-video-v69] planned={len(cases)} rejected={len(rejected)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
