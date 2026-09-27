#!/usr/bin/env python3
"""Freeze V68 completed tracks as an episode-split V69 sequence manifest; no tracking."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held_fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    entries = []
    # A stopped builder's final merged manifest may be absent or stale; atomic complete records are authoritative.
    for record in sorted(root.glob("shard_*/*/complete.json")):
        entry = json.loads(record.read_text())["entry"]
        teacher = record.parent / "teacher.pt"
        data = torch.load(teacher, map_location="cpu", mmap=True, weights_only=False)
        case = data["case"]
        entries.append({**entry, "path": str(teacher.relative_to(root)), "group": case["group"],
                        "episode_index": case["record"]["episode_index"], "case": case,
                        "raw_frames": len(data["native"]["tracks"]), "raw_tracks": data["native"]["tracks"].shape[1]})
    by_source = defaultdict(set)
    for entry in entries:
        by_source[entry["source"]].add((entry["group"], entry["episode_index"]))
    held = set()
    rng = random.Random(args.seed)
    for source, group in sorted(by_source.items()):
        episodes = sorted(group)
        rng.shuffle(episodes)
        held.update((source, *episode) for episode in episodes[:round(len(episodes)*args.held_fraction)])
    for entry in entries:
        entry["partition"] = "held" if (entry["source"], entry["group"], entry["episode_index"]) in held else "train"
    counts = Counter(f"{entry['source']}/{entry['partition']}" for entry in entries)
    write_json(args.output, {"contract": "object_video_sequence_measurements_v1", "root": str(root), "entries": entries,
                            "source_counts": dict(counts), "seed": args.seed, "held_fraction": args.held_fraction,
                            "split_unit": "source/group/episode", "sampling": "episode_uniform_within_recorded_collection",
                            "upstream_selection": "inherited_v68; not claimed uniform over original full corpus"})
    print(json.dumps({"manifest": args.output, "clips": len(entries), "counts": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
