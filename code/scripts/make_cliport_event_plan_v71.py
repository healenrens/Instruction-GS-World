#!/usr/bin/env python3
"""Fix scene-disjoint ordered-actor compositions before any simulation or training."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random


COLORS = ("red", "green", "blue", "yellow", "cyan", "brown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-scenes", type=int, default=1000)
    parser.add_argument("--development-scenes", type=int, default=100)
    parser.add_argument("--test-scenes", type=int, default=200)
    parser.add_argument("--first-scene-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    held = set(zip(COLORS, COLORS[1:]+COLORS[:1]))
    pairs = list(combinations(COLORS, 2))
    rng = random.Random(args.seed)
    scenes, atoms = [], {name: set() for name in ("train", "development", "test")}
    counts = Counter()
    scene_seed = args.first_scene_seed
    for partition, count in (("train", args.train_scenes), ("development", args.development_scenes),
                              ("test", args.test_scenes)):
        for index in range(count):
            actors = pairs[index % len(pairs)]
            remaining = [color for color in COLORS if color not in actors]
            rng.shuffle(remaining)
            colors = [*remaining[:3], *actors, remaining[3]]
            programs = []
            for order in ((0, 1), (1, 0)):
                actor_order = tuple(colors[3+i] for i in order)
                unseen = actor_order in held
                if partition == "train" and unseen:
                    continue
                programs.append({"order": list(order), "actor_colors": actor_order,
                                 "combination": "held_actor_order" if unseen else "seen_actor_order"})
                atoms[partition].update((colors[3+i], colors[i]) for i in order)
                counts[f"{partition}/{'held' if unseen else 'seen'}"] += 1
            scenes.append({"scene_seed": scene_seed, "partition": partition,
                           "colors_bowls_then_blocks": colors, "programs": programs})
            scene_seed += 1
    payload = {"contract": "cliport_actor_order_plan_v1", "seed": args.seed, "scenes": scenes,
               "held_actor_orders": sorted(held), "program_counts": dict(counts),
               "partition_unit": "scene_seed; all programs of one scene stay together",
               "held_variable": "ordered pair of actor colors; not a new atomic physical skill",
               "atomic_predicate": "put actor-color block into target-color bowl",
               "training_atomic_predicates": sorted(atoms["train"]),
               "development_atomic_predicates_missing_from_train": sorted(atoms["development"]-atoms["train"]),
               "test_atomic_predicates_missing_from_train": sorted(atoms["test"]-atoms["train"]),
               "selection_uses_future_motion_or_success": False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: value for key, value in payload.items() if key != "scenes"}), flush=True)


if __name__ == "__main__":
    main()
