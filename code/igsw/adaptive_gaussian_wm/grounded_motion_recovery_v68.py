"""Reuse completed raw tracks in a separate build with the current selection rule."""

from pathlib import Path
import shutil

import torch

from .grounded_motion_jobs_v68 import reserve_arguments
from .grounded_motion_sources_v68 import select_data_cases
from .tracker_visual_review_v67 import read_json


def recover_plan(args, rank, world):
    root = Path(args.recover_from)
    shard = root / f"shard_{rank:04d}"
    plan = read_json(shard / "cases.json")
    cases = plan["cases"]
    selection = read_json(shard / "selection.json")
    if "reserve" in plan:
        reserve = plan["reserve"]
    else:
        # Keep the old case IDs and review subset. Only unused episodes can replace failures.
        assigned = set()
        for path in sorted(root.glob("shard_*/cases.json")):
            for case in read_json(path)["cases"]:
                assigned.add((case["source"], case["group"], case["record"]["episode_index"]))
        candidates, _ = select_data_cases(reserve_arguments(args))
        reserve = [case for case in candidates
                   if (case["source"], case["group"], case["record"]["episode_index"]) not in assigned][rank::world]
    selection.update(recovered_plan=str(shard / "cases.json"), completed_tracks_reselected=True)
    return cases, reserve, selection


def recover_tracks(args, rank, case, directory):
    if not args.recover_from:
        return None
    old = Path(args.recover_from) / f"shard_{rank:04d}" / case["case_id"]
    if not (old / "complete.json").is_file():
        return None
    payload = torch.load(old / "teacher.pt", map_location="cpu", weights_only=False)
    payload["parent_teacher"] = str(old / "teacher.pt")
    if case["render"]:
        refinement = read_json(old / "refinement.json")
        for name in ["refinement.json", *(view["overlay"] for view in refinement["views"])]:
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old / name, destination)
    return payload
