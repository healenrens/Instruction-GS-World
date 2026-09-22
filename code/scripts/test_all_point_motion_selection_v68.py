#!/usr/bin/env python3
"""One CPU regression covering role-free selection, real loader masks and review."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from igsw.adaptive_gaussian_wm.grounded_motion_export_v68 import export_motion_teacher
from igsw.adaptive_gaussian_wm.grounded_motion_dataset_v68 import GroundedMotionDatasetV68
from igsw.adaptive_gaussian_wm.grounded_motion_training_review_v68 import render_training_sample
from igsw.adaptive_gaussian_wm.grounded_motion_jobs_v68 import interleave_sources, append_replacement, target_counts
from igsw.adaptive_gaussian_wm.grounded_motion_recovery_v68 import recover_plan, recover_tracks


def main():
    frames, points, fps = 101, 8, 10
    tracks = torch.full((frames, points, 2), 16.)
    tracks[..., 0] += torch.arange(frames)[:, None] / 100 * torch.arange(points)[None]
    valid = torch.ones(frames, points, dtype=torch.bool)
    roles = ["object_candidate", "robot_context", "scene_context", "unknown"] * 2
    queries = {"frames": torch.zeros(points, dtype=torch.long), "xy": tracks[0],
               "metadata": [{"role": role, "region_id": f"r{i}", "refined_support": False} for i, role in enumerate(roles)]}
    native = {"tracks": tracks, "visibility": valid, "in_bounds": valid}
    evidence = {"valid": valid, "span_px": torch.arange(points).float(), "raw_span_px": torch.arange(points).float(),
                "moving": torch.zeros(points, dtype=torch.bool), "compensated_coordinates": tracks,
                "threshold_px": torch.full((points,), 1000.)}
    relay_valid = valid.clone()
    relay_valid[:, 7] = False
    relay = {"valid": relay_valid, "consistent": torch.zeros(points, dtype=torch.bool)}
    case = {"first_frame": 0, "record": {"fps": fps}}
    args = argparse.Namespace(motion_top_fraction=.75, source_revision="regression")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = export_motion_teacher(root, case, queries, native, evidence, relay,
                                       {"valid": valid[:, 0]}, {}, {}, args)
        assert report["object_target_ids"] == [2, 3, 4, 5, 6, 7]
        for row in queries["metadata"]:
            row["role"] = "robot_context"
        changed = export_motion_teacher(root, case, queries, native, evidence, relay,
                                        {"valid": valid[:, 0]}, {}, {}, args)
        assert changed["object_target_ids"] == report["object_target_ids"]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"root": str(root), "entries": [{"path": "teacher.pt", "source": "robotwin",
                            "camera": "cam_high", "case_id": "test", "object_targets": 6, "rendered": True}]}))
        dataset = GroundedMotionDatasetV68(manifest, points=8)
        def decode(_case, indices):
            return torch.full((len(indices), 3, 32, 48), 80, dtype=torch.uint8)
        with patch("igsw.adaptive_gaussian_wm.grounded_motion_dataset_v68.decode_case", decode):
            sample = dataset[(0, 0)]
        assert set(sample["point_ids"].tolist()) == {-1, 2, 3, 4, 5, 6, 7}
        assert (sample["roles"] == -1).all() and (sample["region_ids"] == -1).all()
        assert sample["target_valid"].sum(1).tolist() == [5, 5]
        report = render_training_sample(sample, root / "preview", width=320)
        assert report["counts"]["current_queries"] == 6
        assert report["counts"]["short_supervised"] == report["counts"]["long_supervised"] == 5
        assert {r["track_id"] for r in report["points"]} == {2, 3, 4, 5, 6, 7}
        # Recovery keeps original clips/review IDs; bad media only consumes unused same-source reserves.
        old_shard = root / "old/shard_0000"
        old_shard.mkdir(parents=True)
        planned = [{"case_id": f"{source}_{i}", "source": source, "group": source,
                    "record": {"episode_index": i}, "render": i == 0}
                   for source in ("robotwin", "agibot") for i in range(2)]
        (old_shard / "cases.json").write_text(json.dumps({"cases": planned}))
        (old_shard / "selection.json").write_text(json.dumps({"review_by_source": {"robotwin": 1, "agibot": 1}}))
        settings = argparse.Namespace(recover_from=str(root / "old"), cases_per_source=2,
                                      case_manifest="", replacement_cases_per_source=0)
        spare = {**planned[2], "case_id": "agibot_spare", "record": {"episode_index": 2}}
        with patch("igsw.adaptive_gaussian_wm.grounded_motion_recovery_v68.select_data_cases",
                   return_value=(planned + [spare], {})):
            recovered, reserve, _ = recover_plan(settings, 0, 1)
        queue = interleave_sources(recovered)
        assert [c["source"] for c in queue] == ["robotwin", "agibot", "robotwin", "agibot"]
        assert append_replacement(queue[1], queue, reserve) == "agibot_spare"
        assert queue[-1]["render"] and target_counts(queue) == {"robotwin": 2, "agibot": 2}
        old_case = old_shard / planned[0]["case_id"]
        old_case.mkdir()
        (old_case / "complete.json").write_text("{}")
        (old_case / "refinement.json").write_text('{"views": []}')
        old_payload = torch.load(root / "teacher.pt", weights_only=False)
        torch.save(old_payload, old_case / "teacher.pt")
        new_case = root / "recovered"
        new_case.mkdir()
        restored = recover_tracks(settings, 0, planned[0], new_case)
        assert torch.equal(restored["native"]["tracks"], tracks)
        assert restored["parent_teacher"] == str(old_case / "teacher.pt")
        assert (new_case / "refinement.json").is_file() and not (new_case / "teacher.pt").exists()
        assert recover_tracks(settings, 0, planned[1], new_case) is None
    print("all-point selection -> Dataset -> preview regression passed")


if __name__ == "__main__":
    main()
