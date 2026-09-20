#!/usr/bin/env python3
"""Reselect completed tracks globally, then visualize the actual training Dataset."""

import argparse
from collections import Counter
import html
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from igsw.adaptive_gaussian_wm.grounded_background_motion_v68 import motion_evidence
from igsw.adaptive_gaussian_wm.grounded_motion_export_v68 import CONTRACT, SELECTION_POLICY, export_motion_teacher
from igsw.adaptive_gaussian_wm.grounded_motion_training_review_v68 import render_motion_selection, write_training_review
from igsw.adaptive_gaussian_wm.grounded_motion_review_v68 import write_review_bundle
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json, decode_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--motion_top_fraction", type=float, default=.75)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", default="0")
    parser.add_argument("--display_width", type=int, default=960)
    args = parser.parse_args()
    source, out = Path(args.input).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Snapshot only completed cases. Never continue the original data-generation job.
    snapshot_path = out / "input_completed.json"
    if snapshot_path.is_file():
        snapshot = read_json(snapshot_path)
    else:
        snapshot = {"input": str(source), "complete_records": [str(p) for p in sorted(source.glob("shard_*/*/complete.json"))]}
        write_json(snapshot_path, snapshot)
    entries = []
    links = ["<!doctype html><meta charset='utf-8'><title>All-point motion training data</title>",
             "<style>body{font:16px system-ui;max-width:1300px;margin:24px auto}video,img{max-width:100%}section{border-top:1px solid #bbb;padding:20px 0}</style>",
             "<h1>所有类别轨迹统一筛选前75%</h1><p><a href='training_samples/index.html'>先看实际训练样本：历史RGB、当前点、+1秒/+3秒监督</a></p>",
             "<p>下面黄色轨迹是整个clip的入选池，不是单次训练的256个点。黄色只表示被选中，不表示物体类别。"
             "物体、夹爪、机械臂、背景、未知点一起排名。无效时刻不绘制点或跨遮挡连线。</p>"]
    partition = "train"
    for number, record_path in enumerate(snapshot["complete_records"]):
        record = read_json(record_path)
        old_entry = record["entry"]
        old_directory = Path(record_path).parent
        relative = Path(old_directory.parent.name) / old_directory.name
        directory = out / relative
        directory.mkdir(parents=True, exist_ok=True)
        completed = directory / "complete.json"
        partition = old_entry["partition"]
        if completed.is_file():
            entry = read_json(completed)["entry"]
        else:
            data = torch.load(old_directory / "teacher.pt", map_location="cpu", weights_only=False)
            config = {**data["configuration"], "motion_top_fraction": args.motion_top_fraction,
                      "source_revision": args.source_revision, "selection_policy": SELECTION_POLICY,
                      "tracks_source_revision": data["source_revision"]}
            settings = argparse.Namespace(**config)
            evidence = motion_evidence(data["native"], data["background_motion"], settings)
            report = export_motion_teacher(directory, data["case"], data["queries"], data["native"], evidence,
                                           data["relay_evidence"], data["background_motion"], data["sampling"],
                                           data["role_evidence"], settings)
            if old_entry["rendered"]:
                rgb = decode_case(data["case"], data["native"]["frame_indices"])
                valid = evidence["valid"] & data["relay_evidence"]["valid"]
                render_motion_selection(directory, rgb, data["native"], valid, report["object_target_ids"],
                                        args.display_width, data["case"]["record"]["fps"])
                for name in ("episode_overview.png", "camera_overview.png", "camera_mapping.json", "source_review.json"):
                    if (old_directory / name).is_file():
                        shutil.copy2(old_directory / name, directory / name)
                del rgb
            entry = {**old_entry, "path": str(relative / "teacher.pt"),
                     "object_targets": report["selected_point_count"], "motion_targets": report["selected_point_count"],
                     "selection_policy": SELECTION_POLICY, "parent_teacher": str(old_directory / "teacher.pt")}
            write_json(completed, {"entry": entry, "configuration": config})
            del data
        entries.append(entry)
        if entry["rendered"]:
            url = html.escape(relative.as_posix())
            links.append(f"<section><h2>{html.escape(entry['case_id'])}</h2><p>入选池: {entry['motion_targets']} points</p>"
                         f"<video controls preload='none' src='{url}/selected_motion.mp4'></video>"
                         f"<p><a href='{url}/motion_filter.json'>全部点的排名、入选和有效帧</a></p></section>")
        print(f"[motion-reselect-v68] completed={number+1}/{len(snapshot['complete_records'])} "
              f"case={entry['case_id']} selected={entry['motion_targets']}", flush=True)
    manifest = out / "training_manifest.json"
    write_json(manifest, {"contract": CONTRACT, "root": str(out), "entries": entries,
               "status": "completed_snapshot", "partition": partition, "source_revision": args.source_revision,
               "selection_policy": SELECTION_POLICY, "motion_top_fraction": args.motion_top_fraction,
               "completed_clips": len(entries), "clips_by_source": dict(Counter(e["source"] for e in entries)),
               "parent": snapshot, "teacher_only": True, "future_used_for_selection": True,
               "not_a_completed_400_clip_build": True})
    write_training_review(manifest, out / "training_samples", points=args.points, seed=args.seed,
                          epochs=tuple(int(e) for e in args.epochs.split(",")))
    (out / "index.html").write_text("\n".join(links), encoding="utf-8")
    metadata = [out / "index.html", manifest, snapshot_path, *sorted((out / "training_samples").rglob("*"))]
    write_review_bundle(out, entries, metadata)
    print(f"[motion-reselect-v68] clips={len(entries)} preview={out / 'training_samples/index.html'} "
          f"bundle={out / 'review_bundle.zip'}", flush=True)


if __name__ == "__main__":
    main()
