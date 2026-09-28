#!/usr/bin/env python3
"""Copy native media and teachers into a relocatable dataset without re-encoding or tracking."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import load_object_video_manifest_v69, resolve_object_video_case_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def make_plan(manifest_path, limit):
    manifest = load_object_video_manifest_v69(manifest_path)
    entries = manifest["entries"][:limit] if limit else manifest["entries"]
    assets, clips, missing = {}, [], []
    for entry in entries:
        teacher = Path(manifest["root"]) / entry["path"]
        case = resolve_object_video_case_v69(entry["case"], manifest["root"])
        media = Path(case["record"]["path"])
        absent = [str(path) for path in (teacher, media) if not path.is_file()]
        if absent:
            missing.append({"case_id": entry["case_id"], "source": entry["source"], "missing": absent})
            continue
        key = str(media)
        if key not in assets:
            assets[key] = {"source": key, "path": f"media/{len(assets):08d}{media.suffix}", "bytes": media.stat().st_size}
        portable_case = {**case, "record": {**case["record"], "path": assets[key]["path"]}}
        portable_entry = {**entry, "path": f"clips/{len(clips):08d}/teacher.pt", "case": portable_case}
        clips.append({"source_teacher": str(teacher), "entry": portable_entry, "teacher_bytes": teacher.stat().st_size})
    return {"contract": "portable_object_video_export_v1", "source_manifest": str(Path(manifest_path).resolve()),
            "manifest_metadata": {key: value for key, value in manifest.items() if key not in ("root", "entries")},
            "requested_clips": len(entries), "assets": list(assets.values()), "clips": clips, "missing": missing,
            "estimated_media_bytes": sum(asset["bytes"] for asset in assets.values()),
            "estimated_teacher_bytes": sum(clip["teacher_bytes"] for clip in clips)}


def copy_media(asset, root):
    target = root / asset["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    reused = target.is_file()
    if not reused:
        temporary = target.with_suffix(target.suffix + ".partial")
        # Materialize symlinks as regular files; packed videos are copied once in their original encoding.
        shutil.copyfile(asset["source"], temporary, follow_symlinks=True)
        temporary.replace(target)
    return {"path": asset["path"], "bytes": target.stat().st_size, "reused": reused}


def copy_teacher(clip, root):
    target = root / clip["entry"]["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    reused = target.is_file()
    if not reused:
        teacher = torch.load(clip["source_teacher"], map_location="cpu", mmap=True, weights_only=False)
        teacher["case"] = clip["entry"]["case"]
        temporary = target.with_suffix(".partial.pt")
        torch.save(teacher, temporary)
        temporary.replace(target)
    return {"path": clip["entry"]["path"], "bytes": target.stat().st_size, "reused": reused}


def export_phase(rows, operation, root, workers, phase):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(operation, row, root) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress = {"phase": phase, "completed": len(results), "total": len(rows), "last_file": result["path"]}
            write_json(root / "progress.json", progress)
            if len(results) == 1 or len(results) % 25 == 0 or len(results) == len(rows):
                print(json.dumps(progress), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--items", type=int, default=0, help="0 exports all clips in the input manifest.")
    parser.add_argument("--archive", default="", help="Default: OUT.tar, outside the data directory.")
    parser.add_argument("--no_archive", action="store_true", help="Export a directory for rsync without a second full tar copy.")
    parser.add_argument("--plan_only", action="store_true", help="Write and display the fixed copy plan and byte counts only.")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "export_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    else:
        plan = make_plan(args.manifest, args.items)
        write_json(plan_path, plan)
    print(json.dumps({"event": "portable_copy_plan", "requested_clips": plan["requested_clips"],
                      "included_clips": len(plan["clips"]), "missing_clips": len(plan["missing"]),
                      "unique_media_files": len(plan["assets"]), "media_bytes": plan["estimated_media_bytes"],
                      "teacher_bytes": plan["estimated_teacher_bytes"], "out": str(out),
                      "estimated_output_bytes_including_archive": (plan["estimated_media_bytes"]+plan["estimated_teacher_bytes"])*(1 if args.no_archive else 2),
                      "note": "packed source videos may include unused episodes; no re-encoding or hash checks"}), flush=True)
    if args.plan_only:
        return
    media_results = export_phase(plan["assets"], copy_media, out, args.workers, "copy_media")
    teacher_results = export_phase(plan["clips"], copy_teacher, out, args.workers, "copy_teachers")
    entries = [clip["entry"] for clip in plan["clips"]]
    counts = dict(Counter(f"{entry['source']}/{entry['partition']}" for entry in entries))
    write_json(out / "manifest.json", {**plan["manifest_metadata"], "root": ".", "entries": entries,
                                       "source_counts": counts, "portable_contract": "object_video_bundle_v1",
                                       "media_encoding": "original_file_bytes", "runtime_external_data_dependencies": False})
    report = {"status": "completed_with_missing_source_skips" if plan["missing"] else "completed",
              "requested_clips": plan["requested_clips"], "exported_clips": len(entries), "source_counts": counts,
              "missing": plan["missing"], "unique_media_files": len(media_results),
              "media_bytes": sum(row["bytes"] for row in media_results),
              "teacher_bytes": sum(row["bytes"] for row in teacher_results),
              "media_reused": sum(row["reused"] for row in media_results),
              "teachers_reused": sum(row["reused"] for row in teacher_results),
              "video_decode_validation_performed": False,
              "preserved": ["native RGB encoding", "frame offsets", "timestamps", "raw tracks", "relay", "selection", "train/held split"],
              "not_included": ["model weights", "Python environment", "training checkpoints"]}
    write_json(out / "export_report.json", report)
    (out / "README.md").write_text(
        "# Portable V69 dataset\n\nUse manifest.json with the V69 portable-path loader. "
        "The root and operational media/teacher paths are relative to this directory. "
        "Original paths inside export_plan.json or teacher provenance are historical metadata, not runtime dependencies.\n\n"
        "Media files retain their original encoding, including unused frames in packed videos. "
        "No video was re-encoded, resized, re-tracked or decoded for validation. "
        "Missing source files are listed in export_report.json.\n\n"
        "Model weights, code and the Python environment must be supplied separately. "
        "When migrating training checkpoints too, their saved dataset/asset paths require a separate relocation step.\n", encoding="utf-8")
    if not args.no_archive:
        archive = Path(args.archive).resolve() if args.archive else out.with_name(out.name + ".tar")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            temporary = archive.with_suffix(archive.suffix + ".partial")
            members = [out/name for name in ("manifest.json", "export_report.json", "export_plan.json", "README.md")]
            members += [out/asset["path"] for asset in plan["assets"]]
            members += [out/clip["entry"]["path"] for clip in plan["clips"]]
            with tarfile.open(temporary, "w") as stream:
                for number, path in enumerate(sorted(members), 1):
                    stream.add(path, arcname=(Path(out.name) / path.relative_to(out)).as_posix(), recursive=False)
                    if number == 1 or number % 100 == 0 or number == len(members):
                        progress = {"phase": "archive", "completed": number, "total": len(members), "archive": str(archive)}
                        write_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
            temporary.replace(archive)
        print(json.dumps({"archive": str(archive), "archive_bytes": archive.stat().st_size}), flush=True)
    write_json(out / "progress.json", {"phase": "completed", "clips": len(entries), "manifest": str(out / "manifest.json")})
    print(json.dumps({"event": "portable_export_complete", "manifest": str(out / "manifest.json"), "report": str(out / "export_report.json")}), flush=True)


if __name__ == "__main__":
    main()
