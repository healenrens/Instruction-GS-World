#!/usr/bin/env python3
"""Publish completed worker manifests without changing the workers' target data."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json
from igsw.adaptive_gaussian_wm.grounded_motion_review_v68 import write_review_bundle
from igsw.adaptive_gaussian_wm.grounded_motion_export_v68 import SELECTION_POLICY
from igsw.adaptive_gaussian_wm.grounded_motion_training_review_v68 import write_training_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    worker_record = json.loads((root / "workers.json").read_text())
    workers = worker_record["count"]
    entries, manifests = [], []
    review_metadata = [root / name for name in ("index.html", "training_manifest.json", "workers.json")]
    links = ["<!doctype html><meta charset='utf-8'><h1>V68 data shards</h1>"]
    for directory in [root / f"shard_{i:04d}" for i in range(workers)]:
        path = directory / "training_manifest.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            manifests.append(payload)
            for entry in payload["entries"]:
                entries.append({**entry, "path": str(Path(directory.name) / entry["path"])})
        links.append(f"<p><a href='{html.escape(directory.name)}/index.html'>{html.escape(directory.name)} review</a></p>")
        review_metadata.extend(directory / name for name in ("index.html", "selection.json", "training_manifest.json", "decode_failures.json"))
    clips_by_source = dict(Counter(e["source"] for e in entries))
    review_by_source = dict(Counter(e["source"] for e in entries if e["rendered"]))
    rendered_clips = sum(review_by_source.values())
    links.insert(1, f"<p>Training clips: {len(entries)}; visualized clips: {rendered_clips}</p>"
                   f"<p>Per source: {html.escape(json.dumps(clips_by_source))}</p>")
    if manifests:
        write_json(root / "training_manifest.json", {"contract": manifests[0]["contract"], "root": str(root.resolve()), "entries": entries,
            "status": "completed" if len(manifests) == workers and all(m["status"] == "completed" for m in manifests) else "partial",
            "partition": manifests[0]["partition"], "configuration": manifests[0]["configuration"],
            "clips_by_source": clips_by_source, "review_by_source": review_by_source,
            "completed_clips": len(entries), "rendered_clips": rendered_clips,
            "decode_skipped_clips": sum(m.get("decode_skipped_clips", 0) for m in manifests),
            "source_revision": manifests[0]["source_revision"], "shards": len(manifests), "teacher_only": True})
        if rendered_clips and manifests[0]["configuration"].get("selection_policy") == SELECTION_POLICY:
            write_training_review(root / "training_manifest.json", root / "training_samples",
                                  seed=manifests[0]["configuration"]["seed"])
            links.insert(1, "<p><a href='training_samples/index.html'>实际训练loader样本（默认256点，epoch 0）</a></p>")
            review_metadata.extend((root / "training_samples").rglob("*"))
    (root / "index.html").write_text("\n".join(links), encoding="utf-8")
    if worker_record["configuration"]["render"] or worker_record["configuration"]["operation"] == "cameras":
        if worker_record["configuration"]["operation"] == "cameras":
            for rank in range(workers):
                review_metadata.extend((root / f"shard_{rank:04d}").rglob("*"))
        write_review_bundle(root, entries, review_metadata)
    print(f"[motion-data-v68] merged clips={len(entries)} rendered={rendered_clips} "
          f"by_source={clips_by_source} review_by_source={review_by_source} manifest={root / 'training_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
