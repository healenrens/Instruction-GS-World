#!/usr/bin/env python3
"""Publish completed worker manifests without changing the workers' target data."""
import argparse
import html
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    worker_record = json.loads((root / "workers.json").read_text())
    workers = worker_record["count"]
    entries, manifests = [], []
    links = ["<!doctype html><meta charset='utf-8'><h1>V68 data shards</h1>"]
    for directory in [root / f"shard_{i:04d}" for i in range(workers)]:
        path = directory / "training_manifest.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            manifests.append(payload)
            for entry in payload["entries"]:
                entries.append({**entry, "path": str(Path(directory.name) / entry["path"])})
        links.append(f"<p><a href='{html.escape(directory.name)}/index.html'>{html.escape(directory.name)} review</a></p>")
    (root / "index.html").write_text("\n".join(links), encoding="utf-8")
    if manifests:
        write_json(root / "training_manifest.json", {"contract": manifests[0]["contract"], "root": str(root.resolve()), "entries": entries,
            "status": "completed" if len(manifests) == workers and all(m["status"] == "completed" for m in manifests) else "partial",
            "partition": manifests[0]["partition"], "configuration": manifests[0]["configuration"],
            "source_revision": manifests[0]["source_revision"], "shards": len(manifests), "teacher_only": True})
    if worker_record["configuration"]["render"] or worker_record["configuration"]["operation"] == "cameras":
        with zipfile.ZipFile(root / "review_bundle.zip", "w", compression=zipfile.ZIP_STORED) as bundle:
            artifacts = [root / name for name in ("index.html", "training_manifest.json", "workers.json")]
            for rank in range(workers):
                artifacts.extend((root / f"shard_{rank:04d}").rglob("*"))
            for path in sorted(artifacts):
                if path.is_file() and path.suffix in (".html", ".json", ".mp4", ".png"):
                    bundle.write(path, path.relative_to(root))
    print(f"[motion-data-v68] merged clips={len(entries)} manifest={root / 'training_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
