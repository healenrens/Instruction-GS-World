#!/usr/bin/env python3
"""Package completed cases without loading models or modifying a running build."""

import argparse
from collections import Counter
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.adaptive_gaussian_wm.grounded_motion_review_v68 import write_data_gallery, write_review_bundle
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    entries = []
    metadata = [root / "partial_index.html", root / "partial_manifest.json"]
    links = ["<!doctype html><meta charset='utf-8'><h1>V68 partial review</h1>",
             "<p>Only completed cases are included. This is not the complete requested dataset.</p>"]
    for shard in sorted(root.glob("shard_*")):
        local = [read_json(path)["entry"] for path in sorted(shard.glob("*/complete.json"))]
        selection = read_json(shard / "selection.json")
        write_json(shard / "partial_manifest.json", {"status": "partial_snapshot", "entries": local,
                   "completed_clips": len(local), "rendered_clips": sum(e["rendered"] for e in local)})
        write_data_gallery(shard, local, selection, index_name="partial_index.html", manifest_name="partial_manifest.json")
        entries.extend({**entry, "path": str(Path(shard.name) / entry["path"])} for entry in local)
        metadata.extend(shard / name for name in ("partial_index.html", "partial_manifest.json", "selection.json", "decode_failures.json"))
        links.append(f"<p><a href='{html.escape(shard.name)}/partial_index.html'>{html.escape(shard.name)}: {len(local)} completed clips</a></p>")
    counts = dict(Counter(e["source"] for e in entries))
    report = {"status": "partial_snapshot", "entries": entries, "completed_clips": len(entries),
              "rendered_clips": sum(e["rendered"] for e in entries), "clips_by_source": counts,
              "not_for_training": True}
    write_json(root / "partial_manifest.json", report)
    links.insert(2, f"<p>Completed clips: {len(entries)}; by source: {html.escape(str(counts))}</p>")
    (root / "partial_index.html").write_text("\n".join(links), encoding="utf-8")
    write_review_bundle(root, entries, metadata, name="partial_review_bundle.zip")
    print(f"[motion-data-v68] partial_clips={len(entries)} by_source={counts} bundle={root / 'partial_review_bundle.zip'}", flush=True)


if __name__ == "__main__":
    main()
