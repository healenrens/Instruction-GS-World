#!/usr/bin/env python3
"""Download the authors' CLIPort archive and inspect its actual data fields."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import time
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image


DATA_URL = "https://www.dropbox.com/scl/fi/qfpd20jvkzdb2ilj6eilh/PutInBowl.zip?rlkey=pb5dnlbfk0o08ox9bt7nvigx6&st=s1ysr4hq&dl=1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default=DATA_URL)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "PutInBowl.zip"
    if not archive_path.is_file():
        partial = root / "PutInBowl.zip.partial"
        received, last_print = 0, time.monotonic()
        with urlopen(args.url, timeout=120) as source, partial.open("wb") as destination:
            while chunk := source.read(8*1024*1024):
                destination.write(chunk)
                received += len(chunk)
                if time.monotonic()-last_print >= 15:
                    print(json.dumps({"event": "benchmark_archive_download", "bytes": received}), flush=True)
                    last_print = time.monotonic()
        partial.replace(archive_path)
    with ZipFile(archive_path) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        by_episode = defaultdict(list)
        for entry in files:
            path = PurePosixPath(entry.filename)
            components = path.parts
            episode_position = next((i for i, part in enumerate(components)
                                     if part.startswith("episode") and part[len("episode"):].isdigit()), None)
            if episode_position is not None:
                by_episode[str(PurePosixPath(*components[:episode_position+1]))].append(entry.filename)
        splits = Counter()
        sample_episodes = []
        for episode, names in sorted(by_episode.items()):
            components = PurePosixPath(episode).parts
            split = next((part for part in components if part in ("train", "val", "valid", "test", "eval")), "unspecified")
            splits[split] += 1
            if sum(row["split"] == split for row in sample_episodes) >= 2:
                continue
            caption = next((name for name in names if PurePosixPath(name).name == "task_description.txt"), None)
            colors = sorted(name for name in names if "color" in PurePosixPath(name).parts and name.endswith(".png"))
            dimensions = None
            if colors:
                with archive.open(colors[0]) as stream, Image.open(stream) as image:
                    dimensions = {"width": image.width, "height": image.height, "mode": image.mode}
            sample_episodes.append({"episode": episode, "split": split,
                "caption": archive.read(caption).decode("utf-8").strip() if caption else None,
                "color_frames": len(colors), "image": dimensions,
                "file_examples": names[:12],
                "non_color_file_examples": [name for name in names if name not in colors][:20]})
        evidence_files = [entry.filename for entry in files if any(
            word in PurePosixPath(entry.filename).name.lower() or word in PurePosixPath(entry.filename).parent.name.lower()
            for word in ("mask", "segm", "pose", "depth", "camera", "metadata", "timestamp", "fps"))]
        report = {"source_url": args.url, "archive": str(archive_path.resolve()),
                  "archive_bytes": archive_path.stat().st_size, "files": len(files),
                  "uncompressed_bytes": sum(entry.file_size for entry in files),
                  "episodes": len(by_episode), "split_counts": dict(splits),
                  "extensions": dict(Counter(PurePosixPath(entry.filename).suffix for entry in files)),
                  "sample_episodes": sample_episodes, "possible_truth_or_time_files": len(evidence_files),
                  "possible_truth_or_time_examples": evidence_files[:40], "extracted": args.extract,
                  "truth_semantics_verified": False, "physical_fps_assumed": False}
        if args.extract:
            archive.extractall(root / "extracted")
    (root / "archive_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
