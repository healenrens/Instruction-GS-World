#!/usr/bin/env python3
"""Prepare an RGB-only experiment manifest from the existing portable collection."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.data import SequenceConfig, prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-root")
    parser.add_argument("--language-manifest", default="")
    parser.add_argument("--history-seconds", type=float, default=3.0)
    parser.add_argument("--future-seconds", type=float, default=5.0)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--future-frames", type=int, default=25)
    args = parser.parse_args()
    config = SequenceConfig(args.history_seconds, args.future_seconds,
                            args.history_frames, args.future_frames)
    result = prepare_manifest(args.manifest, args.output, config,
                              args.language_manifest, args.media_root)
    report = result["report"]
    print(json.dumps({"manifest": str(Path(args.output).resolve()),
                      **{k: v for k, v in report.items() if k != "excluded"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
