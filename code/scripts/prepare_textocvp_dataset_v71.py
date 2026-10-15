#!/usr/bin/env python3
"""Index the authors' CLIPort images and captions without extracting or re-encoding RGB."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.image_sequences import prepare_textocvp_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--future-frames", type=int, default=9)
    args = parser.parse_args()
    manifest = prepare_textocvp_manifest(args.archive, args.out, args.history_frames, args.future_frames)
    print(json.dumps({"output": args.out, **manifest["report"]}), flush=True)


if __name__ == "__main__":
    main()
