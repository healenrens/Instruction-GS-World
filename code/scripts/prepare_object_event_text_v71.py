#!/usr/bin/env python3
"""Cache complete instructions using existing local Qwen weights, without RGB."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.data import write_json
from igsw.object_event_wm.text import FrozenInstructionEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    if index_path.is_file():
        metadata = json.loads(index_path.read_text())
    else:
        manifest = json.loads(Path(args.manifest).read_text())
        instructions = sorted({window["instruction"] for entry in manifest["entries"]
                               for window in entry["language_windows"]})
        metadata = {"instructions": instructions, "manifest": str(Path(args.manifest).resolve()),
                    "model": str(Path(args.model).resolve()), "input": "instruction_only",
                    "truncation": False}
    encoder = FrozenInstructionEncoder(metadata["model"]).cuda().eval()
    metadata["width"] = encoder.width
    write_json(index_path, metadata)
    pending = [i for i in range(len(metadata["instructions"])) if not (out / f"{i:07d}.pt").is_file()]
    for first in range(0, len(pending), args.batch):
        indices = pending[first:first+args.batch]
        values = encoder([metadata["instructions"][i] for i in indices])
        for index, value in zip(indices, values):
            path = out / f"{index:07d}.pt"
            temporary = path.with_suffix(".partial")
            torch.save(value, temporary)
            temporary.replace(path)
        print(json.dumps({"event": "instruction_features_saved", "new": min(first+args.batch, len(pending)),
                          "pending_at_start": len(pending), "instructions": len(metadata["instructions"])}), flush=True)
    write_json(out / "completion.json", {"instructions": len(metadata["instructions"]), "width": encoder.width})


if __name__ == "__main__":
    main()
