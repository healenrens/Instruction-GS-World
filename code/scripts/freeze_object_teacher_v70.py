#!/usr/bin/env python3
"""Create an explicit, small Stage 2 dependency without optimizer or tracking state."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    destination = Path(args.output).resolve()
    saved = torch.load(source, map_location="cpu", weights_only=False)
    fixed = {name: saved[name] for name in ("args", "config", "model", "perception", "step")}
    fixed["snapshot_source"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a preparation rerun from changing existing labels' teacher.
    with destination.open("xb") as stream:
        torch.save(fixed, stream)
    print(json.dumps({"fixed_teacher": str(destination), "step": fixed["step"],
                      "source": str(source), "architecture": fixed["config"]["architecture"]}), flush=True)


if __name__ == "__main__":
    main()
