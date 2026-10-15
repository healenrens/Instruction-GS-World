#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.simulation_data import prepare_simulation_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect recorded CLIPort RGB to a fixed prediction horizon.")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = prepare_simulation_manifest(args.collection, args.out)
    print({"manifest": args.out, "cases": len(manifest["entries"]), "sequence": manifest["sequence"]}, flush=True)
