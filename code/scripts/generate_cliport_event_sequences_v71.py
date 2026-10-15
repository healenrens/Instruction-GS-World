#!/usr/bin/env python3
"""Run paired event orders in CLIPort and record independent simulator truth."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-report", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/metadata/object_event_benchmark_environment_v71.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="17,18")
    parser.add_argument("--history-seconds", type=float, default=3.)
    parser.add_argument("--future-seconds", type=float, default=10.)
    parser.add_argument("--plan", default="", help="Scene-disjoint program plan; overrides --seeds.")
    parser.add_argument("--tail-seconds", type=float, default=1.)
    parser.add_argument("--source-revision", default="")
    args = parser.parse_args()
    environment = json.loads(Path(args.environment_report).read_text())
    repository = Path(environment["cliport_source"])
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from igsw.object_event_wm.cliport_sequence import generate
    result = generate(repository/"cliport/environments/assets", args.out,
                      [int(seed) for seed in args.seeds.split(",")],
                      args.history_seconds, args.tail_seconds, args.source_revision,
                      args.future_seconds, args.plan)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
