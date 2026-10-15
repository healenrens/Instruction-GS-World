#!/usr/bin/env python3
"""Summarize saved per-case results without loading models, video or simulator."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.paired_event_summary import summarize_paired_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    report = {"cases_file": str(Path(args.cases).resolve()), **summarize_paired_events(cases)}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(output), "aggregate": report["aggregate"],
                      "excluded_unpaired_scenes": report["excluded_unpaired_scenes"]}), flush=True)


if __name__ == "__main__":
    main()
