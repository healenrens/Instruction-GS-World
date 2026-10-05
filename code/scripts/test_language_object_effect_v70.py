#!/usr/bin/env python3
"""Full-model distributed update/save/resume experiment with real windows."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = Path(__file__).with_name("train_language_object_effect_v70.py")
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={args.nproc_per_node}"]
    subprocess.run(launcher+[str(Path(__file__).with_name("export_language_effect_labels_v70.py")),
                    "--manifest", args.manifest, "--limit", str(2 * args.nproc_per_node),
                    "--probe_output", str(out/"teacher_future_swap.json")], check=True)
    common = launcher+[str(train),
              "--manifest", args.manifest, "--model_path", args.model_path,
              "--batch", str(args.batch), "--accum", "1", "--steps", "2", "--workers", "0",
              "--checkpoint_every", "1", "--snapshot_every", "0", "--retain", "1",
              "--log_every", "1", "--trace", "--swanlab_mode", "disabled"]
    uninterrupted, resumed = out/"uninterrupted", out/"resumed"
    subprocess.run(common+["--out", str(uninterrupted)], check=True)
    subprocess.run(common+["--out", str(resumed), "--stop_after", "1"], check=True)
    subprocess.run(common+["--out", str(resumed), "--resume", str(resumed/"latest.json")], check=True)
    mismatches = []
    rank_counts = []
    for rank in range(args.nproc_per_node):
        filename = f"trace_rank_{rank:05d}.jsonl"
        first = [json.loads(line) for line in (uninterrupted/filename).read_text().splitlines()]
        second = [json.loads(line) for line in (resumed/filename).read_text().splitlines()]
        rank_counts.append({"rank": rank, "event_count": len(first), "resumed_event_count": len(second)})
        if not first or len(first) != len(second):
            mismatches.append({"rank": rank, "field": "event_count"})
        for index, (a, b) in enumerate(zip(first, second)):
            location = {"rank": rank, "event": index}
            if a["event"] != b["event"]:
                mismatches.append({**location, "field": "event_type"})
                continue
            if a["event"] == "microbatch":
                exact = ("samples", "rank", "step", "microbatch", "epoch", "cursor", "noise", "tau", "target_mean", "history_times")
                for name in exact:
                    if a[name] != b[name]:
                        mismatches.append({**location, "field": name, "first": a[name], "resumed": b[name]})
                if not torch.allclose(torch.tensor(a["loss"]), torch.tensor(b["loss"]), atol=1e-5, rtol=1e-4):
                    mismatches.append({**location, "field": "loss", "first": a["loss"], "resumed": b["loss"]})
            else:
                for name in a["scalars"]:
                    if not torch.allclose(torch.tensor(a["scalars"][name]), torch.tensor(b["scalars"][name]), atol=1e-5, rtol=1e-4):
                        mismatches.append({**location, "field": name})
    report = {"test": "full_qwen_and_expert_real_window_resume", "world_size": args.nproc_per_node,
              "rank_event_counts": rank_counts, "mismatches": mismatches,
              "numeric_tolerance": {"atol": 1e-5, "rtol": 1e-4},
              "passed": not mismatches}
    swap = json.loads((out/"teacher_future_swap.json").read_text())
    report["teacher_future_swap"] = swap
    report["passed"] &= bool(swap["cases"]) and all(row.get("history_query_max_difference", float("inf")) < 1e-6 for row in swap["cases"])
    updates = json.loads((uninterrupted/"module_update_report.json").read_text())
    report["module_updates"] = updates
    report["passed"] &= sorted(row["rank"] for row in updates["ranks"]) == list(range(args.nproc_per_node))
    report["passed"] &= all(value == 0 for value in updates["optimizer_exclusions"].values())
    report["passed"] &= all(row["finite_grad_norm"] and
                            all(probe["changed_elements"] > 0 for probe in row["updates"].values())
                            for row in updates["ranks"])
    (out/"resume_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert report["passed"], "full-model resume trajectory differs; see resume_report.json"


if __name__ == "__main__":
    main()
