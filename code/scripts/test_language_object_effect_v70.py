#!/usr/bin/env python3
"""One full-model, single-GPU update/save/resume experiment with real windows."""

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
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = Path(__file__).with_name("train_language_object_effect_v70.py")
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_language_effect_labels_v70.py")),
                    "--manifest", args.manifest, "--limit", "2",
                    "--probe_output", str(out/"teacher_future_swap.json")], check=True)
    common = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(train),
              "--manifest", args.manifest, "--model_path", args.model_path,
              "--batch", str(args.batch), "--accum", "1", "--steps", "2", "--workers", "0",
              "--checkpoint_every", "1", "--snapshot_every", "0", "--retain", "1",
              "--log_every", "1", "--trace", "--swanlab_mode", "disabled"]
    uninterrupted, resumed = out/"uninterrupted", out/"resumed"
    subprocess.run(common+["--out", str(uninterrupted)], check=True)
    subprocess.run(common+["--out", str(resumed), "--stop_after", "1"], check=True)
    subprocess.run(common+["--out", str(resumed), "--resume", str(resumed/"latest.json")], check=True)
    first = [json.loads(line) for line in (uninterrupted/"trace_rank_00000.jsonl").read_text().splitlines()]
    second = [json.loads(line) for line in (resumed/"trace_rank_00000.jsonl").read_text().splitlines()]
    mismatches = []
    for index, (a, b) in enumerate(zip(first, second)):
        if a["event"] == "microbatch":
            exact = ("samples", "step", "epoch", "cursor", "noise", "tau", "target_mean", "history_times")
            for name in exact:
                if a[name] != b[name]:
                    mismatches.append({"event": index, "field": name, "first": a[name], "resumed": b[name]})
            if not torch.allclose(torch.tensor(a["loss"]), torch.tensor(b["loss"]), atol=1e-5, rtol=1e-4):
                mismatches.append({"event": index, "field": "loss", "first": a["loss"], "resumed": b["loss"]})
        else:
            for name in a["scalars"]:
                if not torch.allclose(torch.tensor(a["scalars"][name]), torch.tensor(b["scalars"][name]), atol=1e-5, rtol=1e-4):
                    mismatches.append({"event": index, "field": name})
    report = {"test": "single_gpu_full_qwen_and_expert_real_window_resume", "event_count": len(first),
              "resumed_event_count": len(second), "mismatches": mismatches,
              "numeric_tolerance": {"atol": 1e-5, "rtol": 1e-4},
              "passed": len(first) == len(second) and not mismatches}
    swap = json.loads((out/"teacher_future_swap.json").read_text())
    report["teacher_future_swap"] = swap
    report["passed"] &= all(row.get("history_query_max_difference", float("inf")) < 1e-6 for row in swap["cases"])
    updates = json.loads((uninterrupted/"module_update_report.json").read_text())
    report["module_updates"] = updates
    report["passed"] &= all(value == 0 for value in updates["optimizer_exclusions"].values())
    report["passed"] &= all(row["finite_grad_norm"] and
                            all(probe["changed_elements"] > 0 for probe in row["updates"].values())
                            for row in updates["ranks"])
    (out/"resume_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert report["passed"], "full-model resume trajectory differs; see resume_report.json"


if __name__ == "__main__":
    main()
