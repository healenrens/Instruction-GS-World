#!/usr/bin/env python3
"""One real-data LPWM update/recovery/generation test, using local pretrained loss assets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_baselines.lpwm import LPWMConfig, LPWMVideoModel
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.prediction import history_batch
from igsw.object_event_wm.training import device_batch, read_checkpoint, restore_rng, rng_state


def equal_state(first, second):
    if torch.is_tensor(first):
        return torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_state(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal_state(a, b) for a, b in zip(first, second))
    return first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "manifest", "text-cache", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--partition", default="engineering")
    parser.add_argument("--source-revision", default="")
    args = parser.parse_args()
    root = Path(args.out)
    scripts = Path(__file__).parent
    common = [sys.executable, str(scripts/"train_lpwm_object_events_v71.py"), "--assets", args.assets,
              "--manifest", args.manifest, "--text-cache", args.text_cache, "--partition", args.partition,
              "--steps", "2", "--batch", "1", "--accum", "1", "--workers", "0", "--warmup-epochs", "0",
              "--log-every", "1", "--save-every", "1", "--snapshot-every", "0", "--source-revision", args.source_revision]
    subprocess.run([*common, "--out", str(root/"uninterrupted")], check=True)
    subprocess.run([*common, "--out", str(root/"recovered"), "--stop-after", "1"], check=True)
    initial = read_checkpoint(root/"recovered/recovery_00000001.pt")
    config = LPWMConfig(**initial["model_config"])
    assets = initial["baseline_assets"]
    torch.cuda.set_device(0)
    model = LPWMVideoModel(config, assets["repository"]).cuda()
    model.load_state_dict(initial["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.native["lr"])
    optimizer.load_state_dict(initial["optimizer"])
    restore_rng(initial["rng"][0])
    roundtrip = {"model_equal": equal_state(model.state_dict(), initial["model"]),
                 "optimizer_equal": equal_state(optimizer.state_dict(), initial["optimizer"]),
                 "rng_equal": equal_state(rng_state(), initial["rng"][0])}
    write_json(root/"checkpoint_roundtrip.json", roundtrip)
    assert all(roundtrip.values()), roundtrip
    del model, optimizer, initial
    torch.cuda.empty_cache()
    subprocess.run([*common, "--out", str(root/"recovered"), "--resume", str(root/"recovered/latest.json")], check=True)
    full, recovered = read_checkpoint(root/"uninterrupted/latest.json"), read_checkpoint(root/"recovered/latest.json")
    first = [json.loads(line) for line in (root/"uninterrupted/train.jsonl").read_text().splitlines()]
    second = [json.loads(line) for line in (root/"recovered/train.jsonl").read_text().splitlines()]
    same_cases = [row["case_ids_rank0"] for row in first] == [row["case_ids_rank0"] for row in second]
    difference = max(float((value.cpu()-recovered["model"][key].cpu()).abs().max())
                     for key, value in full["model"].items() if value.is_floating_point())
    optimizer_steps = sorted({float(row["step"]) for row in recovered["optimizer"]["state"].values()})
    finite = all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in first+second)
    assert same_cases and recovered["step"] == 2 and max(optimizer_steps) == 2 and finite
    model = LPWMVideoModel(config, assets["repository"]).cuda().eval()
    model.load_state_dict(recovered["model"])
    dataset = build_video_dataset(args.manifest, partition=args.partition, text_cache=args.text_cache)
    batch = device_batch(collate_object_event_video([dataset[0]]), torch.device("cuda", 0))
    tf = batch["rgb"].shape[1]-batch["history_frames"]
    language = batch["text_tokens"].float()
    prediction = model.predict_history(history_batch(batch), tf, language, seed=17)
    swapped = dict(batch)
    swapped["rgb"] = batch["rgb"].clone()
    swapped["rgb"][:, batch["history_frames"]:] = swapped["rgb"][:, batch["history_frames"]:].flip(1)
    repeated = model.predict_history(history_batch(swapped), tf, language, seed=17)
    prefix_difference = float((prediction-repeated).abs().max())
    assert prefix_difference < 1e-6 and bool(torch.isfinite(prediction).all())
    report = {"status": "training_and_history_generation_completed", "source_revision": args.source_revision,
              "inventory": model.inventory(),
              "checkpoint_roundtrip": roundtrip, "sample_order_equal": same_cases,
              "resumed_step": recovered["step"], "optimizer_step_values": optimizer_steps,
              "independent_trajectory_parameter_difference": difference,
              "numeric_replay_scope": "exact state roundtrip; no bitwise CUDA grid-sample backward claim",
              "future_swap_history_prediction_difference": prefix_difference,
              "prediction_shape": list(prediction.shape), "finite_losses_and_gradients": finite,
              "scope": "engineering execution with real videos and official pretrained loss weights"}
    write_json(root/"test_report.json", report)
    del model, prediction, repeated
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(scripts/"evaluate_lpwm_rgb_v71.py"),
        "--checkpoint", str(root/"recovered/latest.json"), "--manifest", args.manifest,
        "--partition", args.partition, "--items-per-source", "1", "--out", str(root/"evaluation")], check=True)
    report["evaluation"] = json.loads((root/"evaluation/report.json").read_text())
    report["status"] = "passed"
    write_json(root/"test_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
