#!/usr/bin/env python3
"""Held-data capacity gate for choosing a v29 Gaussian child count."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.carrier_contracts import (  # noqa: E402
    BASIS_GATE_CONTRACT,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.gaussian_basis_audit import audit_batch  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v28_training import ARCHITECTURE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--max_items", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum_gap_recovery", type=float, default=0.70)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.architecture != ARCHITECTURE:
        raise ValueError("basis source checkpoint has the wrong architecture")
    if checkpoint.get("phase") != "representation":
        raise ValueError("basis source checkpoint must be a representation stage")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model.eval()


def main() -> None:
    args = parse_args()
    if min(args.max_items, args.batch) <= 0:
        raise ValueError("audit sizes must be positive")
    if not 0.0 < args.minimum_gap_recovery <= 1.0:
        raise ValueError("minimum gap recovery must be in (0,1]")
    if not torch.cuda.is_available():
        raise ValueError("hierarchical basis audit requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    if status.strip():
        raise ValueError("hierarchical basis audit requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dataset = CausalVisualSequenceDataset(
        args.data, args.split, max_items=args.max_items
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    checkpoint, model = load_model(args.checkpoint, device)
    rows: dict[str, list[torch.Tensor]] = {}
    use_bf16 = torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                history = model.encode_history(batch)
            values = audit_batch(
                history["token_states"][-1],
                batch["history_features"][:, -1],
                batch["history_coordinates"][:, -1],
                batch["history_valid"][:, -1],
            )
            for name, value in values.items():
                rows.setdefault(name, []).append(value.detach().float().cpu())
    raw_metrics = {
        name: float(torch.cat(parts).mean()) for name, parts in rows.items()
    }
    nonfinite_metrics = sorted(
        name for name, value in raw_metrics.items() if not math.isfinite(value)
    )
    metrics = {
        name: value if math.isfinite(value) else None
        for name, value in raw_metrics.items()
    }
    finite_single_gap = math.isfinite(raw_metrics["feature_1"]) and math.isfinite(
        raw_metrics["token"]
    )
    single_gap = (
        max(raw_metrics["feature_1"] - raw_metrics["token"], 0.0)
        if finite_single_gap
        else None
    )
    recovery = {}
    for children in (2, 4, 8):
        feature = raw_metrics[f"feature_{children}"]
        if single_gap is None or not math.isfinite(feature):
            recovery[str(children)] = None
        elif single_gap <= 1e-8:
            recovery[str(children)] = 1.0
        else:
            recovery[str(children)] = 1.0 - max(
                feature - raw_metrics["token"], 0.0
            ) / single_gap
    eligible = [
        children
        for children in (2, 4, 8)
        if recovery[str(children)] is not None
        and recovery[str(children)] >= args.minimum_gap_recovery
    ]
    selected = min(eligible) if eligible else 0
    checks = {
        "nonempty_held_set": len(dataset) > 0,
        "all_metrics_are_finite": not nonfinite_metrics,
        "single_gaussian_gap_is_finite": single_gap is not None,
        "child_basis_recovers_required_gap": bool(eligible),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": BASIS_GATE_CONTRACT,
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": dataset.data_sha256,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "source_checkpoint_version": checkpoint.get("checkpoint_version"),
        "held_split": args.split,
        "held_items": len(dataset),
        "minimum_gap_recovery": args.minimum_gap_recovery,
        "selected_children": selected,
        "single_gaussian_gap": single_gap,
        "gap_recovery": recovery,
        "metrics": metrics,
        "nonfinite_metrics": nonfinite_metrics,
        "checks": checks,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise RuntimeError("hierarchical Gaussian basis gate failed")


if __name__ == "__main__":
    main()
