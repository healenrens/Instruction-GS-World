"""Preflight audited pair data and an optional exact-resume checkpoint."""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch


def absolute_path(path: str, label: str) -> str:
    if not os.path.isabs(path):
        raise ValueError(f"{label} must be an absolute path: {path}")
    return os.path.abspath(path)


def load_audit(path: str, label: str) -> dict:
    with open(path) as handle:
        audit = json.load(handle)
    if audit.get("status") != "ok":
        raise ValueError(f"{label} audit status is not ok: {path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--pair_audit", required=True)
    parser.add_argument("--dino_audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--dino_dim", type=int, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    data = absolute_path(args.data, "data")
    dino = absolute_path(args.dino, "dino")
    pair_audit_path = absolute_path(args.pair_audit, "pair_audit")
    dino_audit_path = absolute_path(args.dino_audit, "dino_audit")
    output = absolute_path(args.out, "out")
    report_path = absolute_path(args.report, "report")
    if not os.path.isdir(data) or not os.path.isdir(dino):
        raise FileNotFoundError("pair and DINO roots must both exist")
    if args.world_size < 1 or args.batch < 1 or args.grad_accum < 1:
        raise ValueError("world_size, batch, and grad_accum must be positive")

    pair_audit = load_audit(pair_audit_path, "pair")
    dino_audit = load_audit(dino_audit_path, "DINO")
    if os.path.abspath(pair_audit["data"]) != data:
        raise ValueError("pair audit data root differs from --data")
    if os.path.abspath(dino_audit["pairs"]) != data:
        raise ValueError("DINO audit pair root differs from --data")
    if os.path.abspath(dino_audit["dino"]) != dino:
        raise ValueError("DINO audit root differs from --dino")
    if int(dino_audit["feature_dim"]) != args.dino_dim:
        raise ValueError("DINO audit feature_dim differs from model dino_dim")

    pair_names = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(data, "*.pt"))
    }
    dino_names = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(dino, "*.pt"))
    }
    if pair_names != dino_names:
        raise ValueError(
            "pair/DINO filenames differ: "
            f"missing={sorted(pair_names - dino_names)[:5]} "
            f"extra={sorted(dino_names - pair_names)[:5]}"
        )
    pair_count = len(pair_names)
    if pair_count != int(pair_audit["count"]) or pair_count != int(dino_audit["count"]):
        raise ValueError("current file count differs from audit reports")
    train_count = int(pair_audit["split_counts"]["train"])
    minimum_train = args.world_size * args.batch * args.grad_accum
    if train_count < minimum_train:
        raise ValueError(
            f"train split has {train_count} pairs but at least {minimum_train} are required"
        )

    resume_summary = None
    if args.resume:
        resume = absolute_path(args.resume, "resume")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if int(checkpoint.get("checkpoint_version", 0)) != 2:
            raise ValueError("resume checkpoint does not support exact DDP recovery")
        if int(checkpoint["world_size"]) != args.world_size:
            raise ValueError("resume checkpoint world size differs from launch world size")
        if len(checkpoint["rng_states"]) != args.world_size:
            raise ValueError("resume checkpoint has incomplete per-rank RNG states")
        resume_summary = {
            "path": resume,
            "phase": checkpoint["phase"],
            "posterior_step": int(checkpoint["posterior_step"]),
            "prior_step": int(checkpoint["prior_step"]),
        }

    result = {
        "status": "ok",
        "data": data,
        "dino": dino,
        "pair_audit": pair_audit_path,
        "dino_audit": dino_audit_path,
        "pair_count": pair_count,
        "train_count": train_count,
        "world_size": args.world_size,
        "batch_per_rank": args.batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.world_size * args.batch * args.grad_accum,
        "out": output,
        "resume": resume_summary,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary = f"{report_path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    os.replace(temporary, report_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
