"""Freeze a node-independent contract before a two-node effect-core launch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re


NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
TRAINING_PROFILE = "effect_anchored_posterior_core_v27_100k_v1"
STRING_OPTIONS = {
    "--data_format": "sequence",
    "--sequence_anchors": "3,5,8",
    "--profile": "full",
    "--amp": "bf16",
    "--language_condition": "off",
    "--rgb_supervision": "on",
    "--posterior_update_scope": "full",
    "--action_anchor": "object_slot",
    "--semantic_action_basis": "rgb",
}
INTEGER_OPTIONS = {
    "--history_frames": 4,
    "--future_frames": 4,
    "--representation_steps": 0,
    "--joint_steps": 100000,
    "--batch": 2,
    "--grad_accum": 8,
    "--workers": 2,
    "--warmup_steps": 5000,
    "--save_every": 1000,
    "--log_every": 20,
    "--seed": 17,
    "--rgb_short_side": 256,
    "--rgb_pad_multiple": 16,
    "--rgb_render_chunk": 8192,
    "--action_residual_dim": 8,
}
FLOAT_OPTIONS = {
    "--lr": 5e-5,
    "--lr_floor": 5e-6,
    "--warmup_fraction": 0.0,
    "--weight_decay": 1e-4,
    "--rgb_loss_weight": 0.5,
    "--rgb_ssim_weight": 0.2,
    "--rgb_change_loss_weight": 1.0,
    "--rgb_change_threshold": 0.04,
    "--language_effect_weight": 0.0,
    "--zero_action_margin_weight": 5.0,
    "--canonical_center_gate": 1.0,
    "--canonical_activity_power": 1.0,
    "--action_residual_gate": 1.0,
    "--action_residual_dropout": 0.0,
}
REQUIRED_FLAGS = ("--posterior_dynamics_gate", "--canonical_activity_gate")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_file(path: str) -> str:
    result = os.path.abspath(path)
    if path != result or not os.path.isfile(result):
        raise ValueError(f"launch evidence must be an absolute file: {path}")
    return result


def option_value(arguments: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"training arguments require exactly one {name}")
    return arguments[positions[0] + 1]


def training_profile(arguments: list[str]) -> dict:
    observed: dict[str, str | int | float | bool] = {}
    for name, expected in STRING_OPTIONS.items():
        value = option_value(arguments, name)
        if value != expected:
            raise ValueError(f"training profile requires {name}={expected}")
        observed[name] = value
    for name, expected in INTEGER_OPTIONS.items():
        value = int(option_value(arguments, name))
        if value != expected:
            raise ValueError(f"training profile requires {name}={expected}")
        observed[name] = value
    for name, expected in FLOAT_OPTIONS.items():
        value = float(option_value(arguments, name))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"training profile requires {name}={expected}")
        observed[name] = value
    for name in REQUIRED_FLAGS:
        if arguments.count(name) != 1:
            raise ValueError(f"training profile requires exactly one {name}")
        observed[name] = True
    return {"name": TRAINING_PROFILE, "options": observed}


def evidence_records(items: list[str]) -> dict[str, dict]:
    records = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or NAME.fullmatch(name) is None or name in records:
            raise ValueError(f"invalid launch evidence declaration: {item}")
        path = absolute_file(path)
        records[name] = {
            "path": path,
            "sha256": file_sha256(path),
            "bytes": os.path.getsize(path),
        }
    if not records:
        raise ValueError("launch contract requires evidence files")
    return records


def contract_payload(args: argparse.Namespace) -> dict:
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    data = os.path.abspath(args.data)
    if args.root != root or not os.path.isdir(root):
        raise ValueError("launch root must be an absolute directory")
    if args.out != out or args.data != data or not os.path.isdir(data):
        raise ValueError("launch output and data paths must be absolute")
    if NAME.fullmatch(args.attempt_key) is None:
        raise ValueError("launch attempt key contains unsafe characters")
    if args.nnodes != 2 or args.nproc_per_node != 8:
        raise ValueError("launch contract requires exactly two nodes x eight GPUs")
    if min(
        args.master_port,
        args.batch_per_gpu,
        args.grad_accum,
    ) <= 0:
        raise ValueError("launch topology and batch values must be positive")
    effective_batch = (
        args.nnodes
        * args.nproc_per_node
        * args.batch_per_gpu
        * args.grad_accum
    )
    if effective_batch != 256:
        raise ValueError(f"launch global batch differs from 256: {effective_batch}")
    training = list(args.training_args)
    if not training or training[0] != "code/scripts/train_adaptive_gaussian_wm.py":
        raise ValueError("launch contract has an unexpected training entrypoint")
    forbidden = {
        "--node_rank",
        "--master_addr",
        "--master_port",
        "--nnodes",
        "--nproc_per_node",
    }
    if forbidden.intersection(training):
        raise ValueError("node-specific torchrun arguments entered training contract")
    if os.path.abspath(option_value(training, "--data")) != data:
        raise ValueError("training data differs from launch contract")
    if os.path.abspath(option_value(training, "--out")) != out:
        raise ValueError("training output differs from launch contract")
    if int(option_value(training, "--batch")) != args.batch_per_gpu:
        raise ValueError("training micro batch differs from launch contract")
    if int(option_value(training, "--grad_accum")) != args.grad_accum:
        raise ValueError("training accumulation differs from launch contract")
    profile = training_profile(training)
    checkpoint = absolute_file(args.checkpoint)
    init_option = "--resume" if args.mode == "resume" else "--init_from"
    other_option = "--init_from" if args.mode == "resume" else "--resume"
    if os.path.abspath(option_value(training, init_option)) != checkpoint:
        raise ValueError("training initialization differs from launch checkpoint")
    if other_option in training:
        raise ValueError("training arguments mix resume and warm-start modes")
    return {
        "schema_version": 2,
        "scope": "future_conditioned_posterior_core_only",
        "deployable_world_model_proven": False,
        "attempt_key": args.attempt_key,
        "mode": args.mode,
        "root": root,
        "out": out,
        "data": data,
        "checkpoint": {
            "path": checkpoint,
            "sha256": file_sha256(checkpoint),
            "bytes": os.path.getsize(checkpoint),
        },
        "topology": {
            "nnodes": args.nnodes,
            "nproc_per_node": args.nproc_per_node,
            "master_addr": args.master_addr,
            "master_port": args.master_port,
            "batch_per_gpu": args.batch_per_gpu,
            "grad_accum": args.grad_accum,
            "effective_global_batch": effective_batch,
        },
        "evidence": evidence_records(args.evidence),
        "training_profile": profile,
        "training_args": training,
    }


def write_contract(path: str, payload: dict) -> str:
    output = os.path.abspath(path)
    if path != output:
        raise ValueError("launch contract output must be absolute")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if os.path.exists(output):
        with open(output, encoding="utf-8") as handle:
            if handle.read() != serialized:
                raise FileExistsError(f"launch contract differs: {output}")
    else:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = f"{output}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temporary, output)
    return hashlib.sha256(serialized.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--attempt_key", required=True)
    parser.add_argument("--mode", choices=("warm_start", "resume"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--master_addr", required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--batch_per_gpu", type=int, required=True)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--training_args", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    payload = contract_payload(args)
    digest = write_contract(args.output, payload)
    print(json.dumps({"status": "ok", "sha256": digest, "output": args.output}))


if __name__ == "__main__":
    main()
