"""Create an auditable warm start with a calibrated Prior token gate."""
from __future__ import annotations

import argparse
import json
import os

import torch


GATE_KEY = "latent_actions.prior_token_conditioner.gate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raw_gate_scale", type=float, required=True)
    args = parser.parse_args()
    if args.raw_gate_scale < 0.0:
        raise ValueError("raw gate scale must be non-negative")

    source_path = os.path.abspath(args.input)
    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = source.get("config", {})
    if not config.get("token_conditioned_prior", False):
        raise ValueError("source checkpoint has no token-conditioned Prior")
    source_state = source.get("model")
    if not isinstance(source_state, dict) or GATE_KEY not in source_state:
        raise ValueError("source checkpoint is missing the token gate")
    state = dict(source_state)
    original_gate = source_state[GATE_KEY].detach().clone()
    state[GATE_KEY] = original_gate * args.raw_gate_scale
    changed = [
        name
        for name in source_state
        if not torch.equal(source_state[name], state[name])
    ]
    if changed != [GATE_KEY]:
        raise ValueError(f"gate calibration changed unexpected tensors: {changed}")

    output = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    artifact = {
        "checkpoint_version": source["checkpoint_version"],
        "checkpoint_kind": "token_conditioned_prior_gate_init",
        "parallelism": "ddp_full_state_dict",
        "source_checkpoint": source_path,
        "model": state,
        "config": config,
        "args": dict(source.get("args", {})),
        "global_step": 0,
        "token_gate_calibration": {
            "raw_gate_scale": args.raw_gate_scale,
            "raw_gate_before": float(original_gate),
            "raw_gate_after": float(state[GATE_KEY]),
            "effective_gate_before": float(torch.tanh(original_gate)),
            "effective_gate_after": float(torch.tanh(state[GATE_KEY])),
        },
    }
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(artifact, temporary)
    os.replace(temporary, output)
    report = {
        "status": "ok",
        "source": source_path,
        "output": output,
        "model_tensor_count": len(state),
        "changed_model_tensors": changed,
        "unchanged_model_tensor_count": len(state) - len(changed),
        **artifact["token_gate_calibration"],
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
