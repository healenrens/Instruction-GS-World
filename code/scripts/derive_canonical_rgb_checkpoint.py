"""Convert an RGB-action residual checkpoint to canonical-only without updates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    warm_start_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--canonical_center_gate", type=float, default=0.1)
    parser.add_argument("--canonical_activity_gate", action="store_true")
    parser.add_argument("--canonical_activity_power", type=float, default=0.5)
    parser.add_argument(
        "--action_residual_dim", type=int, choices=(0, 8, 16), default=0
    )
    parser.add_argument("--action_residual_gate", type=float, default=1.0)
    parser.add_argument("--action_residual_dropout", type=float, default=0.0)
    args = parser.parse_args()

    source_path = os.path.abspath(args.source)
    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_config = AdaptiveGaussianWMConfig(**source["config"])
    if not source_config.rgb_semantic_action:
        raise ValueError("source must use observable RGB semantic actions")
    if (
        source_config.action_residual_dim <= 0
        and args.action_residual_dim > 0
    ):
        raise ValueError("source must contain a residual action bottleneck")
    config = replace(
        source_config,
        action_dim=6 + args.action_residual_dim,
        canonical_center_gate=args.canonical_center_gate,
        canonical_activity_gate=args.canonical_activity_gate,
        canonical_activity_power=args.canonical_activity_power,
        bounded_residual_action=True,
        action_residual_gate=args.action_residual_gate,
        action_residual_dropout=args.action_residual_dropout,
    )
    model = AdaptiveGaussianObjectWorldModel(config)
    report = warm_start_model(model, source)
    if report["missing"] or report["unexpected"] or report["shape_mismatch"]:
        raise ValueError(f"incomplete canonical conversion: {report}")

    saved_args = dict(source.get("args", {}))
    saved_args.update(
        {
            "init_from": source_path,
            "canonical_center_gate": args.canonical_center_gate,
            "canonical_activity_gate": args.canonical_activity_gate,
            "canonical_activity_power": args.canonical_activity_power,
            "action_residual_dim": args.action_residual_dim,
            "action_residual_gate": args.action_residual_gate,
            "action_residual_dropout": args.action_residual_dropout,
            "semantic_action_basis": "rgb",
        }
    )
    artifact = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "parallelism": "ddp_full_state_dict",
        "artifact_kind": "warm_start_model_only",
        "resume_supported": False,
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "config": config.to_dict(),
        "args": saved_args,
        "phase": "converted",
        "phase_step": 0,
        "global_step": 0,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(artifact, temporary)
    os.replace(temporary, output)
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
