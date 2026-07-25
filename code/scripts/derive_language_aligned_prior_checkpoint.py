"""Add the existing language-effect alignment head to a Prior checkpoint."""
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--initialization_seed", type=int, default=1705)
    args = parser.parse_args()
    if args.weight <= 0.0:
        raise ValueError("language alignment weight must be positive")

    source = torch.load(
        args.input,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if source.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("source must use the current checkpoint version")
    source_config = AdaptiveGaussianWMConfig(**source["config"])
    if source_config.language_effect_weight != 0.0:
        raise ValueError("source already contains language-effect alignment")
    target_config = replace(
        source_config,
        language_effect_weight=args.weight,
    )
    torch.manual_seed(args.initialization_seed)
    model = AdaptiveGaussianObjectWorldModel(target_config)
    warm_start = warm_start_model(model, source)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("language_effect_alignment.")
    }
    if set(warm_start["missing"]) != expected_missing:
        raise ValueError(
            "warm-start missing tensors are not exactly the new alignment head"
        )
    for field in ("unexpected", "shape_mismatch", "transformed", "dropped"):
        if warm_start[field]:
            raise ValueError(f"unexpected warm-start {field}: {warm_start[field]}")
    target_state = model.state_dict()
    changed_existing = [
        name
        for name, value in source["model"].items()
        if not torch.equal(value, target_state[name])
    ]
    if changed_existing:
        raise ValueError(f"existing model tensors changed: {changed_existing}")

    output = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    source_args = dict(source.get("args", {}))
    source_args["language_effect_weight"] = args.weight
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_kind": "language_aligned_prior_init",
            "parallelism": "ddp_full_state_dict",
            "source_checkpoint": os.path.abspath(args.input),
            "model": target_state,
            "config": target_config.to_dict(),
            "args": source_args,
            "global_step": 0,
        },
        output,
    )
    report = {
        "status": "ok",
        "source": os.path.abspath(args.input),
        "output": output,
        "checkpoint_version": CHECKPOINT_VERSION,
        "language_effect_weight": args.weight,
        "initialization_seed": args.initialization_seed,
        "existing_model_tensor_count": len(source["model"]),
        "new_alignment_tensors": sorted(expected_missing),
        "new_alignment_parameter_count": sum(
            value.numel()
            for name, value in target_state.items()
            if name in expected_missing
        ),
        "existing_model_tensors_exact": True,
        "warm_start": warm_start,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
