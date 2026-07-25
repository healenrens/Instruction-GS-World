"""Add the exact-token instruction cross-attention branch to a Prior."""
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
from igsw.adaptive_gaussian_wm.conditioning import (  # noqa: E402
    InstructionConditionStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--initialization_seed", type=int, default=2017)
    args = parser.parse_args()

    source_path = os.path.abspath(args.input)
    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if source.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("source must use the current checkpoint version")
    source_config = AdaptiveGaussianWMConfig(**source["config"])
    if source_config.token_conditioned_prior:
        raise ValueError("source already has token-conditioned Prior")
    target_config = replace(source_config, token_conditioned_prior=True)
    condition_store = InstructionConditionStore(args.condition_cache)
    if condition_store.token_features is None:
        raise ValueError("token-conditioned Prior requires a token cache")
    if condition_store.feature_dim != target_config.condition_dim:
        raise ValueError("token cache and model condition dimensions differ")
    torch.manual_seed(args.initialization_seed)
    model = AdaptiveGaussianObjectWorldModel(target_config)
    warm_start = warm_start_model(model, source)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("latent_actions.prior_token_conditioner.")
    }
    if set(warm_start["missing"]) != expected_missing:
        raise ValueError("missing tensors are not exactly the token conditioner")
    for field in ("unexpected", "shape_mismatch", "transformed", "dropped"):
        if warm_start[field]:
            raise ValueError(f"unexpected warm-start {field}: {warm_start[field]}")
    state = model.state_dict()
    changed_existing = [
        name
        for name, value in source["model"].items()
        if not torch.equal(value, state[name])
    ]
    if changed_existing:
        raise ValueError(f"existing model tensors changed: {changed_existing}")
    conditioner = model.latent_actions.prior_token_conditioner
    if conditioner is None or float(
        torch.tanh(conditioner.gate.detach())
    ) != 0.0:
        raise ValueError("token conditioner must start at an exact zero gate")

    output = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    source_args = dict(source.get("args", {}))
    source_args.update(
        {
            "token_conditioned_prior": True,
            "condition_cache": condition_store.path,
            "condition_feature_sha256": condition_store.feature_sha256,
            "condition_token_sha256": condition_store.token_feature_sha256,
        }
    )
    artifact = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": "token_conditioned_prior_init",
        "parallelism": "ddp_full_state_dict",
        "source_checkpoint": source_path,
        "model": state,
        "config": target_config.to_dict(),
        "args": source_args,
        "global_step": 0,
    }
    torch.save(artifact, output)
    report = {
        "status": "ok",
        "source": source_path,
        "output": output,
        "checkpoint_version": CHECKPOINT_VERSION,
        "initialization_seed": args.initialization_seed,
        "existing_model_tensor_count": len(source["model"]),
        "existing_model_tensors_exact": True,
        "new_token_conditioner_tensors": sorted(expected_missing),
        "new_token_conditioner_parameter_count": sum(
            value.numel()
            for name, value in state.items()
            if name in expected_missing
        ),
        "initial_token_gate": 0.0,
        "condition_cache": condition_store.path,
        "condition_feature_sha256": condition_store.feature_sha256,
        "condition_token_sha256": condition_store.token_feature_sha256,
        "warm_start": warm_start,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
