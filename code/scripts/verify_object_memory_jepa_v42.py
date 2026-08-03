#!/usr/bin/env python3
"""Server gate for stable correspondence Object Memory JEPA v42."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/pfs/public/xuhaoming/.cache")

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.dirname(__file__))

import verify_object_memory_jepa_v39 as base  # noqa: E402

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.v40_verification import (  # noqa: E402
    verify_v40_state_contracts,
)
from igsw.adaptive_gaussian_wm.v41_verification import (  # noqa: E402
    verify_v41_correspondence_contracts,
)
from igsw.adaptive_gaussian_wm.v42_runtime_contracts import (  # noqa: E402
    gate_contract_fields,
    runtime_contract_fields,
)
from igsw.adaptive_gaussian_wm.v42_verification import (  # noqa: E402
    verify_v42_objective_parts,
    verify_v42_stability_contracts,
)


ARCHITECTURE = "object_memory_v3"


def build_model(dataset, args, device):
    config = replace(
        AdaptiveGaussianWMConfig.object_memory_correspondence_full(
            dataset.feature_dim
        ),
        dual_horizon_dynamics=True,
        goal_rollout_weight=args.goal_rollout_weight,
        path_consistency_weight=args.path_consistency_weight,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    checkpoint_report = {"checkpoint": "not_provided"}
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        base.require(
            checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
            "checkpoint is not v42",
        )
        base.require(
            checkpoint.get("config") == config.to_dict(),
            "checkpoint config differs",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_report = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": base.file_sha256(args.checkpoint),
            "checkpoint_phase": checkpoint.get("phase"),
            "checkpoint_phase_step": checkpoint.get("phase_step"),
        }
    return model, checkpoint_report


def main() -> None:
    args = base.parse_args()
    base.require(torch.cuda.is_available(), "v42 verifier requires CUDA")
    local_gpus = torch.cuda.device_count()
    base.require(local_gpus > 0, "no visible CUDA GPU")
    if args.expected_local_gpus != "auto":
        base.require(
            args.expected_local_gpus.isdigit()
            and int(args.expected_local_gpus) == local_gpus,
            f"visible GPU count is {local_gpus}, expected {args.expected_local_gpus}",
        )
    commit = base.verify_repository()
    manifest_sha256, _ = base.verify_manifest(args.data)
    dataset = base.build_dataset(args)
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    temporal = base.verify_temporal_contract(dataset, args)
    device = torch.device("cuda:0")
    dino_amp = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    runtime = base.JitDinoFeatureRuntime(
        device,
        dino_amp,
        args.jit_dino_batch,
        args.goal_stability_threshold,
    )
    batch, goal_scan = base.select_v39_gate_batch(
        dataset,
        runtime,
        device,
        args.goal_gate_candidates,
    )
    del runtime
    torch.cuda.empty_cache()
    model, checkpoint = build_model(dataset, args, device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    model.eval()
    with torch.no_grad(), amp_context():
        causal = base.verify_v39_causal_paths(model, batch)
    training = base.verify_model_paths(model, batch, amp_context)
    objective_contracts = verify_v42_objective_parts(
        training["representation_lifecycle_objectives"]
    )
    correspondence_gradients = set(
        training["representation_correspondence_gradient_parameters"]
    )
    required_correspondence_gradients = {
        "object_memory.correspondence.dustbin_logit",
        "object_memory.correspondence.residual.3.weight",
    }
    base.require(
        required_correspondence_gradients <= correspondence_gradients,
        "representation loss does not train correspondence parameters",
    )
    correspondence_module = model.object_memory.correspondence
    base.require(correspondence_module is not None, "correspondence module is missing")
    base.configure_v28_stage(
        model,
        SimpleNamespace(architecture=ARCHITECTURE, training_stage="representation"),
    )
    weights = base.representation_weights()
    lifecycle_transport = verify_v40_state_contracts(
        model,
        batch,
        amp_context,
        weights,
    )
    correspondence = verify_v41_correspondence_contracts(
        model,
        batch,
        amp_context,
        weights,
    )
    stability = verify_v42_stability_contracts(model)
    report = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": base.READOUT_BACKEND,
        **gate_contract_fields(ARCHITECTURE),
        **runtime_contract_fields(ARCHITECTURE),
        "feature_source": "jit",
        "feature_contract": base.FEATURE_CONTRACT,
        "jit_dino_model": base.JIT_DINO_MODEL,
        "jit_dino_image_size": base.JIT_DINO_IMAGE_SIZE,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "goal_stability_threshold": args.goal_stability_threshold,
        **goal_scan,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "control_hz": float(dataset.control_hz),
        "temporal_contract": base.DYNAMIC_DUAL_HORIZON_CONTRACT,
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
        "local_gpu_count": local_gpus,
        "gpu_policy": args.expected_local_gpus,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        **checkpoint,
        **temporal,
        **causal,
        **training,
        **lifecycle_transport,
        **correspondence,
        **stability,
        **objective_contracts,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
