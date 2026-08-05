#!/usr/bin/env python3
"""Server gate for model-owned DINO and compact Object-Region JEPA v43."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
import subprocess
import sys

import torch
from torch.utils.data._utils.collate import default_collate

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
from igsw.adaptive_gaussian_wm.dynamic_dual_horizon_dataset import (  # noqa: E402
    DYNAMIC_DUAL_HORIZON_CONTRACT,
    DynamicDualHorizonEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    JIT_DINO_IMAGE_SIZE,
    JIT_DINO_MODEL,
    file_sha256,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v42_runtime_contracts import (  # noqa: E402
    gate_contract_fields,
)
from igsw.adaptive_gaussian_wm.v43_curriculum import curriculum_at  # noqa: E402
from igsw.adaptive_gaussian_wm.v43_stage_contracts import (  # noqa: E402
    validate_v43_initialization,
    validate_v43_warm_start_report,
)
from igsw.adaptive_gaussian_wm.v43_verification import (  # noqa: E402
    verify_v43_action_factorization,
    verify_v43_persistence_contracts,
)
from verify_object_memory_jepa_v39 import (  # noqa: E402
    verify_manifest,
    verify_temporal_contract,
)


ARCHITECTURE = "object_region_memory_v1"
FEATURE_CONTRACT = "model_owned_trainable_dinov2_l_1024_region_768"
READOUT_BACKEND = "offline_probe_only"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--init_from", default="")
    parser.add_argument("--expected_local_gpus", default="auto")
    parser.add_argument("--jit_dino_batch", type=int, default=16)
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--goal_stability_threshold", type=float, default=0.05)
    parser.add_argument("--goal_rollout_weight", type=float, default=1.0)
    parser.add_argument("--path_consistency_weight", type=float, default=0.25)
    args = parser.parse_args()
    for name in ("data", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    for name in ("teacher_sidecar", "init_from"):
        value = getattr(args, name)
        require(not value or os.path.isabs(value), f"--{name} must be absolute")
    require(args.jit_dino_batch > 0, "DINO frame batch must be positive")
    return args


def repository_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "v43 verifier rejects tracked worktree changes")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def build_dataset(args):
    return DynamicDualHorizonEpisodeDataset(
        args.data,
        "train",
        history_frames_min=1,
        history_frames_max=4,
        history_span_frames=args.history_span_frames,
        short_horizon_frames=30,
        goal_query_seconds=args.goal_query_seconds,
        goal_tail_guard_frames=args.goal_tail_guard_frames,
        goal_probe_frames=args.goal_probe_frames,
        max_items=32,
        teacher_sidecar=args.teacher_sidecar,
        feature_source="jit",
    )


def build_model(dataset, args, device):
    config = replace(
        AdaptiveGaussianWMConfig.object_region_memory_full(dataset.feature_dim),
        dino_frame_batch=args.jit_dino_batch,
        goal_stability_threshold=args.goal_stability_threshold,
        goal_rollout_weight=args.goal_rollout_weight,
        path_consistency_weight=args.path_consistency_weight,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    warm_start = {"checkpoint": "not_provided"}
    if args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", weights_only=False, mmap=True
        )
        validate_v43_initialization(
            checkpoint,
            argparse.Namespace(
                architecture=ARCHITECTURE, training_stage="representation"
            ),
        )
        report = warm_start_model(model, checkpoint)
        validate_v43_warm_start_report(
            report, argparse.Namespace(architecture=ARCHITECTURE)
        )
        warm_start = {
            "checkpoint": os.path.abspath(args.init_from),
            "checkpoint_sha256": file_sha256(args.init_from),
            "warm_start_loaded": report["loaded"],
            "warm_start_missing": len(report["missing"]),
        }
    return model, warm_start


def verify_parameter_contract(model) -> dict:
    blocks = model.online_dino.backbone.blocks
    lower = [parameter for block in blocks[:12] for parameter in block.parameters()]
    upper = [parameter for block in blocks[12:] for parameter in block.parameters()]
    upper_ids = {id(parameter) for parameter in upper}
    upper_ids.update(id(parameter) for parameter in model.online_dino.backbone.norm.parameters())
    frozen_backbone = [
        parameter
        for parameter in model.online_dino.backbone.parameters()
        if id(parameter) not in upper_ids
    ]
    projector = list(model.online_dino.projector.parameters())
    require(frozen_backbone and not any(parameter.requires_grad for parameter in frozen_backbone),
            "DINO patch embedding/lower backbone is not frozen")
    require(upper and all(parameter.requires_grad for parameter in upper),
            "DINO upper 12 blocks are not trainable")
    require(projector and all(parameter.requires_grad for parameter in projector),
            "DINO region projector is not trainable")
    require(
        not any(parameter.requires_grad for parameter in model.target_dino.parameters()),
        "EMA DINO has trainable parameters",
    )
    require(model.latent_actions is None, "v43 unexpectedly contains a History Prior")
    require(model.gaussian_readout is None and model.change_readout is None,
            "v43 unexpectedly contains a core dense readout")
    return {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "trainable_parameter_count": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "dino_lower_trainable_tensors": sum(
            parameter.requires_grad for parameter in lower
        ),
        "dino_upper_trainable_tensors": sum(
            parameter.requires_grad for parameter in upper
        ),
        "dino_projector_trainable_tensors": sum(
            parameter.requires_grad for parameter in projector
        ),
    }


def verify_curriculum(model) -> dict:
    probes = {
        step: curriculum_at(step, model.config)
        for step in (0, 5000, 7000, 20000, 22000, 50000)
    }
    require(probes[0].dynamics_weight == 0.0, "Dynamics starts before step 5k")
    require(probes[5000].dynamics_weight == 0.0, "Dynamics ramp boundary differs")
    require(probes[7000].dynamics_weight == 1.0, "Dynamics ramp is not 2k")
    require(probes[20000].posterior_weight == 0.0, "posterior boundary differs")
    require(probes[22000].posterior_weight == 1.0, "posterior ramp is not 2k")
    return {
        f"curriculum_{step}": {
            "phase": value.phase,
            "dynamics_weight": value.dynamics_weight,
            "posterior_weight": value.posterior_weight,
        }
        for step, value in probes.items()
    }


def gradient_contract(model, batch, amp_context) -> tuple[dict, dict]:
    model.train()
    model.set_curriculum_step(22000)
    model.zero_grad(set_to_none=True)
    with amp_context():
        output = model(batch, collect_diagnostics=True)
    require(bool(torch.isfinite(output["loss"])), "v43 loss is non-finite")
    output["loss"].backward()
    groups = {
        "dino_lower": [],
        "dino_upper": [],
        "dino_projector": [],
        "allocator_budget": [],
        "target": [],
        "region": [],
        "posterior": [],
        "root_dynamics": [],
    }
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if name.startswith("online_dino.backbone.blocks."):
            block = int(name.split(".")[3])
            groups["dino_lower" if block < 12 else "dino_upper"].append(parameter)
        elif name.startswith("online_dino.projector."):
            groups["dino_projector"].append(parameter)
        elif name.startswith("allocator.budget_head."):
            groups["allocator_budget"].append(parameter)
        elif name.startswith("target_"):
            groups["target"].append(parameter)
        elif name.startswith(("region_transformer.", "region_memory.", "region_dynamics.")):
            groups["region"].append(parameter)
        elif name.startswith("region_effect_posterior."):
            groups["posterior"].append(parameter)
        elif name.startswith("dynamics."):
            groups["root_dynamics"].append(parameter)
    require(not nonfinite, f"non-finite gradients: {nonfinite}")
    require(not any(parameter.grad is not None for parameter in groups["dino_lower"]),
            "frozen DINO blocks received gradients")
    require(any(parameter.grad is not None for parameter in groups["dino_upper"]),
            "trainable DINO blocks received no gradients")
    require(not any(parameter.grad is not None for parameter in groups["target"]),
            "EMA target received gradients")
    for name in (
        "dino_projector",
        "allocator_budget",
        "region",
        "posterior",
        "root_dynamics",
    ):
        require(any(parameter.grad is not None for parameter in groups[name]),
                f"{name} received no gradients")
    result = {
        f"{name}_gradient_tensors": sum(
            parameter.grad is not None for parameter in parameters
        )
        for name, parameters in groups.items()
    }
    result["loss"] = float(output["loss"].detach())
    result["goal_valid_fraction"] = float(
        output["future_horizon_valid"][:, 1].float().mean()
    )
    model.zero_grad(set_to_none=True)
    return output, result


@torch.no_grad()
def causal_contract(model, batch, amp_context) -> dict:
    model.eval()
    changed = dict(batch)
    changed["future_jit_rgb"] = batch["future_jit_rgb"].flip(1)
    with amp_context():
        reference = model(batch)
        altered = model(changed)
    online_difference = difference(
        reference["online"]["roots"]["slots"],
        altered["online"]["roots"]["slots"],
    )
    target_difference = difference(
        reference["target"]["roots"]["slots"],
        altered["target"]["roots"]["slots"],
    )
    posterior_difference = difference(
        reference["short_action"], altered["short_action"]
    )
    require(online_difference < 1e-6, "future content reached online history path")
    require(target_difference > 1e-6, "future swap did not change EMA targets")
    require(posterior_difference > 1e-6, "future swap did not change posterior")
    return {
        "history_future_swap_max_difference": online_difference,
        "target_future_swap_max_difference": target_difference,
        "posterior_future_swap_max_difference": posterior_difference,
    }


def state_contract(model, output, amp_context) -> dict:
    regions = output["online"]["regions"]
    active = (regions["presence"][:, -1] > 0.5).float()
    active_count = active.sum(dim=1)
    scene_fraction = (
        regions["owner"][:, -1, :, -2] * active
    ).sum(dim=1) / active_count.clamp_min(1.0)
    root_environment_weight = output["online"]["root_environment_weight"][:, -1]
    transient_exclusion_fraction = (
        ((root_environment_weight < 1.0).float() * active).sum(dim=1)
        / active_count.clamp_min(1.0)
    )
    require(bool(((active_count >= 64) & (active_count <= 256)).all()),
            "active region count is outside [64,256]")
    require(bool((scene_fraction <= 0.2501).all()), "scene region quota exceeded")
    require(
        bool((transient_exclusion_fraction <= 0.2501).all()),
        "transient root exclusion quota exceeded",
    )
    tensors = (
        regions["feature"],
        regions["center"],
        regions["covariance"],
        regions["owner"],
        regions["presence"],
        regions["identity_key"],
    )
    require(all(bool(torch.isfinite(value).all()) for value in tensors),
            "compact state contains non-finite values")
    transformer_inputs = output["online"]["transformer_inputs"]
    token_activation = torch.stack(
        [
            state.activation.squeeze(-1)
            for state in output["online"]["token_states"]
        ],
        dim=1,
    )
    token_active = token_activation > 0.5
    perturbed_inputs = transformer_inputs + (
        ~token_active
    )[..., None].to(transformer_inputs.dtype) * 100.0
    model.eval()
    with torch.no_grad(), amp_context():
        reference = model.region_transformer._run(
            transformer_inputs, token_active
        )
        perturbed = model.region_transformer._run(
            perturbed_inputs, token_active
        )
    inactive_leakage = float(
        (
            (reference.float() - perturbed.float()).abs()
            * token_active[..., None].float()
        ).max()
    )
    require(inactive_leakage < 1e-6, "inactive regions leaked into active states")
    return {
        "active_region_count": float(active_count.mean()),
        "scene_region_fraction": float(scene_fraction.mean()),
        "transient_root_exclusion_fraction": float(
            transient_exclusion_fraction.mean()
        ),
        "region_feature_shape": list(regions["feature"].shape),
        "root_feature_shape": list(output["online"]["roots"]["slots"].shape),
        "inactive_region_leakage_max_difference": inactive_leakage,
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v43 verifier requires CUDA")
    gpu_count = torch.cuda.device_count()
    require(gpu_count > 0, "v43 verifier sees no CUDA GPU")
    if args.expected_local_gpus != "auto":
        require(args.expected_local_gpus.isdigit(), "expected GPU count is invalid")
        require(int(args.expected_local_gpus) == gpu_count,
                f"visible GPU count is {gpu_count}")
    commit = repository_commit()
    manifest_sha256, _ = verify_manifest(args.data)
    dataset = build_dataset(args)
    temporal = verify_temporal_contract(dataset, args)
    sample = dataset[(0, 2)]
    batch = move_to_device(default_collate([sample]), torch.device("cuda:0"))
    model, warm_start = build_model(dataset, args, torch.device("cuda:0"))
    parameter_contract = verify_parameter_contract(model)
    curriculum = verify_curriculum(model)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    output, gradients = gradient_contract(model, batch, amp_context)
    states = state_contract(model, output, amp_context)
    persistence = verify_v43_persistence_contracts(model, output, amp_context)
    factorization = verify_v43_action_factorization(
        model, batch, output, amp_context
    )
    causal = causal_contract(model, batch, amp_context)
    report = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": READOUT_BACKEND,
        "feature_source": "jit",
        "feature_contract": FEATURE_CONTRACT,
        "jit_dino_model": JIT_DINO_MODEL,
        "jit_dino_image_size": JIT_DINO_IMAGE_SIZE,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "dino_trainable_blocks": 12,
        "dino_target": "ema_no_grad",
        "region_feature_dim": 768,
        "curriculum_boundaries": [5000, 20000, 50000],
        "goal_stability_threshold": args.goal_stability_threshold,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "control_hz": float(dataset.control_hz),
        "temporal_contract": DYNAMIC_DUAL_HORIZON_CONTRACT,
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
        "local_gpu_count": gpu_count,
        "gpu_policy": args.expected_local_gpus,
        **gate_contract_fields(ARCHITECTURE),
        **warm_start,
        **temporal,
        **parameter_contract,
        **curriculum,
        **gradients,
        **states,
        **persistence,
        **factorization,
        **causal,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
