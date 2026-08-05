#!/usr/bin/env python3
"""Server-only gate for the dual DINO/Wan-VAE Object-Region JEPA v44."""
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
from igsw.adaptive_gaussian_wm.dual_encoder_temporal_dataset import (  # noqa: E402
    DUAL_ENCODER_TEMPORAL_CONTRACT,
    DualEncoderDynamicEpisodeDataset,
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
from igsw.adaptive_gaussian_wm.v43_verification import (  # noqa: E402
    verify_v43_action_factorization,
    verify_v43_persistence_contracts,
)
from igsw.adaptive_gaussian_wm.v44_model_runtime import (  # noqa: E402
    _video_feature_sequences,
)
from igsw.adaptive_gaussian_wm.v44_stage_contracts import (  # noqa: E402
    validate_v44_initialization,
    validate_v44_warm_start_report,
    v44_gate_fields,
)
from igsw.adaptive_gaussian_wm.video_vae_contract import (  # noqa: E402
    validate_video_vae_artifact,
)
from igsw.adaptive_gaussian_wm.v43_model_runtime import _encode_sequence  # noqa: E402
from verify_object_memory_jepa_v39 import (  # noqa: E402
    verify_manifest,
    verify_temporal_contract,
)
from verify_object_region_jepa_v43 import (  # noqa: E402
    gradient_contract,
    state_contract,
    verify_curriculum,
    verify_parameter_contract,
)


ARCHITECTURE = "object_region_dual_encoder_v1"
FEATURE_CONTRACT = "model_owned_dinov2_l_plus_frozen_wan_vae_region_768"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video_vae_model", required=True)
    parser.add_argument("--video_vae_contract", required=True)
    parser.add_argument("--video_vae_pythonpath", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--init_from", default="")
    parser.add_argument("--expected_local_gpus", default="auto")
    parser.add_argument("--jit_dino_batch", type=int, default=16)
    parser.add_argument("--video_vae_batch", type=int, default=1)
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--goal_stability_threshold", type=float, default=0.05)
    parser.add_argument("--goal_gate_candidates", type=int, default=32)
    parser.add_argument("--goal_rollout_weight", type=float, default=1.0)
    parser.add_argument("--path_consistency_weight", type=float, default=0.25)
    args = parser.parse_args()
    for name in (
        "data", "output", "video_vae_model", "video_vae_contract",
        "video_vae_pythonpath",
    ):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(os.path.isdir(args.video_vae_pythonpath),
            "--video_vae_pythonpath must be an existing directory")
    for name in ("teacher_sidecar", "init_from"):
        value = getattr(args, name)
        require(not value or os.path.isabs(value), f"--{name} must be absolute")
    require(args.jit_dino_batch > 0, "DINO frame batch must be positive")
    require(args.video_vae_batch > 0, "video VAE batch must be positive")
    return args


def repository_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "v44 verifier rejects tracked worktree changes")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def build_dataset(args, tail_guard: int | None = None):
    return DualEncoderDynamicEpisodeDataset(
        args.data,
        "train",
        history_frames_min=1,
        history_frames_max=4,
        history_span_frames=args.history_span_frames,
        short_horizon_frames=30,
        goal_query_seconds=args.goal_query_seconds,
        goal_tail_guard_frames=(
            args.goal_tail_guard_frames if tail_guard is None else tail_guard
        ),
        goal_probe_frames=args.goal_probe_frames,
        max_items=max(16, args.goal_gate_candidates),
        teacher_sidecar=args.teacher_sidecar,
        feature_source="jit",
        video_clip_frames=5,
    )


def verify_video_temporal_contract(dataset, args) -> dict:
    sample = dataset[(0, 4)]
    history = sample["history_video_control_indices"]
    short = sample["short_video_control_indices"]
    goal = sample["goal_video_control_indices"]
    anchor = int(sample["anchor_control_index"])
    require(len(history) == len(short) == len(goal) == 5, "VAE clips are not 5 frames")
    require(int(history[-1]) == anchor, "history VAE clip does not end at t0")
    require(bool((history <= anchor).all()), "future frame leaked into history VAE clip")
    require(int(short[0]) == anchor and int(short[-1]) == anchor + 30,
            "short VAE clip endpoints differ")
    require(bool((short[1:] > short[:-1]).all()), "short VAE clip is not ordered")
    require(int(goal[0]) == int(short[-1]), "goal VAE clip does not start at short")
    require(bool((goal[1:] > goal[:-1]).all()), "goal VAE clip is not ordered")
    goal_horizon_seconds = (int(goal[-1]) - anchor) / dataset.control_hz
    changed = build_dataset(args, args.goal_tail_guard_frames + 1)[(0, 4)]
    require(torch.equal(history, changed["history_video_control_indices"]),
            "terminal target changed the history VAE clip")
    require(torch.equal(short, changed["short_video_control_indices"]),
            "terminal target changed the short VAE clip")
    return {
        "video_clip_frames": 5,
        "history_video_future_leakage": False,
        "history_video_controls": history.tolist(),
        "short_video_controls": short.tolist(),
        "goal_video_controls": goal.tolist(),
        "minimum_goal_tail_frames": 4,
        "sample_goal_horizon_seconds": goal_horizon_seconds,
    }


def build_model(dataset, args, device):
    config = replace(
        AdaptiveGaussianWMConfig.object_region_dual_encoder_full(
            dataset.feature_dim
        ),
        dino_frame_batch=args.jit_dino_batch,
        goal_stability_threshold=args.goal_stability_threshold,
        goal_rollout_weight=args.goal_rollout_weight,
        path_consistency_weight=args.path_consistency_weight,
        video_vae_model=args.video_vae_model,
        video_vae_contract=args.video_vae_contract,
        video_vae_batch=args.video_vae_batch,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    warm_start = {"checkpoint": "not_provided"}
    if args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", weights_only=False, mmap=True
        )
        namespace = argparse.Namespace(
            architecture=ARCHITECTURE, training_stage="representation"
        )
        validate_v44_initialization(checkpoint, namespace)
        report = warm_start_model(model, checkpoint)
        validate_v44_warm_start_report(report, namespace)
        model.set_curriculum_step(0)
        warm_start = {
            "checkpoint": os.path.abspath(args.init_from),
            "checkpoint_sha256": file_sha256(args.init_from),
            "warm_start_loaded": report["loaded"],
            "warm_start_missing": len(report["missing"]),
        }
    return model, warm_start


@torch.no_grad()
def verify_dual_encoder_separation(model, batch, output) -> dict:
    model.eval()
    require(not isinstance(model.video_vae, torch.nn.Module),
            "frozen VAE was registered in the checkpoint module tree")
    require(not any(name.startswith("video_vae") for name, _ in model.named_parameters()),
            "external VAE appeared in optimizer parameters")
    require(model.video_vae._vae is not None, "video VAE was not exercised")
    require(not any(parameter.requires_grad for parameter in model.video_vae._vae.parameters()),
            "video VAE contains trainable parameters")
    features = model.online_dino(
        batch["history_jit_rgb"], batch["history_jit_valid"]
    )
    auxiliary, auxiliary_valid = _video_feature_sequences(
        features,
        batch["history_jit_rgb"].shape[1],
        output["video_features"],
        target=False,
    )
    reference = _encode_sequence(
        model, features, batch["history_times"], False, False,
        auxiliary, auxiliary_valid,
    )
    altered = _encode_sequence(
        model, features, batch["history_times"], False, False,
        -auxiliary, auxiliary_valid,
    )
    root_difference = difference(reference["roots"]["slots"], altered["roots"]["slots"])
    owner_difference = difference(reference["regions"]["owner"], altered["regions"]["owner"])
    identity_difference = difference(
        reference["regions"]["identity_key"], altered["regions"]["identity_key"]
    )
    region_difference = difference(
        reference["regions"]["feature"], altered["regions"]["feature"]
    )
    require(root_difference < 1e-6, "VAE detail changed DINO object roots")
    require(owner_difference < 1e-6, "VAE detail changed DINO region owners")
    require(identity_difference < 1e-6, "VAE detail changed DINO identities")
    require(region_difference > 1e-6, "VAE detail did not reach region features")
    detail = output["online"]["regions"]["detail_latent"]
    require(detail.shape[-1] == 96 and bool(torch.isfinite(detail).all()),
            "region detail target is invalid")
    detail_gate = output["online"]["regions"]["detail_gate"]
    require(
        bool(torch.isfinite(detail_gate).all())
        and bool(((detail_gate >= 0.0) & (detail_gate <= 1.0)).all()),
        "region video-detail gate is invalid",
    )
    return {
        "vae_root_max_difference": root_difference,
        "vae_owner_max_difference": owner_difference,
        "vae_identity_max_difference": identity_difference,
        "vae_region_feature_max_difference": region_difference,
        "video_vae_parameter_count": sum(
            parameter.numel() for parameter in model.video_vae._vae.parameters()
        ),
        "video_detail_gate_mean": float(detail_gate.float().mean()),
    }


@torch.no_grad()
def verify_future_isolation(model, batch, amp_context) -> dict:
    model.eval()
    model.set_curriculum_step(22000)
    changed = dict(batch)
    for name in (
        "future_jit_rgb",
        "goal_probe_jit_rgb",
        "short_video_rgb",
        "goal_video_rgb",
    ):
        changed[name] = batch[name].flip(-1)
    with amp_context():
        reference = model(batch)
        altered = model(changed)
    online_root = difference(
        reference["online"]["roots"]["slots"],
        altered["online"]["roots"]["slots"],
    )
    online_region = difference(
        reference["online"]["regions"]["feature"],
        altered["online"]["regions"]["feature"],
    )
    history_video = difference(
        reference["video_features"]["history"].feature,
        altered["video_features"]["history"].feature,
    )
    target_root = difference(
        reference["target"]["roots"]["slots"],
        altered["target"]["roots"]["slots"],
    )
    target_region = difference(
        reference["target"]["regions"]["feature"],
        altered["target"]["regions"]["feature"],
    )
    posterior = difference(reference["short_action"], altered["short_action"])
    short_video = difference(
        reference["video_features"]["short"].feature,
        altered["video_features"]["short"].feature,
    )
    require(online_root < 1e-6 and online_region < 1e-6,
            "future content reached the online history state")
    require(history_video < 1e-6,
            "future content changed the causal history VAE feature")
    require(target_root > 1e-6 and target_region > 1e-6,
            "future perturbation did not change EMA compact targets")
    require(posterior > 1e-6,
            "future perturbation did not change the effect posterior")
    require(short_video > 1e-6,
            "future perturbation did not change the video-detail target")
    return {
        "history_future_swap_root_max_difference": online_root,
        "history_future_swap_region_max_difference": online_region,
        "history_future_swap_video_max_difference": history_video,
        "target_future_swap_root_max_difference": target_root,
        "target_future_swap_region_max_difference": target_region,
        "posterior_future_swap_max_difference": posterior,
        "video_target_future_swap_max_difference": short_video,
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v44 verifier requires CUDA")
    gpu_count = torch.cuda.device_count()
    require(gpu_count > 0, "v44 verifier sees no CUDA GPU")
    if args.expected_local_gpus != "auto":
        require(args.expected_local_gpus.isdigit(), "expected GPU count is invalid")
        require(int(args.expected_local_gpus) == gpu_count,
                f"visible GPU count is {gpu_count}")
    commit = repository_commit()
    manifest_sha256, _ = verify_manifest(args.data)
    artifact = validate_video_vae_artifact(
        args.video_vae_model, args.video_vae_contract, verify_hashes=True
    )
    dataset = build_dataset(args)
    temporal = verify_temporal_contract(dataset, args)
    video_temporal = verify_video_temporal_contract(dataset, args)
    batch = move_to_device(
        default_collate([dataset[(0, 2)]]), torch.device("cuda:0")
    )
    model, warm_start = build_model(dataset, args, torch.device("cuda:0"))
    parameters = verify_parameter_contract(model)
    curriculum = verify_curriculum(model)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    output, gradients = gradient_contract(model, batch, amp_context)
    causality = verify_future_isolation(model, batch, amp_context)
    states = state_contract(model, output, amp_context)
    persistence = verify_v43_persistence_contracts(model, output, amp_context)
    factorization = verify_v43_action_factorization(model, batch, output, amp_context)
    separation = verify_dual_encoder_separation(model, batch, output)
    gate_args = argparse.Namespace(
        video_vae_model=args.video_vae_model,
        video_vae_contract=args.video_vae_contract,
        video_vae_pythonpath=args.video_vae_pythonpath,
        video_vae_clip_frames=5,
        video_vae_short_side=256,
        video_vae_batch=args.video_vae_batch,
        video_vae_contract_sha256="",
    )
    report = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": "offline_probe_only",
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
        "goal_gate_candidates": args.goal_gate_candidates,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "control_hz": float(dataset.control_hz),
        "temporal_contract": DUAL_ENCODER_TEMPORAL_CONTRACT,
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
        "local_gpu_count": gpu_count,
        "gpu_policy": args.expected_local_gpus,
        "video_vae_file_count": len(artifact["files"]),
        **gate_contract_fields(ARCHITECTURE),
        **v44_gate_fields(gate_args),
        **warm_start,
        **temporal,
        **video_temporal,
        **parameters,
        **curriculum,
        **gradients,
        **causality,
        **states,
        **persistence,
        **factorization,
        **separation,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
