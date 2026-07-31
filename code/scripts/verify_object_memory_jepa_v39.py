#!/usr/bin/env python3
"""Server gate for dynamic-history, dual-horizon Object Memory JEPA v39."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.dynamic_dual_horizon_dataset import (  # noqa: E402
    DYNAMIC_DUAL_HORIZON_CONTRACT,
    DynamicDualHorizonEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    build_training_sampler,
)
from igsw.adaptive_gaussian_wm.jit_dino_runtime import (  # noqa: E402
    JitDinoFeatureRuntime,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    JIT_DINO_IMAGE_SIZE,
    JIT_DINO_MODEL,
    RGB_EPISODE_CACHE_VERSION,
    file_sha256,
    validate_manifest,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_ROOT,
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    LEROBOT_SOURCE_KIND,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)
from igsw.adaptive_gaussian_wm.v28_training import (  # noqa: E402
    configure_v28_stage,
)
from igsw.adaptive_gaussian_wm.v39_causal_verification import (  # noqa: E402
    verify_v39_causal_paths,
)
from igsw.adaptive_gaussian_wm.v39_gate_sampling import (  # noqa: E402
    select_v39_gate_batch,
)


FEATURE_CONTRACT = "jit_backbone_native_dinov2_l_1024"
READOUT_BACKEND = "change_only_object_residual"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected_local_gpus", default="auto")
    parser.add_argument("--jit_dino_batch", type=int, default=16)
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--goal_stability_threshold", type=float, default=0.05)
    parser.add_argument("--goal_gate_candidates", type=int, default=32)
    parser.add_argument("--goal_rollout_weight", type=float, default=1.0)
    parser.add_argument("--path_consistency_weight", type=float, default=0.25)
    args = parser.parse_args()
    for name in ("data", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    for name in ("teacher_sidecar", "checkpoint"):
        value = getattr(args, name)
        require(not value or os.path.isabs(value), f"--{name} must be absolute")
    require(args.jit_dino_batch > 0, "--jit_dino_batch must be positive")
    require(args.goal_gate_candidates >= 4, "goal gate requires four candidates")
    return args


def verify_repository() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    require(not status.strip(), "v39 verifier rejects tracked worktree changes")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def verify_manifest(data: str) -> tuple[str, dict]:
    path = os.path.join(data, EPISODE_MANIFEST_NAME)
    checksum = os.path.join(data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(path), "RGB episode manifest is missing")
    require(os.path.isfile(checksum), "RGB manifest checksum is missing")
    result = subprocess.run(
        ["sha256sum", "-c", "--status", EPISODE_VERIFIED_NAME],
        cwd=data,
        check=False,
    )
    require(result.returncode == 0, "RGB manifest checksum failed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, path)
    source = manifest["source"]
    require(
        manifest["episode_cache_version"] == RGB_EPISODE_CACHE_VERSION,
        "data is not the RGB-only episode cache",
    )
    require(source.get("kind") == LEROBOT_SOURCE_KIND, "source is not LeRobot-v3")
    require(
        os.path.realpath(source.get("path", ""))
        == os.path.realpath(LEROBOT_DEFAULT_ROOT),
        "RoboTwin source root is not authoritative",
    )
    require(
        tuple(source.get("variants", ())) == LEROBOT_DEFAULT_VARIANTS,
        "source variants differ",
    )
    require(
        float(source.get("expected_source_fps", 0.0)) == LEROBOT_EXPECTED_FPS
        and int(source.get("source_frame_stride", 0)) == 1
        and float(manifest.get("control_hz", 0.0)) == LEROBOT_EXPECTED_FPS,
        "RoboTwin source is not native stride-1 30 Hz",
    )
    return file_sha256(path), manifest


def build_dataset(args, tail_guard: int | None = None):
    return DynamicDualHorizonEpisodeDataset(
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
    )


def verify_temporal_contract(dataset, args) -> dict:
    samples = {length: dataset[(0, length)] for length in range(1, 5)}
    anchor = int(samples[1]["anchor_control_index"])
    future = samples[1]["future_control_indices"]
    for length, sample in samples.items():
        controls = sample["history_control_indices"]
        require(len(controls) == length, "dynamic history length differs")
        require(int(controls[-1]) == anchor, "history does not end at t0")
        require(bool((controls[1:] > controls[:-1]).all()), "history is not ordered")
        require(
            torch.equal(sample["future_control_indices"], future),
            "future target changed with history length",
        )
        require(
            torch.equal(sample["future_times"], samples[1]["future_times"]),
            "future query scale changed with history length",
        )
    require(int(future[0]) - anchor == 30, "short target is not +30 frames")
    require(
        torch.allclose(
            samples[1]["future_times"],
            torch.tensor([1.0, args.goal_query_seconds]),
        ),
        "short/goal query scales differ",
    )
    changed_goal = build_dataset(
        args, tail_guard=args.goal_tail_guard_frames + 1
    )[(0, 4)]
    require(
        torch.equal(samples[4]["history_control_indices"], changed_goal["history_control_indices"]),
        "terminal target changed history sampling",
    )
    require(
        torch.equal(samples[4]["history_jit_rgb"], changed_goal["history_jit_rgb"]),
        "terminal target changed history content",
    )
    require(
        int(samples[4]["future_control_indices"][1])
        - int(changed_goal["future_control_indices"][1])
        == 1,
        "goal-tail guard did not move only the terminal target",
    )
    rank_sequences = []
    for rank in range(2):
        iterator = iter(build_training_sampler(dataset, 2, rank, 17, 4, 4))
        rank_sequences.append([next(iterator)[1] for _ in range(16)])
    require(rank_sequences[0] == rank_sequences[1], "DDP ranks use different H")
    microbatches = [
        rank_sequences[0][index : index + 4] for index in range(0, 16, 4)
    ]
    require(
        all(len(set(values)) == 1 for values in microbatches),
        "one microbatch contains mixed history lengths",
    )
    require(
        set(values[0] for values in microbatches) == {1, 2, 3, 4},
        "history rotation does not cover H=1..4",
    )
    return {
        "history_lengths": [1, 2, 3, 4],
        "history_span_frames": list(dataset.history_span_frames),
        "short_horizon_frames": 30,
        "goal_query_seconds": args.goal_query_seconds,
        "goal_tail_guard_frames": args.goal_tail_guard_frames,
        "goal_probe_frames": args.goal_probe_frames,
        "history_microbatch_rotation": [values[0] for values in microbatches],
        "goal_history_content_max_difference": max_difference(
            samples[4]["history_jit_rgb"].float(),
            changed_goal["history_jit_rgb"].float(),
        ),
    }


def representation_weights() -> AdaptiveGaussianLossWeights:
    return AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=1.0,
        allocator=0.2,
        slot=0.2,
        action=0.0,
        action_specificity=0.0,
        geometry=0.25,
        rgb=0.0,
    )


def posterior_weights() -> AdaptiveGaussianLossWeights:
    return replace(representation_weights(), action_specificity=1.0)


def build_model(dataset, args, device):
    config = replace(
        AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim),
        dual_horizon_dynamics=True,
        goal_rollout_weight=args.goal_rollout_weight,
        path_consistency_weight=args.path_consistency_weight,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    checkpoint_report = {"checkpoint": "not_provided"}
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        require(
            checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
            "checkpoint is not v39",
        )
        require(checkpoint.get("config") == config.to_dict(), "checkpoint config differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_report = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_phase": checkpoint.get("phase"),
            "checkpoint_phase_step": checkpoint.get("phase_step"),
        }
    return model, checkpoint_report


def verify_model_paths(model, batch: dict, amp_context) -> dict:
    configure_v28_stage(
        model,
        SimpleNamespace(architecture="object_memory_v1", training_stage="representation"),
    )
    model.train()
    model.zero_grad(set_to_none=True)
    fixed_history_mask = torch.zeros(
        batch["history_times"].shape[0],
        batch["history_times"].shape[1],
        model.config.object_slots,
        device=batch["history_times"].device,
        dtype=torch.bool,
    )
    with amp_context():
        representation = model(
            batch,
            history_mask=fixed_history_mask,
            phase="object_memory_representation_loss",
            loss_weights=representation_weights(),
        )
    require(bool(torch.isfinite(representation["loss"])), "representation loss is non-finite")
    require(
        not bool(representation["future_horizon_valid"][:, 1].any()),
        "representation stage supervised the terminal target",
    )
    representation["loss"].backward()
    nonfinite = []
    core_gradients = 0
    action_gradients = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if name.startswith(("allocator.", "object_aggregator.", "object_memory.", "dynamics.")):
            core_gradients += 1
        if name.startswith(("latent_actions.posterior.", "effect_composer.")):
            action_gradients += 1
    require(not nonfinite, f"non-finite gradients: {nonfinite}")
    require(core_gradients > 0, "representation core received no gradients")
    require(action_gradients == 0, "representation updated posterior/composer")
    representation_loss = float(representation["loss"].detach())
    short_prediction = representation["predicted_future_slots"][:, :1].detach().clone()
    model.zero_grad(set_to_none=True)
    del representation
    torch.cuda.empty_cache()
    future_changed = dict(batch)
    future_changed["future_features"] = batch["future_features"].roll(1, dims=-1)
    with torch.no_grad(), amp_context():
        changed_representation = model(
            future_changed,
            history_mask=fixed_history_mask,
            phase="object_memory_representation_loss",
            loss_weights=representation_weights(),
        )
    action_free_difference = max_difference(
        short_prediction,
        changed_representation["predicted_future_slots"][:, :1],
    )
    require(action_free_difference == 0.0, "future content reached action-free short")
    del changed_representation
    torch.cuda.empty_cache()
    model.requires_grad_(True)
    for target_module in (
        model.target_allocator,
        model.target_object_aggregator,
        model.target_object_memory,
    ):
        if target_module is not None:
            target_module.requires_grad_(False)
    configure_v28_stage(
        model,
        SimpleNamespace(architecture="object_memory_v1", training_stage="posterior"),
    )
    model.zero_grad(set_to_none=True)
    with amp_context():
        posterior = model(
            batch,
            history_mask=fixed_history_mask,
            phase="posterior_dynamics_loss",
            loss_weights=posterior_weights(),
        )
    require("rollout_goal_slots" in posterior, "posterior path has no goal rollout")
    require(
        posterior["predicted_future_slots"].shape[1] == 2
        and posterior["rollout_goal_slots"].shape[1] == 1,
        "direct/rollout horizon shapes differ",
    )
    require(bool(torch.isfinite(posterior["loss"])), "posterior loss is non-finite")
    require(
        "action_specificity_ranking" in posterior["parts"],
        "posterior negative-ranking objective is missing",
    )
    posterior["loss"].backward()
    posterior_gradients = 0
    composer_gradients = 0
    posterior_nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            posterior_nonfinite.append(name)
        if name.startswith("latent_actions.posterior."):
            posterior_gradients += 1
        if name.startswith("effect_composer."):
            composer_gradients += 1
    require(not posterior_nonfinite, f"non-finite posterior gradients: {posterior_nonfinite}")
    require(posterior_gradients > 0, "Posterior received no gradients")
    require(composer_gradients > 0, "effect composer received no gradients")
    return {
        "representation_loss": representation_loss,
        "representation_core_gradient_tensors": core_gradients,
        "representation_action_gradient_tensors": action_gradients,
        "action_free_short_future_swap_max_difference": action_free_difference,
        "posterior_loss": float(posterior["loss"].detach()),
        "posterior_action_gradient_tensors": posterior_gradients,
        "posterior_composer_gradient_tensors": composer_gradients,
        "goal_valid_fraction": float(
            batch["future_horizon_valid"][:, 1].float().mean()
        ),
        "goal_stability_error": float(batch["goal_stability_error"].mean()),
    }


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v39 verifier requires CUDA")
    local_gpus = torch.cuda.device_count()
    require(local_gpus > 0, "no visible CUDA GPU")
    if args.expected_local_gpus != "auto":
        require(
            args.expected_local_gpus.isdigit()
            and int(args.expected_local_gpus) == local_gpus,
            f"visible GPU count is {local_gpus}, expected {args.expected_local_gpus}",
        )
    commit = verify_repository()
    manifest_sha256, _ = verify_manifest(args.data)
    dataset = build_dataset(args)
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    temporal = verify_temporal_contract(dataset, args)
    device = torch.device("cuda:0")
    dino_amp = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    runtime = JitDinoFeatureRuntime(
        device,
        dino_amp,
        args.jit_dino_batch,
        args.goal_stability_threshold,
    )
    batch, goal_scan = select_v39_gate_batch(
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
        causal = verify_v39_causal_paths(model, batch)
    training = verify_model_paths(model, batch, amp_context)
    report = {
        "status": "passed",
        "architecture": "object_memory_v1",
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": READOUT_BACKEND,
        "feature_source": "jit",
        "feature_contract": FEATURE_CONTRACT,
        "jit_dino_model": JIT_DINO_MODEL,
        "jit_dino_image_size": JIT_DINO_IMAGE_SIZE,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "goal_stability_threshold": args.goal_stability_threshold,
        **goal_scan,
        "goal_rollout_weight": args.goal_rollout_weight,
        "path_consistency_weight": args.path_consistency_weight,
        "control_hz": float(dataset.control_hz),
        "temporal_contract": DYNAMIC_DUAL_HORIZON_CONTRACT,
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
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
