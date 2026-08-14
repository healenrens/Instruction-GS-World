"""GPU verifier for the v48 pure-video object-state contract."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.recurrent_slot_state import (  # noqa: E402
    evidence_normalized_attention,
)
from igsw.adaptive_gaussian_wm.slot_contrast_world_model import (  # noqa: E402
    SlotContrastObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.slot_contrast_objective import (  # noqa: E402
    temporal_slot_contrast,
)
from igsw.adaptive_gaussian_wm.stable_normalization import (  # noqa: E402
    stable_rms_normalize,
    stable_unit_normalize,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TEMPORAL_OBJECT_VIDEO_CONTRACT,
    TemporalObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.v48_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    SlotContrastConfig,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def gradient_contract(model, features, batch, amp_context) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
        )
    require(bool(torch.isfinite(output["loss"])), "v48 loss is non-finite")
    output["loss"].backward()
    missing, nonfinite, gradients = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        gradients.append(parameter.grad.detach().float().square().sum())
    require(not missing, f"v48 has unused parameters: {missing}")
    require(not nonfinite, f"v48 has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, "v48 produced zero gradient norm")
    metrics = {
        "loss": float(output["loss"].detach()),
        "gradient_norm": float(norm),
        **{name: float(value) for name, value in output["parts"].items()},
    }
    model.zero_grad(set_to_none=True)
    return metrics


def numerical_stability_contract(device: torch.device) -> dict[str, float]:
    tiny = torch.full((4, 128), 1e-30, device=device, requires_grad=True)
    stable_unit_normalize(tiny).sum().backward()
    require(bool(torch.isfinite(tiny.grad).all()), "tiny-vector normalization gradient failed")

    recurrent = torch.full((4, 256), 1e30, device=device, requires_grad=True)
    bounded = stable_rms_normalize(recurrent)
    bounded.sum().backward()
    bounded_rms = bounded.float().square().mean(dim=-1).sqrt()
    require(bool(torch.isfinite(recurrent.grad).all()), "recurrent RMS gradient failed")
    require(float(bounded_rms.max()) <= 1.001, "recurrent RMS normalization failed")

    competition = torch.tensor(
        [[[1e-30, 1e-30], [0.5, 0.5]]],
        device=device,
        requires_grad=True,
    )
    normalized, support = evidence_normalized_attention(competition)
    (normalized.sum() + support.sum()).backward()
    require(
        bool(torch.isfinite(competition.grad).all()),
        "low-evidence slot normalization gradient failed",
    )

    parameter = torch.nn.Parameter(torch.zeros(4, device=device))
    parameter.grad = torch.full_like(parameter, 1e30)
    preclip = clip_finite_grad_norm_((("synthetic", parameter),), 5.0)
    postclip = torch.linalg.vector_norm(parameter.grad.double())
    require(bool(torch.isfinite(preclip)), "stable clipping returned a non-finite norm")
    require(float(postclip) <= 5.001, "stable clipping exceeded its bound")

    identities = torch.eye(4, device=device)[:2]
    projected = identities[None, None].expand(1, 4, 2, 4).contiguous()
    activity = torch.ones(1, 4, 2, device=device)
    contrast_loss, retrieval = temporal_slot_contrast(
        projected, activity, 0.02, 0.1
    )
    require(bool(torch.isfinite(contrast_loss)), "multi-positive contrast is non-finite")
    require(float(retrieval) == 1.0, "multi-positive identity retrieval failed")
    return {
        "tiny_normalization_gradient_max": float(tiny.grad.abs().max()),
        "recurrent_state_rms_max": float(bounded_rms.max()),
        "low_evidence_gradient_max": float(competition.grad.abs().max()),
        "huge_finite_preclip_norm": float(preclip),
        "huge_finite_postclip_norm": float(postclip),
        "multi_positive_contrast_loss": float(contrast_loss),
        "multi_positive_retrieval_top1": float(retrieval),
    }


def long_sequence_gradient_contract(
    model, features, amp_context
) -> dict[str, float]:
    frames = 32
    source_frames = features.patches.shape[1]
    repeats = (frames + source_frames - 1) // source_frames
    patches = features.patches.repeat(1, repeats, 1, 1)[:, :frames]
    coordinates = features.coordinates.repeat(1, repeats, 1, 1)[:, :frames]
    valid = features.valid.repeat(1, repeats, 1)[:, :frames]
    frame_times = torch.arange(
        frames, device=patches.device, dtype=torch.float32
    )[None].expand(len(patches), -1) / 30.0
    observation_mask = torch.ones(
        len(patches), frames, device=patches.device, dtype=torch.bool
    )
    observation_mask[:, 4::5] = False
    model.zero_grad(set_to_none=True)
    with amp_context():
        output = model(
            patches, coordinates, valid, frame_times, observation_mask
        )
    require(bool(torch.isfinite(output["loss"])), "long v48 loss is non-finite")
    output["loss"].backward()
    predictor_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("state_encoder.predictor.") and parameter.grad is not None
    ]
    require(predictor_gradients, "long v48 path did not train the predictor")
    predictor_max = max(float(gradient.abs().max()) for gradient in predictor_gradients)
    require(
        bool(torch.isfinite(torch.tensor(predictor_max))),
        "long v48 predictor gradient is non-finite",
    )
    preclip = clip_finite_grad_norm_(model.named_parameters(), 5.0)
    model.zero_grad(set_to_none=True)
    return {
        "long_sequence_frames": float(frames),
        "long_sequence_gradient_norm": float(preclip),
        "long_sequence_predictor_gradient_max": predictor_max,
    }


@torch.no_grad()
def structural_contract(model, features, batch, amp_context) -> dict[str, float]:
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
        )
    batch_size, frames, patches = features.valid.shape
    slots = model.config.object_slots
    require(
        output["slots"].shape == (batch_size, frames, slots, model.config.slot_dim),
        "v48 slot state shape differs",
    )
    require(
        output["assignment"].shape == (batch_size, frames, patches, slots),
        "v48 assignment shape differs",
    )
    require(
        output["reconstruction"].shape == features.patches.shape,
        "v48 reconstruction shape differs",
    )
    partition = output["assignment"].sum(dim=-1)
    partition_error = maximum_difference(
        partition[features.valid], torch.ones_like(partition[features.valid])
    )
    require(partition_error < 1e-5, "v48 slots do not partition valid patches")
    require(
        bool(torch.isfinite(output["reconstruction"]).all()),
        "v48 reconstruction is non-finite",
    )
    require(
        bool(torch.isfinite(output["slots"]).all()),
        "v48 slots are non-finite",
    )
    return {
        "owner_partition_max_error": partition_error,
        "slot_count": float(slots),
        "patch_count": float(patches),
        "active_slot_count": float(output["parts"]["slot_active_count"]),
        "assignment_entropy": float(output["parts"]["slot_assignment_entropy"]),
        "reconstruction_error": float(output["parts"]["loss_reconstruction"]),
        "frame_mean_error": float(output["parts"]["diagnostic_frame_mean_error"]),
    }


@torch.no_grad()
def causal_prefix_contract(model, features, batch, amp_context) -> dict[str, float]:
    midpoint = features.patches.shape[1] // 2
    changed = features.patches.clone()
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        reference = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
        )
        altered = model(
            changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
        )
    difference = maximum_difference(
        reference["slots"][:, :midpoint], altered["slots"][:, :midpoint]
    )
    require(difference < 1e-6, "future content changed the causal slot prefix")
    return {"future_swap_prefix_max_difference": difference}


@torch.no_grad()
def masked_frame_contract(model, features, batch, amp_context) -> dict[str, float]:
    index = features.patches.shape[1] // 2
    mask = batch["observation_mask"].clone()
    mask[:, index] = False
    changed = features.patches.clone()
    changed[:, index] = changed[:, index].flip(1)
    with amp_context():
        reference = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            mask,
        )
        altered = model(
            changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            mask,
        )
    slot_difference = maximum_difference(reference["slots"], altered["slots"])
    reconstruction_difference = maximum_difference(
        reference["reconstruction"], altered["reconstruction"]
    )
    require(slot_difference < 1e-6, "masked frame content changed slot state")
    require(
        reconstruction_difference < 1e-6,
        "masked frame content changed model reconstruction",
    )
    return {
        "masked_content_slot_max_difference": slot_difference,
        "masked_content_reconstruction_max_difference": reconstruction_difference,
    }


@torch.no_grad()
def temporal_and_decoder_contract(model, features, batch, amp_context) -> dict[str, float]:
    full_mask = torch.ones_like(batch["observation_mask"])
    with amp_context():
        ordered = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            full_mask,
        )
        reversed_output = model(
            features.patches.flip(1),
            features.coordinates.flip(1),
            features.valid.flip(1),
            batch["frame_times"],
            full_mask,
        )
        original_decode, _ = model.decode_frame(
            ordered["slots"][:, -1], features.coordinates[:, -1]
        )
        moved_decode, _ = model.decode_frame(
            ordered["slots"][:, -1], features.coordinates[:, -1].roll(1, dims=1)
        )
    order_difference = maximum_difference(
        ordered["slots"][:, -1], reversed_output["slots"][:, -1]
    )
    position_difference = maximum_difference(original_decode, moved_decode)
    require(order_difference > 1e-6, "v48 slot state ignores temporal order")
    require(position_difference > 1e-6, "v48 decoder ignores spatial queries")
    return {
        "ordered_reversed_last_slot_max_difference": order_difference,
        "decoder_coordinate_swap_max_difference": position_difference,
    }


@torch.no_grad()
def slot_deletion_diagnostic(model, features, batch, amp_context) -> dict[str, float]:
    full_mask = torch.ones_like(batch["observation_mask"])
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            full_mask,
        )
    slots = output["slots"][:, -1]
    assignment = output["assignment"][:, -1]
    selected = output["mass"][:, -1].argmax(dim=1)
    slot_valid = torch.ones(
        slots.shape[:2], dtype=torch.bool, device=slots.device
    )
    slot_valid[torch.arange(len(slots), device=slots.device), selected] = False
    with amp_context():
        deleted, _ = model.decode_frame(
            slots, features.coordinates[:, -1], slot_valid
        )
    target = F.normalize(features.patches[:, -1].float(), dim=-1, eps=1e-6)
    full_error = 1.0 - (output["reconstruction"][:, -1].float() * target).sum(dim=-1)
    deleted_error = 1.0 - (deleted.float() * target).sum(dim=-1)
    delta = deleted_error - full_error
    selected_weight = assignment[
        torch.arange(len(slots), device=slots.device),
        :,
        selected,
    ] * features.valid[:, -1].float()
    local = (delta * selected_weight).sum() / selected_weight.sum().clamp_min(1.0)
    outside_weight = features.valid[:, -1].float() * (1.0 - selected_weight)
    outside = (delta * outside_weight).sum() / outside_weight.sum().clamp_min(1.0)
    return {
        "slot_deletion_local_error_increase": float(local),
        "slot_deletion_outside_error_increase": float(outside),
        "slot_deletion_locality_gap": float(local - outside),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v48 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = SlotContrastConfig()
    dataset = TemporalObjectVideoDataset(
        args.data,
        "train",
        max_items=64,
        seed=args.seed,
        record_manifest_hash=False,
    )
    samples = [dataset[(index, args.chunk_length)] for index in range(2)]
    forbidden = {
        "instruction",
        "condition_feature",
        "teacher_sidecar",
        "segmentation",
        "action",
        "dino",
    }
    require(not forbidden.intersection(samples[0]), "v48 dataset exposed supervision")
    batch = default_collate(samples)
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    batch["observation_mask"][:, args.chunk_length // 2] = False
    encoder = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    require(
        not any(parameter.requires_grad for parameter in encoder.backbone.parameters()),
        "v48 DINO teacher is not frozen",
    )
    features = encoder(batch)
    model = SlotContrastObjectWorldModel(config).to(device).train()
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    structural = structural_contract(model, features, batch, amp_context)
    gradients = gradient_contract(model, features, batch, amp_context)
    numerical = numerical_stability_contract(device)
    long_sequence = long_sequence_gradient_contract(model, features, amp_context)
    model.eval()
    causal = causal_prefix_contract(model, features, batch, amp_context)
    masked = masked_frame_contract(model, features, batch, amp_context)
    temporal = temporal_and_decoder_contract(model, features, batch, amp_context)
    deletion = slot_deletion_diagnostic(model, features, batch, amp_context)
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": TEMPORAL_OBJECT_VIDEO_CONTRACT,
        "git_commit": args.source_revision,
        "data": os.path.abspath(args.data),
        "historical_checkpoint_used": False,
        "teacher_sidecar_used": False,
        "language_used": False,
        "explicit_action_used": False,
        "instance_segmentation_used": False,
        "dino_model": config.dino_model_name,
        "dino_fully_frozen": True,
        "dino_checkpoint": os.path.abspath(args.dino_checkpoint),
        "object_slots": config.object_slots,
        "explicit_scene_owner": False,
        "free_center_regression": False,
        "dynamics_trained": False,
        "latent_effect_trained": False,
        "core_objective": "frozen_dino_reconstruction_plus_temporal_slot_contrast",
        **structural,
        **gradients,
        **numerical,
        **long_sequence,
        **causal,
        **masked,
        **temporal,
        **deletion,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
