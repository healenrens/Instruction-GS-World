"""Multisource CUDA and numerical stability verification for v53."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.semantic_object_world_model_v53 import (  # noqa: E402
    SemanticObjectLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.gradient_health import (  # noqa: E402
    clip_finite_grad_norm_,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v53_checkpointing import validate_init_from  # noqa: E402
from igsw.adaptive_gaussian_wm.v53_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    DECODER_CONTRACT,
    STAGES,
    SemanticObjectWorldModelConfig,
)
from igsw.adaptive_gaussian_wm.v53_training_loop import select_stage_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init_from", default="")
    parser.add_argument("--chunk_length", type=int, default=3)
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--dino_frame_batch", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--stability_steps_per_source", type=int, default=2)
    parser.add_argument("--stability_batch_per_source", type=int, default=2)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _require_finite(name: str, value: torch.Tensor) -> None:
    require(bool(torch.isfinite(value).all()), f"v53 non-finite tensor: {name}")


def _require_finite_model(model, optimizer) -> None:
    for name, parameter in model.named_parameters():
        _require_finite(f"parameter.{name}", parameter)
    for parameter, state in optimizer.state.items():
        parameter_name = next(
            name for name, candidate in model.named_parameters() if candidate is parameter
        )
        for state_name, value in state.items():
            if torch.is_tensor(value):
                _require_finite(f"optimizer.{parameter_name}.{state_name}", value)


def _batch_for_source(dataset, source_index: int, round_index: int, args):
    start = dataset.source_probe_indices[source_index]
    offset = round_index * args.stability_batch_per_source
    items = [
        dataset[(start + offset + item, args.chunk_length)]
        for item in range(args.stability_batch_per_source)
    ]
    observed = {int(item["source_index"]) for item in items}
    require(observed == {source_index}, "v53 source stability batch crossed sources")
    return default_collate(items)


def _mixed_source_batch(dataset, args):
    items = [
        dataset[(index, args.chunk_length)] for index in dataset.source_probe_indices
    ]
    return default_collate(items)


def _verify_numerical_stability(model, dino, dataset, device, amp_context, args):
    require(args.stability_steps_per_source > 0, "stability steps must be positive")
    require(args.stability_batch_per_source > 0, "stability batch must be positive")
    require(
        len(dataset.source_probe_indices) == len(dataset.source_names),
        "v53 verifier cannot probe every data source",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    batches = []
    labels = []
    for round_index in range(args.stability_steps_per_source):
        for source_index, source_name in enumerate(dataset.source_names):
            batches.append(_batch_for_source(dataset, source_index, round_index, args))
            labels.append(source_name)
    batches.append(_mixed_source_batch(dataset, args))
    labels.append("mixed")
    source_losses: dict[str, list[float]] = {}
    maximum_gradients: dict[str, float] = {}
    last_output = last_features = last_batch = None
    for cpu_batch, label in zip(batches, labels, strict=True):
        batch = select_stage_frames(move_to_device(cpu_batch, device), args.stage)
        features = dino(batch)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            output = model(
                features.patches,
                features.coordinates,
                features.valid,
                batch["frame_times"],
            )
        _require_finite("loss", output["loss"])
        for name, value in vars(output["encoding"]).items():
            _require_finite(f"encoding.{name}", value)
        for name, value in output["parts"].items():
            _require_finite(f"metric.{name}", value)
        output["loss"].backward()
        missing = [name for name, parameter in trainable if parameter.grad is None]
        require(not missing, f"v53 trainable parameters lack gradients: {missing}")
        for name, parameter in trainable:
            _require_finite(f"gradient.{name}", parameter.grad)
            maximum_gradients[name] = max(
                maximum_gradients.get(name, 0.0),
                float(parameter.grad.detach().abs().max().float()),
            )
        clip_finite_grad_norm_(trainable, 5.0)
        optimizer.step()
        _require_finite_model(model, optimizer)
        source_losses.setdefault(label, []).append(float(output["loss"].detach()))
        last_output, last_features, last_batch = output, features, batch
    return {
        "output": last_output,
        "features": last_features,
        "batch": last_batch,
        "trainable": trainable,
        "source_losses": {
            name: sum(values) / len(values) for name, values in source_losses.items()
        },
        "maximum_gradient": max(maximum_gradients.values()),
        "coordinate_basis_weight_maximum_gradient": maximum_gradients.get(
            "tokenizer.decoder.coordinate_basis.2.weight", 0.0
        ),
        "updates": len(batches),
    }


def main() -> None:
    args = parse_args()
    for name in ("data_index", "dino_checkpoint", "output", "init_from"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    require(torch.cuda.is_available(), "v53 verifier requires one visible CUDA device")
    require(os.path.isfile(args.data_index), "v53 multisource index is missing")
    require(
        os.path.isfile(args.dino_checkpoint), "v53 frozen DINO checkpoint is missing"
    )
    if args.stage == "dynamics":
        require(
            bool(args.init_from), "v53 dynamics verifier requires tokenizer init_from"
        )
        require(os.path.isfile(args.init_from), "v53 tokenizer init_from is missing")
    elif args.init_from:
        raise ValueError("v53 tokenizer verifier does not accept init_from")
    config = SemanticObjectWorldModelConfig()
    config.validate()
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        str(args.chunk_length),
        args.temporal_step_ms,
        max_items=0,
        seed=173,
    )
    device = torch.device("cuda:0")
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    model = SemanticObjectLatentWorldModel(config).to(device)
    if args.init_from:
        checkpoint = torch.load(
            args.init_from, map_location="cpu", weights_only=False, mmap=True
        )
        validate_init_from(checkpoint, config)
        model.load_state_dict(checkpoint["model"], strict=True)
    model.configure_stage(args.stage)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    stability = _verify_numerical_stability(
        model, dino, dataset, device, amp_context, args
    )
    output = stability["output"]
    features = stability["features"]
    trainable = stability["trainable"]
    require(not features.patches.requires_grad, "frozen DINO features require grad")
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    require(
        not frozen_with_grad,
        f"v53 frozen modules received gradients: {frozen_with_grad}",
    )
    with torch.no_grad():
        encoding = model.tokenizer(
            features.patches, features.coordinates, features.valid
        )
        swapped = features.patches.clone()
        swapped[:, -1] = swapped.flip(0)[:, -1]
        swapped_encoding = model.tokenizer(
            swapped, features.coordinates, features.valid
        )
        source_difference = (
            (encoding.slots[:, 0] - swapped_encoding.slots[:, 0]).abs().max()
        )
        target_difference = (
            (encoding.slots[:, -1] - swapped_encoding.slots[:, -1]).abs().max()
        )
    require(float(source_difference) < 1e-6, "v53 source state reads future content")
    require(
        float(target_difference) > 1e-5,
        "v53 target state ignores changed future content",
    )
    parts = {
        name: float(value.detach().float()) for name, value in output["parts"].items()
    }
    if args.stage == "tokenizer":
        require(
            parts["object_effective_slot_count"] > 1.0,
            "v53 tokenizer collapsed in verifier",
        )
    else:
        require(
            "dynamics_correct_gain_over_zero" in parts,
            "v53 dynamics diagnostics missing",
        )
        require(
            "dynamics_correct_gain_over_shuffled" in parts,
            "v53 effect diagnostics missing",
        )
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "decoder_contract": DECODER_CONTRACT,
        "stage": args.stage,
        "git_commit": args.source_revision,
        "data_index": args.data_index,
        "source_count": len(dataset.source_names),
        "source_names": list(dataset.source_names),
        "object_slots": config.object_slots,
        "scene_slots": config.scene_slots,
        "action_dim": config.action_dim,
        "frozen_dino": True,
        "point_tracker_used": False,
        "rgb_reconstruction_used": False,
        "instance_segmentation_used": False,
        "language_used": False,
        "explicit_action_used": False,
        "historical_checkpoint_used": False,
        "source_future_swap_max_difference": float(source_difference),
        "target_future_swap_max_difference": float(target_difference),
        "trainable_parameter_tensors": len(trainable),
        "gradient_parameter_tensors": len(trainable),
        "numerical_stability_status": "passed",
        "numerical_stability_updates": stability["updates"],
        "numerical_stability_source_losses": stability["source_losses"],
        "maximum_parameter_gradient": stability["maximum_gradient"],
        "coordinate_basis_weight_maximum_gradient": stability[
            "coordinate_basis_weight_maximum_gradient"
        ],
        "loss": float(output["loss"].detach().float()),
        "metrics": parts,
    }
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
