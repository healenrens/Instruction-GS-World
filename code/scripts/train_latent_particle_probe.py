"""Train and evaluate one strict-causal latent particle world-model probe."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.metrics import (  # noqa: E402
    MetricAccumulator,
    active_mover_recall,
    add_point_metrics,
    add_sample_metrics,
)
from igsw.latent_particle_wm.models import ParticleWorldModel, WorldModelConfig  # noqa: E402
from igsw.latent_particle_wm.objectives import particle_world_model_loss  # noqa: E402
from igsw.latent_particle_wm.probe_data import ParticleProbeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--kind",
        choices=sorted(ParticleWorldModel.VALID_KINDS),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--local_dim", type=int, default=8)
    parser.add_argument("--global_dim", type=int, default=12)
    parser.add_argument("--mixtures", type=int, default=4)
    parser.add_argument("--kl_weight", type=float, default=2e-3)
    parser.add_argument("--free_bits", type=float, default=0.02)
    parser.add_argument("--move_weight", type=float, default=6.0)
    parser.add_argument("--appearance_weight", type=float, default=0.15)
    parser.add_argument("--visibility_weight", type=float, default=0.1)
    parser.add_argument("--effect_weight", type=float, default=0.1)
    parser.add_argument("--usage_weight", type=float, default=0.0)
    parser.add_argument("--usage_margin", type=float, default=0.005)
    parser.add_argument("--alignment_weight", type=float, default=0.0)
    parser.add_argument("--two_stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--posterior_fraction", type=float, default=0.67)
    parser.add_argument("--prior_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def slice_batch(batch: dict, keep: torch.Tensor) -> dict:
    return {key: value[keep] for key, value in batch.items()}


def update_metrics(
    accumulators: dict[str, MetricAccumulator],
    key: str,
    batch: dict,
    posterior: torch.Tensor,
    posterior_visibility: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_visibility: torch.Tensor,
    prior_samples: torch.Tensor,
) -> None:
    accumulator = accumulators[key]
    zero = torch.zeros_like(posterior)
    always_visible = torch.full_like(posterior_visibility, 10.0)
    add_point_metrics(accumulator, "zero_motion", batch, zero, always_visible)
    add_point_metrics(accumulator, "posterior", batch, posterior, posterior_visibility)
    add_point_metrics(accumulator, "prior_mean", batch, prior_mean, prior_visibility)
    add_sample_metrics(accumulator, batch, prior_samples)
    recall, count = active_mover_recall(batch)
    accumulator.add("data/active_mover_recall", recall, count)


@torch.no_grad()
def evaluate(
    model: ParticleWorldModel,
    cache: dict,
    split: str,
    batch_size: int,
    prior_samples_count: int,
    device: torch.device,
) -> dict:
    dataset = ParticleProbeDataset(cache, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    accumulators: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    causal_prior_difference = 0.0
    posterior_future_sensitivity = 0.0
    kl_sum = 0.0
    kl_count = 0
    model.eval()

    for batch_index, cpu_batch in enumerate(loader):
        batch = to_device(cpu_batch, device)
        output = model(batch, sample_posterior=False)
        prior_mean, prior_visibility = model.predict_prior(batch, samples=1, sample=False)
        prior_samples, _ = model.predict_prior(batch, samples=prior_samples_count, sample=True)
        update_metrics(
            accumulators,
            "all",
            batch,
            output["prediction"],
            output["visibility_logits"],
            prior_mean[0],
            prior_visibility[0],
            prior_samples,
        )
        for horizon in batch["horizon"].unique().tolist():
            keep = batch["horizon"] == horizon
            update_metrics(
                accumulators,
                f"horizon_{int(horizon):02d}",
                slice_batch(batch, keep),
                output["prediction"][keep],
                output["visibility_logits"][keep],
                prior_mean[0, keep],
                prior_visibility[0, keep],
                prior_samples[:, keep],
            )
        if split == "heldtask":
            task_names = [cache["clips"]["task"][int(index)] for index in batch["clip_index"]]
            for task in sorted(set(task_names)):
                keep = torch.tensor([name == task for name in task_names], device=device)
                update_metrics(
                    accumulators,
                    f"task_{task}",
                    slice_batch(batch, keep),
                    output["prediction"][keep],
                    output["visibility_logits"][keep],
                    prior_mean[0, keep],
                    prior_visibility[0, keep],
                    prior_samples[:, keep],
                )
        kl, _ = model.prior_fitting_loss(output, batch["valid"], free_bits=0.0)
        kl_sum += float(kl) * len(batch["state"])
        kl_count += len(batch["state"])

        if batch_index == 0 and len(batch["state"]) > 1:
            shuffled = dict(batch)
            order = torch.arange(len(batch["state"]) - 1, -1, -1, device=device)
            for key in ("target", "valid", "visible", "motion_valid", "target_xyz"):
                shuffled[key] = batch[key][order]
            first_prior = model.prior_parameters(batch)
            second_prior = model.prior_parameters(shuffled)
            for key in first_prior:
                causal_prior_difference = max(
                    causal_prior_difference,
                    float((first_prior[key] - second_prior[key]).abs().max()),
                )
            if model.config.kind != "deterministic":
                second_output = model(shuffled, sample_posterior=False)
                for key in ("local_q_mu", "global_q_mu"):
                    if key in output:
                        posterior_future_sensitivity = max(
                            posterior_future_sensitivity,
                            float((output[key] - second_output[key]).abs().max()),
                        )

    result = {key: accumulator.compute() for key, accumulator in accumulators.items()}
    result["invariants"] = {
        "future_swap_prior_max_abs_difference": causal_prior_difference,
        "future_swap_posterior_max_abs_difference": posterior_future_sensitivity,
        "mean_q_prior_kl": kl_sum / max(kl_count, 1),
    }
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_dataset = ParticleProbeDataset(cache, "train")
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    config = WorldModelConfig(
        kind=args.kind,
        state_dim=int(cache["clips"]["state"].shape[-1]),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        local_latent_dim=args.local_dim,
        global_latent_dim=args.global_dim,
        mixture_components=args.mixtures,
    )
    model = ParticleWorldModel(config).to(device)
    def make_optimizer(steps: int):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
        return optimizer, scheduler

    posterior_epochs = args.epochs
    if args.two_stage and args.kind != "deterministic":
        posterior_epochs = max(1, min(args.epochs - 1, round(args.epochs * args.posterior_fraction)))
        model.set_training_phase("posterior")
    total_steps = args.epochs * len(loader)
    phase_steps = posterior_epochs * len(loader)
    optimizer, scheduler = make_optimizer(phase_steps)
    history = []
    step = 0
    started = time.time()
    model.train()

    for epoch in range(args.epochs):
        phase = "posterior"
        if args.two_stage and args.kind != "deterministic" and epoch >= posterior_epochs:
            phase = "prior"
            if epoch == posterior_epochs:
                model.set_training_phase("prior")
                remaining_steps = (args.epochs - posterior_epochs) * len(loader)
                optimizer, scheduler = make_optimizer(remaining_steps)
        sums: dict[str, float] = defaultdict(float)
        count = 0
        for cpu_batch in loader:
            batch = to_device(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch, sample_posterior=True)
                if phase == "prior":
                    loss, kl_parts = model.prior_fitting_loss(
                        output,
                        batch["valid"],
                        args.free_bits,
                    )
                    parts = {
                        "loss": loss.detach(),
                        "motion": output["prediction"].sum().detach() * 0.0,
                        "appearance": output["prediction"].sum().detach() * 0.0,
                        "visibility": output["prediction"].sum().detach() * 0.0,
                        "kl": loss.detach(),
                        "effect": output["prediction"].sum().detach() * 0.0,
                        "usage": output["prediction"].sum().detach() * 0.0,
                        "alignment": output["prediction"].sum().detach() * 0.0,
                        **kl_parts,
                    }
                else:
                    warmup = min(1.0, step / max(total_steps * 0.2, 1.0))
                    loss, parts = particle_world_model_loss(
                        model,
                        batch,
                        output,
                        kl_weight=0.0 if args.two_stage else args.kl_weight * warmup,
                        free_bits=args.free_bits,
                        move_weight=args.move_weight,
                        appearance_weight=args.appearance_weight,
                        visibility_weight=args.visibility_weight,
                        effect_weight=args.effect_weight,
                        usage_weight=args.usage_weight,
                        usage_margin=args.usage_margin,
                        alignment_weight=args.alignment_weight,
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for key, value in parts.items():
                sums[key] += float(value)
            count += 1
            step += 1
        epoch_result = {key: value / max(count, 1) for key, value in sums.items()}
        epoch_result.update({"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0], "phase": phase})
        history.append(epoch_result)
        print(
            f"[probe-train] kind={args.kind} epoch={epoch + 1}/{args.epochs} "
            f"loss={epoch_result['loss']:.5f} motion={epoch_result['motion']:.5f} "
            f"kl={epoch_result['kl']:.5f} phase={phase} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "model.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "cache_version": cache["version"],
        },
        checkpoint_path,
    )
    evaluation = {
        split: evaluate(
            model,
            cache,
            split,
            args.batch_size,
            args.prior_samples,
            device,
        )
        for split in ("train", "heldseed", "heldtask")
    }
    result = {
        "status": "ok",
        "kind": args.kind,
        "config": config.to_dict(),
        "args": vars(args),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "elapsed_seconds": time.time() - started,
        "history": history,
        "evaluation": evaluation,
        "checkpoint": os.path.abspath(checkpoint_path),
    }
    result_path = os.path.join(args.out, "metrics.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"status": "ok", "kind": args.kind, "metrics": result_path}, indent=2))


if __name__ == "__main__":
    main()
