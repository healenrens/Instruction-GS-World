"""Analyze whether global latent actions transfer as scene-independent effects."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.models import ParticleWorldModel, WorldModelConfig  # noqa: E402
from igsw.latent_particle_wm.objectives import global_effect_target  # noqa: E402
from igsw.latent_particle_wm.probe_data import ParticleProbeDataset  # noqa: E402


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def collect(model: ParticleWorldModel, cache: dict, split: str, device: torch.device) -> dict:
    dataset = ParticleProbeDataset(cache, split)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    values: dict[str, list[torch.Tensor]] = {
        "latent": [],
        "q_std": [],
        "effect": [],
        "effect_prediction": [],
        "prior_entropy": [],
        "component": [],
    }
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        output = model(batch, sample_posterior=False)
        effect = global_effect_target(batch["target"], batch["motion_valid"])
        values["latent"].append(output["global_q_mu"].float().cpu())
        values["q_std"].append(output["global_q_log_std"].exp().float().cpu())
        values["effect"].append(effect.float().cpu())
        values["effect_prediction"].append(output["effect_prediction"].float().cpu())
        if "global_p_logits" in output:
            probability = output["global_p_logits"].softmax(dim=-1)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
            distance = (
                output["global_q_mu"][:, None] - output["global_p_mu"]
            ).square().mean(dim=-1)
            values["prior_entropy"].append(entropy.float().cpu())
            values["component"].append(distance.argmin(dim=-1).cpu())
    return {
        key: torch.cat(items) if items else torch.empty(0)
        for key, items in values.items()
    }


def normalized(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / std.clamp_min(1e-6)


def fit_effect_readout(train: dict, split: dict) -> dict:
    latent_mean = train["latent"].mean(dim=0)
    latent_std = train["latent"].std(dim=0)
    effect_mean = train["effect"].mean(dim=0)
    effect_std = train["effect"].std(dim=0)
    x_train = normalized(train["latent"], latent_mean, latent_std)
    y_train = normalized(train["effect"], effect_mean, effect_std)
    x_train = torch.cat((x_train, torch.ones(len(x_train), 1)), dim=-1)
    ridge = torch.eye(x_train.shape[1]) * 1e-3
    weight = torch.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
    x = normalized(split["latent"], latent_mean, latent_std)
    x = torch.cat((x, torch.ones(len(x), 1)), dim=-1)
    target = normalized(split["effect"], effect_mean, effect_std)
    prediction = x @ weight
    residual = (prediction - target).square().sum(dim=0)
    total = (target - target.mean(dim=0)).square().sum(dim=0).clamp_min(1e-8)
    r2 = 1.0 - residual / total
    return {
        "linear_effect_r2_mean": float(r2.mean()),
        "linear_effect_r2_per_dim": [float(value) for value in r2],
        "effect_head_mae": float((split["effect_prediction"] - split["effect"]).abs().mean()),
    }


def nearest_effect_transfer(train: dict, split: dict, seed: int) -> dict:
    train_latent = F_normalize(train["latent"])
    query_latent = F_normalize(split["latent"])
    nearest_effect = []
    chunk = 256
    for start in range(0, len(query_latent), chunk):
        distance = torch.cdist(query_latent[start : start + chunk], train_latent)
        nearest = distance.argmin(dim=1)
        nearest_effect.append(train["effect"][nearest])
    nearest_effect = torch.cat(nearest_effect)
    effect_scale = train["effect"].std(dim=0).clamp_min(1e-6)
    nearest_error = ((nearest_effect - split["effect"]) / effect_scale).norm(dim=-1)
    generator = torch.Generator().manual_seed(seed)
    random_index = torch.randint(len(train["effect"]), (len(split["effect"]),), generator=generator)
    random_error = ((train["effect"][random_index] - split["effect"]) / effect_scale).norm(dim=-1)
    return {
        "nearest_effect_error": float(nearest_error.mean()),
        "random_effect_error": float(random_error.mean()),
        "nearest_over_random": float(nearest_error.mean() / random_error.mean().clamp_min(1e-8)),
    }


def F_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def distance_alignment(values: dict, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    count = min(20000, len(values["latent"]) * 10)
    first = torch.randint(len(values["latent"]), (count,), generator=generator)
    second = torch.randint(len(values["latent"]), (count,), generator=generator)
    latent_distance = (values["latent"][first] - values["latent"][second]).norm(dim=-1)
    effect_distance = (values["effect"][first] - values["effect"][second]).norm(dim=-1)
    return float(torch.corrcoef(torch.stack((latent_distance, effect_distance)))[0, 1])


def summarize(train: dict, split: dict, seed: int) -> dict:
    result = {
        **fit_effect_readout(train, split),
        **nearest_effect_transfer(train, split, seed),
        "latent_effect_distance_correlation": distance_alignment(split, seed),
        "posterior_latent_std_across_data": float(split["latent"].std(dim=0).mean()),
        "posterior_distribution_std": float(split["q_std"].mean()),
    }
    if len(split["prior_entropy"]):
        count = int(split["component"].max()) + 1
        histogram = torch.bincount(split["component"], minlength=count).float()
        result.update(
            {
                "prior_mixture_entropy": float(split["prior_entropy"].mean()),
                "nearest_component_fraction": [float(value) for value in histogram / histogram.sum()],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = WorldModelConfig(**checkpoint["config"])
    model = ParticleWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = {
        split: collect(model, cache, split, device)
        for split in ("train", "heldseed", "heldtask")
    }
    result = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "cache": os.path.abspath(args.cache),
        "splits": {
            split: summarize(values["train"], values[split], args.seed + offset)
            for offset, split in enumerate(("train", "heldseed", "heldtask"))
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
