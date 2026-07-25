"""Real-pair regression test for correlated actions, rendering, and flow prior."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.action_field import (  # noqa: E402
    ActionFieldConfig,
    CorrelatedActionField,
)
from igsw.latent_particle_wm.action_objectives import (  # noqa: E402
    ActionLossWeights,
    posterior_joint_loss,
)
from igsw.latent_particle_wm.pair_data import CausalPairDataset  # noqa: E402


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_height", type=int, default=98)
    parser.add_argument("--render_width", type=int, default=130)
    args = parser.parse_args()
    torch.manual_seed(17)
    device = torch.device(args.device)
    dataset = CausalPairDataset(
        args.data,
        "train",
        control_rows=16,
        control_cols=16,
        active_count=192,
        dino_root=args.dino,
    )
    batch = next(iter(DataLoader(Subset(dataset, [0, 1]), batch_size=2)))
    batch = move_to_device(batch, device)
    model = CorrelatedActionField(
        ActionFieldConfig(hidden_dim=64, layers=1, heads=4, dino_dim=32)
    ).to(device)

    model.set_training_phase("posterior")
    prior_context = model.encode_current(batch)["prior_context"]
    order = torch.tensor([1, 0], device=device)
    shuffled = dict(batch)
    future_fields = (
        "target",
        "motion_valid",
        "visible",
        "matched",
        "target_xyz",
        "tracker_image_jump",
        "tracker_depth_jump",
        "rgb1",
        "dino1",
    )
    for key in future_fields:
        shuffled[key] = batch[key][order]
    shuffled_context = model.encode_current(shuffled)["prior_context"]
    prior_difference = float((prior_context - shuffled_context).abs().max().detach())
    if prior_difference != 0.0:
        raise ValueError(f"future-conditioned prior context: {prior_difference}")

    output = model.forward_posterior(batch)
    shuffled_output = model.forward_posterior(shuffled)
    posterior_difference = float(
        (output["actions"] - shuffled_output["actions"]).abs().max().detach()
    )
    if posterior_difference <= 1e-6:
        raise ValueError("posterior actions do not respond to future targets")
    loss, parts = posterior_joint_loss(
        model,
        batch,
        output,
        ActionLossWeights(),
        args.render_height,
        args.render_width,
    )
    loss.backward()
    posterior_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not posterior_gradients or not all(
        bool(torch.isfinite(gradient).all()) for gradient in posterior_gradients
    ):
        raise ValueError("posterior stage produced missing or nonfinite gradients")
    field_gradient = float(model.field_head.weight.grad.abs().mean())
    if field_gradient <= 0.0:
        raise ValueError("Gaussian rendering did not reach the motion field head")

    model.zero_grad(set_to_none=True)
    model.set_training_phase("prior")
    prior_loss = model.prior_matching_loss(batch)
    prior_loss.backward()
    flow_gradients = [
        parameter.grad
        for parameter in model.flow_prior.parameters()
        if parameter.grad is not None
    ]
    nonprior_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("flow_prior.") and parameter.grad is not None
    ]
    if not flow_gradients or nonprior_gradients:
        raise ValueError("prior phase did not isolate flow-prior parameters")
    if not all(bool(torch.isfinite(gradient).all()) for gradient in flow_gradients):
        raise ValueError("flow-prior stage produced nonfinite gradients")

    report = {
        "status": "ok",
        "posterior_loss": float(loss.detach()),
        "prior_loss": float(prior_loss.detach()),
        "prior_future_swap_max_abs_difference": prior_difference,
        "posterior_future_swap_max_abs_difference": posterior_difference,
        "posterior_gradient_tensors": len(posterior_gradients),
        "flow_gradient_tensors": len(flow_gradients),
        "field_head_mean_abs_gradient": field_gradient,
        "loss_parts": {key: float(value) for key, value in parts.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
