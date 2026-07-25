"""Remote structural test for causal Object-Slot continuous action anchors."""
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
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.synthetic import make_synthetic_batch  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _gradient_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.float().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    ) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--residual_dim",
        type=int,
        choices=(0, 8, 16, 58),
        required=True,
    )
    parser.add_argument(
        "--residual_gate",
        type=float,
        choices=(1.0, 0.25, 0.1),
        default=1.0,
    )
    parser.add_argument(
        "--residual_dropout",
        type=float,
        choices=(0.0, 0.5, 0.75),
        default=0.0,
    )
    parser.add_argument("--semantic_basis", choices=("fixed", "learned"), default="fixed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this contract test must run on the remote CUDA host")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        condition_dim=16,
        action_tokens=4,
        action_dim=6 + args.residual_dim,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        learned_semantic_action_basis=args.semantic_basis == "learned",
        semantic_action_basis_weight=(
            0.1 if args.semantic_basis == "learned" else 0.0
        ),
        bounded_residual_action=(
            args.residual_gate < 1.0 or args.residual_dropout > 0.0
        ),
        action_residual_gate=args.residual_gate,
        action_residual_dropout=args.residual_dropout,
        action_query_modulation=True,
        prior_query_residual=True,
    )
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    batch = make_synthetic_batch(
        config.feature_dim,
        4,
        1,
        1,
        8,
        device,
        paired_futures=True,
        max_objects=config.object_slots,
        semantic_branch_strength=0.2,
    )
    batch["condition_feature"] = torch.randn(4, config.condition_dim, device=device)
    order = torch.arange(4, device=device).reshape(-1, 2).flip(1).flatten()
    swapped = {
        name: (
            value[order]
            if torch.is_tensor(value)
            and value.shape[:1] == (4,)
            and name.startswith("future_")
            else value
        )
        for name, value in batch.items()
    }
    instruction_swapped = dict(batch)
    instruction_swapped["condition_feature"] = batch["condition_feature"][order]

    model.eval()
    with torch.no_grad():
        history = model.encode_history(batch)
        future_history = model.encode_history(swapped)
        _, target = model.encode_targets(batch)
        _, swapped_target = model.encode_targets(swapped)
        condition = model.encode_condition(batch)
        swapped_condition = model.encode_condition(instruction_swapped)
        history_scale = signed_gap_scale(
            batch["history_times"],
            config.gap_reference,
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            config.gap_reference,
        )
        prior = model.prior_context(
            history,
            future_scale,
            history_scale,
            condition,
        )
        prior_future = model.prior_context(
            future_history,
            future_scale,
            history_scale,
            condition,
        )
        prior_instruction = model.prior_context(
            history,
            future_scale,
            history_scale,
            swapped_condition,
        )
        actions = model.latent_actions.posterior(
            history["slots"],
            history["activity"],
            target["slots"],
            target["activity"],
            future_scale,
            history["center"],
            target["center"],
            condition,
        )
        swapped_actions = model.latent_actions.posterior(
            future_history["slots"],
            future_history["activity"],
            swapped_target["slots"],
            swapped_target["activity"],
            future_scale,
            future_history["center"],
            swapped_target["center"],
            condition,
        )
        center_delta = target["center"] - history["center"][:, -1, None]
        expected_center = torch.tanh(
            torch.cat(
                (center_delta, center_delta.norm(dim=-1, keepdim=True)),
                dim=-1,
            )
            / 0.25
        )
        slot_delta = target["slots"] - history["slots"][:, -1, None]
        expected_semantic = torch.tanh(
            slot_delta @ model.latent_actions.posterior.semantic_basis()
            / 0.25
        )
        history_mask = torch.zeros(
            history["slots"].shape[:3],
            device=device,
            dtype=torch.bool,
        )
        matched = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions,
            history_mask,
            history["center"],
            condition,
        )
        permuted = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions.roll(1, dims=-2),
            history_mask,
            history["center"],
            condition,
        )
    report = {
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "action_residual_gate": config.action_residual_gate,
        "action_residual_dropout": config.action_residual_dropout,
        "semantic_action_basis": args.semantic_basis,
        "history_future_swap_max": _difference(
            history["slots"],
            future_history["slots"],
        ),
        "prior_future_swap_max": _difference(prior, prior_future),
        "prior_instruction_swap_max": _difference(prior, prior_instruction),
        "posterior_future_swap_max": _difference(actions, swapped_actions),
        "center_anchor_max_error": _difference(actions[..., :3], expected_center),
        "semantic_anchor_max_error": _difference(
            actions[..., 3:6],
            expected_semantic,
        ),
        "object_permutation_slot_max": _difference(
            matched.future_slots,
            permuted.future_slots,
        ),
    }
    _require(report["history_future_swap_max"] < 1e-6, "future changed history")
    _require(report["prior_future_swap_max"] < 1e-6, "future changed Prior")
    _require(report["prior_instruction_swap_max"] > 1e-6, "Prior ignored language")
    _require(
        report["posterior_future_swap_max"] > 1e-6,
        "Posterior ignored future",
    )
    _require(report["center_anchor_max_error"] < 1e-6, "center anchor is not exact")
    _require(
        report["semantic_anchor_max_error"] < 1e-6,
        "semantic anchor is not exact",
    )
    _require(
        report["object_permutation_slot_max"] > 1e-6,
        "Dynamics ignored slot identity",
    )

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch, history_mask=history_mask)
    loss = output["predicted_future_slots"].square().mean()
    loss.backward()
    report["posterior_gradient_norm"] = _gradient_norm(
        model.latent_actions.posterior
    )
    report["dynamics_gradient_norm"] = _gradient_norm(model.dynamics)
    if config.action_residual_dim == 0:
        _require(
            report["posterior_gradient_norm"] == 0.0,
            "canonical-only Posterior unexpectedly has learned residual gradients",
        )
    else:
        _require(report["posterior_gradient_norm"] > 0.0, "Posterior has no gradient")
    _require(report["dynamics_gradient_norm"] > 0.0, "Dynamics has no gradient")
    report["status"] = "ok"
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
