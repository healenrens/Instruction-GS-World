"""Remote CPU contract for canonical plus bottlenecked residual actions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.action_posterior import (  # noqa: E402
    ObjectDeltaActionPosterior,
)
from igsw.adaptive_gaussian_wm.action_embedding import (  # noqa: E402
    gate_canonical_center,
    residual_action_dropout,
)
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    _resize_action_tensor,
    warm_start_model,
)
from igsw.adaptive_gaussian_wm.config import (  # noqa: E402
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.dynamics import (  # noqa: E402
    JointObjectLatentDynamics,
)
from igsw.adaptive_gaussian_wm.model import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.observed_action import (  # noqa: E402
    rgb_logit_action,
)


RESIDUAL_DIMS = (0, 8, 16)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _posterior_contract(residual_dim: int) -> dict:
    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=6 + residual_dim,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        action_query_modulation=True,
        prior_query_residual=True,
    )
    posterior = ObjectDeltaActionPosterior(config)
    batch = 2
    history = torch.randn(batch, 1, config.object_slots, config.object_dim)
    future = history[:, -1, None] + 0.05 * torch.randn(
        batch,
        1,
        config.object_slots,
        config.object_dim,
    )
    activity = torch.ones(batch, 1, config.object_slots)
    history_centers = torch.randn(batch, 1, config.object_slots, 2)
    future_centers = history_centers[:, -1, None] + 0.05 * torch.randn(
        batch,
        1,
        config.object_slots,
        2,
    )
    actions = posterior(
        history,
        activity,
        future,
        activity,
        torch.ones(batch, 1),
        history_centers,
        future_centers,
    )
    center_delta = future_centers - history_centers[:, -1, None]
    expected_center = torch.tanh(
        torch.cat(
            (center_delta, center_delta.norm(dim=-1, keepdim=True)),
            dim=-1,
        )
        / 0.25
    )
    expected_semantic = torch.tanh(
        (future - history[:, -1, None]) @ posterior.semantic_projection
        / 0.25
    )
    center_error = float(
        (actions[..., :3] - expected_center).detach().abs().max()
    )
    semantic_error = float(
        (actions[..., 3:6] - expected_semantic).detach().abs().max()
    )
    _require(actions.shape[-1] == 6 + residual_dim, "action width mismatch")
    _require(center_error == 0.0, "center anchor is not exact")
    _require(semantic_error == 0.0, "semantic anchor is not exact")
    _require(
        (posterior.output is None) == (residual_dim == 0),
        "residual head presence mismatch",
    )
    if residual_dim > 0:
        actions[..., 6:].square().mean().backward()
        gradient_norm = sum(
            float(parameter.grad.square().sum())
            for parameter in posterior.parameters()
            if parameter.grad is not None
        ) ** 0.5
        _require(gradient_norm > 0.0, "residual head has no gradient")
    else:
        gradient_norm = 0.0
    return {
        "action_dim": config.action_dim,
        "action_residual_dim": config.action_residual_dim,
        "center_anchor_max_error": center_error,
        "semantic_anchor_max_error": semantic_error,
        "residual_head_present": posterior.output is not None,
        "residual_gradient_norm": gradient_norm,
    }


def _dropout_contract() -> dict:
    torch.manual_seed(83)
    actions = torch.randn(32, 2, 8, 14, requires_grad=True)
    dropped = residual_action_dropout(actions, 0.75, True)
    _require(
        torch.equal(dropped[..., :6], actions[..., :6]),
        "dropout changed canonical actions",
    )
    kept = dropped[..., 6:].abs().sum(dim=-1) > 0.0
    _require(kept.any() and (~kept).any(), "dropout did not produce both states")
    dropped.square().mean().backward()
    _require(
        actions.grad is not None and float(actions.grad[..., :6].norm()) > 0.0,
        "canonical actions lost gradients",
    )
    _require(
        float(actions.grad[..., 6:][kept].norm()) > 0.0,
        "kept residual actions lost gradients",
    )
    _require(
        float(actions.grad[..., 6:][~kept].abs().max()) == 0.0,
        "dropped residual actions retained gradients",
    )
    evaluation = residual_action_dropout(actions.detach(), 0.75, False)
    _require(torch.equal(evaluation, actions.detach()), "evaluation used dropout")
    return {"keep_fraction": float(kept.float().mean())}


def _center_gate_contract() -> dict:
    actions = torch.randn(2, 1, 4, 6)
    gated = gate_canonical_center(actions, 0.1)
    _require(
        torch.allclose(gated[..., :3], 0.1 * actions[..., :3]),
        "center gate did not attenuate geometry",
    )
    _require(
        torch.equal(gated[..., 3:], actions[..., 3:]),
        "center gate changed semantic channels",
    )
    return {
        "center_scale": float(
            gated[..., :3].norm() / actions[..., :3].norm()
        ),
        "semantic_max_error": float(
            (gated[..., 3:] - actions[..., 3:]).abs().max()
        ),
    }


def _center_gate_dynamics_contract() -> dict:
    base = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=6,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        bounded_residual_action=True,
        action_query_modulation=True,
        prior_query_residual=True,
    )
    reference = JointObjectLatentDynamics(base).eval()
    gated = JointObjectLatentDynamics(
        replace(base, canonical_center_gate=0.1)
    ).eval()
    gated.load_state_dict(reference.state_dict(), strict=True)
    history = torch.randn(2, 1, 4, base.object_dim)
    activity = torch.ones(2, 1, 4)
    history_scale = torch.zeros(2, 1)
    future_scale = torch.ones(2, 1)
    actions = torch.randn(2, 1, 4, 6)
    expected = reference(
        history, activity, history_scale, future_scale,
        gate_canonical_center(actions, 0.1),
    ).future_slots
    actual = gated(
        history, activity, history_scale, future_scale, actions,
    ).future_slots
    error = float((actual - expected).detach().abs().max())
    _require(error == 0.0, "Dynamics did not apply the center gate")
    _require(gated.residual_action_input is None, "R=0 kept residual projection")
    return {"routing_max_error": error, "residual_projection_present": False}


def _canonical_only_warm_start_contract() -> dict:
    source_config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=14,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        rgb_semantic_action=True,
        bounded_residual_action=True,
        rgb_supervision=True,
    )
    source = AdaptiveGaussianObjectWorldModel(source_config)
    target = AdaptiveGaussianObjectWorldModel(
        replace(source_config, action_dim=6, canonical_center_gate=0.1)
    )
    report = warm_start_model(
        target,
        {
            "checkpoint_version": 18,
            "config": source_config.to_dict(),
            "model": source.state_dict(),
        },
    )
    projection = "dynamics.residual_action_input.weight"
    _require(projection in report["dropped"], "residual projection was not dropped")
    _require(not report["unexpected"], "canonical warm-start has unexpected keys")
    _require(not report["shape_mismatch"], "canonical warm-start has shape mismatch")
    return {
        "dropped_residual_projection": report["dropped"][projection],
        "missing_count": len(report["missing"]),
        "transformed_count": len(report["transformed"]),
    }


def _learned_basis_contract() -> dict:
    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=14,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        learned_semantic_action_basis=True,
        semantic_action_basis_weight=0.1,
    )
    posterior = ObjectDeltaActionPosterior(config)
    history = torch.randn(2, 1, 4, config.object_dim)
    future = history[:, -1, None] + 0.05 * torch.randn(2, 1, 4, config.object_dim)
    activity = torch.ones(2, 1, 4)
    centers = torch.randn(2, 1, 4, 2)
    actions = posterior(
        history,
        activity,
        future,
        activity,
        torch.ones(2, 1),
        centers,
        centers + 0.05,
    )
    actions[..., 3:6].square().mean().backward()
    gradient = posterior.semantic_projection.grad
    _require(gradient is not None and float(gradient.norm()) > 0.0, "basis has no gradient")
    norms = posterior.semantic_basis().norm(dim=0)
    norm_error = float((norms.detach() - 1.0).abs().max())
    _require(norm_error < 1e-6, "basis is not normalized")
    return {
        "gradient_norm": float(gradient.norm()),
        "max_column_norm_error": norm_error,
    }


def _rgb_action_contract() -> dict:
    config = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=14,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        rgb_semantic_action=True,
        rgb_supervision=True,
    )
    posterior = ObjectDeltaActionPosterior(config)
    history = torch.randn(2, 1, 4, config.object_dim)
    future = history[:, -1, None] + 0.05 * torch.randn(
        2, 1, 4, config.object_dim
    )
    activity = torch.ones(2, 1, 4)
    centers = torch.randn(2, 1, 4, 2)
    current_rgb = 0.2 + 0.6 * torch.rand(2, 4, 3)
    future_rgb = (
        current_rgb[:, None] + 0.1 * torch.randn(2, 1, 4, 3)
    ).clamp(0.05, 0.95)
    actions = posterior(
        history,
        activity,
        future,
        activity,
        torch.ones(2, 1),
        centers,
        centers + 0.05,
        None,
        current_rgb,
        future_rgb,
    )
    expected = rgb_logit_action(current_rgb, future_rgb)
    error = float((actions[..., 3:6] - expected).detach().abs().max())
    _require(error == 0.0, "RGB action anchor is not exact")
    _require(posterior.semantic_projection is None, "RGB action kept a slot basis")
    return {
        "semantic_anchor_max_error": error,
        "action_rms": float(
            actions[..., 3:6].detach().square().mean().sqrt()
        ),
    }


def _rgb_warm_start_contract() -> dict:
    base = replace(
        AdaptiveGaussianWMConfig.tiny(12),
        action_tokens=4,
        action_dim=14,
        object_aligned_actions=True,
        canonical_center_action=True,
        canonical_semantic_action=True,
        bounded_residual_action=True,
        action_residual_gate=0.1,
        action_residual_dropout=0.75,
        rgb_supervision=True,
    )
    source = AdaptiveGaussianObjectWorldModel(base)
    target = AdaptiveGaussianObjectWorldModel(
        replace(base, rgb_semantic_action=True)
    )
    source.dynamics.action_input.weight.data.fill_(7.0)
    initial_rgb_columns = target.dynamics.action_input.weight[:, 3:6].clone()
    report = warm_start_model(
        target,
        {
            "checkpoint_version": 17,
            "config": base.to_dict(),
            "model": source.state_dict(),
        },
    )
    weight = target.dynamics.action_input.weight
    _require(
        bool((weight[:, :3] == 7.0).all()),
        "center action columns were not warm-started",
    )
    _require(
        torch.equal(weight[:, 3:6], initial_rgb_columns),
        "RGB semantic columns reused incompatible slot semantics",
    )
    projection = "latent_actions.posterior.semantic_projection"
    _require(projection in report["dropped"], "slot basis was not dropped")
    return {
        "semantic_transform": report["transformed"][
            "dynamics.action_input.weight"
        ]["transform"],
        "dropped_slot_basis": report["dropped"][projection],
    }


def _warm_start_contract(residual_dim: int) -> dict:
    target_action_dim = 6 + residual_dim
    checkpoint = {
        "checkpoint_version": 13,
        "config": {
            "action_dim": 64,
            "canonical_semantic_action": True,
        },
    }
    posterior_source = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    posterior_target = torch.zeros(residual_dim, 4)
    posterior = _resize_action_tensor(
        "latent_actions.posterior.output.3.weight",
        posterior_source,
        posterior_target,
        checkpoint,
        target_action_dim,
    )
    if residual_dim > 0:
        _require(posterior is not None, "posterior residual resize failed")
        _require(
            torch.equal(posterior[0], posterior_source[6 : 6 + residual_dim]),
            "posterior residual slice used the wrong source dimensions",
        )

    dynamics_source = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64)
    dynamics_target = torch.zeros(3, target_action_dim)
    dynamics = _resize_action_tensor(
        "dynamics.action_input.weight",
        dynamics_source,
        dynamics_target,
        checkpoint,
        target_action_dim,
    )
    _require(dynamics is not None, "Dynamics action resize failed")
    _require(
        torch.equal(dynamics[0], dynamics_source[:, :target_action_dim]),
        "Dynamics action prefix copy failed",
    )

    tail_dim = 9
    flow_source = torch.arange(
        2 * (2 * 64 + tail_dim),
        dtype=torch.float32,
    ).reshape(2, 2 * 64 + tail_dim)
    flow_target = torch.zeros(2, 2 * target_action_dim + tail_dim)
    flow = _resize_action_tensor(
        "latent_actions.prior.input_projection.weight",
        flow_source,
        flow_target,
        checkpoint,
        target_action_dim,
    )
    _require(flow is not None, "Prior input resize failed")
    _require(
        torch.equal(flow[0][:, :target_action_dim], flow_source[:, :target_action_dim]),
        "Prior latent prefix copy failed",
    )
    _require(
        torch.equal(flow[0][:, 2 * target_action_dim :], flow_source[:, 128:]),
        "Prior context copy failed",
    )
    return {
        "posterior_transform": posterior[1] if posterior is not None else None,
        "dynamics_transform": dynamics[1],
        "flow_transform": flow[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "status": "ok",
        "checkpoint_version": CHECKPOINT_VERSION,
        "dropout": _dropout_contract(),
        "center_gate": _center_gate_contract(),
        "center_gate_dynamics": _center_gate_dynamics_contract(),
        "canonical_only_warm_start": _canonical_only_warm_start_contract(),
        "learned_basis": _learned_basis_contract(),
        "rgb_action": _rgb_action_contract(),
        "rgb_warm_start": _rgb_warm_start_contract(),
        "layouts": {
            str(residual_dim): {
                "posterior": _posterior_contract(residual_dim),
                "warm_start": _warm_start_contract(residual_dim),
            }
            for residual_dim in RESIDUAL_DIMS
        },
    }
    _require(CHECKPOINT_VERSION == 27, "checkpoint version was not upgraded")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
