#!/usr/bin/env python3
"""CPU contracts for v60 change calibration and residual Dynamics."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.gated_residual_object_transition_v60 import (  # noqa: E402
    GatedResidualObjectTransitionModel,
)
from igsw.adaptive_gaussian_wm.object_transition_objective_v60 import (  # noqa: E402
    object_transition_objective_v60,
)
from igsw.adaptive_gaussian_wm.query_persistent_state_v58 import (  # noqa: E402
    QueryPersistentState,
)
from igsw.adaptive_gaussian_wm.query_transition_target_v60 import (  # noqa: E402
    GatedQueryTransitionTarget,
    _teacher_change_distance,
)
from igsw.adaptive_gaussian_wm.v60_config import (  # noqa: E402
    GatedResidualTransitionConfig,
)


def tiny_config():
    return GatedResidualTransitionConfig(
        patch_dim=16,
        model_dim=32,
        identity_dim=8,
        dynamic_dim=8,
        heads=4,
        effect_dim=8,
        transition_layers=1,
    )


def fixture(config):
    torch.manual_seed(7)
    batch, frames, patches = 4, 2, 5
    horizons = len(config.dynamic_horizons)
    semantic = F.normalize(torch.randn(batch, config.patch_dim), dim=-1)
    future = semantic[:, None].repeat(1, horizons, 1)
    future[2:, :, 0] += 0.8
    future = F.normalize(future, dim=-1)
    source_geometry = torch.zeros(batch, config.target_geometry_dim)
    future_geometry = source_geometry[:, None].repeat(1, horizons, 1)
    future_geometry[2:] += 0.2
    source_visibility = torch.full((batch,), 0.9)
    future_visibility = source_visibility[:, None].repeat(1, horizons)
    valid = torch.ones(batch, horizons, dtype=torch.bool)
    provisional = GatedQueryTransitionTarget(
        source_semantic=semantic,
        source_geometry=source_geometry,
        source_visibility=source_visibility,
        future_semantic=future,
        future_geometry=future_geometry,
        future_visibility=future_visibility,
        delta_seconds=torch.tensor(config.dynamic_horizons)
        .float()[None]
        .repeat(batch, 1),
        pair_valid=valid,
        motion_active=valid,
        change_distance=torch.zeros(batch, horizons),
        change_strength=torch.zeros(batch, horizons),
    )
    distance = _teacher_change_distance(provisional, config)
    strength = 1.0 - torch.exp(
        -F.relu(distance - config.change_noise_floor) / config.change_scale
    )
    target = GatedQueryTransitionTarget(
        **{
            **provisional.__dict__,
            "change_distance": distance,
            "change_strength": strength,
        }
    )
    support = torch.full((batch, frames, patches), 0.2)
    center = torch.rand(batch, frames, 2)
    covariance = torch.eye(2)[None, None].repeat(batch, frames, 1, 1) * 0.02
    visibility_logits = torch.logit(torch.full((batch, frames), 0.9))
    source = QueryPersistentState(
        support_logits=torch.logit(support),
        support=support,
        identity=F.normalize(torch.randn(batch, config.identity_dim), dim=-1),
        identity_sequence=F.normalize(
            torch.randn(batch, frames, config.identity_dim), dim=-1
        ),
        dynamic=torch.randn(batch, frames, config.dynamic_dim),
        center=center,
        covariance=covariance,
        visibility_logits=visibility_logits,
        visibility=torch.sigmoid(visibility_logits),
        pooled_semantic=semantic[:, None].repeat(1, frames, 1),
    )
    return source, target


def prediction_delta(prediction, base):
    return (
        (prediction.semantic - base.semantic).abs().mean()
        + (prediction.geometry - base.geometry).abs().mean()
        + (prediction.visibility_logits - base.visibility_logits).abs().mean()
    )


def main():
    config = tiny_config()
    config.validate()
    source, target = fixture(config)
    model = GatedResidualObjectTransitionModel(config)
    effect = model.posterior(target)
    gated = model.change_gate(effect)
    base, correct = model.dynamics(source, effect, target.delta_seconds, gated.gate)
    _, low = model.dynamics(
        source, effect, target.delta_seconds, torch.full_like(gated.gate, 0.1)
    )
    _, high = model.dynamics(
        source, effect, target.delta_seconds, torch.full_like(gated.gate, 0.9)
    )
    assert float(prediction_delta(high.prediction, base).detach()) > float(
        prediction_delta(low.prediction, base).detach()
    )
    assert float(target.change_strength[:2].max()) < 1e-6
    assert float(target.change_strength[2:].min()) > 0.5
    shuffled_effect = effect.roll(1, dims=0)
    shuffled_gate = gated.gate.roll(1, dims=0)
    _, shuffled = model.dynamics(
        source, shuffled_effect, target.delta_seconds, shuffled_gate
    )
    output = {
        "effect": effect,
        "change_gate_logits": gated.gate_logits,
        "change_gate": gated.gate,
        "base": base,
        "correct": correct.prediction,
        "zero": base,
        "shuffled": shuffled.prediction,
    }
    loss, parts = object_transition_objective_v60(output, target, config)
    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(torch.stack(tuple(parts.values()))).all())
    loss.backward()
    assert model.change_gate.network[-1].weight.grad is not None
    assert model.dynamics.semantic_delta.weight.grad is not None
    print(
        {
            "status": "passed",
            "tests": 4,
            "low_change_strength": float(target.change_strength[:2].mean()),
            "high_change_strength": float(target.change_strength[2:].mean()),
        }
    )


if __name__ == "__main__":
    main()
