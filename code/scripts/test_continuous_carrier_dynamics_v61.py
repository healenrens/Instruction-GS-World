"""CPU tensor contracts for object-bound v61 latent-effect Dynamics."""

from __future__ import annotations

import os
from types import SimpleNamespace
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.carrier_dynamics_objective_v61 import (  # noqa: E402
    carrier_dynamics_objective_v61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_dynamics_v61 import (  # noqa: E402
    EFFECT_CAPACITIES,
    ContinuousCarrierEffectPosteriorV61,
    EffectConditionedCarrierDynamicsV61,
    shuffled_effect_v61,
    zero_effect_v61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_state_v61 import (  # noqa: E402
    CarrierStateV61,
    ContinuousObjectStateV61,
    ObjectRootStateV61,
)
from igsw.adaptive_gaussian_wm.v61_config import config_for_variant  # noqa: E402


def _normalized(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def synthetic_state(config, batch=3):
    carriers, roots, tokens = config.carrier_count, config.object_roots, 16
    support = torch.softmax(torch.randn(batch, 1, carriers, tokens), dim=-1)
    owner = torch.softmax(torch.randn(batch, 1, carriers, config.total_owners), dim=-1)
    carrier = CarrierStateV61(
        feature=_normalized(batch, 1, carriers, config.student_dim),
        identity=_normalized(batch, 1, carriers, config.identity_dim),
        dynamic=torch.randn(batch, 1, carriers, config.dynamic_dim),
        center=torch.rand(batch, 1, carriers, 2) * 1.8 - 0.9,
        covariance=torch.eye(2)[None, None, None]
        .expand(batch, 1, carriers, -1, -1)
        .clone()
        * 0.05,
        presence=torch.rand(batch, 1, carriers) * 0.5 + 0.25,
        visibility=torch.rand(batch, 1, carriers) * 0.5 + 0.25,
        support=support,
    )
    root = ObjectRootStateV61(
        feature=_normalized(batch, 1, roots, config.student_dim),
        identity=_normalized(batch, 1, roots, config.identity_dim),
        dynamic=torch.randn(batch, 1, roots, config.dynamic_dim),
        center=torch.rand(batch, 1, roots, 2) * 1.8 - 0.9,
        relative_scale=torch.rand(batch, 1, roots) * 0.5 + 0.1,
        presence=torch.rand(batch, 1, roots) * 0.5 + 0.25,
        visibility=torch.rand(batch, 1, roots) * 0.5 + 0.25,
        owner=owner,
    )
    return ContinuousObjectStateV61(carrier, root)


class Harness(nn.Module):
    def __init__(self, config, capacity):
        super().__init__()
        self.config = config
        self.state_model = nn.Module()
        self.state_model.identity_to_dino = nn.Linear(
            config.identity_dim, config.teacher_projection_dim
        )
        self.state_model.requires_grad_(False)
        self.effect_posterior = ContinuousCarrierEffectPosteriorV61(config, capacity)
        self.dynamics = EffectConditionedCarrierDynamicsV61(
            config, EFFECT_CAPACITIES[capacity][1]
        )


def run_capacity(capacity):
    config = config_for_variant("siglip2_dino_object")
    model = Harness(config, capacity)
    source, target = synthetic_state(config), synthetic_state(config)
    effect = model.effect_posterior(source, target)
    delta = torch.full((3,), 0.4)
    correct = model.dynamics(source, effect, delta)
    zero = model.dynamics(source, zero_effect_v61(effect), delta)
    shuffled = model.dynamics(source, shuffled_effect_v61(effect), delta)
    batch, frames, points = 3, 2, 12
    carrier_assignment = torch.softmax(
        torch.randn(batch, frames, points, config.carrier_count), dim=-1
    )
    root_assignment = torch.softmax(
        torch.randn(batch, frames, points, config.object_roots), dim=-1
    )
    evidence = SimpleNamespace(
        coordinates=torch.rand(batch, frames, points, 2) * 1.8 - 0.9,
        sampled_features=_normalized(batch, frames, points, config.dino_dim),
        visibility=torch.ones(batch, frames, points, dtype=torch.bool),
    )
    relation = SimpleNamespace(
        lifecycle_known=torch.ones(batch, frames, points, dtype=torch.bool),
        visibility=torch.ones(batch, frames, points),
        presence=torch.ones(batch, frames, points),
    )
    loss, parts = carrier_dynamics_objective_v61(
        model,
        source,
        target,
        correct,
        zero,
        shuffled,
        effect,
        carrier_assignment,
        root_assignment,
        evidence,
        relation,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"v61 {capacity} Dynamics loss is non-finite")
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"v61 {capacity} leaves parameters unused: {missing}")
    factors, dimensions, binding = EFFECT_CAPACITIES[capacity]
    if tuple(effect.value.shape) != (batch, factors, dimensions):
        raise RuntimeError(f"v61 {capacity} latent effect shape differs")
    if float((effect.owner.sum(dim=-1) - 1.0).abs().max()) >= 1e-5:
        raise RuntimeError(f"v61 {capacity} owner partition differs")
    if not all(bool(torch.isfinite(value).all()) for value in parts.values()):
        raise RuntimeError(f"v61 {capacity} diagnostics are non-finite")
    return {
        "capacity": capacity,
        "binding": binding,
        "loss": float(loss.detach()),
        "effect_shape": tuple(effect.value.shape),
        "metric_count": len(parts),
    }


def main():
    torch.manual_seed(17)
    reports = [run_capacity(capacity) for capacity in EFFECT_CAPACITIES]
    print({"status": "passed", "capacities": reports})


if __name__ == "__main__":
    main()
