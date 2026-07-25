"""Contract tests for action interventions, probes, and temporal slots."""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.action_interventions import (  # noqa: E402
    component_actions,
    current_object_support,
    replace_history,
    spatial_locality,
    state_locality,
    support_to_rgb,
    swap_one_object_action,
)
from igsw.adaptive_gaussian_wm.action_probe import (  # noqa: E402
    explained_fraction,
    fit_ridge,
    per_row_group_mse,
)
from igsw.adaptive_gaussian_wm.temporal_slot_evaluation import (  # noqa: E402
    sequence_metrics,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_action_components() -> None:
    posterior = torch.arange(2 * 3 * 2 * 8, dtype=torch.float32).reshape(2, 3, 2, 8)
    variants = component_actions(posterior, 6)
    require(bool((variants["canonical_only"][..., 6:] == 0).all()), "residual remained")
    require(bool((variants["residual_only"][..., :6] == 0).all()), "canonical remained")
    donor = -posterior
    swapped = swap_one_object_action(posterior, donor, 1)
    require(torch.equal(swapped[:, :, 0], posterior[:, :, 0]), "other object changed")
    require(torch.equal(swapped[:, :, 1], donor[:, :, 1]), "selected object did not change")
    batch = {"history_features": torch.ones(2, 1, 3, 4), "future_features": torch.ones(2, 1, 3, 4)}
    history = {"history_features": torch.zeros_like(batch["history_features"])}
    replaced = replace_history(batch, history)
    require(float(replaced["history_features"].sum()) == 0.0, "history was not replaced")
    require(torch.equal(replaced["future_features"], batch["future_features"]), "future changed")


def test_object_locality() -> None:
    token_assignment = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]
    )
    output = {
        "history_token_states": [
            SimpleNamespace(
                assignment=token_assignment,
                activation=torch.ones(1, 2, 1),
            )
        ],
        "history_slot_states": [
            SimpleNamespace(assignment=torch.eye(2)[None])
        ],
    }
    support = current_object_support(output)
    require(torch.equal(support.argmax(dim=1), torch.tensor([[0, 0, 1, 1]])), "support identity failed")
    valid_rgb = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    rgb_support = support_to_rgb(support, torch.tensor([[2, 2]]), valid_rgb)
    reference = torch.zeros(1, 1, 3, 4, 4)
    intervention = reference.clone()
    intervention[:, :, :, :2] = 1.0
    spatial = spatial_locality(
        reference, intervention, rgb_support, valid_rgb, 0
    )
    require(float(spatial["mass_lift"]) > 1.5, "localized RGB effect lacked lift")
    state = torch.zeros(1, 1, 2, 4)
    changed = state.clone()
    changed[:, :, 0] = 1.0
    effect, own = state_locality(state, changed, 0)
    require(float(effect) > 0.0 and float(own) > 0.999, "slot locality failed")


def test_action_probe() -> None:
    generator = torch.Generator().manual_seed(17)
    action = torch.randn(600, 2, generator=generator)
    history = torch.randn(600, 3, generator=generator)
    delta = action @ torch.tensor([[1.5, -0.5], [0.25, 2.0]])
    absolute = torch.cat((history[:, :2], history[:, 2:]), dim=1)
    absolute = torch.cat((absolute, torch.zeros(600, 1)), dim=1)
    absolute = absolute + torch.cat((delta, delta), dim=1)
    train = slice(0, 500)
    test = slice(500, None)
    action_delta = fit_ridge(action[train], delta[train], 0.1)
    action_absolute = fit_ridge(action[train], absolute[train], 0.1)
    joint_absolute = fit_ridge(
        torch.cat((history[train], action[train]), dim=1),
        absolute[train],
        0.1,
    )
    target_delta = delta[test]
    target_absolute = absolute[test]
    groups = {"all": torch.arange(target_absolute.shape[1])}
    delta_error = per_row_group_mse(
        action_delta.predict(action[test]), target_delta, {"all": torch.arange(2)}
    )["all"]
    delta_baseline = target_delta.square().mean(dim=1)
    action_error = per_row_group_mse(
        action_absolute.predict(action[test]), target_absolute, groups
    )["all"]
    joint_error = per_row_group_mse(
        joint_absolute.predict(torch.cat((history[test], action[test]), dim=1)),
        target_absolute,
        groups,
    )["all"]
    require(explained_fraction(delta_error, delta_baseline) > 0.99, "action did not decode delta")
    require(float(joint_error.mean()) < 1e-4, "history plus action did not decode state")
    require(float(action_error.mean()) > 100.0 * float(joint_error.mean()), "action leaked absolute state")


def test_temporal_slots() -> None:
    batch, frames, objects, dimension = 2, 3, 3, 4
    identity = torch.eye(dimension)[:objects]
    tracking = identity[None, None].expand(batch, frames, -1, -1).clone()
    features = tracking.clone()
    centers = torch.tensor(
        [[[-0.8, 0.0], [0.0, 0.0], [0.8, 0.0]]]
    ).expand(batch, frames, -1, -1).clone()
    centers[:, 1:] += torch.tensor([0.02, 0.01])
    activity = torch.ones(batch, frames, objects)
    support = torch.eye(objects)[None, None].expand(batch, frames, -1, -1)
    metrics = sequence_metrics(tracking, features, centers, activity, support)
    for name in (
        "tracking_retrieval_accuracy",
        "feature_retrieval_accuracy",
        "center_retrieval_accuracy",
        "support_retrieval_accuracy",
    ):
        require(float(metrics[name].min()) > 0.999, f"{name} lost identity")
    for name in (
        "tracking_identity_margin",
        "feature_identity_margin",
        "center_identity_margin",
        "support_identity_margin",
    ):
        require(float(metrics[name].min()) > 0.0, f"{name} is not positive")


def main() -> None:
    test_action_components()
    test_object_locality()
    test_action_probe()
    test_temporal_slots()
    print(
        json.dumps(
            {
                "status": "ok",
                "action_components": True,
                "object_locality": True,
                "action_probe": True,
                "temporal_slots": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
