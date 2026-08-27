#!/usr/bin/env python3
"""CPU contracts for V60 observation-grounded reconstruction metrics."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    ObjectTransitionPrediction,
)
from igsw.adaptive_gaussian_wm.object_observation_evaluation_v60 import (  # noqa: E402
    ObjectObservationAccumulator,
    ObjectObservationTarget,
    observation_reconstruction_statistics_v60,
)


def unit(index, dim):
    value = torch.zeros(dim)
    value[index] = 1.0
    return value


def prediction(semantic, geometry, visibility):
    return ObjectTransitionPrediction(
        semantic=semantic,
        geometry=geometry,
        visibility_logits=torch.logit(visibility.clamp(1e-4, 1.0 - 1e-4)),
    )


def fixture():
    batch, horizons, patches, points, dim = 2, 3, 9, 3, 8
    source_semantic = unit(0, dim)[None].repeat(batch, 1)
    future_semantic = unit(1, dim)[None, None].repeat(batch, horizons, 1)
    source_patches = source_semantic[:, None].repeat(1, patches, 1)
    future_patches = future_semantic[:, :, None].repeat(1, 1, patches, 1)
    axis = torch.linspace(-1.0, 1.0, 3)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(1, 1, patches, 2)
    coordinates = coordinates.repeat(batch, horizons, 1, 1)
    actual_support = torch.exp(-coordinates.square().sum(dim=-1) / 0.25)
    track_features = future_semantic[:, :, None].repeat(1, 1, points, 1)
    track_weights = torch.ones(batch, horizons, points)
    observation = ObjectObservationTarget(
        source_patches=source_patches,
        future_patches=future_patches,
        future_valid=torch.ones(batch, horizons, patches, dtype=torch.bool),
        relative_patch_coordinates=coordinates,
        future_track_features=track_features,
        future_track_weights=track_weights,
        actual_support=actual_support,
        pair_valid=torch.ones(batch, horizons, dtype=torch.bool),
    )
    source_geometry = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.0])
    future_geometry = source_geometry[None, None].repeat(batch, horizons, 1)
    source_visibility = torch.full((batch,), 0.9)
    future_visibility = torch.full((batch, horizons), 0.9)
    target = SimpleNamespace(
        source_semantic=source_semantic,
        source_geometry=source_geometry[None].repeat(batch, 1),
        source_visibility=source_visibility,
        future_semantic=future_semantic,
        future_geometry=future_geometry,
        future_visibility=future_visibility,
        pair_valid=observation.pair_valid,
    )
    correct = prediction(future_semantic, future_geometry, future_visibility)
    base_semantic = source_semantic[:, None].repeat(1, horizons, 1)
    base = prediction(base_semantic, future_geometry, future_visibility)
    shuffled = prediction(-future_semantic, future_geometry, future_visibility)
    output = {
        "correct": correct,
        "base": base,
        "zero": base,
        "shuffled": shuffled,
    }
    return output, target, observation


def main():
    output, target, observation = fixture()
    statistics = observation_reconstruction_statistics_v60(
        output, target, observation, support_sigma=0.10
    )
    accumulator = ObjectObservationAccumulator()
    accumulator.update(statistics)
    metrics = accumulator.finalize()
    assert metrics["actual_support_oracle_support_iou"] == 1.0
    assert metrics["semantic_compression_floor"] < 1e-6
    assert (
        abs(
            metrics["correct_composite_object_error"]
            - metrics["teacher_state_oracle_composite_object_error"]
        )
        < 1e-6
    )
    assert metrics["correct_dense_object_semantic_error"] < 1e-6
    assert metrics["persistence_dense_object_semantic_error"] > 0.9
    assert metrics["prediction_gain_over_persistence"] > 0.0
    assert metrics["teacher_state_gain_over_persistence"] > 0.0
    assert metrics["teacher_state_oracle_covariance_psd_fraction"] == 1.0
    selected = torch.tensor([True, False])
    selected_statistics = observation_reconstruction_statistics_v60(
        output, target, observation, support_sigma=0.10, sample_mask=selected
    )
    selected_accumulator = ObjectObservationAccumulator()
    selected_accumulator.update(selected_statistics)
    selected_metrics = selected_accumulator.finalize()
    assert bool(torch.isfinite(torch.tensor(tuple(selected_metrics.values()))).all())
    print(
        {
            "status": "passed",
            "tests": 9,
            "semantic_compression_floor": metrics["semantic_compression_floor"],
            "prediction_gain_over_persistence": metrics[
                "prediction_gain_over_persistence"
            ],
        }
    )


if __name__ == "__main__":
    main()
