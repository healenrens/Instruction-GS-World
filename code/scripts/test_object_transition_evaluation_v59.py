#!/usr/bin/env python3
"""CPU regression tests for v59 unseen-window evaluation."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.group_balanced_sampler import (  # noqa: E402
    DistributedGroupBalancedSampler,
)
from igsw.adaptive_gaussian_wm.latent_object_transition_v59 import (  # noqa: E402
    ObjectTransitionPrediction,
)
from igsw.adaptive_gaussian_wm.object_transition_evaluation_v59 import (  # noqa: E402
    TransitionEvaluationAccumulator,
    bootstrap_macro_gains_v59,
    macro_metrics_v59,
)
from igsw.adaptive_gaussian_wm.query_transition_target_v59 import (  # noqa: E402
    QueryTransitionTarget,
)
from igsw.adaptive_gaussian_wm.v59_config import (  # noqa: E402
    ObjectTransitionConfig,
)
from igsw.adaptive_gaussian_wm.v59_evaluation_sampling import (  # noqa: E402
    training_base_indices_v59,
)


class TinyDataset:
    balance_sampling = True
    sampling_group_targets_may_undersample = True
    sampling_group_spans = ((0, 20), (20, 40))
    sampling_group_targets = (20, 20)
    dynamic_history_lengths = (1, 2, 3, 4)

    def __len__(self):
        return 40


def test_sampler_slice():
    sampler = DistributedGroupBalancedSampler(
        TinyDataset(),
        TinyDataset.sampling_group_spans,
        TinyDataset.sampling_group_targets,
        num_replicas=2,
        rank=0,
        seed=5,
        samples_per_rank=20,
        require_full_group_coverage=False,
    )
    sampler.set_epoch(7)
    full = list(sampler)
    assert sampler.sample_indices(3, 11).tolist() == full[3:11]


def test_training_exclusion():
    checkpoint = {
        "global_step": 4,
        "world_size": 2,
        "args": {"batch": 2, "grad_accum": 1, "seed": 5},
    }
    excluded = training_base_indices_v59(TinyDataset(), checkpoint)
    assert len(excluded) > 0
    assert len(excluded) <= 16


def transition_fixture():
    batch, horizons, dim = 2, 3, 768
    source_semantic = torch.zeros(batch, dim)
    source_semantic[:, 0] = 1.0
    future_semantic = torch.zeros(batch, horizons, dim)
    future_semantic[..., 1] = 1.0
    source_geometry = torch.zeros(batch, 5)
    future_geometry = torch.full((batch, horizons, 5), 0.2)
    source_visibility = torch.ones(batch)
    future_visibility = torch.ones(batch, horizons)
    target = QueryTransitionTarget(
        source_semantic=source_semantic,
        source_geometry=source_geometry,
        source_visibility=source_visibility,
        future_semantic=future_semantic,
        future_geometry=future_geometry,
        future_visibility=future_visibility,
        delta_seconds=torch.ones(batch, horizons),
        pair_valid=torch.ones(batch, horizons, dtype=torch.bool),
        motion_active=torch.ones(batch, horizons, dtype=torch.bool),
    )
    correct = ObjectTransitionPrediction(
        semantic=future_semantic,
        geometry=future_geometry,
        visibility_logits=torch.full((batch, horizons), 20.0),
    )
    zero = ObjectTransitionPrediction(
        semantic=source_semantic[:, None].expand_as(future_semantic),
        geometry=source_geometry[:, None].expand_as(future_geometry),
        visibility_logits=torch.full((batch, horizons), 20.0),
    )
    shuffled = ObjectTransitionPrediction(
        semantic=-future_semantic,
        geometry=-future_geometry,
        visibility_logits=torch.full((batch, horizons), -20.0),
    )
    output = {
        "correct": correct,
        "zero": zero,
        "shuffled": shuffled,
        "effect": torch.linspace(-1.0, 1.0, batch * horizons * 4 * 32).reshape(
            batch, horizons, 4, 32
        ),
    }
    return output, target


def test_exact_accumulator():
    output, target = transition_fixture()
    config = ObjectTransitionConfig()
    accumulator = TransitionEvaluationAccumulator(config.dynamic_horizons)
    accumulator.update(output, target, config)
    metrics = accumulator.finalize()
    assert metrics["sample_count"] == 2.0
    assert metrics["motion_active_count"] == 6.0
    assert metrics["gain_over_zero"] > 0.99
    assert metrics["gain_over_shuffled"] > 0.99
    assert metrics["gain_over_persistence"] > 0.99
    assert metrics["h4_gain_over_persistence"] > 0.99
    assert metrics["effect_std"] > 0.0
    assert metrics["effect_prediction_delta"] > 0.0


def test_macro_and_bootstrap():
    output, target = transition_fixture()
    config = ObjectTransitionConfig()
    records = []
    for _ in range(4):
        accumulator = TransitionEvaluationAccumulator(config.dynamic_horizons)
        accumulator.update(output, target, config)
        records.append(accumulator.finalize())
    macro = macro_metrics_v59(records)
    bootstrap = bootstrap_macro_gains_v59(records, 100, 9)
    assert macro["gain_over_persistence"] > 0.99
    assert bootstrap["persistence"]["ci95_low"] > 0.99


def main():
    test_sampler_slice()
    test_training_exclusion()
    test_exact_accumulator()
    test_macro_and_bootstrap()
    print({"status": "passed", "tests": 4})


if __name__ == "__main__":
    main()
