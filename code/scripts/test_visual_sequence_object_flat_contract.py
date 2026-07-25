"""Synthetic contracts for matched flat baselines and their evidence gate."""
from __future__ import annotations

from copy import deepcopy
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from igsw.adaptive_gaussian_wm.matched_flat_world_model import (  # noqa: E402
    MatchedFlatLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.matched_flat_objective import (  # noqa: E402
    FLAT_TRAIN_METRICS,
    flat_training_objective,
)
from igsw.adaptive_gaussian_wm.matched_flat_evaluation import (  # noqa: E402
    RGB_REGIONS,
    rgb_region_errors_by_query,
)
from igsw.adaptive_gaussian_wm.matched_flat_rgb import (  # noqa: E402
    batch_rgb_grids,
    render_rgb_grid,
)
from igsw.adaptive_gaussian_wm.flat_baseline_checkpointing import (  # noqa: E402
    initialize_flat_dynamics,
)
from igsw.adaptive_gaussian_wm.temporal_region_evaluation import (  # noqa: E402
    TemporalRegionConfig,
)
from verify_visual_sequence_object_flat_gate import verify  # noqa: E402
from verify_devup_preflight import REQUIRED_SCALE_CHECKS  # noqa: E402
from task_group_test_fixtures import object_flat_task_evidence  # noqa: E402
def test_model_contract() -> dict:
    torch.manual_seed(7)
    batch, history_count, future_count = 2, 4, 4
    grid_count, feature_dim = 6, 8
    model = MatchedFlatLatentWorldModel(
        feature_dim=feature_dim,
        rgb_channels=3,
        model_dim=16,
        state_tokens=3,
        action_tokens=3,
        action_dim=14,
        action_residual_dim=8,
        dynamics_layers=2,
        heads=4,
        checkpoint_blocks=True,
    )
    source_state = {}
    for name, value in model.state_dict().items():
        if name.startswith("dynamics.") and name.split(".")[1].isdigit():
            suffix = name.removeprefix("dynamics.")
            source_state[f"dynamics.blocks.{suffix}"] = value.detach().clone() + 0.001
    initialization = initialize_flat_dynamics(
        model,
        {
            "checkpoint_version": 27,
            "global_step": 12000,
            "model": source_state,
        },
        "/tmp/object.pt",
        "a" * 64,
    )
    if initialization["loaded_tensors"] != len(source_state):
        raise AssertionError("shared Dynamics initialization is incomplete")
    history = torch.randn(batch, history_count, grid_count, feature_dim)
    future = history[:, -1:] + 0.2 * torch.randn(
        batch,
        future_count,
        grid_count,
        feature_dim,
    )
    rgb_height, rgb_width, content_width = 8, 10, 9
    history_rgb_frames = torch.randint(
        0,
        256,
        (batch, history_count, 3, rgb_height, rgb_width),
        dtype=torch.uint8,
    )
    future_rgb_frames = torch.randint(
        0,
        256,
        (batch, future_count, 3, rgb_height, rgb_width),
        dtype=torch.uint8,
    )
    history_rgb_frames[..., content_width:] = 0
    future_rgb_frames[..., content_width:] = 0
    history_rgb_valid = torch.zeros(
        batch,
        history_count,
        rgb_height,
        rgb_width,
        dtype=torch.bool,
    )
    future_rgb_valid = torch.zeros(
        batch,
        future_count,
        rgb_height,
        rgb_width,
        dtype=torch.bool,
    )
    history_rgb_valid[..., :content_width] = True
    future_rgb_valid[..., :content_width] = True
    batch_data = {
        "history_features": history,
        "future_features": future,
        "future_valid": torch.ones(batch, future_count, grid_count, dtype=torch.bool),
        "history_rgb": history_rgb_frames,
        "history_rgb_valid": history_rgb_valid,
        "future_rgb": future_rgb_frames,
        "future_rgb_valid": future_rgb_valid,
        "feature_grid_hw": torch.tensor([[2, 3]]).expand(batch, -1),
    }
    history_rgb, future_rgb = batch_rgb_grids(batch_data)
    coordinates = torch.randn(batch, future_count, grid_count, 2)
    history_coordinates = coordinates[:, :1].expand(
        -1,
        history_count,
        -1,
        -1,
    )
    history_scale = torch.linspace(-0.8, 0.0, history_count)[None].expand(
        batch,
        -1,
    )
    future_scale = torch.linspace(0.1, 0.8, future_count)[None].expand(batch, -1)
    output = model(
        history,
        history_coordinates,
        history_scale,
        history_rgb,
        future,
        coordinates,
        future_scale,
        future_rgb,
    )
    expected_prediction = (batch, future_count, grid_count, feature_dim)
    expected_rgb = (batch, future_count, grid_count, 3)
    expected_action = (batch, future_count, 3, 14)
    if output["history_feature_prediction"].shape != expected_prediction:
        raise AssertionError("history flat prediction shape differs")
    if output["posterior_feature_prediction"].shape != expected_prediction:
        raise AssertionError("posterior flat prediction shape differs")
    if output["history_rgb_grid_prediction"].shape != expected_rgb:
        raise AssertionError("history flat RGB prediction shape differs")
    if output["posterior_rgb_grid_prediction"].shape != expected_rgb:
        raise AssertionError("posterior flat RGB prediction shape differs")
    if output["posterior_actions"].shape != expected_action:
        raise AssertionError("flat posterior action shape differs")
    swapped = model(
        history,
        history_coordinates,
        history_scale,
        history_rgb,
        future.flip(0),
        coordinates,
        future_scale,
        future_rgb.flip(0),
    )
    history_feature_difference = (
        output["history_feature_prediction"]
        - swapped["history_feature_prediction"]
    ).abs().max()
    history_rgb_difference = (
        output["history_rgb_grid_prediction"]
        - swapped["history_rgb_grid_prediction"]
    ).abs().max()
    history_difference = torch.maximum(
        history_feature_difference,
        history_rgb_difference,
    )
    action_difference = (
        output["posterior_actions"] - swapped["posterior_actions"]
    ).square().mean().sqrt()
    if float(history_difference.detach()) > 1e-7:
        raise AssertionError("history-only baseline depends on future")
    if float(action_difference.detach()) <= 1e-6:
        raise AssertionError("flat posterior does not respond to future")
    fixed_feature, fixed_rgb = model.dynamics_readout(
        output["history_state"],
        history[:, -1],
        history_rgb[:, -1],
        coordinates,
        future_scale,
        output["posterior_actions"],
    )
    fixed_feature_difference = (
        fixed_feature - output["posterior_feature_prediction"]
    ).abs().max()
    fixed_rgb_difference = (
        fixed_rgb - output["posterior_rgb_grid_prediction"]
    ).abs().max()
    fixed_difference = torch.maximum(fixed_feature_difference, fixed_rgb_difference)
    if float(fixed_difference.detach()) > 1e-7:
        raise AssertionError("flat future bypasses the action bottleneck")
    loss, metrics = flat_training_objective(
        output,
        batch_data,
        SimpleNamespace(
            change_loss_weight=1.0,
            history_loss_weight=1.0,
            rgb_loss_weight=0.5,
            rgb_ssim_weight=0.2,
            rgb_change_loss_weight=1.0,
            rgb_change_threshold=0.04,
        ),
    )
    if tuple(metrics) != FLAT_TRAIN_METRICS or not bool(torch.isfinite(loss)):
        raise AssertionError("flat DINO+RGB objective contract differs")
    rendered = render_rgb_grid(
        output["posterior_rgb_grid_prediction"],
        future_rgb_valid,
        2,
        3,
    )
    if rendered.shape != future_rgb_frames.shape:
        raise AssertionError("flat RGB rendering shape differs")
    if bool((rendered[..., content_width:] != 0).any()):
        raise AssertionError("flat RGB rendering writes into padding")
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
    ]
    if missing:
        raise AssertionError(f"flat baseline gradient contract failed: {missing}")
    return {
        "prediction_shape": list(expected_prediction),
        "rgb_prediction_shape": list(expected_rgb),
        "action_shape": list(expected_action),
        "history_future_swap_max_difference": float(history_difference.detach()),
        "posterior_future_swap_rms_difference": float(action_difference.detach()),
        "fixed_action_max_difference": float(fixed_difference.detach()),
        "training_metric_count": len(metrics),
        "initialized_dynamics_tensors": initialization["loaded_tensors"],
        "initialized_parameter_fraction": initialization[
            "flat_parameter_fraction"
        ],
    }

def interval(relative: float, positive: bool = True) -> dict:
    return {
        "clusters": 120,
        "absolute_improvement": relative,
        "relative_improvement": relative,
        "cluster_win_fraction": 0.8,
        "cluster_standard_error": 0.001,
        "critical_value": 2.0,
        "ci95_lower": 0.01 if positive else -0.01,
        "ci95_upper": 0.03,
        "positive_ci95_lower": positive,
    }

def test_region_contract() -> dict:
    batch_size, history_frames, future_frames = 2, 4, 4
    height, width = 24, 28
    history = torch.zeros(
        (batch_size, history_frames, 3, height, width),
        dtype=torch.uint8,
    )
    future = torch.zeros(
        (batch_size, future_frames, 3, height, width),
        dtype=torch.uint8,
    )
    future[:, :, :, 7:15, 9:18] = 255
    valid_history = torch.ones(
        (batch_size, history_frames, height, width),
        dtype=torch.bool,
    )
    valid_future = torch.ones(
        (batch_size, future_frames, height, width),
        dtype=torch.bool,
    )
    target = future.float() / 255.0
    predictions = {
        "object_posterior": target,
        "flat_posterior": target * 0.75,
        "flat_history": torch.zeros_like(target),
        "copy": torch.zeros_like(target),
    }
    regions = rgb_region_errors_by_query(
        predictions,
        {
            "history_rgb": history,
            "future_rgb": future,
            "history_rgb_valid": valid_history,
            "future_rgb_valid": valid_future,
        },
        TemporalRegionConfig(),
    )
    for region in RGB_REGIONS:
        if not bool(regions[region]["nonempty"].all()):
            raise AssertionError(f"synthetic {region} region is empty")
        for name in predictions:
            if not bool(torch.isfinite(regions[region][name]).all()):
                raise AssertionError(f"non-finite {region} metric for {name}")
    if not bool(
        (
            regions["change"]["object_posterior"]
            < regions["change"]["flat_posterior"]
        ).all()
    ):
        raise AssertionError("observed-change metric does not rank predictions")
    return {
        region: float(regions[region]["coverage"].mean())
        for region in RGB_REGIONS
    }

def fake_report(split: str, samples: int, clusters: int) -> dict:
    feature = {
        "flat_history_over_copy": interval(0.03),
        "flat_posterior_over_history": interval(0.08),
        "flat_posterior_over_copy": interval(0.1),
        "object_over_flat_posterior": interval(0.04),
        "object_over_flat_history": interval(0.12),
        "object_over_copy": interval(0.15),
    }
    change = deepcopy(feature)
    return {
        "status": "ok",
        "object_checkpoint": "/tmp/object.pt",
        "object_checkpoint_sha256": "a" * 64,
        "object_checkpoint_global_step": 12000,
        "flat_checkpoint": "/tmp/flat.pt",
        "flat_checkpoint_sha256": "c" * 64,
        "flat_checkpoint_global_step": 12000,
        "data": "/tmp/data",
        "data_sha256": "b" * 64,
        "task_source_index_sha256": "c" * 64,
        "split": split,
        "requested_max_items": samples,
        "available_samples": samples,
        "history_frames": 4,
        "future_frames": 4,
        "anchors": [3, 5, 8],
        "action_contract": {
            "type": "continuous",
            "tokens": 16,
            "dimensions": 14,
            "state_tokens": 16,
            "layout": "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8",
            "object_source": "future_conditioned_object_posterior_oracle",
            "flat_source": "future_conditioned_unstructured_latent_posterior_oracle",
            "dynamics_future_access": "latent_action_only",
            "flat_posterior_future_modalities": ["dino", "rgb"],
        },
        "parameter_count": {
            "object_model": 1000,
            "total_parameters": 900,
            "flat_to_object_ratio": 0.9,
        },
        "training_contract": {
            "optimizer_contract": "posterior_core_matched_flat_optimization_v1",
            "object_checkpoint_phase_steps": 12000,
            "flat_additional_steps": 12000,
            "object_effective_global_batch": 256,
            "flat_effective_global_batch": 256,
            "shared_dynamics_initialization": "object_dynamics_blocks_only",
            "optimization_budget_bias": "flat_receives_additional_updates_after_object_source",
            "modality_matching": "dino_rgb",
            "flat_rgb_supervision": True,
            "flat_posterior_observes_future_rgb": True,
            "history_encoder_input": "dino_only",
        },
        "evaluation": {
            "samples": samples,
            "clusters": clusters,
            "task_group_evidence": object_flat_task_evidence(
                split, samples, clusters),
            "action_statistics": {
                "flat_posterior": {"rms": 0.2, "sample_std_mean": 0.1}
            },
            "causal_probe": {
                "posterior_responds_to_future": True,
                "history_has_no_future_input": True,
                "dynamics_has_no_direct_future_input": True,
            },
            "comparison": {
                "feature_mse": feature,
                "change_weighted_feature_mse": change,
                "rgb_distance": deepcopy(feature),
                "change_weighted_rgb_distance": deepcopy(change),
            },
            "object_vs_flat_noninferiority_5pct": {
                "feature_mse": {
                    "relative_margin": 0.05,
                    "noninferior_ci95": True,
                },
                "rgb_distance": {
                    "relative_margin": 0.05,
                    "noninferior_ci95": True,
                },
            },
            "rgb_regions": {
                "mask_source": "ground_truth_current_and_future_rgb_for_evaluation_only",
                "config": {},
                "regions": {
                    region: {
                        "frames": samples * 4,
                        "clusters": clusters,
                        "mean_coverage": 0.2 if region == "change" else 0.7,
                        "mean": {},
                        "comparison": deepcopy(change),
                    }
                    for region in ("change", "static")
                },
                "object_vs_flat_static_noninferiority_5pct": {
                    "relative_margin": 0.05,
                    "noninferior_ci95": True,
                },
            },
            "by_future_query": {
                str(index): {
                    "comparison": {
                        "change_weighted_feature_mse": change,
                        "change_weighted_rgb_distance": deepcopy(change),
                    }
                }
                for index in range(4)
            },
        },
    }

def test_gate_contract() -> dict:
    heldseed = fake_report("heldseed", 1024, 120)
    heldtask = fake_report("heldtask", 450, 110)
    passed = verify(heldseed, heldtask, 12000, 1024, 450, 100, 0.03)
    if passed["status"] != "pass":
        raise AssertionError(f"valid object-flat evidence failed: {passed}")
    if {check["name"] for check in passed["checks"]} != REQUIRED_SCALE_CHECKS:
        raise AssertionError("object-flat and DevUp scale checks diverged")
    weak = deepcopy(heldtask)
    weak_gain = weak["evaluation"]["comparison"][
        "change_weighted_feature_mse"
    ]["object_over_flat_posterior"]
    weak_gain.update(relative_improvement=0.01, positive_ci95_lower=False)
    failed = verify(heldseed, weak, 12000, 1024, 450, 100, 0.03)
    expected = "heldtask.object_beats_flat_on_change_feature"
    if failed["status"] != "fail" or expected not in failed["failed_checks"]:
        raise AssertionError("weak object representation did not fail the gate")
    weak_region = deepcopy(heldtask)
    region_gain = weak_region["evaluation"]["rgb_regions"]["regions"][
        "change"
    ]["comparison"]["object_over_flat_posterior"]
    region_gain.update(relative_improvement=0.01, positive_ci95_lower=False)
    failed_region = verify(heldseed, weak_region, 12000, 1024, 450, 100, 0.03)
    region_check = "heldtask.object_beats_flat_on_observed_change_rgb"
    if (
        failed_region["status"] != "fail"
        or region_check not in failed_region["failed_checks"]
    ):
        raise AssertionError("weak observed-change result did not fail the gate")
    weak_static = deepcopy(heldtask)
    weak_static["evaluation"]["rgb_regions"][
        "object_vs_flat_static_noninferiority_5pct"
    ]["noninferior_ci95"] = False
    failed_static = verify(heldseed, weak_static, 12000, 1024, 450, 100, 0.03)
    static_check = "heldtask.object_static_rgb_noninferior"
    if (
        failed_static["status"] != "fail"
        or static_check not in failed_static["failed_checks"]
    ):
        raise AssertionError("static degradation did not fail the gate")
    invalid_identity = deepcopy(heldtask)
    invalid_identity["flat_checkpoint_sha256"] = "d" * 64
    invalid_identity["evaluation"]["samples"] -= 1
    invalid_identity["evaluation"]["by_future_query"].pop("0")
    invalid_identity["evaluation"]["task_group_evidence"]["task_names"] = []
    invalid_identity["evaluation"]["task_group_evidence"]["comparison"]["change_weighted_feature_mse"]["object_over_flat_posterior"]["positive_task_fraction"] = 0.0
    failed_identity = verify(heldseed, invalid_identity, 12000, 1024, 450, 100, 0.03)
    required_failures = {"cross_split_identity", "heldtask.sample_coverage", "heldtask.future_query_contract", "heldtask.task_group_contract", "heldtask.object_beats_flat_by_task_on_change_feature"}
    if not required_failures.issubset(failed_identity["failed_checks"]):
        raise AssertionError("identity and coverage corruption did not fail")
    return {
        "valid_status": passed["status"],
        "weak_status": failed["status"],
        "weak_failed_check": expected,
        "weak_region_failed_check": region_check,
        "weak_static_failed_check": static_check,
        "identity_coverage_failures": sorted(required_failures),
    }
def main() -> None:
    print(
        {
            "model": test_model_contract(),
            "regions": test_region_contract(),
            "gate": test_gate_contract(),
        }
    )


if __name__ == "__main__":
    main()
