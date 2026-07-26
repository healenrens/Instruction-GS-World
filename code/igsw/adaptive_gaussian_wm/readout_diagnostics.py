"""Round-trip and change-region diagnostics for Gaussian feature readout."""
from __future__ import annotations

import math

import torch

from .change_objectives import dense_feature_loss
from .diagnostic_statistics import ratio_moments
from .readout_runtime import residual_future_features


def _current_targets(
    batch: dict,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = batch["history_features"][:, -1:]
    valid = batch["history_valid"][:, -1:]
    return current.expand_as(reference), valid.expand(reference.shape[:-1])


def _scene_mean_baseline(
    batch: dict,
    reference: torch.Tensor,
) -> torch.Tensor:
    current = batch["history_features"][:, -1].detach().float()
    valid = batch["history_valid"][:, -1].detach().float()
    denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (current * valid[..., None]).sum(dim=1) / denominator
    return mean[:, None, None].expand_as(reference)


def _oracle_future_readout(model, batch: dict, output: dict):
    current_tokens = output["history_token_states"][-1]
    current_slots = output["history_slot_states"][-1]
    oracle_state = model.gaussian_readout(
        output["target_future_slots"].detach(),
        current_tokens,
        current_slots.assignment.detach(),
        predicted_features=output["target_future_object_features"].detach(),
        current_object_features=current_slots.feature.detach(),
        predicted_centers=output["target_future_centers"].detach(),
        current_object_centers=current_slots.center.detach(),
        predicted_relative_scale=(
            output["target_future_relative_scale"].detach()
        ),
        current_relative_scale=current_slots.relative_scale.detach(),
        predicted_relative_disparity=(
            output["target_future_relative_disparity"].detach()
        ),
        current_relative_disparity=current_slots.relative_disparity.detach(),
    )
    direct, coverage = model.gaussian_readout.splat_features(
        oracle_state,
        batch["future_coordinates"],
    )
    rendered = residual_future_features(
        direct,
        output["residual_reference_features"].detach(),
        batch,
    )
    return rendered, coverage


def _change_region_masks(
    batch: dict,
    coverage: torch.Tensor,
    fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    future = batch["future_features"].detach().float()
    current = batch["history_features"][:, -1:].detach().float().expand_as(future)
    current_valid = batch["history_valid"][:, -1:].expand_as(
        batch["future_valid"]
    )
    valid = batch["future_valid"] & current_valid & (coverage > 1e-4)
    score = (future - current).square().mean(dim=-1).sqrt()
    flat_valid = valid.flatten(0, 1)
    flat_score = score.flatten(0, 1)
    flat_dynamic = torch.zeros_like(flat_valid)
    for row in range(flat_valid.shape[0]):
        indices = torch.nonzero(flat_valid[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("readout diagnostic has no valid future patches")
        count = max(1, math.ceil(float(indices.numel()) * fraction))
        selected = torch.topk(
            flat_score[row, indices], count, sorted=False
        ).indices
        flat_dynamic[row, indices[selected]] = True
    dynamic = flat_dynamic.unflatten(0, valid.shape[:2])
    return dynamic, valid & ~dynamic, valid


def _region_metrics(
    name: str,
    model_prediction: torch.Tensor,
    oracle_prediction: torch.Tensor,
    persistence: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    coverage: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model_loss = dense_feature_loss(
        model_prediction, target, mask, coverage
    )
    oracle_loss = dense_feature_loss(
        oracle_prediction, target, mask, coverage
    )
    persistence_loss = dense_feature_loss(
        persistence, target, mask, coverage
    )
    result = {
        f"readout_{name}_model_feature": model_loss,
        f"readout_{name}_oracle_feature": oracle_loss,
        f"readout_{name}_persistence_feature": persistence_loss,
        f"readout_{name}_model_gain_over_persistence": (
            persistence_loss - model_loss
        ),
        f"readout_{name}_oracle_gain_over_persistence": (
            persistence_loss - oracle_loss
        ),
    }
    result.update(
        ratio_moments(
            f"readout_{name}_model_relative_gain_over_persistence",
            persistence_loss - model_loss,
            persistence_loss,
        )
    )
    result.update(
        ratio_moments(
            f"readout_{name}_oracle_relative_gain_over_persistence",
            persistence_loss - oracle_loss,
            persistence_loss,
        )
    )
    return result


@torch.no_grad()
def gaussian_readout_diagnostics(
    model,
    batch: dict,
    output: dict,
) -> dict[str, torch.Tensor]:
    """Separate readout capacity, Dynamics error, and static-scene dilution."""
    reference = output["residual_reference_features"].detach().float()
    reference_coverage = output["residual_reference_coverage"].detach().float()
    current_target, current_valid = _current_targets(batch, reference)
    scene_mean = _scene_mean_baseline(batch, reference)
    token_reconstruction = output["history_token_states"][
        -1
    ].reconstructed_features.detach().float()[:, None].expand_as(reference)
    current_loss = dense_feature_loss(
        reference, current_target, current_valid, reference_coverage
    )
    token_loss = dense_feature_loss(
        token_reconstruction, current_target, current_valid, reference_coverage
    )
    scene_loss = dense_feature_loss(
        scene_mean, current_target, current_valid, reference_coverage
    )
    result = {
        "readout_current_feature": current_loss,
        "readout_current_token_feature": token_loss,
        "readout_current_scene_mean_feature": scene_loss,
        "readout_current_gain_over_scene_mean": scene_loss - current_loss,
        "readout_current_gap_to_token_reconstruction": current_loss - token_loss,
    }
    result.update(
        ratio_moments(
            "readout_current_relative_gain_over_scene_mean",
            scene_loss - current_loss,
            scene_loss,
        )
    )

    oracle, oracle_coverage = _oracle_future_readout(model, batch, output)
    model_prediction = output["rendered_future_features"].detach().float()
    target = batch["future_features"].detach().float()
    persistence = batch["history_features"][:, -1:].detach().float().expand_as(
        target
    )
    common_coverage = torch.minimum(
        output["render_coverage"].detach().float(),
        oracle_coverage.detach().float(),
    )
    comparable = _region_metrics(
        "comparable",
        model_prediction,
        oracle,
        persistence,
        target,
        batch["future_valid"],
        common_coverage,
    )
    result.update(comparable)
    result["readout_comparable_model_gap_to_oracle"] = (
        comparable["readout_comparable_model_feature"]
        - comparable["readout_comparable_oracle_feature"]
    )

    dynamic, static, valid = _change_region_masks(batch, common_coverage)
    result.update(
        _region_metrics(
            "dynamic",
            model_prediction,
            oracle,
            persistence,
            target,
            dynamic,
            common_coverage,
        )
    )
    result.update(
        _region_metrics(
            "static",
            model_prediction,
            oracle,
            persistence,
            target,
            static,
            common_coverage,
        )
    )
    result.update(
        ratio_moments(
            "readout_change_region_fraction",
            dynamic.float().sum(),
            valid.float().sum(),
        )
    )
    for name, coverage in (
        ("current", reference_coverage),
        ("model", output["render_coverage"].detach().float()),
        ("oracle", oracle_coverage.detach().float()),
        ("common", common_coverage),
    ):
        valid_mask = current_valid if name == "current" else batch["future_valid"]
        result.update(
            ratio_moments(
                f"readout_{name}_coverage_fraction",
                ((coverage > 1e-4) & valid_mask).float().sum(),
                valid_mask.float().sum(),
            )
        )
    return result
