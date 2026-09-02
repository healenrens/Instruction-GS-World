"""Compact object-shared motion models for the v66 G0 audit."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectMotionFieldFitV66:
    translation: torch.Tensor
    affine: torch.Tensor
    mode_centers: torch.Tensor
    mode_scale: torch.Tensor
    mode_coefficients: torch.Tensor
    reference_center: torch.Tensor
    translation_valid: torch.Tensor
    affine_valid: torch.Tensor
    field_valid: torch.Tensor
    translation_prefix_error: torch.Tensor
    affine_prefix_error: torch.Tensor
    field_prefix_error: torch.Tensor


def _weighted_center(coordinates, weight):
    center = (coordinates.float() * weight[..., None]).sum(dim=-2)
    return center / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _relative_mode_centers(evidence, membership, start, stop, config):
    coordinates = evidence.coordinates[:, start:stop].float()
    visible = evidence.visibility[:, start:stop].float()
    weight = visible * membership[:, None].float()
    weight = weight * evidence.reliability[:, None]
    frame_center = _weighted_center(coordinates, weight)
    relative = coordinates - frame_center[:, :, None]
    track_weight = weight.sum(dim=1)
    pooled = (relative * weight[..., None]).sum(dim=1)
    pooled = pooled / track_weight.clamp_min(1e-6)[..., None]
    available = track_weight > 0.0
    normalized_weight = track_weight / track_weight.amax(dim=-1, keepdim=True).clamp_min(
        1e-6
    )
    selected = torch.zeros_like(available)
    minimum_distance = torch.full_like(track_weight, float("inf"))
    centers = []
    batch_index = torch.arange(len(pooled), device=pooled.device)
    selection_score = normalized_weight.masked_fill(~available, -1.0)
    for _ in range(config.local_motion_modes):
        index = selection_score.argmax(dim=-1)
        center = pooled[batch_index, index]
        centers.append(center)
        selected[batch_index, index] = True
        distance = (pooled - center[:, None]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        selection_score = minimum_distance * (0.25 + 0.75 * normalized_weight)
        selection_score = selection_score.masked_fill(~available | selected, -1.0)
    centers = torch.stack(centers, dim=1)
    object_scale = (pooled.square().sum(dim=-1) * track_weight).sum(dim=-1)
    object_scale = (object_scale / track_weight.sum(dim=-1).clamp_min(1e-6)).sqrt()
    mode_scale = (object_scale * config.local_mode_width_scale).clamp_min(
        config.minimum_local_mode_scale
    )
    valid = available.sum(dim=-1) >= config.local_motion_modes
    return centers, mode_scale, valid


def _radial_design(relative, centers, scale):
    distance = (relative[..., None, :] - centers[:, None, None]).square().sum(dim=-1)
    radial = torch.exp(-distance / (2.0 * scale[:, None, None, None].square()))
    return radial / radial.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _robust_linear_fit(design, target, base_weight, ridge, config):
    batch, _, width = design.shape
    identity = torch.eye(width, device=design.device, dtype=torch.float32)[None]
    weight = base_weight.float()
    coefficients = torch.zeros(batch, width, 2, device=design.device)
    for _ in range(config.transition_irls_steps):
        gram = torch.einsum("bni,bn,bnj->bij", design, weight, design)
        rhs = torch.einsum("bni,bn,bnd->bid", design, weight, target)
        coefficients = torch.linalg.solve(gram + ridge * identity, rhs)
        prediction = torch.einsum("bni,bid->bnd", design, coefficients)
        residual = (prediction - target).norm(dim=-1).clamp_min(1e-6)
        robust = (config.transition_huber_delta / residual).clamp(max=1.0)
        weight = base_weight * robust
    prediction = torch.einsum("bni,bid->bnd", design, coefficients)
    residual = (prediction - target).norm(dim=-1)
    rms = (residual.square() * base_weight).sum(dim=-1)
    rms = (rms / base_weight.sum(dim=-1).clamp_min(1e-6)).sqrt()
    return coefficients, rms


def _horizon_data(evidence, membership, centers, mode_scale, start, stop, horizon):
    source = evidence.coordinates[:, start : stop - horizon].float()
    target = evidence.coordinates[:, start + horizon : stop].float()
    visible = evidence.visibility[:, start : stop - horizon]
    visible = visible & evidence.visibility[:, start + horizon : stop]
    weight = visible.float() * membership[:, None].float()
    weight = weight * evidence.reliability[:, None]
    source_center = _weighted_center(source, weight)
    relative = source - source_center[:, :, None]
    radial = _radial_design(relative, centers, mode_scale)
    return source, target, weight, radial


def fit_object_motion_models_v66(evidence, membership, start, stop, config):
    centers, mode_scale, center_valid = _relative_mode_centers(
        evidence, membership, start, stop, config
    )
    translation, affine, modes = [], [], []
    translation_error, affine_error, field_error = [], [], []
    horizon_valid = []
    for horizon in config.transition_horizons:
        source, target, weight, radial = _horizon_data(
            evidence, membership, centers, mode_scale, start, stop, horizon
        )
        flat_weight = weight.flatten(1)
        flat_source = source.flatten(1, 2)
        flat_target = target.flatten(1, 2)
        translation_design = torch.ones(
            len(source), flat_source.shape[1], 1, device=source.device
        )
        translation_fit, translation_rms = _robust_linear_fit(
            translation_design,
            flat_target - flat_source,
            flat_weight,
            config.transition_ridge,
            config,
        )
        affine_design = torch.cat(
            (flat_source, torch.ones_like(flat_source[..., :1])), dim=-1
        )
        affine_fit, affine_rms = _robust_linear_fit(
            affine_design,
            flat_target,
            flat_weight,
            config.transition_ridge,
            config,
        )
        affine_prediction = torch.einsum(
            "bni,bid->bnd", affine_design, affine_fit
        )
        radial_design = radial.flatten(1, 2)
        mode_fit, _ = _robust_linear_fit(
            radial_design,
            flat_target - affine_prediction,
            flat_weight,
            config.motion_field_ridge,
            config,
        )
        field_prediction = affine_prediction + torch.einsum(
            "bnk,bkd->bnd", radial_design, mode_fit
        )
        residual = (field_prediction - flat_target).norm(dim=-1)
        field_rms = (residual.square() * flat_weight).sum(dim=-1)
        field_rms = (field_rms / flat_weight.sum(dim=-1).clamp_min(1e-6)).sqrt()
        visible_tracks = ((weight > 0.0).any(dim=1)).sum(dim=-1)
        horizon_valid.append(visible_tracks >= config.minimum_component_tracks)
        translation.append(translation_fit[:, 0])
        affine.append(affine_fit)
        modes.append(mode_fit)
        translation_error.append(translation_rms)
        affine_error.append(affine_rms)
        field_error.append(field_rms)
    reference_weight = evidence.visibility[:, stop - 1].float() * membership.float()
    reference_weight = reference_weight * evidence.reliability
    reference_center = _weighted_center(
        evidence.coordinates[:, stop - 1].float(), reference_weight
    )
    reference_valid = (reference_weight > 0.0).sum(dim=-1) >= config.local_motion_modes
    horizon_valid = torch.stack(horizon_valid, dim=1).all(dim=1)
    tensors = (
        translation,
        affine,
        centers,
        mode_scale,
        modes,
        reference_center,
    )
    tensors = tuple(
        torch.stack(value, dim=1) if isinstance(value, list) else value
        for value in tensors
    )
    translation, affine, centers, mode_scale, modes, reference_center = tensors
    return ObjectMotionFieldFitV66(
        translation=translation.detach(),
        affine=affine.detach(),
        mode_centers=centers.detach(),
        mode_scale=mode_scale.detach(),
        mode_coefficients=modes.detach(),
        reference_center=reference_center.detach(),
        translation_valid=horizon_valid.detach(),
        affine_valid=horizon_valid.detach(),
        field_valid=(horizon_valid & center_valid & reference_valid).detach(),
        translation_prefix_error=torch.stack(translation_error, dim=1).mean(1).detach(),
        affine_prefix_error=torch.stack(affine_error, dim=1).mean(1).detach(),
        field_prefix_error=torch.stack(field_error, dim=1).mean(1).detach(),
    )


def predict_translation_v66(source, fit):
    return source[:, None].float() + fit.translation[:, :, None]


def predict_affine_v66(source, fit):
    design = torch.cat((source.float(), torch.ones_like(source[..., :1])), dim=-1)
    return torch.einsum("bpi,bhid->bhpd", design, fit.affine)


def predict_motion_field_v66(source, fit):
    affine = predict_affine_v66(source, fit)
    relative = source.float() - fit.reference_center[:, None]
    distance = (
        relative[:, :, None] - fit.mode_centers[:, None]
    ).square().sum(dim=-1)
    radial = torch.exp(
        -distance / (2.0 * fit.mode_scale[:, None, None].square())
    )
    radial = radial / radial.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    residual = torch.einsum("bpk,bhkd->bhpd", radial, fit.mode_coefficients)
    return affine + residual


def roll_motion_field_fit_v66(fit):
    index = torch.roll(torch.arange(len(fit.translation), device=fit.translation.device), 1)
    return ObjectMotionFieldFitV66(
        **{
            field: getattr(fit, field).index_select(0, index)
            for field in fit.__dataclass_fields__
        }
    )
