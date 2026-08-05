"""Server-only regressions for v42 roots and v43 persistent regions."""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .object_correspondence import _augmented_optimal_transport
from .relative_geometry import ObjectGeometryState
from .scale import signed_gap_scale


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sinkhorn_stress(model) -> dict[str, float]:
    config = model.config
    slots = config.object_slots
    count = slots * slots
    device = next(model.object_memory.parameters()).device
    scores = torch.linspace(
        -config.correspondence_logit_clip,
        config.correspondence_logit_clip,
        count,
        device=device,
    ).reshape(1, slots, slots)
    checkerboard = torch.where(
        torch.arange(count, device=device).reshape(slots, slots) % 2 == 0,
        scores.new_tensor(config.correspondence_logit_clip),
        scores.new_tensor(-config.correspondence_logit_clip),
    )[None]
    scores = torch.cat((scores, checkerboard))
    dustbin = scores.new_zeros(len(scores), 1, 1)
    transport, unmatched, discovery = _augmented_optimal_transport(
        scores, dustbin, config.correspondence_sinkhorn_iterations
    )
    row_error = float(
        (transport.sum(dim=-1) + unmatched - 1.0).abs().max()
    )
    column_error = float(
        (transport.sum(dim=-2) + discovery - 1.0).abs().max()
    )
    tolerance = config.correspondence_mass_tolerance
    _require(row_error < tolerance, "root Sinkhorn row mass regressed")
    _require(column_error < tolerance, "root Sinkhorn column mass regressed")
    return {
        "root_stress_row_mass_max_error": row_error,
        "root_stress_column_mass_max_error": column_error,
    }


def _root_persistence(model, output, amp_context) -> dict[str, float]:
    state = output["online"]["last_root"]
    tokens = output["online"]["token_states"][-1]
    delta = state.slots.new_ones(state.slots.shape[0])
    scaled = replace(
        state,
        center=1.7 * state.center,
        relative_scale=1.7 * state.relative_scale,
    )
    occluded = replace(state, activity=torch.zeros_like(state.activity))
    with amp_context():
        predicted = model.object_memory.predict(state, delta)
        scaled_prediction = model.object_memory.predict(scaled, delta)
        corrected = model.object_memory.correct(predicted, occluded, tokens)
        observation = model.object_aggregator(
            tokens,
            predicted.tracking_slots,
            predicted.center,
            predicted.identity_key,
        )
    identity_prediction = _difference(predicted.identity_key, state.identity_key)
    identity_occlusion = _difference(corrected.identity_key, predicted.identity_key)
    existence_occlusion = _difference(corrected.existence, predicted.existence)
    update_gate = float(corrected.update_gate.float().abs().max())
    transport_scale = _difference(
        scaled_prediction.center, 1.7 * predicted.center
    )
    norm_error = float(
        (state.identity_key.float().norm(dim=-1) - 1.0).abs().max()
    )
    _require(identity_prediction == 0.0, "root prediction changed identity")
    _require(identity_occlusion == 0.0, "root occlusion changed identity")
    _require(existence_occlusion == 0.0, "root occlusion deleted a track")
    _require(update_gate == 0.0, "root occlusion opened the correction gate")
    _require(transport_scale < 1e-5, "root relative transport regressed")
    _require(norm_error < 1e-4, "root identity normalization regressed")
    object_count = predicted.identity_key.shape[1]
    identity_dim = predicted.identity_key.shape[2]
    feature_dim = predicted.decoded_feature.shape[2]
    identity_basis = torch.eye(
        object_count,
        identity_dim,
        device=predicted.identity_key.device,
        dtype=torch.float32,
    )[None].expand(predicted.identity_key.shape[0], -1, -1)
    feature_basis = torch.eye(
        object_count,
        feature_dim,
        device=predicted.decoded_feature.device,
        dtype=predicted.decoded_feature.dtype,
    )[None].expand(predicted.identity_key.shape[0], -1, -1)
    permutation = torch.arange(
        object_count - 1,
        -1,
        -1,
        device=predicted.identity_key.device,
    )
    synthetic_prediction = replace(
        predicted,
        identity_key=identity_basis,
        decoded_feature=feature_basis,
        existence=torch.zeros_like(predicted.existence),
    )
    synthetic_observation = replace(
        observation,
        tracking_slots=identity_basis[:, permutation].to(
            observation.tracking_slots.dtype
        ),
        decoded_feature=feature_basis[:, permutation],
        activity=torch.ones_like(observation.activity),
    )
    base_geometry = ObjectGeometryState(
        center=synthetic_prediction.center,
        relative_scale=synthetic_prediction.relative_scale,
        relative_disparity=synthetic_prediction.relative_disparity,
        relations=synthetic_prediction.relations,
    )
    synthetic_geometry = ObjectGeometryState(
        center=base_geometry.center[:, permutation],
        relative_scale=base_geometry.relative_scale[:, permutation],
        relative_disparity=base_geometry.relative_disparity[:, permutation],
        relations=base_geometry.relations[:, permutation][:, :, permutation],
    )
    with amp_context():
        retrieval = model.object_memory.correspondence(
            synthetic_prediction,
            synthetic_observation,
            synthetic_geometry,
        )
    expected = permutation[None].expand(retrieval.transport.shape[0], -1)
    retrieval_accuracy = float(
        (retrieval.transport.argmax(dim=-1) == expected).float().mean()
    )
    _require(retrieval_accuracy == 1.0, "root identity reappearance regressed")
    return {
        "root_identity_prediction_max_difference": identity_prediction,
        "root_identity_occlusion_max_difference": identity_occlusion,
        "root_existence_occlusion_max_difference": existence_occlusion,
        "root_occlusion_update_gate_max": update_gate,
        "root_transport_scale_equivariance_max_difference": transport_scale,
        "root_identity_norm_max_error": norm_error,
        "root_identity_reappearance_accuracy": retrieval_accuracy,
    }


def _region_persistence(model, output, amp_context) -> dict[str, float]:
    state = output["online"]["last_region"]
    root_state = output["online"]["last_root"]
    delta = state.feature.new_full((state.feature.shape[0],), 0.1)
    with amp_context():
        predicted_root = model.object_memory.predict(root_state, delta)
        predicted = model.region_memory.predict(state, delta, predicted_root)
        occluded = replace(
            state,
            activation=predicted.presence,
            presence=predicted.presence,
            visibility=torch.zeros_like(state.visibility),
            observed=torch.zeros_like(state.observed),
        )
        corrected = model.region_memory.correct(predicted, occluded)
    object_owned = (
        predicted.owner[..., :-2].sum(dim=-1) > 0.5
    ) & (predicted.presence > 0.1)
    _require(bool(object_owned.any()), "region memory produced no object-owned region")
    identity_error = F.cosine_similarity(
        corrected.identity_key.float(), predicted.identity_key.float(), dim=-1
    )
    identity_error = (
        (1.0 - identity_error) * object_owned.float()
    ).sum() / object_owned.float().sum().clamp_min(1.0)
    update_gate = (
        corrected.update_gate.float() * object_owned.float()
    ).abs().max()
    _require(float(identity_error) < 1e-5, "region occlusion changed identity")
    _require(float(update_gate) < 1e-6, "region occlusion opened correction")
    region_count = state.feature.shape[1]
    identity_dim = state.identity_key.shape[-1]
    test_count = min(16, region_count, identity_dim)
    identity = torch.zeros_like(state.identity_key)
    identity[:, :test_count, :test_count] = torch.eye(
        test_count, device=identity.device, dtype=identity.dtype
    )
    relative_center = torch.zeros_like(state.relative_center)
    relative_center[:, :test_count, 0] = torch.linspace(
        -0.75, 0.75, test_count, device=identity.device
    )
    owner = torch.zeros_like(state.owner)
    owner[..., 0] = 1.0
    presence = torch.zeros_like(state.presence)
    presence[:, :test_count] = 1.0
    synthetic = replace(
        state,
        identity_key=identity,
        relative_center=relative_center,
        owner=owner,
        presence=presence,
        visibility=presence,
    )
    permutation = torch.arange(region_count, device=identity.device)
    permutation[:test_count] = torch.arange(
        test_count - 1, -1, -1, device=identity.device
    )
    observation = replace(
        synthetic,
        feature=synthetic.feature[:, permutation],
        center=synthetic.center[:, permutation],
        covariance=synthetic.covariance[:, permutation],
        owner=synthetic.owner[:, permutation],
        relative_center=synthetic.relative_center[:, permutation],
        presence=synthetic.presence[:, permutation],
        visibility=synthetic.visibility[:, permutation],
        identity_key=synthetic.identity_key[:, permutation],
    )
    with amp_context():
        retrieval = model.region_memory.correspondence(synthetic, observation)
    expected = torch.arange(
        test_count - 1, -1, -1, device=identity.device
    )[None].expand(identity.shape[0], -1)
    retrieval_accuracy = float(
        (retrieval.matrix[:, :test_count].argmax(dim=-1) == expected)
        .float()
        .mean()
    )
    _require(retrieval_accuracy == 1.0, "region reappearance retrieval failed")
    return {
        "region_occlusion_identity_error": float(identity_error),
        "region_occlusion_update_gate_max": float(update_gate),
        "region_identity_reappearance_accuracy": retrieval_accuracy,
    }


@torch.no_grad()
def verify_v43_persistence_contracts(model, output, amp_context) -> dict[str, float]:
    _require(
        model.config.architecture in (
            "object_region_memory_v1",
            "object_region_dual_encoder_v1",
        ),
        "v43 persistence verifier received another architecture",
    )
    model.eval()
    result = _sinkhorn_stress(model)
    result.update(_root_persistence(model, output, amp_context))
    result.update(_region_persistence(model, output, amp_context))
    return result


@torch.no_grad()
def verify_v43_action_factorization(
    model,
    batch: dict,
    output: dict,
    amp_context,
) -> dict[str, float]:
    root = output["root_prediction"]
    region = output["region_prediction"]
    _require(root is not None and region is not None, "v43 factorization has no prediction")
    online = output["online"]
    roots = online["roots"]
    future_scale = signed_gap_scale(
        batch["future_times"], model.config.gap_reference
    )
    history_scale = signed_gap_scale(
        batch["history_times"], model.config.gap_reference
    )
    zero_actions = torch.zeros_like(output["short_action"])[:, None].expand(
        -1, future_scale.shape[1], -1, -1
    )
    conditioned_actions = torch.stack(
        (output["short_action"], output["composed_action"]), dim=1
    )
    model.eval()
    with amp_context():
        conditioned_root = model.dynamics(
            roots["slots"],
            roots["activity"],
            history_scale,
            future_scale,
            conditioned_actions,
            history_mask=torch.zeros_like(roots["activity"], dtype=torch.bool),
            history_centers=roots["center"],
            history_relative_scale=roots["relative_scale"],
            history_relative_disparity=roots["relative_disparity"],
            history_relations=roots["relations"],
            history_existence=roots["existence"],
        )
        conditioned_region = model.region_dynamics(
            online["regions"],
            conditioned_root.future_slots,
            conditioned_root.future_centers,
            future_scale,
            conditioned_actions,
            base_root_future_slots=conditioned_root.base_future_slots,
        )
        isolated_base_region = model.region_dynamics(
            online["regions"],
            conditioned_root.base_future_slots,
            conditioned_root.future_centers,
            future_scale,
            zero_actions,
            base_root_future_slots=conditioned_root.base_future_slots,
        )
        zero_root = model.dynamics(
            roots["slots"],
            roots["activity"],
            history_scale,
            future_scale,
            zero_actions,
            history_mask=torch.zeros_like(roots["activity"], dtype=torch.bool),
            history_centers=roots["center"],
            history_relative_scale=roots["relative_scale"],
            history_relative_disparity=roots["relative_disparity"],
            history_relations=roots["relations"],
            history_existence=roots["existence"],
        )
        zero_region = model.region_dynamics(
            online["regions"],
            zero_root.future_slots,
            zero_root.future_centers,
            future_scale,
            zero_actions,
            base_root_future_slots=zero_root.base_future_slots,
        )
        perturbed_history = dict(online["regions"])
        inactive = online["regions"]["presence"][:, -1] <= 0.5
        perturbed_feature = online["regions"]["feature"].clone()
        perturbed_feature[:, -1] = perturbed_feature[:, -1] + (
            inactive[..., None].to(perturbed_feature.dtype) * 100.0
        )
        perturbed_history["feature"] = perturbed_feature
        perturbed_region = model.region_dynamics(
            perturbed_history,
            zero_root.future_slots,
            zero_root.future_centers,
            future_scale,
            zero_actions,
            base_root_future_slots=zero_root.base_future_slots,
        )
    root_base_difference = _difference(
        zero_root.future_slots, conditioned_root.base_future_slots
    )
    region_base_difference = _difference(
        isolated_base_region.base_future_feature,
        conditioned_region.base_future_feature,
    )
    root_zero_residual = float(zero_root.action_slot_residual.float().abs().max())
    region_zero_residual = _difference(
        zero_region.future_feature, zero_region.base_future_feature
    )
    region_zero_presence = _difference(
        zero_region.future_presence, zero_region.base_future_presence
    )
    region_zero_visibility = _difference(
        zero_region.future_visibility, zero_region.base_future_visibility
    )
    active = (~inactive)[:, None, :, None].to(zero_region.future_feature.dtype)
    inactive_dynamics_leakage = float(
        (
            (
                zero_region.future_feature.float()
                - perturbed_region.future_feature.float()
            ).abs()
            * active.float()
        ).max()
    )
    predicted_active = conditioned_region.future_presence > 0.5
    predicted_active_count = predicted_active.sum(dim=-1)
    predicted_scene_fraction = float(
        (
            conditioned_region.future_owner[..., -2]
            * predicted_active.to(conditioned_region.future_owner.dtype)
        ).sum()
        / predicted_active.float().sum().clamp_min(1.0)
    )
    _require(root_base_difference < 1e-6, "root base depends on latent effect")
    _require(
        region_base_difference < 1e-6,
        "region base reads conditioned roots: "
        f"max_difference={region_base_difference:.9g}",
    )
    _require(root_zero_residual == 0.0, "zero root effect has a residual")
    _require(
        region_zero_residual == 0.0,
        "zero region effect has a residual: "
        f"max_difference={region_zero_residual:.9g}",
    )
    _require(region_zero_presence < 1e-6, "zero effect changed region presence")
    _require(region_zero_visibility < 1e-6, "zero effect changed region visibility")
    _require(
        inactive_dynamics_leakage < 1e-6,
        "inactive region content leaked through Dynamics",
    )
    _require(
        predicted_scene_fraction <= model.config.region_scene_fraction + 1e-6,
        "predicted region scene quota regressed",
    )
    _require(
        bool(
            (
                (predicted_active_count >= model.config.min_active_tokens)
                & (predicted_active_count <= model.config.max_micro_tokens)
            ).all()
        ),
        "predicted active region count left the configured range",
    )
    return {
        "root_action_free_base_max_difference": root_base_difference,
        "region_action_free_base_max_difference": region_base_difference,
        "root_zero_effect_residual_max": root_zero_residual,
        "region_zero_effect_residual_max": region_zero_residual,
        "region_zero_effect_presence_max_difference": region_zero_presence,
        "region_zero_effect_visibility_max_difference": region_zero_visibility,
        "inactive_region_dynamics_leakage_max_difference": (
            inactive_dynamics_leakage
        ),
        "predicted_region_scene_fraction": predicted_scene_fraction,
        "predicted_active_region_count": float(
            predicted_active_count.float().mean()
        ),
    }
