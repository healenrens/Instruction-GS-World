"""Server-only invariants for persistent identity and relative object transport."""

from __future__ import annotations

from dataclasses import replace

import torch

from .relative_transport import centers_from_support_transport


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@torch.no_grad()
def verify_v40_state_contracts(model, batch: dict, amp_context, loss_weights) -> dict:
    _require(model.config.architecture == "object_memory_v2", "model is not v40")
    _require(model.config.persistent_identity_key, "identity key is disabled")
    _require(model.config.relative_transport_dynamics, "relative transport is disabled")
    _require(model.config.factorized_lifecycle, "factorized lifecycle is disabled")
    _require(
        model.object_aggregator.identity_anchor_projection is not None,
        "identity key does not condition tracking queries",
    )
    history_mask = torch.zeros(
        batch["history_times"].shape[0],
        batch["history_times"].shape[1],
        model.config.object_slots,
        device=batch["history_times"].device,
        dtype=torch.bool,
    )
    model.eval()
    with amp_context():
        output = model(
            batch,
            history_mask=history_mask,
            phase="object_memory_representation_loss",
            loss_weights=loss_weights,
        )
    required = (
        "predicted_future_transport_units",
        "predicted_future_survival",
        "predicted_future_birth",
        "predicted_future_observability",
        "predicted_future_in_frame",
        "online_history_identity_keys",
    )
    missing = [name for name in required if output.get(name) is None]
    _require(not missing, f"v40 model outputs are missing: {missing}")

    current_center = output["online_history_centers"][:, -1, None]
    current_scale = output["online_history_relative_scale"][:, -1, None]
    transport = output["predicted_future_transport_units"]
    reconstructed_center = centers_from_support_transport(
        current_center,
        current_scale,
        transport,
    )
    transport_formula_difference = _difference(
        reconstructed_center,
        output["predicted_future_centers"],
    )
    _require(
        transport_formula_difference < 1e-5,
        "future centers do not follow support-normalized transport",
    )
    scale_factor = 1.7
    scaled_center = centers_from_support_transport(
        scale_factor * current_center,
        scale_factor * current_scale,
        transport,
    )
    transport_scale_equivariance = _difference(
        scaled_center,
        scale_factor * reconstructed_center,
    )
    _require(
        transport_scale_equivariance < 1e-5,
        "relative transport is not scale equivariant",
    )

    current_existence = output["online_history_existence"][:, -1, None]
    survival = output["predicted_future_survival"]
    birth = output["predicted_future_birth"]
    expected_existence = current_existence * survival + (
        1.0 - current_existence
    ) * birth
    existence_composition_difference = _difference(
        expected_existence,
        output["predicted_future_existence"],
    )
    expected_visibility = (
        expected_existence
        * output["predicted_future_in_frame"]
        * output["predicted_future_observability"]
    )
    visibility_composition_difference = _difference(
        expected_visibility,
        output["predicted_future_visibility"],
    )
    hierarchy_violation = float(
        torch.relu(
            output["predicted_future_visibility"].float()
            - output["predicted_future_existence"].float()
        ).max()
    )
    _require(existence_composition_difference < 1e-5, "existence is not factorized")
    _require(visibility_composition_difference < 1e-5, "visibility is not factorized")
    _require(hierarchy_violation < 1e-6, "visibility exceeds existence")

    state = output["history_slot_states"][-1]
    token_state = output["history_token_states"][-1]
    delta_time = torch.ones(
        state.slots.shape[0],
        device=state.slots.device,
        dtype=state.slots.dtype,
    )
    predicted = model.object_memory.predict(state, delta_time)
    memory_scale_factor = 1.7
    scaled_state = replace(
        state,
        center=memory_scale_factor * state.center,
        relative_scale=memory_scale_factor * state.relative_scale,
    )
    scaled_prediction = model.object_memory.predict(scaled_state, delta_time)
    memory_transport_scale_equivariance = _difference(
        scaled_prediction.center,
        memory_scale_factor * predicted.center,
    )
    prediction_identity_difference = _difference(
        predicted.identity_key,
        state.identity_key,
    )
    occluded_observation = replace(
        state,
        activity=torch.zeros_like(state.activity),
    )
    corrected = model.object_memory.correct(
        predicted,
        occluded_observation,
        token_state,
    )
    occluded_identity_difference = _difference(
        corrected.identity_key,
        predicted.identity_key,
    )
    occluded_existence_difference = _difference(
        corrected.existence,
        predicted.existence,
    )
    occluded_update_gate_max = float(corrected.update_gate.float().abs().max())
    identity_norm_error = float(
        (
            output["online_history_identity_keys"].float().norm(dim=-1) - 1.0
        ).abs().max()
    )
    _require(prediction_identity_difference == 0.0, "prediction changed identity")
    _require(
        memory_transport_scale_equivariance < 1e-5,
        "object-memory transport is not scale equivariant",
    )
    _require(occluded_identity_difference == 0.0, "occlusion changed identity")
    _require(occluded_existence_difference == 0.0, "occlusion deleted object")
    _require(occluded_update_gate_max == 0.0, "occlusion opened correction gate")
    _require(identity_norm_error < 1e-4, "persistent identity keys are not normalized")
    return {
        "identity_prediction_carry_max_difference": prediction_identity_difference,
        "identity_occlusion_carry_max_difference": occluded_identity_difference,
        "identity_occlusion_existence_max_difference": occluded_existence_difference,
        "identity_occlusion_update_gate_max": occluded_update_gate_max,
        "identity_key_norm_max_error": identity_norm_error,
        "memory_transport_scale_equivariance_max_difference": (
            memory_transport_scale_equivariance
        ),
        "transport_formula_max_difference": transport_formula_difference,
        "transport_scale_equivariance_max_difference": transport_scale_equivariance,
        "lifecycle_existence_composition_max_difference": (
            existence_composition_difference
        ),
        "lifecycle_visibility_composition_max_difference": (
            visibility_composition_difference
        ),
        "lifecycle_visibility_hierarchy_max_violation": hierarchy_violation,
    }
