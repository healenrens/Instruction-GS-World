"""Server-only invariants for causal object correspondence and track presence."""

from __future__ import annotations

from dataclasses import replace

import torch

from .object_slots import ObjectSlotState
from .relative_geometry import ObjectGeometryState, pool_object_geometry


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _permute_observation(
    observation: ObjectSlotState,
    permutation: torch.Tensor,
) -> ObjectSlotState:
    return replace(
        observation,
        slots=observation.slots[:, permutation],
        tracking_slots=observation.tracking_slots[:, permutation],
        assignment=observation.assignment[:, :, permutation],
        activity=observation.activity[:, permutation],
        center=observation.center[:, permutation],
        feature=observation.feature[:, permutation],
        decoded_center=observation.decoded_center[:, permutation],
        decoded_feature=observation.decoded_feature[:, permutation],
    )


def _permute_geometry(
    geometry: ObjectGeometryState,
    permutation: torch.Tensor,
) -> ObjectGeometryState:
    return replace(
        geometry,
        center=geometry.center[:, permutation],
        relative_scale=geometry.relative_scale[:, permutation],
        relative_disparity=geometry.relative_disparity[:, permutation],
        relations=geometry.relations[:, permutation][:, :, permutation],
    )


@torch.no_grad()
def verify_v41_correspondence_contracts(
    model,
    batch: dict,
    amp_context,
    loss_weights,
) -> dict[str, float]:
    _require(model.config.architecture == "object_memory_v3", "model is not v41")
    _require(
        model.config.causal_object_correspondence,
        "causal correspondence is disabled",
    )
    _require(model.config.track_presence_semantics, "track presence is disabled")
    module = model.object_memory.correspondence
    _require(module is not None, "object memory has no correspondence module")
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
        "online_history_association",
        "online_history_association_match",
        "online_history_association_unmatched",
        "online_history_association_discovery",
        "online_history_observation_confidence",
        "online_history_track_presence",
    )
    missing = [name for name in required if output.get(name) is None]
    _require(not missing, f"v41 model outputs are missing: {missing}")
    state = output["history_slot_states"][-1]
    token_state = output["history_token_states"][-1]
    _require(
        state.identity_key.dtype == torch.float32,
        "persistent identity state is not float32",
    )
    delta_time = torch.ones(
        state.slots.shape[0],
        device=state.slots.device,
        dtype=state.slots.dtype,
    )
    with amp_context():
        predicted = model.object_memory.predict(state, delta_time)
        observation = model.object_aggregator(
            token_state,
            predicted.tracking_slots,
            predicted.center,
            predicted.identity_key,
        )
        geometry = pool_object_geometry(
            token_state,
            observation.assignment,
            observation.activity,
        )
        correspondence = module(predicted, observation, geometry)
        corrected = model.object_memory.correct(predicted, observation, token_state)
    correspondence_tensors = (
        correspondence.transport,
        correspondence.unmatched_probability,
        correspondence.discovery_probability,
        correspondence.entropy,
        correspondence.identity_similarity,
        correspondence.support_distance,
    )
    _require(
        all(bool(torch.isfinite(value).all()) for value in correspondence_tensors),
        "correspondence produced non-finite values",
    )
    row_error = _difference(
        correspondence.transport.sum(dim=-1)
        + correspondence.unmatched_probability,
        torch.ones_like(correspondence.match_probability),
    )
    column_error = _difference(
        correspondence.transport.sum(dim=-2)
        + correspondence.discovery_probability,
        torch.ones_like(correspondence.discovery_probability),
    )
    _require(row_error < 5e-4, "correspondence row mass is not conserved")
    _require(column_error < 5e-4, "correspondence column mass is not conserved")

    permutation = torch.arange(
        observation.slots.shape[1] - 1,
        -1,
        -1,
        device=observation.slots.device,
    )
    permuted_observation = _permute_observation(observation, permutation)
    with amp_context():
        permuted = model.object_memory.correct(
            predicted,
            permuted_observation,
            token_state,
        )
    permutation_slot_difference = _difference(corrected.slots, permuted.slots)
    permutation_center_difference = _difference(corrected.center, permuted.center)
    permutation_identity_difference = _difference(
        corrected.identity_key,
        permuted.identity_key,
    )
    _require(
        max(
            permutation_slot_difference,
            permutation_center_difference,
            permutation_identity_difference,
        )
        < 2e-3,
        "observation slot permutation changed persistent track order",
    )

    scale = 1.7
    translation = predicted.center.new_tensor((0.2, -0.15))
    transformed_prediction = replace(
        predicted,
        center=scale * predicted.center + translation,
        relative_scale=scale * predicted.relative_scale,
    )
    transformed_geometry = replace(
        geometry,
        center=scale * geometry.center + translation,
        relative_scale=scale * geometry.relative_scale,
    )
    with amp_context():
        transformed = module(
            transformed_prediction,
            observation,
            transformed_geometry,
        )
    geometry_invariance = _difference(
        correspondence.transport,
        transformed.transport,
    )
    _require(
        geometry_invariance < 5e-4,
        "correspondence is not translation/scale invariant",
    )

    synthetic_observation = replace(
        observation,
        tracking_slots=predicted.identity_key[:, permutation],
        decoded_feature=predicted.decoded_feature[:, permutation],
        activity=torch.ones_like(observation.activity),
    )
    synthetic_geometry = _permute_geometry(
        ObjectGeometryState(
            center=predicted.center,
            relative_scale=predicted.relative_scale,
            relative_disparity=predicted.relative_disparity,
            relations=predicted.relations,
        ),
        permutation,
    )
    with amp_context():
        synthetic = module(predicted, synthetic_observation, synthetic_geometry)
    expected = permutation[None].expand(synthetic.transport.shape[0], -1)
    retrieval_accuracy = float(
        (synthetic.transport.argmax(dim=-1) == expected).float().mean()
    )
    _require(retrieval_accuracy == 1.0, "identity reappearance retrieval failed")

    expected_birth = (
        1.0 - predicted.existence
    ) * corrected.observation_confidence
    birth_formula_difference = _difference(corrected.birth_evidence, expected_birth)
    _require(birth_formula_difference < 1e-6, "birth evidence is not factorized")
    observation_hierarchy_violation = float(
        torch.relu(
            corrected.observation_confidence.float() - corrected.existence.float()
        ).max()
    )
    _require(
        observation_hierarchy_violation < 1e-6,
        "observation confidence exceeds track presence",
    )
    return {
        "persistent_identity_state_dtype_fp32": 1.0,
        "correspondence_row_mass_max_difference": row_error,
        "correspondence_column_mass_max_difference": column_error,
        "correspondence_permutation_slot_max_difference": (
            permutation_slot_difference
        ),
        "correspondence_permutation_center_max_difference": (
            permutation_center_difference
        ),
        "correspondence_permutation_identity_max_difference": (
            permutation_identity_difference
        ),
        "correspondence_geometry_invariance_max_difference": geometry_invariance,
        "correspondence_synthetic_reappearance_accuracy": retrieval_accuracy,
        "track_birth_evidence_formula_max_difference": birth_formula_difference,
        "track_observation_hierarchy_max_violation": (
            observation_hierarchy_violation
        ),
    }
