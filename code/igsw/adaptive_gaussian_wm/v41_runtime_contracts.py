"""Versioned identity, transport, and lifecycle metadata for Object Memory."""

from __future__ import annotations


def gate_contract_fields(architecture: str) -> dict[str, object]:
    if architecture == "object_memory_v3":
        return {
            "identity_contract": "causal_sinkhorn_identity_v2",
            "correspondence_contract": "causal_augmented_sinkhorn_current_only_v1",
            "transport_contract": "support_normalized_relative_transport_v1",
            "lifecycle_contract": "track_presence_discovery_observation_v2",
            "correspondence_temperature": 0.5,
            "correspondence_sinkhorn_iterations": 64,
            "correspondence_dustbin_logit": 0.0,
            "correspondence_residual_scale": 0.1,
        }
    if architecture == "object_memory_v2":
        return {
            "identity_contract": "persistent_identity_key_v1",
            "transport_contract": "support_normalized_relative_transport_v1",
            "lifecycle_contract": "survival_birth_observability_v1",
        }
    return {}


def runtime_contract_fields(architecture: str) -> dict[str, object]:
    fields = gate_contract_fields(architecture)
    if architecture == "object_memory_v3":
        fields.update(
            diagnostics_contract="object_correspondence_presence_training_v1",
            lifecycle_semantics="latent_track_presence_not_physical_existence",
        )
    elif architecture == "object_memory_v2":
        fields["diagnostics_contract"] = "object_lifecycle_transport_training_v1"
    else:
        fields.update(
            diagnostics_contract="dynamic_dual_horizon_training_v1",
            identity_contract="disabled",
            transport_contract="absolute_center_residual",
            lifecycle_contract="joint_existence_visibility",
        )
    return fields
