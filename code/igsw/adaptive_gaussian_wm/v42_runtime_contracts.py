"""Stable correspondence and calibrated track-presence runtime metadata."""

from __future__ import annotations


def gate_contract_fields(architecture: str) -> dict[str, object]:
    if architecture == "object_region_memory_v1":
        return {
            "identity_contract": "causal_sinkhorn_identity_v2",
            "correspondence_contract": (
                "bounded_identity_preserving_sinkhorn_current_only_v3"
            ),
            "transport_contract": "support_normalized_relative_transport_v1",
            "lifecycle_contract": "track_presence_anchored_calibrated_v3",
            "region_contract": "persistent_object_scene_transient_regions_v1",
            "region_owner_count": 18,
            "region_budget": [64, 256],
            "maximum_scene_fraction": 0.25,
        }
    if architecture == "object_memory_v3":
        return {
            "identity_contract": "causal_sinkhorn_identity_v2",
            "correspondence_contract": (
                "bounded_identity_preserving_sinkhorn_current_only_v3"
            ),
            "transport_contract": "support_normalized_relative_transport_v1",
            "lifecycle_contract": "track_presence_anchored_calibrated_v3",
            "correspondence_temperature": 0.5,
            "correspondence_sinkhorn_iterations": 256,
            "correspondence_dustbin_logit": 0.0,
            "correspondence_residual_scale": 0.1,
            "correspondence_logit_clip": 4.0,
            "correspondence_mass_tolerance": 5e-4,
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
    if architecture == "object_region_memory_v1":
        fields.update(
            diagnostics_contract="compact_object_region_jepa_v1",
            dense_diagnostic_contract="offline_frozen_probe_only_v1",
            lifecycle_semantics="root_and_region_track_presence",
            readout_backend="offline_probe_only",
        )
    elif architecture == "object_memory_v3":
        fields.update(
            diagnostics_contract="stable_correspondence_presence_training_v2",
            dense_diagnostic_contract="comparable_dense_dino_persistence_v2",
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
