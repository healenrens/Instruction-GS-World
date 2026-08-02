"""Assemble model outputs without widening the causal forward path."""

from __future__ import annotations

import torch

from .dynamics_runtime import factorized_result_fields


def assemble_model_output(
    model,
    history: dict,
    target_history: dict,
    target_future: dict,
    future_output,
    history_output,
    predicted_future_centers: torch.Tensor,
    predicted_history_centers: torch.Tensor,
    zero_action_future_centers: torch.Tensor,
    effects,
    history_mask: torch.Tensor,
    readout_fields: dict,
    rgb_fields: dict,
    condition: torch.Tensor | None,
) -> dict:
    action_rgb = effects.action_rgb
    result = {
        "predicted_future_slots": future_output.future_slots,
        "predicted_future_centers": predicted_future_centers,
        "predicted_future_object_features": model.object_aggregator.decode_feature(
            future_output.future_slots
        ),
        "zero_action_future_slots": history_output.future_slots,
        "zero_action_future_centers": zero_action_future_centers,
        "predicted_history_slots": history_output.history_slots,
        "predicted_history_centers": predicted_history_centers,
        "target_future_slots": target_future["slots"],
        "target_future_activity": target_future["activity"],
        "target_future_visibility": target_future.get(
            "visibility", target_future["activity"]
        ),
        "target_future_existence": target_future.get(
            "existence", target_future["activity"]
        ),
        "target_future_track_presence": target_future.get(
            "existence", target_future["activity"]
        ),
        "target_future_observation_confidence": target_future.get(
            "observation_confidence", target_future["activity"]
        ),
        "target_future_in_frame": target_future.get(
            "in_frame", target_future["activity"]
        ),
        "target_future_relative_scale": target_future.get("relative_scale"),
        "target_future_relative_disparity": target_future.get(
            "relative_disparity"
        ),
        "target_future_relations": target_future.get("relations"),
        "target_future_centers": target_future["center"],
        "target_future_object_features": target_future["feature"],
        "current_object_rgb": action_rgb[0],
        "target_future_object_rgb": action_rgb[1],
        "target_history_slots": target_history["slots"],
        "target_history_activity": target_history["activity"],
        "target_history_visibility": target_history.get(
            "visibility", target_history["activity"]
        ),
        "target_history_existence": target_history.get(
            "existence", target_history["activity"]
        ),
        "target_history_track_presence": target_history.get(
            "existence", target_history["activity"]
        ),
        "target_history_observation_confidence": target_history.get(
            "observation_confidence", target_history["activity"]
        ),
        "target_history_in_frame": target_history.get(
            "in_frame", target_history["activity"]
        ),
        "target_history_relative_scale": target_history.get("relative_scale"),
        "target_history_relative_disparity": target_history.get(
            "relative_disparity"
        ),
        "target_history_relations": target_history.get("relations"),
        "target_history_centers": target_history["center"],
        "target_history_object_features": target_history["feature"],
        "online_history_slots": history["slots"],
        "online_history_centers": history["center"],
        "online_history_visibility": history.get("visibility", history["activity"]),
        "online_history_existence": history.get("existence", history["activity"]),
        "online_history_track_presence": history.get(
            "existence", history["activity"]
        ),
        "online_history_observation_confidence": history.get(
            "observation_confidence", history["activity"]
        ),
        "online_history_in_frame": history.get("in_frame", history["activity"]),
        "online_history_relative_scale": history.get("relative_scale"),
        "online_history_relative_disparity": history.get("relative_disparity"),
        "online_history_relations": history.get("relations"),
        "online_history_identity_keys": history.get("identity_key"),
        "online_history_identity_similarity": history.get("identity_similarity"),
        "online_history_association": history.get("association_matrix"),
        "online_history_association_match": history.get("association_match"),
        "online_history_association_unmatched": history.get(
            "association_unmatched"
        ),
        "online_history_association_discovery": history.get(
            "association_discovery"
        ),
        "online_history_association_entropy": history.get("association_entropy"),
        "online_history_association_support_distance": history.get(
            "association_support_distance"
        ),
        "online_history_birth_evidence": history.get("birth_evidence"),
        "online_history_object_features": torch.stack(
            [state.decoded_feature for state in history["slot_states"]], dim=1
        ),
        "target_history_identity_keys": target_history.get("identity_key"),
        "target_future_identity_keys": target_future.get("identity_key"),
        "target_history_association": target_history.get("association_matrix"),
        "target_future_association": target_future.get("association_matrix"),
        "posterior_actions": effects.posterior,
        "hierarchical_effects": effects.selected,
        "dynamics_actions": effects.dynamics,
        "prior_context": effects.prior_context,
        "history_mask": history_mask,
        "language_condition": condition,
        "history_token_states": history["token_states"],
        "history_slot_states": history["slot_states"],
        "dual_horizon": model.config.dual_horizon_dynamics,
        "lifecycle_focal_gamma": model.config.lifecycle_focal_gamma,
        "transport_max_support_units": model.config.transport_max_support_units,
        **readout_fields,
        **rgb_fields,
    }
    if model.config.dual_horizon_dynamics:
        posterior_composed = model.effect_composer(
            effects.posterior[:, 0], effects.posterior[:, 1]
        )
        result.update(
            short_effect=effects.selected[:, 0],
            tail_effect=effects.selected[:, 1],
            composed_goal_effect=effects.dynamics[:, 1],
            posterior_short_effect=effects.posterior[:, 0],
            posterior_tail_effect=effects.posterior[:, 1],
            posterior_composed_goal_effect=posterior_composed,
        )
    result.update(
        factorized_result_fields(
            future_output,
            history_output,
            target_future,
            target_history,
        )
    )
    if model.config.track_presence_semantics:
        result.update(
            predicted_future_track_presence=result["predicted_future_existence"],
            predicted_future_track_presence_logits=result[
                "predicted_future_existence_logits"
            ],
            zero_action_future_track_presence=result[
                "zero_action_future_existence"
            ],
            lifecycle_semantics="latent_track_presence_not_physical_existence",
        )
    return result
