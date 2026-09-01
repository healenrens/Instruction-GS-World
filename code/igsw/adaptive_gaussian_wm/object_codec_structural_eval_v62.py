"""Counterfactual structural evaluation of a trained v62 E0 object codec."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .object_transition_metrics_v62 import (
    compact_object_baseline_errors_v62,
    decoded_field_errors_v62,
    lifecycle_error_v62,
    weighted_mean_v62,
)
from .teacher_object_codec_v62 import frame_observation_v62


POINT_FIELDS = ("coordinates", "dino", "siglip", "support", "visibility", "membership")


def _slice_frame(frame, indices):
    return {
        name: value[:, indices] if name in POINT_FIELDS else value
        for name, value in frame.items()
    }


def _query_swap(frame):
    membership = torch.roll(frame["membership"], shifts=1, dims=0)
    return {
        **frame,
        "membership": membership,
        "support": membership * frame["visibility"].float(),
        "lifecycle": torch.roll(frame["lifecycle"], shifts=1, dims=0),
    }


def _all_scene(frame):
    zeros = torch.zeros_like(frame["membership"])
    return {**frame, "membership": zeros, "support": zeros}


def _merge_all(frame):
    membership = torch.ones_like(frame["membership"])
    membership = membership * frame["object_valid"][:, None].float()
    return {
        **frame,
        "membership": membership,
        "support": membership * frame["visibility"].float(),
    }


def _field_total(state, decoded, frame):
    errors = decoded_field_errors_v62(decoded, frame)
    lifecycle = lifecycle_error_v62(state, frame)
    total = (
        errors["support_bce"]
        + errors["dino_cosine_error"]
        + errors["siglip_cosine_error"]
        + 0.25 * errors["visibility_bce"]
        + 0.25 * lifecycle
    )
    return total, errors, lifecycle


def _baseline_total(frame, covariance_floor):
    errors = compact_object_baseline_errors_v62(frame, covariance_floor)
    total = (
        errors["support_bce"]
        + errors["dino_cosine_error"]
        + errors["siglip_cosine_error"]
        + 0.25 * errors["visibility_bce"]
        + 0.25 * errors["lifecycle_cross_entropy"]
    )
    return total, errors


def _average(values):
    return sum(values) / len(values)


def _average_fields(values):
    return {
        name: _average([entry[name] for entry in values]) for name in values[0]
    }


def _positive_fraction(frame):
    valid = frame["object_valid"][:, None].float()
    return (frame["support"].float() * valid).sum() / valid.expand_as(
        frame["support"]
    ).sum().clamp_min(1.0)


def _carrier_ablation(state, frame):
    support = frame["support"].float()
    mass = (state.assignment.float() * support[:, None]).sum(dim=-1)
    selected = mass.argmax(dim=-1)
    mask = F.one_hot(selected, num_classes=state.carriers.shape[1]).bool()
    return replace(
        state,
        carriers=state.carriers.masked_fill(mask[..., None], 0.0),
        presence=torch.where(mask, state.presence.new_full((), 1e-4), state.presence),
        visibility=torch.where(
            mask, state.visibility.new_full((), 1e-4), state.visibility
        ),
    )


def _carrier_swap(state):
    return replace(state, carriers=torch.roll(state.carriers, shifts=1, dims=0))


def _decoded_change(reference, changed, frame):
    support_change = (
        torch.sigmoid(reference.support_logits.float())
        - torch.sigmoid(changed.support_logits.float())
    ).abs()
    semantic_change = 1.0 - F.cosine_similarity(
        reference.dino.float(), changed.dino.float(), dim=-1
    )
    semantic_change = (
        semantic_change
        + 1.0
        - F.cosine_similarity(reference.siglip.float(), changed.siglip.float(), dim=-1)
    )
    value = support_change + semantic_change
    valid = frame["object_valid"][:, None].float()
    inside = frame["support"].float() * valid
    outside = (1.0 - frame["support"].float()) * valid
    inside_change = weighted_mean_v62(value, inside)
    outside_change = weighted_mean_v62(value, outside)
    return inside_change, outside_change


def _identity_consistency(states, valid):
    reference = states[0].identity.float()
    errors = [
        1.0 - F.cosine_similarity(reference, state.identity.float(), dim=-1)
        for state in states[1:]
    ]
    return sum(weighted_mean_v62(error, valid) for error in errors) / len(errors)


@torch.no_grad()
def evaluate_object_codec_structure_v62(model, observation):
    output = model(observation)
    frames = [
        frame_observation_v62(observation, index)
        for index in range(observation.coordinates.shape[1])
    ]
    normal_totals = []
    baseline_totals = []
    baseline_fields = []
    full_context_holdout_totals = []
    full_context_holdout_fields = []
    holdout_totals = []
    holdout_fields = []
    holdout_baseline_totals = []
    holdout_baseline_fields = []
    holdout_positive_fractions = []
    query_swap_totals = []
    all_scene_totals = []
    merge_all_totals = []
    carrier_delete_totals = []
    carrier_swap_totals = []
    inside_changes = []
    outside_changes = []
    split_states = []
    point_count = frames[0]["coordinates"].shape[1]
    observed_indices = torch.arange(
        0, point_count, 2, device=frames[0]["coordinates"].device
    )
    held_indices = torch.arange(
        1, point_count, 2, device=frames[0]["coordinates"].device
    )
    for index, frame in enumerate(frames):
        normal_state = output["states"][index]
        normal_decoded = output["decoded"][index]
        normal_total, _, _ = _field_total(normal_state, normal_decoded, frame)
        normal_totals.append(normal_total)
        baseline_total, baseline = _baseline_total(
            frame, model.config.covariance_floor
        )
        baseline_totals.append(baseline_total)
        baseline_fields.append(baseline)

        observed_frame = _slice_frame(frame, observed_indices)
        held_frame = _slice_frame(frame, held_indices)
        full_context_decoded = model.decoder(
            normal_state, held_frame["coordinates"]
        )
        full_context_total, full_context_errors, full_context_lifecycle = _field_total(
            normal_state, full_context_decoded, held_frame
        )
        full_context_holdout_totals.append(full_context_total)
        full_context_holdout_fields.append(
            {
                **full_context_errors,
                "lifecycle_cross_entropy": full_context_lifecycle,
            }
        )
        held_state = model.codec(observed_frame)
        held_decoded = model.decoder(held_state, held_frame["coordinates"])
        held_total, held_errors, held_lifecycle = _field_total(
            held_state, held_decoded, held_frame
        )
        holdout_totals.append(held_total)
        holdout_fields.append(
            {**held_errors, "lifecycle_cross_entropy": held_lifecycle}
        )
        held_baseline_total, held_baseline = _baseline_total(
            held_frame, model.config.covariance_floor
        )
        holdout_baseline_totals.append(held_baseline_total)
        holdout_baseline_fields.append(held_baseline)
        holdout_positive_fractions.append(_positive_fraction(held_frame))

        query_state = model.codec(_query_swap(frame))
        query_decoded = model.decoder(query_state, frame["coordinates"])
        query_swap_totals.append(_field_total(query_state, query_decoded, frame)[0])

        scene_state = model.codec(_all_scene(frame))
        scene_decoded = model.decoder(scene_state, frame["coordinates"])
        all_scene_totals.append(_field_total(scene_state, scene_decoded, frame)[0])

        merge_state = model.codec(_merge_all(frame))
        merge_decoded = model.decoder(merge_state, frame["coordinates"])
        merge_all_totals.append(_field_total(merge_state, merge_decoded, frame)[0])

        deleted_state = _carrier_ablation(normal_state, frame)
        deleted_decoded = model.decoder(deleted_state, frame["coordinates"])
        carrier_delete_totals.append(
            _field_total(deleted_state, deleted_decoded, frame)[0]
        )
        inside, outside = _decoded_change(normal_decoded, deleted_decoded, frame)
        inside_changes.append(inside)
        outside_changes.append(outside)

        swapped_state = _carrier_swap(normal_state)
        swapped_decoded = model.decoder(swapped_state, frame["coordinates"])
        carrier_swap_totals.append(
            _field_total(swapped_state, swapped_decoded, frame)[0]
        )
        split_states.append(
            normal_state if index == 0 else model.codec(_query_swap(frame))
        )

    normal_total = _average(normal_totals)
    baseline_total = _average(baseline_totals)
    baseline = _average_fields(baseline_fields)
    full_context_total = _average(full_context_holdout_totals)
    full_context_fields = _average_fields(full_context_holdout_fields)
    holdout_total = _average(holdout_totals)
    holdout = _average_fields(holdout_fields)
    holdout_baseline_total = _average(holdout_baseline_totals)
    holdout_baseline = _average_fields(holdout_baseline_fields)
    inside_change = _average(inside_changes)
    outside_change = _average(outside_changes)
    valid = observation.object_valid.float()
    normal_identity = _identity_consistency(output["states"], valid)
    split_identity = _identity_consistency(split_states, valid)
    result = {
        "normal_absolute_error": normal_total,
        "compact_baseline_absolute_error": baseline_total,
        "normal_gap_recovery": 1.0 - normal_total / baseline_total.clamp_min(1e-6),
        "continuous_full_context_absolute_error": full_context_total,
        "continuous_full_context_ratio_to_normal": full_context_total
        / normal_total.clamp_min(1e-6),
        "continuous_holdout_absolute_error": holdout_total,
        "continuous_holdout_ratio_to_normal": holdout_total
        / normal_total.clamp_min(1e-6),
        "continuous_holdout_ratio": holdout_total / normal_total.clamp_min(1e-6),
        "continuous_holdout_ratio_to_full_context": holdout_total
        / full_context_total.clamp_min(1e-6),
        "continuous_holdout_error_increase_over_full_context": holdout_total
        - full_context_total,
        "continuous_holdout_compact_baseline_absolute_error": holdout_baseline_total,
        "continuous_holdout_gap_recovery": 1.0
        - holdout_total / holdout_baseline_total.clamp_min(1e-6),
        "continuous_holdout_support_gap_recovery": 1.0
        - holdout["support_bce"]
        / holdout_baseline["support_bce"].clamp_min(1e-6),
        "continuous_holdout_semantic_gap_recovery": 1.0
        - (holdout["dino_cosine_error"] + holdout["siglip_cosine_error"])
        / (
            holdout_baseline["dino_cosine_error"]
            + holdout_baseline["siglip_cosine_error"]
        ).clamp_min(1e-6),
        "continuous_holdout_lifecycle_gap_recovery": 1.0
        - holdout["lifecycle_cross_entropy"]
        / holdout_baseline["lifecycle_cross_entropy"].clamp_min(1e-6),
        "continuous_holdout_support_ratio_to_full_context": holdout["support_bce"]
        / full_context_fields["support_bce"].clamp_min(1e-6),
        "continuous_holdout_semantic_ratio_to_full_context": (
            holdout["dino_cosine_error"] + holdout["siglip_cosine_error"]
        )
        / (
            full_context_fields["dino_cosine_error"]
            + full_context_fields["siglip_cosine_error"]
        ).clamp_min(1e-6),
        "continuous_holdout_positive_point_fraction": _average(
            holdout_positive_fractions
        ),
        "query_swap_absolute_error": _average(query_swap_totals),
        "query_swap_error_increase": _average(query_swap_totals) - normal_total,
        "all_scene_absolute_error": _average(all_scene_totals),
        "all_scene_error_increase": _average(all_scene_totals) - normal_total,
        "merge_all_absolute_error": _average(merge_all_totals),
        "merge_all_error_increase": _average(merge_all_totals) - normal_total,
        "carrier_delete_absolute_error": _average(carrier_delete_totals),
        "carrier_delete_error_increase": _average(carrier_delete_totals)
        - normal_total,
        "carrier_swap_absolute_error": _average(carrier_swap_totals),
        "carrier_swap_error_increase": _average(carrier_swap_totals) - normal_total,
        "carrier_delete_inside_change": inside_change,
        "carrier_delete_outside_change": outside_change,
        "carrier_delete_locality_ratio": inside_change / outside_change.clamp_min(1e-6),
        "normal_temporal_identity_error": normal_identity,
        "split_by_time_identity_error": split_identity,
        "split_by_time_identity_increase": split_identity - normal_identity,
        "object_valid_fraction": observation.object_valid.float().mean(),
    }
    result.update(
        {f"compact_baseline_{name}": value for name, value in baseline.items()}
    )
    result.update(
        {
            f"continuous_full_context_{name}": value
            for name, value in full_context_fields.items()
        }
    )
    result.update(
        {f"continuous_holdout_{name}": value for name, value in holdout.items()}
    )
    result.update(
        {
            f"continuous_holdout_compact_baseline_{name}": value
            for name, value in holdout_baseline.items()
        }
    )
    result.update({f"normal_{name}": value for name, value in output["parts"].items()})
    return result
