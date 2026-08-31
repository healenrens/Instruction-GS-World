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
    return (
        errors["support_bce"]
        + errors["dino_cosine_error"]
        + errors["siglip_cosine_error"]
        + 0.25 * errors["visibility_bce"]
        + 0.25 * errors["lifecycle_cross_entropy"]
    )


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
    holdout_totals = []
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
        baseline_totals.append(_baseline_total(frame, model.config.covariance_floor))

        observed_frame = _slice_frame(frame, observed_indices)
        held_frame = _slice_frame(frame, held_indices)
        held_state = model.codec(observed_frame)
        held_decoded = model.decoder(held_state, held_frame["coordinates"])
        holdout_totals.append(_field_total(held_state, held_decoded, held_frame)[0])

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

    def average(values):
        return sum(values) / len(values)

    normal_total = average(normal_totals)
    baseline_total = average(baseline_totals)
    inside_change = average(inside_changes)
    outside_change = average(outside_changes)
    valid = observation.object_valid.float()
    normal_identity = _identity_consistency(output["states"], valid)
    split_identity = _identity_consistency(split_states, valid)
    result = {
        "normal_absolute_error": normal_total,
        "compact_baseline_absolute_error": baseline_total,
        "oracle_absolute_error": normal_total.new_zeros(()),
        "normal_gap_recovery": 1.0 - normal_total / baseline_total.clamp_min(1e-6),
        "continuous_holdout_absolute_error": average(holdout_totals),
        "continuous_holdout_ratio": average(holdout_totals)
        / normal_total.clamp_min(1e-6),
        "query_swap_absolute_error": average(query_swap_totals),
        "query_swap_error_increase": average(query_swap_totals) - normal_total,
        "all_scene_absolute_error": average(all_scene_totals),
        "all_scene_error_increase": average(all_scene_totals) - normal_total,
        "merge_all_absolute_error": average(merge_all_totals),
        "merge_all_error_increase": average(merge_all_totals) - normal_total,
        "carrier_delete_absolute_error": average(carrier_delete_totals),
        "carrier_delete_error_increase": average(carrier_delete_totals) - normal_total,
        "carrier_swap_absolute_error": average(carrier_swap_totals),
        "carrier_swap_error_increase": average(carrier_swap_totals) - normal_total,
        "carrier_delete_inside_change": inside_change,
        "carrier_delete_outside_change": outside_change,
        "carrier_delete_locality_ratio": inside_change / outside_change.clamp_min(1e-6),
        "normal_temporal_identity_error": normal_identity,
        "split_by_time_identity_error": split_identity,
        "split_by_time_identity_increase": split_identity - normal_identity,
        "object_valid_fraction": observation.object_valid.float().mean(),
    }
    result.update({f"normal_{name}": value for name, value in output["parts"].items()})
    return result
