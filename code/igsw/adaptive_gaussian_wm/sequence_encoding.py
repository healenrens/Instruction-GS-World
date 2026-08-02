"""Shared causal sequence encoding for legacy slots and persistent memory."""
from __future__ import annotations

import torch

from .gpstoken import GPSTokenState, LearnableGPSTokenAllocator
from .object_memory import ObjectMemoryState, ObjectMemoryTransition
from .object_slots import ObjectSlotAggregator, ObjectSlotState


def validate_visual_sequence(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    times: torch.Tensor,
) -> None:
    if features.ndim != 4:
        raise ValueError("features must have shape [B,T,N,C]")
    if coordinates.shape != (*features.shape[:3], 2):
        raise ValueError("coordinates must have shape [B,T,N,2]")
    if valid.shape != features.shape[:3]:
        raise ValueError("valid must have shape [B,T,N]")
    if times.shape != features.shape[:2]:
        raise ValueError("times must have shape [B,T]")


def _stack_states(
    token_states: list[GPSTokenState],
    slot_states: list[ObjectSlotState | ObjectMemoryState],
    last_memory: ObjectMemoryState | None,
) -> dict:
    result = {
        "token_states": token_states,
        "slot_states": slot_states,
        "slots": torch.stack([state.slots for state in slot_states], dim=1),
        "tracking_slots": torch.stack(
            [state.tracking_slots for state in slot_states], dim=1
        ),
        "activity": torch.stack(
            [state.activity for state in slot_states], dim=1
        ),
        "center": torch.stack([state.center for state in slot_states], dim=1),
        "feature": torch.stack(
            [state.decoded_feature for state in slot_states], dim=1
        ),
        "last_memory": last_memory,
    }
    if last_memory is None:
        result["legacy_anchor_slots"] = slot_states[0].tracking_slots
        result["legacy_anchor_centers"] = slot_states[0].center
    if last_memory is not None:
        memory_states = [
            state for state in slot_states if isinstance(state, ObjectMemoryState)
        ]
        if len(memory_states) != len(slot_states):
            raise ValueError("persistent sequence contains a legacy slot state")
        for name in (
            "relative_scale",
            "relative_disparity",
            "relations",
            "existence",
            "in_frame",
            "visibility",
            "update_gate",
            "identity_key",
            "identity_similarity",
            "association_matrix",
            "association_match",
            "association_unmatched",
            "association_discovery",
            "association_entropy",
            "association_support_distance",
            "observation_confidence",
            "birth_evidence",
        ):
            result[name] = torch.stack(
                [getattr(state, name) for state in memory_states],
                dim=1,
            )
    return result


def encode_visual_sequence(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    times: torch.Tensor,
    allocator: LearnableGPSTokenAllocator,
    aggregator: ObjectSlotAggregator,
    memory: ObjectMemoryTransition | None = None,
    initial_memory: ObjectMemoryState | None = None,
    previous_time: torch.Tensor | None = None,
    initial_anchor_slots: torch.Tensor | None = None,
    initial_anchor_centers: torch.Tensor | None = None,
) -> dict:
    validate_visual_sequence(features, coordinates, valid, times)
    if initial_memory is not None and memory is None:
        raise ValueError("initial_memory requires a persistent memory module")
    if initial_memory is not None and previous_time is None:
        raise ValueError("continued memory encoding requires previous_time")
    if (initial_anchor_slots is None) != (initial_anchor_centers is None):
        raise ValueError("legacy anchor slots and centers must be provided together")
    if initial_anchor_slots is not None and memory is not None:
        raise ValueError("persistent memory cannot use a legacy anchor")

    token_states: list[GPSTokenState] = []
    slot_states: list[ObjectSlotState | ObjectMemoryState] = []
    anchor_slots = initial_anchor_slots
    anchor_centers = initial_anchor_centers
    previous_memory = initial_memory
    for index in range(features.shape[1]):
        token_state = allocator(
            features[:, index],
            coordinates[:, index],
            valid[:, index],
        )
        if memory is None:
            slot_state = aggregator(token_state, anchor_slots, anchor_centers)
            if anchor_slots is None:
                anchor_slots = slot_state.tracking_slots
                anchor_centers = slot_state.center
        elif previous_memory is None:
            observation = aggregator(token_state)
            slot_state = memory.initialize(observation, token_state)
        else:
            reference_time = (
                previous_time if index == 0 else times[:, index - 1]
            )
            if reference_time is None:
                raise ValueError("memory prediction has no reference time")
            predicted = memory.predict(
                previous_memory,
                times[:, index] - reference_time,
            )
            observation = aggregator(
                token_state,
                predicted.tracking_slots,
                predicted.center,
                predicted.identity_key if memory.persistent_identity_key else None,
            )
            slot_state = memory.correct(predicted, observation, token_state)
        token_states.append(token_state)
        slot_states.append(slot_state)
        if memory is not None:
            if not isinstance(slot_state, ObjectMemoryState):
                raise ValueError("memory encoder returned a legacy slot state")
            previous_memory = slot_state
    return _stack_states(token_states, slot_states, previous_memory)
