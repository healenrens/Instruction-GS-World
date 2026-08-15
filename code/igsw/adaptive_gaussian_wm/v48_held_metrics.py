"""Held-video measurements for the v48 recurrent slot representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.shape != weight.shape:
        raise ValueError("v48 held value and weight shapes differ")
    flattened_value = value.flatten(1)
    flattened_weight = weight.flatten(1).float()
    return (flattened_value * flattened_weight).sum(dim=1) / flattened_weight.sum(
        dim=1
    ).clamp_min(1.0)


def _cosine_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("v48 held prediction and target shapes differ")
    return 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1, eps=1e-6
    )


def _activity(
    assignment: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    weight = assignment.float() * valid[..., None].float()
    return weight.sum(dim=2) / valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)


def _centers(
    assignment: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    weight = assignment.float() * valid[..., None].float()
    mass = weight.sum(dim=2)
    center = torch.einsum("btnk,btnd->btkd", weight, coordinates.float())
    return center / mass[..., None].clamp_min(1e-6)


def encode_fully_observed(model, patches, coordinates, valid, frame_times) -> dict:
    observed = torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
    slots, encoder_assignment = model.state_encoder(
        patches, coordinates, valid, frame_times, observed
    )
    reconstruction, assignment = [], []
    for frame_index in range(patches.shape[1]):
        decoded, mixed = model.decode_frame(
            slots[:, frame_index], coordinates[:, frame_index]
        )
        reconstruction.append(decoded)
        assignment.append(mixed)
    assignment_tensor = torch.stack(assignment, dim=1)
    return {
        "slots": slots,
        "contrast_slots": model.contrast_projector(slots),
        "encoder_assignment": encoder_assignment,
        "reconstruction": torch.stack(reconstruction, dim=1),
        "assignment": assignment_tensor,
        "activity": _activity(assignment_tensor, valid),
        "center": _centers(assignment_tensor, coordinates, valid),
    }


def _identity_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source = F.normalize(source.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("b...kd,b...jd->b...kj", source, target)
    slot_count = similarity.shape[-1]
    identity = torch.arange(slot_count, device=source.device)
    same = similarity[..., identity, identity]
    diagonal = torch.eye(slot_count, device=source.device, dtype=torch.bool)
    wrong = similarity.masked_fill(diagonal, -torch.inf).max(dim=-1).values
    retrieval = similarity.argmax(dim=-1).eq(identity).float()
    return {
        "same_cosine": _sample_mean(same, weight),
        "best_wrong_cosine": _sample_mean(wrong, weight),
        "identity_margin": _sample_mean(same - wrong, weight),
        "retrieval_top1": _sample_mean(retrieval, weight),
    }


def base_state_metrics(model, patches, valid, state) -> dict[str, torch.Tensor]:
    target = F.normalize(patches.float(), dim=-1, eps=1e-6)
    patch_weight = valid.float()
    reconstruction_error = _cosine_error(state["reconstruction"], target)
    valid_count = patch_weight.sum(dim=2, keepdim=True).clamp_min(1.0)
    frame_mean = (target * patch_weight[..., None]).sum(dim=2, keepdim=True)
    frame_mean = F.normalize(frame_mean / valid_count[..., None], dim=-1, eps=1e-6)
    frame_mean_error = _cosine_error(frame_mean.expand_as(target), target)
    activity = state["activity"].float()
    assignment = state["assignment"].float().clamp_min(1e-8)
    entropy = -(assignment * assignment.log()).sum(dim=-1)
    entropy = entropy / torch.log(
        assignment.new_tensor(float(model.config.object_slots))
    )
    normalized_slots = F.normalize(state["slots"].float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("btkd,btjd->btkj", normalized_slots, normalized_slots)
    slot_count = model.config.object_slots
    pairwise = (similarity.sum(dim=(-1, -2)) - slot_count) / (
        slot_count * (slot_count - 1)
    )
    temporal_weight = torch.minimum(activity[:, :-1], activity[:, 1:])
    identity = _identity_metrics(
        state["contrast_slots"][:, :-1],
        state["contrast_slots"][:, 1:],
        temporal_weight,
    )
    center_motion = (
        state["center"][:, 1:].float() - state["center"][:, :-1].float()
    ).norm(dim=-1)
    metrics = {
        "reconstruction_error": _sample_mean(reconstruction_error, patch_weight),
        "last_frame_reconstruction_error": _sample_mean(
            reconstruction_error[:, -1], patch_weight[:, -1]
        ),
        "frame_mean_error": _sample_mean(frame_mean_error, patch_weight),
        "reconstruction_gain_over_frame_mean": _sample_mean(
            frame_mean_error - reconstruction_error, patch_weight
        ),
        "active_slot_count": (activity >= model.config.active_slot_fraction)
        .float()
        .mean(dim=1)
        .sum(dim=-1),
        "assignment_entropy": _sample_mean(entropy, patch_weight),
        "slot_pairwise_cosine": pairwise.mean(dim=1),
        "temporal_center_motion": _sample_mean(center_motion, temporal_weight),
    }
    metrics.update({f"temporal_{name}": value for name, value in identity.items()})
    return metrics


def _ordered_target_prediction(
    model,
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    frame_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation_mask = torch.ones(
        patches.shape[:2], device=patches.device, dtype=torch.bool
    )
    observation_mask[:, -1] = False
    slots, _ = model.state_encoder(
        patches, coordinates, valid, frame_times, observation_mask
    )
    prediction, _ = model.decode_frame(slots[:, -1], coordinates[:, -1])
    return slots[:, -1], prediction


def _permute_early_history(
    value: torch.Tensor, permutation: torch.Tensor
) -> torch.Tensor:
    last_history = value.shape[1] - 2
    return torch.cat(
        (
            value[:, permutation],
            value[:, last_history : last_history + 1],
            value[:, -1:],
        ),
        dim=1,
    )


def temporal_order_metrics(
    model,
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    frame_times: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    early_frames = patches.shape[1] - 2
    if early_frames < 1:
        raise ValueError("v48 order evaluation needs at least three frames")
    ordered_slots, ordered = _ordered_target_prediction(
        model, patches, coordinates, valid, frame_times
    )
    reverse = torch.arange(early_frames - 1, -1, -1, device=patches.device)
    generator = torch.Generator().manual_seed(seed)
    shuffle = torch.randperm(early_frames, generator=generator).to(patches.device)

    def predict(permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _ordered_target_prediction(
            model,
            _permute_early_history(patches, permutation),
            _permute_early_history(coordinates, permutation),
            _permute_early_history(valid, permutation),
            frame_times,
        )

    reversed_slots, reversed_prediction = predict(reverse)
    shuffled_slots, shuffled_prediction = predict(shuffle)
    target = patches[:, -1]
    target_valid = valid[:, -1]
    persistence_valid = target_valid & valid[:, -2]
    ordered_error = _sample_mean(_cosine_error(ordered, target), target_valid)
    reversed_error = _sample_mean(
        _cosine_error(reversed_prediction, target), target_valid
    )
    shuffled_error = _sample_mean(
        _cosine_error(shuffled_prediction, target), target_valid
    )
    persistence_error = _sample_mean(
        _cosine_error(patches[:, -2], target), persistence_valid
    )
    slot_weight = torch.ones(
        ordered_slots.shape[:2], device=patches.device, dtype=torch.float32
    )
    reversed_state = _sample_mean(
        _cosine_error(ordered_slots, reversed_slots), slot_weight
    )
    shuffled_state = _sample_mean(
        _cosine_error(ordered_slots, shuffled_slots), slot_weight
    )
    return {
        "ordered_target_error": ordered_error,
        "reversed_target_error": reversed_error,
        "shuffled_target_error": shuffled_error,
        "persistence_target_error": persistence_error,
        "ordered_gain_over_reversed": reversed_error - ordered_error,
        "ordered_gain_over_shuffled": shuffled_error - ordered_error,
        "ordered_gain_over_persistence": persistence_error - ordered_error,
        "reversed_state_difference": reversed_state,
        "shuffled_state_difference": shuffled_state,
    }


def synthetic_blackout_metrics(
    model,
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    frame_times: torch.Tensor,
    full_state: dict,
) -> dict[str, torch.Tensor]:
    frames = patches.shape[1]
    blackout_start = max(1, frames // 3)
    blackout_end = min(frames - 1, blackout_start + max(2, frames // 4))
    observation_mask = torch.ones(
        patches.shape[:2], device=patches.device, dtype=torch.bool
    )
    observation_mask[:, blackout_start:blackout_end] = False
    slots, _ = model.state_encoder(
        patches, coordinates, valid, frame_times, observation_mask
    )
    hidden_predictions = []
    for frame_index in range(blackout_start, blackout_end):
        decoded, _ = model.decode_frame(
            slots[:, frame_index], coordinates[:, frame_index]
        )
        hidden_predictions.append(decoded)
    hidden_prediction = torch.stack(hidden_predictions, dim=1)
    hidden_target = patches[:, blackout_start:blackout_end]
    hidden_valid = valid[:, blackout_start:blackout_end]
    persistence = patches[:, blackout_start - 1 : blackout_start].expand_as(
        hidden_target
    )
    persistence_valid = hidden_valid & valid[
        :, blackout_start - 1 : blackout_start
    ].expand_as(hidden_valid)
    prediction_error = _sample_mean(
        _cosine_error(hidden_prediction, hidden_target), hidden_valid
    )
    persistence_error = _sample_mean(
        _cosine_error(persistence, hidden_target), persistence_valid
    )
    projected = model.contrast_projector(slots)
    pre_index = blackout_start - 1
    hidden_index = blackout_end - 1
    reappearance_index = blackout_end
    hidden_weight = torch.minimum(
        full_state["activity"][:, pre_index],
        full_state["activity"][:, hidden_index],
    )
    reappearance_weight = torch.minimum(
        full_state["activity"][:, pre_index],
        full_state["activity"][:, reappearance_index],
    )
    hidden_identity = _identity_metrics(
        projected[:, pre_index], projected[:, hidden_index], hidden_weight
    )
    reappearance_identity = _identity_metrics(
        projected[:, pre_index], projected[:, reappearance_index], reappearance_weight
    )
    post_state_difference = _sample_mean(
        _cosine_error(
            slots[:, reappearance_index],
            full_state["slots"][:, reappearance_index],
        ),
        reappearance_weight,
    )
    result = {
        "blackout_prediction_error": prediction_error,
        "blackout_persistence_error": persistence_error,
        "blackout_gain_over_persistence": persistence_error - prediction_error,
        "reappearance_state_difference_from_full": post_state_difference,
        "blackout_fraction": prediction_error.new_full(
            prediction_error.shape,
            (blackout_end - blackout_start) / frames,
        ),
    }
    result.update({f"hidden_{name}": value for name, value in hidden_identity.items()})
    result.update(
        {f"reappearance_{name}": value for name, value in reappearance_identity.items()}
    )
    return result


def slot_deletion_metrics(
    model,
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    state: dict,
) -> dict[str, torch.Tensor]:
    slots = state["slots"][:, -1]
    target = patches[:, -1]
    target_valid = valid[:, -1]
    full_prediction = state["reconstruction"][:, -1]
    full_assignment = state["assignment"][:, -1]
    activity = state["activity"][:, -1]
    batch, slot_count, slot_dim = slots.shape
    patch_count = target.shape[1]
    repeated_slots = slots[:, None].expand(batch, slot_count, slot_count, slot_dim)
    repeated_slots = repeated_slots.reshape(batch * slot_count, slot_count, slot_dim)
    repeated_coordinates = coordinates[:, -1, None].expand(
        batch, slot_count, patch_count, 2
    )
    repeated_coordinates = repeated_coordinates.reshape(
        batch * slot_count, patch_count, 2
    )
    slot_valid = ~torch.eye(slot_count, device=slots.device, dtype=torch.bool)
    slot_valid = slot_valid[None].expand(batch, slot_count, slot_count)
    deleted, _ = model.decode_frame(
        repeated_slots,
        repeated_coordinates,
        slot_valid.reshape(batch * slot_count, slot_count),
    )
    deleted = deleted.reshape(batch, slot_count, patch_count, -1)
    deleted_error = _cosine_error(deleted, target[:, None].expand_as(deleted))
    valid_slots = target_valid[:, None].expand(batch, slot_count, patch_count)
    deleted_error = (deleted_error * valid_slots.float()).sum(
        dim=2
    ) / valid_slots.float().sum(dim=2).clamp_min(1.0)
    full_error = _sample_mean(_cosine_error(full_prediction, target), target_valid)
    error_increase = deleted_error - full_error[:, None]
    change = _cosine_error(deleted, full_prediction[:, None].expand_as(deleted))
    owner = full_assignment.argmax(dim=-1)
    slot_index = torch.arange(slot_count, device=slots.device)[None, :, None]
    owned = owner[:, None].eq(slot_index) & valid_slots
    outside = ~owned & valid_slots
    change_mass = (change * valid_slots.float()).sum(dim=2).clamp_min(1e-8)
    owned_change = (change * owned.float()).sum(dim=2)
    precision = owned_change / change_mass
    area = owned.float().sum(dim=2) / valid_slots.float().sum(dim=2).clamp_min(1.0)
    enrichment = precision / area.clamp_min(1e-6)
    inside_mean = owned_change / owned.float().sum(dim=2).clamp_min(1.0)
    outside_mean = (change * outside.float()).sum(dim=2) / outside.float().sum(
        dim=2
    ).clamp_min(1.0)
    active = (
        (activity >= model.config.active_slot_fraction) & owned.any(dim=2)
    ).float()

    def active_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)

    return {
        "deletion_error_increase": active_mean(error_increase),
        "deletion_positive_utility_fraction": active_mean(
            error_increase.gt(0.0).float()
        ),
        "deletion_change_locality_precision": active_mean(precision),
        "deletion_owned_area_fraction": active_mean(area),
        "deletion_locality_enrichment": active_mean(enrichment),
        "deletion_inside_outside_change_ratio": active_mean(
            inside_mean / outside_mean.clamp_min(1e-8)
        ),
        "deletion_evaluated_slots": active.sum(dim=1),
    }
