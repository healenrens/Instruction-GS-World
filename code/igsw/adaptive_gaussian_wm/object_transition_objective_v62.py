"""E1 objective and matched intervention construction for v62."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed_statistics import gather_batch_without_grad
from .object_effect_posterior_v62 import ObjectEffectV62
from .object_transition_metrics_v62 import (
    decoded_field_errors_v62,
    lifecycle_error_v62,
    state_geometry_error_v62,
    transition_error_sum_v62,
)


def object_change_magnitude_v62(source, target):
    source_weight = source.presence.float()
    target_weight = target.presence.float()
    source_center = (source.center.float() * source_weight[..., None]).sum(dim=1)
    source_center = source_center / source_weight.sum(dim=1, keepdim=True).clamp_min(
        1e-6
    )
    target_center = (target.center.float() * target_weight[..., None]).sum(dim=1)
    target_center = target_center / target_weight.sum(dim=1, keepdim=True).clamp_min(
        1e-6
    )
    semantic = (target.carriers.float() - source.carriers.float()).square().mean((1, 2))
    return (target_center - source_center).norm(dim=-1) + semantic.sqrt()


def matched_shuffle_effect_v62(effect, source_index, magnitude):
    global_effect = gather_batch_without_grad(effect.value)
    global_source = gather_batch_without_grad(source_index.long())
    magnitude_bin = torch.bucketize(
        magnitude.detach(), magnitude.new_tensor((0.05, 0.20))
    )
    global_bin = gather_batch_without_grad(magnitude_bin)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    start = rank * len(effect.value)
    selected, exact = [], []
    positions = torch.arange(len(global_effect), device=global_effect.device)
    for local_index in range(len(effect.value)):
        global_index = start + local_index
        same_source = global_source == source_index[local_index]
        same_bin = global_bin == magnitude_bin[local_index]
        candidate = same_source & same_bin & (positions != global_index)
        source_candidate = same_source & (positions != global_index)
        candidate = torch.where(candidate.any(), candidate, source_candidate)
        candidate = torch.where(candidate.any(), candidate, positions != global_index)
        distance = (positions - global_index).remainder(len(global_effect))
        distance = distance.masked_fill(~candidate, len(global_effect) + 1)
        chosen = distance.argmin()
        selected.append(global_effect[chosen])
        exact.append((same_source[chosen] & same_bin[chosen]).float())
    return ObjectEffectV62(torch.stack(selected)), torch.stack(exact).mean()


def _branch_errors(model, state, target_state, target_frame):
    decoded = model.decoder(state, target_frame["coordinates"])
    field = decoded_field_errors_v62(decoded, target_frame)
    lifecycle = lifecycle_error_v62(state, target_frame)
    geometry, center, scale = state_geometry_error_v62(
        state, target_state, target_frame["object_valid"].float()
    )
    total = transition_error_sum_v62(field, lifecycle, geometry)
    return total, field, lifecycle, center, scale


def object_transition_objective_v62(model, output, target_frame):
    config = model.config
    correct = _branch_errors(model, output["correct"], output["target"], target_frame)
    zero = _branch_errors(model, output["zero"], output["target"], target_frame)
    shuffled = _branch_errors(model, output["shuffled"], output["target"], target_frame)
    persistence = _branch_errors(
        model, output["source"], output["target"], target_frame
    )
    correct_total, zero_total = correct[0], zero[0]
    shuffled_total, persistence_total = shuffled[0], persistence[0]
    intervention = F.relu(
        correct_total - zero_total.detach() + config.intervention_margin
    )
    intervention = intervention + F.relu(
        correct_total - shuffled_total.detach() + config.intervention_margin
    )
    intervention = intervention + F.relu(
        correct_total - persistence_total.detach() + config.intervention_margin
    )
    total = correct_total + config.intervention_weight * intervention
    transport = output["transport"].float().clamp_min(1e-8)
    transport_entropy = -(transport * transport.log()).sum(dim=-1).mean()
    effect_flat = gather_batch_without_grad(output["effect"].value).float().flatten(1)
    centered_effect = effect_flat - effect_flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered_effect)
    spectrum = singular.square()
    spectrum_mass = spectrum.sum()
    spectrum = spectrum / spectrum_mass.clamp_min(1e-8)
    effect_effective_rank = (
        -(spectrum * spectrum.clamp_min(1e-8).log()).sum()
    ).exp() * (spectrum_mass > 1e-8).float()
    target_change = object_change_magnitude_v62(output["source"], output["target"])
    predicted_change = object_change_magnitude_v62(output["source"], output["correct"])
    zero_change = object_change_magnitude_v62(output["source"], output["zero"])
    identity_copy_error = 1.0 - F.cosine_similarity(
        output["source"].identity.float(), output["correct"].identity.float(), dim=-1
    )
    parts = {
        "loss": total.detach(),
        "transition_correct_absolute_error": correct_total.detach(),
        "transition_zero_absolute_error": zero_total.detach(),
        "transition_shuffled_absolute_error": shuffled_total.detach(),
        "transition_persistence_absolute_error": persistence_total.detach(),
        "correct_gain_over_zero": (zero_total - correct_total).detach(),
        "correct_gain_over_shuffled": (shuffled_total - correct_total).detach(),
        "correct_gain_over_persistence": (persistence_total - correct_total).detach(),
        "intervention_margin_loss": intervention.detach(),
        "matched_shuffle_fraction": output["matched_shuffle_fraction"].detach(),
        "effect_rms": output["effect"].value.float().square().mean().sqrt().detach(),
        "effect_batch_variance": effect_flat.var(dim=0, unbiased=False).mean().detach(),
        "effect_effective_rank": effect_effective_rank.detach(),
        "transport_entropy": transport_entropy.detach(),
        "target_change_magnitude": target_change.mean().detach(),
        "predicted_change_magnitude": predicted_change.mean().detach(),
        "zero_change_magnitude": zero_change.mean().detach(),
        "identity_copy_cosine_error": identity_copy_error.mean().detach(),
        "correct_support_bce": correct[1]["support_bce"].detach(),
        "correct_dino_cosine_error": correct[1]["dino_cosine_error"].detach(),
        "correct_siglip_cosine_error": correct[1]["siglip_cosine_error"].detach(),
        "correct_visibility_bce": correct[1]["visibility_bce"].detach(),
        "correct_lifecycle_cross_entropy": correct[2].detach(),
        "correct_center_error": correct[3].detach(),
        "correct_relative_scale_error": correct[4].detach(),
    }
    return total, parts
