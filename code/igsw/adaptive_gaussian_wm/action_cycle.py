"""Training-only cross-context cycle consistency for latent actions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .scale import signed_gap_scale


def cross_context_action_cycle_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> torch.Tensor:
    actions = output["posterior_actions"]
    batch_size, future_count = actions.shape[:2]
    group_id = batch.get("group_id")
    if group_id is None:
        shift = 1
    else:
        _, counts = torch.unique_consecutive(
            group_id.flatten(),
            return_counts=True,
        )
        if not bool((counts == counts[0]).all()):
            raise ValueError("action cycle requires equal-size context groups")
        shift = int(counts[0])
    order = torch.roll(
        torch.arange(batch_size, device=actions.device),
        shifts=shift,
    )
    transferred_actions = actions[order].detach()
    history_slots = output["online_history_slots"].detach()
    history_centers = output["online_history_centers"].detach()
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    ).detach()
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    history_mask = torch.zeros(
        history_slots.shape[:3],
        device=history_slots.device,
        dtype=torch.bool,
    )
    generated = model.dynamics(
        history_slots,
        history_activity,
        history_scale,
        future_scale,
        transferred_actions,
        history_mask,
        history_centers,
        output.get("language_condition"),
    )
    generated_centers = (
        generated.future_centers
        if generated.future_centers is not None
        else model.object_aggregator.decode_center(generated.future_slots)
    )
    future_activity = history_activity[:, -1, None].expand(
        -1,
        future_count,
        -1,
    )
    recovered_actions = model.latent_actions.posterior(
        history_slots,
        history_activity,
        generated.future_slots,
        future_activity,
        future_scale,
        history_centers,
        generated_centers,
        output.get("language_condition"),
    )
    recovered_code = F.normalize(
        recovered_actions.flatten(-2),
        dim=-1,
    )
    target_code = F.normalize(
        transferred_actions.flatten(-2),
        dim=-1,
    )
    return F.mse_loss(recovered_code, target_code)
