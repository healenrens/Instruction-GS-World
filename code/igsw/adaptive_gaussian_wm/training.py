"""Stage-specific objectives that keep representation and dynamics roles explicit."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import allocator_loss, dense_feature_loss, slot_regularization
from .representation import reconstruct_current
from .rgb_supervision import rgb_reconstruction_loss


def representation_pretrain_loss(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history = model.encode_history(batch)
    allocator_terms = []
    slot_terms = []
    for index, (token_state, slot_state) in enumerate(
        zip(
            history["token_states"],
            history["slot_states"],
            strict=True,
        )
    ):
        allocator_terms.append(
            allocator_loss(
                token_state,
                batch["history_features"][:, index],
                batch["history_valid"][:, index],
            )[0]
        )
        slot_terms.append(slot_regularization(slot_state, token_state)[0])
    allocator = torch.stack(allocator_terms).mean()
    slot = torch.stack(slot_terms).mean()

    reconstruction = reconstruct_current(model, batch, history)
    feature = dense_feature_loss(
        reconstruction["feature"],
        batch["history_features"][:, -1, None],
        batch["history_valid"][:, -1, None],
        reconstruction["feature_coverage"],
    )
    rgb = feature * 0.0
    rgb_object = feature * 0.0
    rgb_parts = {}
    if model.config.rgb_supervision:
        rgb, rgb_parts = rgb_reconstruction_loss(
            reconstruction["rgb"],
            batch["history_rgb"][:, -1:],
            batch["history_rgb_valid"][:, -1:],
            model.config.rgb_ssim_weight,
        )
        current_slots = history["slot_states"][-1]
        predicted_object_rgb = torch.sigmoid(
            model.object_aggregator.decode_rgb_logits(current_slots.slots)
        )
        target_object_rgb = (
            reconstruction["readout_context"].current_object_rgb.detach()
        )
        object_rgb_error = F.smooth_l1_loss(
            predicted_object_rgb,
            target_object_rgb,
            reduction="none",
        ).mean(dim=-1)
        object_weight = current_slots.activity.to(object_rgb_error.dtype)
        rgb_object = (
            object_rgb_error * object_weight
        ).sum() / object_weight.sum().clamp_min(1e-6)
    total = (
        allocator
        + feature
        + 0.5 * slot
        + model.config.rgb_loss_weight * (rgb + rgb_object)
    )
    parts = {
        "total": total,
        "allocator": allocator,
        "feature": feature,
        "slot": slot,
        "rgb_current": rgb,
        "rgb_object": rgb_object,
    }
    parts.update({f"rgb_current_{name}": value for name, value in rgb_parts.items()})
    return total, parts
