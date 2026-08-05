"""Learning-rate groups for Object Memory model generations."""
from __future__ import annotations

import torch


def build_object_memory_optimizer(model, args) -> torch.optim.AdamW:
    action_prefixes = (
        "latent_actions.posterior.", "latent_actions.effect_head.",
        "latent_actions.prior.", "latent_actions.prior_", "dynamics.factor_keys",
        "dynamics.routing_query.", "dynamics.action_", "effect_composer.",
        "region_effect_posterior.", "region_dynamics.action_",
        "region_dynamics.factor_keys", "region_dynamics.route_query.",
    )
    dino_prefixes = tuple(
        f"online_dino.backbone.blocks.{index}." for index in range(12, 24)
    ) + ("online_dino.backbone.norm.",)
    new_prefixes = (
        "online_dino.projector.", "region_transformer.", "region_memory.",
        "region_dynamics.",
    )
    grouped = {"core": [], "action": [], "dino": [], "new_modules": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(action_prefixes):
            group = "action"
        elif name.startswith(dino_prefixes):
            group = "dino"
        elif name.startswith(new_prefixes):
            group = "new_modules"
        else:
            group = "core"
        grouped[group].append(parameter)
    learning_rates = {
        "core": args.core_lr,
        "action": args.action_lr,
        "dino": args.dino_lr,
        "new_modules": args.new_module_lr,
    }
    groups = [
        {"params": parameters, "lr": learning_rates[name], "group_name": name}
        for name, parameters in grouped.items()
        if parameters
    ]
    if not groups:
        raise ValueError("Object Memory optimizer has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
