"""Prior-only training with Dynamics-equivalent endpoint alignment."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .architecture_metrics import normalized_slot_error
from .flow_matching import flow_training_objective
from .mode_set_prior import (
    ModeSetActionPrior,
    ordered_group_responsibility,
)
from .scale import signed_gap_scale
from .synthetic import make_synthetic_batch


def _make_batch(
    model,
    batch_size: int,
    history_frames: int,
    future_steps: int,
    grid_size: int,
    device: torch.device,
    semantic_branch_strength: float,
) -> dict[str, torch.Tensor]:
    config = model.config
    return make_synthetic_batch(
        config.feature_dim,
        batch_size,
        history_frames,
        future_steps,
        grid_size,
        device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        semantic_branch_strength=semantic_branch_strength,
    )


def _future_centers(model, prediction) -> torch.Tensor:
    if prediction.future_centers is not None:
        return prediction.future_centers
    return model.object_aggregator.decode_center(prediction.future_slots)


def dynamics_effect_loss(
    model,
    history: dict[str, torch.Tensor],
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_mask = torch.zeros(
        history["slots"].shape[:3],
        device=history["slots"].device,
        dtype=torch.bool,
    )
    predicted = model.dynamics(
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        predicted_actions,
        history_mask,
        history["center"],
        condition,
    )
    with torch.no_grad():
        target = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            target_actions,
            history_mask,
            history["center"],
            condition,
        )
        target_features = model.object_aggregator.decode_feature(
            target.future_slots
        )
        target_centers = _future_centers(model, target)
    predicted_features = model.object_aggregator.decode_feature(
        predicted.future_slots
    )
    predicted_centers = _future_centers(model, predicted)
    slot = normalized_slot_error(
        predicted.future_slots,
        target.future_slots,
    ).mean()
    feature = normalized_slot_error(
        predicted_features,
        target_features,
    ).mean()
    center = (predicted_centers - target_centers).square().mean()
    return slot + feature + 10.0 * center, {
        "effect_slot": slot,
        "effect_feature": feature,
        "effect_center": center,
    }


def mode_set_effect_objective(
    model,
    history: dict[str, torch.Tensor],
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    target_actions: torch.Tensor,
    context: torch.Tensor,
    group_id: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    prior = model.latent_actions.prior
    if not isinstance(prior, ModeSetActionPrior):
        raise ValueError("mode-set objective requires ModeSetActionPrior")
    history_mask = torch.zeros(
        history["slots"].shape[:3],
        device=history["slots"].device,
        dtype=torch.bool,
    )
    with torch.no_grad():
        target = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            target_actions,
            history_mask,
            history["center"],
        )
        target_features = model.object_aggregator.decode_feature(
            target.future_slots
        )
        target_centers = _future_centers(model, target)
    logits, prototypes = prior._distribution(context)
    slot_costs = []
    feature_costs = []
    center_costs = []
    component_slots = []
    component_features = []
    component_centers = []
    for component in range(prior.components):
        prediction = model.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            prototypes[:, component],
            history_mask,
            history["center"],
        )
        predicted_features = model.object_aggregator.decode_feature(
            prediction.future_slots
        )
        predicted_centers = _future_centers(model, prediction)
        component_slots.append(prediction.future_slots)
        component_features.append(predicted_features)
        component_centers.append(predicted_centers)
        slot_costs.append(
            normalized_slot_error(
                prediction.future_slots,
                target.future_slots,
            )
        )
        feature_costs.append(
            normalized_slot_error(predicted_features, target_features)
        )
        center_costs.append(
            (predicted_centers - target_centers)
            .square()
            .mean(dim=(1, 2, 3))
        )
    slot_cost = torch.stack(slot_costs, dim=-1)
    feature_cost = torch.stack(feature_costs, dim=-1)
    center_cost = torch.stack(center_costs, dim=-1)
    effect_cost = slot_cost + feature_cost + 10.0 * center_cost
    if prior.ordered_assignment:
        responsibility = ordered_group_responsibility(
            target_centers,
            history["center"][:, -1],
            prior.components,
            group_id,
        ).to(effect_cost.dtype)
    else:
        responsibility = prior._responsibilities(
            effect_cost,
            logits,
            group_id,
        )
    weights = (
        (1.0 - prior.responsibility_floor) * responsibility
        + prior.responsibility_floor / prior.components
    )
    effect = (weights.detach() * effect_cost).sum(dim=-1).mean()
    action_cost = (
        prototypes - target_actions[:, None]
    ).square().mean(dim=(2, 3, 4))
    action_fit = (weights.detach() * action_cost).sum(dim=-1).mean()
    logit_fit = -(
        responsibility.detach() * F.log_softmax(logits, dim=-1)
    ).sum(dim=-1).mean()
    usage = responsibility.mean(dim=0)
    usage_fit = (
        usage - usage.new_full(usage.shape, 1.0 / prior.components)
    ).square().mean()
    calibration = logit_fit + usage_fit
    geometry = effect.new_zeros(())
    if prior.geometry_weight > 0.0:
        if group_id is None:
            raise ValueError("mode-set geometry requires grouped futures")
        _, counts = torch.unique_consecutive(
            group_id.flatten(),
            return_counts=True,
        )
        if not bool((counts == prior.components).all()):
            raise ValueError("mode-set geometry requires complete groups")
        groups = counts.numel()

        def distance_spectrum(
            values: torch.Tensor,
            normalize: bool,
        ) -> torch.Tensor:
            if normalize:
                values = F.normalize(values, dim=-1)
            distance = (
                values[:, :, None] - values[:, None, :]
            ).square().mean(dim=(-1, -2, -3))
            upper = torch.triu_indices(
                prior.components,
                prior.components,
                offset=1,
                device=values.device,
            )
            return distance[:, upper[0], upper[1]].sort(dim=-1).values

        def relative_geometry(
            prediction: torch.Tensor,
            target_value: torch.Tensor,
            normalize: bool,
        ) -> torch.Tensor:
            predicted_distance = distance_spectrum(prediction, normalize)
            target_distance = distance_spectrum(target_value, normalize)
            scale = target_distance.mean(dim=-1, keepdim=True).clamp_min(1e-3)
            return F.smooth_l1_loss(
                predicted_distance / scale.detach(),
                target_distance / scale.detach(),
                beta=0.1,
            )

        slot_set = torch.stack(component_slots, dim=1).reshape(
            groups,
            prior.components,
            prior.components,
            *target.future_slots.shape[1:],
        )[:, 0]
        feature_set = torch.stack(component_features, dim=1).reshape(
            groups,
            prior.components,
            prior.components,
            *target_features.shape[1:],
        )[:, 0]
        center_set = torch.stack(component_centers, dim=1).reshape(
            groups,
            prior.components,
            prior.components,
            *target_centers.shape[1:],
        )[:, 0]
        target_slot_set = target.future_slots.reshape(
            groups,
            prior.components,
            *target.future_slots.shape[1:],
        )
        target_feature_set = target_features.reshape(
            groups,
            prior.components,
            *target_features.shape[1:],
        )
        target_center_set = target_centers.reshape(
            groups,
            prior.components,
            *target_centers.shape[1:],
        )
        geometry = (
            relative_geometry(slot_set, target_slot_set, True)
            + relative_geometry(feature_set, target_feature_set, True)
            + relative_geometry(center_set, target_center_set, False)
        )
    effect_weight = max(model.config.prior_effect_weight, 1.0)
    loss = (
        0.05 * action_fit
        + prior.fit_weight * calibration
        + effect_weight * effect
        + prior.geometry_weight * geometry
    )
    return loss, action_fit, effect, {
        "effect_slot": (weights.detach() * slot_cost).sum(dim=-1).mean(),
        "effect_feature": (
            weights.detach() * feature_cost
        ).sum(dim=-1).mean(),
        "effect_center": (
            weights.detach() * center_cost
        ).sum(dim=-1).mean(),
        "effect_geometry": geometry,
    }


def train_prior(
    model,
    steps: int,
    batch_size: int,
    grid_size: int,
    future_steps: int,
    device: torch.device,
    learning_rate: float,
    semantic_branch_strength: float,
    single_history_probability: float,
) -> list[dict[str, float]]:
    parameters = [
        *model.latent_actions.prior.parameters(),
        *model.latent_actions.prior_condition_parameters(),
    ]
    active = {id(parameter) for parameter in parameters}
    original_requires_grad = {
        id(parameter): parameter.requires_grad
        for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in active)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )
    trace = []
    model.train()
    for step in range(1, steps + 1):
        history_frames = (
            1
            if float(torch.rand(())) < single_history_probability
            else 3
        )
        batch = _make_batch(
            model,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            semantic_branch_strength,
        )
        with torch.no_grad():
            history = model.encode_history(batch)
            _, target = model.encode_targets(batch)
            history_scale = signed_gap_scale(
                batch["history_times"],
                model.config.gap_reference,
            )
            future_scale = signed_gap_scale(
                batch["future_times"],
                model.config.gap_reference,
            )
            posterior = model.latent_actions.posterior(
                history["slots"],
                history["activity"],
                target["slots"],
                target["activity"],
                future_scale,
                history["center"],
                target["center"],
            )
        context = model.prior_context(
            history,
            future_scale,
            history_scale,
        )
        if isinstance(model.latent_actions.prior, ModeSetActionPrior):
            loss, flow, effect, effect_parts = mode_set_effect_objective(
                model,
                history,
                history_scale,
                future_scale,
                posterior,
                context,
                batch.get("group_id"),
            )
        else:
            flow, predicted_actions, target_actions = flow_training_objective(
                model.latent_actions.prior,
                posterior,
                context,
                batch.get("group_id"),
            )
            effect_weight = model.config.prior_effect_weight
            if effect_weight > 0.0:
                effect, effect_parts = dynamics_effect_loss(
                    model,
                    history,
                    history_scale,
                    future_scale,
                    predicted_actions,
                    target_actions,
                )
            else:
                effect = flow.new_zeros(())
                effect_parts = {
                    "effect_slot": effect,
                    "effect_feature": effect,
                    "effect_center": effect,
                }
            loss = flow + effect_weight * effect
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "total": float(loss.detach()),
                    "flow": float(flow.detach()),
                    "effect": float(effect.detach()),
                    **{
                        name: float(value.detach())
                        for name, value in effect_parts.items()
                    },
                    "gradient_norm": float(gradient_norm),
                }
            )
    for parameter in model.parameters():
        parameter.requires_grad_(original_requires_grad[id(parameter)])
    return trace
