"""Training objective for a language-free history, time, and image-goal Prior."""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow_matching import flow_training_objective
from .goal_conditioning import (
    ObjectGoalConditioner,
    build_goal_prior_context,
    encode_explicit_goal,
    goal_scale_from_batch,
)
from .goal_contrast import (
    deterministic_goal_objective,
    select_hard_wrong_goal,
)
from .observed_action import posterior_from_targets
from .prior_training import dynamics_effect_loss
from .scale import signed_gap_scale


class GoalPriorObjective(nn.Module):
    """Distill future-conditioned teacher actions into a deployable goal Prior."""

    def __init__(
        self,
        model,
        goal_conditioner: ObjectGoalConditioner,
        effect_weight: float,
        goal_anchor_weight: float,
        goal_rank_weight: float,
        goal_relative_margin: float,
        action_activity_floor: float,
    ):
        super().__init__()
        if model.language_condition is not None:
            raise ValueError("goal Prior requires a language-free base model")
        if model.latent_actions.prior_token_conditioner is not None:
            raise ValueError("goal Prior forbids instruction token conditioning")
        if not model.config.temporal_prior_context:
            raise ValueError("goal Prior requires temporal history conditioning")
        if not model.config.object_aligned_actions:
            raise ValueError("goal Prior requires object-aligned actions")
        self.model = model
        self.goal_conditioner = goal_conditioner
        self.effect_weight = effect_weight
        self.goal_anchor_weight = goal_anchor_weight
        self.goal_rank_weight = goal_rank_weight
        self.goal_relative_margin = goal_relative_margin
        self.action_activity_floor = action_activity_floor

    @property
    def goal_contrast_enabled(self) -> bool:
        return self.goal_anchor_weight > 0.0 or self.goal_rank_weight > 0.0

    def _action_weight(
        self,
        history: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.config.canonical_activity_gate:
            return torch.ones_like(posterior[..., 0])
        confidence = (
            history["activity"][:, -1, None] * target["activity"]
        ).float().clamp(0.0, 1.0).pow(
            self.model.config.canonical_activity_power
        ).detach()
        if confidence.shape != posterior.shape[:-1]:
            raise ValueError("activity and object action tokens must align")
        return self.action_activity_floor + (
            1.0 - self.action_activity_floor
        ) * confidence

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model = self.model
        history_scale = signed_gap_scale(
            batch["history_times"],
            model.config.gap_reference,
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            model.config.gap_reference,
        )
        goal_scale = goal_scale_from_batch(model, batch)
        with torch.no_grad():
            history = model.encode_history(batch)
            _, target = model.encode_targets(batch)
            goal = encode_explicit_goal(model, batch, history)
            posterior = posterior_from_targets(
                model,
                batch,
                history,
                target,
                future_scale,
                condition=None,
            )[0]
        context, goal_tokens = build_goal_prior_context(
            model,
            self.goal_conditioner,
            history,
            goal,
            history_scale,
            future_scale,
            goal_scale,
        )
        action_weight = self._action_weight(history, target, posterior)
        flow, predicted_actions, target_actions = flow_training_objective(
            model.latent_actions.prior,
            posterior,
            context,
            weight=action_weight,
        )

        anchor = flow.new_zeros(())
        rank = flow.new_zeros(())
        contrast_parts: dict[str, torch.Tensor] = {}
        correct_actions = None
        if self.goal_contrast_enabled:
            if "sequence_index" not in batch:
                raise ValueError("goal contrast requires sequence_index")
            wrong_goal, selection_parts = select_hard_wrong_goal(
                history["slots"][:, -1],
                history["activity"][:, -1],
                goal,
                batch["sequence_index"],
            )
            wrong_context, _ = build_goal_prior_context(
                model,
                self.goal_conditioner,
                history,
                wrong_goal,
                history_scale,
                future_scale,
                goal_scale,
            )
            (
                anchor,
                rank,
                contrast_parts,
                correct_actions,
                _,
            ) = deterministic_goal_objective(
                model.latent_actions.prior,
                target_actions,
                context,
                wrong_context,
                self.goal_relative_margin,
                action_weight,
            )
            contrast_parts.update(selection_parts)

        if self.effect_weight > 0.0:
            effect, effect_parts = dynamics_effect_loss(
                model,
                history,
                history_scale,
                future_scale,
                (
                    correct_actions
                    if correct_actions is not None
                    else predicted_actions
                ),
                target_actions,
                condition=None,
            )
        else:
            effect = flow.new_zeros(())
            effect_parts = {
                "effect_slot": effect,
                "effect_feature": effect,
                "effect_center": effect,
            }
        endpoint_error = (
            predicted_actions - target_actions
        ).square().mean(dim=-1)
        endpoint_mse = (endpoint_error * action_weight).sum() / (
            action_weight.sum().clamp_min(1e-6)
        )
        canonical_mse = (
            (
                (
                    predicted_actions[..., :6]
                    - target_actions[..., :6]
                ).square().mean(dim=-1)
                * action_weight
            ).sum()
            / action_weight.sum().clamp_min(1e-6)
            if model.config.canonical_semantic_action
            else endpoint_mse.new_zeros(())
        )
        return {
            "loss": (
                flow
                + self.effect_weight * effect
                + self.goal_anchor_weight * anchor
                + self.goal_rank_weight * rank
            ),
            "flow": flow,
            "effect": effect,
            "goal_anchor": anchor,
            "goal_rank": rank,
            "endpoint_mse": endpoint_mse,
            "canonical_endpoint_mse": canonical_mse,
            "context_rms": context.square().mean().sqrt(),
            "goal_token_rms": goal_tokens.square().mean().sqrt(),
            "goal_gate": torch.sigmoid(
                self.goal_conditioner.gate_logit
            ),
            "action_weight_mean": action_weight.mean(),
            "goal_scale_mean": goal_scale.mean(),
            **effect_parts,
            **contrast_parts,
        }
