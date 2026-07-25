"""Real-data objective for the history-and-instruction action Prior."""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow_matching import flow_training_objective
from .instruction_contrast import (
    deterministic_instruction_objective,
    flow_endpoint_prediction,
)
from .instruction_groups import select_different_task_condition
from .observed_action import posterior_from_targets
from .prior_training import dynamics_effect_loss
from .relative_effect import relative_dynamics_effects
from .scale import signed_gap_scale
from .task_semantic_contrast import task_semantic_contrast


class RealPriorObjective(nn.Module):
    def __init__(
        self,
        model,
        effect_weight: float,
        instruction_anchor_weight: float,
        instruction_rank_weight: float,
        instruction_relative_margin: float,
        action_activity_floor: float,
        dynamics_relative_objective: bool = False,
        paraphrase_positive_weight: float = 0.0,
        paraphrase_features: torch.Tensor | None = None,
        paraphrase_tokens: torch.Tensor | None = None,
        paraphrase_token_valid: torch.Tensor | None = None,
        paraphrase_index: torch.Tensor | None = None,
        condition_task_index: torch.Tensor | None = None,
        task_semantic_contrast_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.effect_weight = effect_weight
        self.instruction_anchor_weight = instruction_anchor_weight
        self.instruction_rank_weight = instruction_rank_weight
        self.instruction_relative_margin = instruction_relative_margin
        self.action_activity_floor = action_activity_floor
        self.dynamics_relative_objective = dynamics_relative_objective
        self.paraphrase_positive_weight = paraphrase_positive_weight
        self.task_semantic_contrast_weight = task_semantic_contrast_weight
        banks = (
            paraphrase_features,
            paraphrase_tokens,
            paraphrase_token_valid,
            paraphrase_index,
            condition_task_index,
        )
        if (
            instruction_anchor_weight > 0.0
            or instruction_rank_weight > 0.0
            or paraphrase_positive_weight > 0.0
            or task_semantic_contrast_weight > 0.0
        ) and any(
            value is None for value in banks
        ):
            raise ValueError("semantic losses require full condition banks")
        self.register_buffer(
            "paraphrase_features",
            paraphrase_features,
            persistent=False,
        )
        self.register_buffer(
            "paraphrase_tokens",
            paraphrase_tokens,
            persistent=False,
        )
        self.register_buffer(
            "paraphrase_token_valid",
            paraphrase_token_valid,
            persistent=False,
        )
        self.register_buffer(
            "paraphrase_index",
            paraphrase_index,
            persistent=False,
        )
        self.register_buffer(
            "condition_task_index",
            condition_task_index,
            persistent=False,
        )

    @property
    def instruction_contrast_enabled(self) -> bool:
        return (
            self.instruction_anchor_weight > 0.0
            or self.instruction_rank_weight > 0.0
            or self.paraphrase_positive_weight > 0.0
        )

    def _instruction_contrast(
        self,
        batch: dict[str, torch.Tensor],
        history: dict[str, torch.Tensor],
        history_scale: torch.Tensor,
        future_scale: torch.Tensor,
        condition: torch.Tensor,
        context: torch.Tensor,
        target_actions: torch.Tensor,
        action_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if any(
            name not in batch
            for name in ("condition_feature", "condition_index", "task_index")
        ):
            raise ValueError(
                "instruction contrast requires condition and task ids"
            )
        wrong_index = select_different_task_condition(
            batch["condition_index"],
            batch["task_index"],
            self.condition_task_index,
            self.paraphrase_features,
        )
        if bool((wrong_index == batch["condition_index"]).any()):
            raise AssertionError("wrong instruction selector returned a match")
        wrong_condition = self.model.language_condition(
            self.paraphrase_features[wrong_index]
        )
        wrong_tokens = None
        wrong_token_valid = None
        if self.model.config.token_conditioned_prior:
            wrong_tokens = self.paraphrase_tokens[wrong_index]
            wrong_token_valid = self.paraphrase_token_valid[wrong_index]
        wrong_context = self.model.prior_context(
            history,
            future_scale,
            history_scale,
            wrong_condition,
            wrong_tokens,
            wrong_token_valid,
        )
        anchor, rank, parts, correct, wrong = (
            deterministic_instruction_objective(
                self.model.latent_actions.prior,
                target_actions,
                context,
                wrong_context,
                self.instruction_relative_margin,
                action_weight,
            )
        )
        positive = None
        positive_available = None
        if self.paraphrase_positive_weight > 0.0:
            positive_index = self.paraphrase_index[
                batch["condition_index"]
            ]
            positive_available = positive_index >= 0
            safe_index = positive_index.clamp_min(0)
            positive_condition = self.model.language_condition(
                self.paraphrase_features[safe_index]
            )
            positive_tokens = None
            positive_token_valid = None
            if self.model.config.token_conditioned_prior:
                positive_tokens = self.paraphrase_tokens[safe_index]
                positive_token_valid = self.paraphrase_token_valid[safe_index]
            positive_context = self.model.prior_context(
                history,
                future_scale,
                history_scale,
                positive_condition,
                positive_tokens,
                positive_token_valid,
            )
            positive = flow_endpoint_prediction(
                self.model.latent_actions.prior,
                torch.zeros_like(target_actions),
                target_actions.new_zeros(target_actions.shape[:2]),
                positive_context,
            )
            parts["instruction_paraphrase_fraction"] = (
                positive_available.float().mean()
            )
        return (
            anchor,
            rank,
            parts,
            correct,
            wrong,
            positive,
            positive_available,
        )

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
        condition = model.encode_condition(batch)
        if condition is None:
            raise ValueError("real Prior training requires language condition")
        with torch.no_grad():
            history = model.encode_history(batch)
            _, target = model.encode_targets(batch)
            posterior = posterior_from_targets(
                model,
                batch,
                history,
                target,
                future_scale,
                condition.detach(),
            )[0]
        context = model.prior_context(
            history,
            future_scale,
            history_scale,
            condition,
            batch.get("condition_tokens"),
            batch.get("condition_token_valid"),
        )
        if model.config.canonical_activity_gate:
            confidence = (
                history["activity"][:, -1, None]
                * target["activity"]
            ).float().clamp(0.0, 1.0).pow(
                model.config.canonical_activity_power
            ).detach()
            if confidence.shape != posterior.shape[:-1]:
                raise ValueError("activity and object action tokens must align")
            action_weight = (
                self.action_activity_floor
                + (1.0 - self.action_activity_floor) * confidence
            )
        else:
            action_weight = torch.ones_like(posterior[..., 0])
        flow, predicted_actions, target_actions = flow_training_objective(
            model.latent_actions.prior,
            posterior,
            context,
            weight=action_weight,
        )
        language_effect = flow.new_zeros(())
        language_parts = {}
        task_semantic = flow.new_zeros(())
        task_semantic_parts = {}
        if self.task_semantic_contrast_weight > 0.0:
            conditioner = model.latent_actions.prior_token_conditioner
            if conditioner is None:
                raise ValueError("task contrast requires token conditioner")
            task_semantic, task_semantic_parts = task_semantic_contrast(
                conditioner,
                self.paraphrase_tokens,
                self.paraphrase_token_valid,
                self.condition_task_index,
            )
        if model.language_effect_alignment is not None:
            if "condition_index" not in batch:
                raise ValueError(
                    "Prior language alignment requires condition ids"
                )
            language_effect, language_parts = model.language_effect_alignment(
                condition,
                model.latent_actions.predict_effect(predicted_actions),
                batch["condition_index"],
            )
        anchor = flow.new_zeros(())
        rank = flow.new_zeros(())
        contrast_parts = {}
        correct_actions = None
        wrong_actions = None
        positive_actions = None
        positive_available = None
        if self.instruction_contrast_enabled:
            (
                anchor,
                rank,
                contrast_parts,
                correct_actions,
                wrong_actions,
                positive_actions,
                positive_available,
            ) = self._instruction_contrast(
                batch,
                history,
                history_scale,
                future_scale,
                condition,
                context,
                target_actions,
                action_weight,
            )
        dynamics_condition = (
            None if model.config.token_conditioned_prior else condition
        )
        paraphrase = flow.new_zeros(())
        if self.effect_weight > 0.0 and self.dynamics_relative_objective:
            effect_actions = (
                correct_actions
                if correct_actions is not None
                else predicted_actions
            )
            action_variants = {"effect": effect_actions}
            if correct_actions is not None and wrong_actions is not None:
                action_variants.update(
                    wrong=wrong_actions,
                )
            if positive_actions is not None:
                action_variants["positive"] = positive_actions
            relative = relative_dynamics_effects(
                model,
                history,
                history_scale,
                future_scale,
                action_variants,
                target_actions,
                dynamics_condition,
            )
            effect = relative["effect"]["total"]
            effect_parts = {
                f"effect_{name}": relative["effect"][name]
                for name in ("slot", "feature", "center")
            }
            if self.instruction_rank_weight > 0.0:
                if "wrong" not in relative:
                    raise ValueError(
                        "relative instruction rank requires contrast actions"
                    )
                action_rank = rank
                requested_margin = (
                    self.instruction_relative_margin
                    * relative["_zero"]["per_sample"]
                )
                correct_effect = relative["effect"]["per_sample"]
                wrong_effect = relative["wrong"]["per_sample"]
                rank = torch.relu(
                    correct_effect + requested_margin - wrong_effect
                ).mean()
                effect_rank = rank
                rank = effect_rank + action_rank
                contrast_parts.update(
                    instruction_action_rank=action_rank,
                    instruction_effect_rank=effect_rank,
                    instruction_correct_effect=correct_effect.mean(),
                    instruction_wrong_effect=wrong_effect.mean(),
                    instruction_effect_relative_advantage=(
                        (
                            wrong_effect - correct_effect
                        ) / relative["_zero"]["per_sample"].clamp_min(1e-8)
                    ).mean(),
                )
            if positive_actions is not None:
                if positive_available is None:
                    raise AssertionError("paraphrase mask is missing")
                positive_effect = relative["positive"]["per_sample"]
                paraphrase = (
                    positive_effect[positive_available].mean()
                    if bool(positive_available.any())
                    else positive_effect.sum() * 0.0
                )
                contrast_parts["instruction_paraphrase_effect"] = paraphrase
        elif self.effect_weight > 0.0:
            effect, effect_parts = dynamics_effect_loss(
                model,
                history,
                history_scale,
                future_scale,
                predicted_actions,
                target_actions,
                dynamics_condition,
            )
        else:
            effect = flow.new_zeros(())
            effect_parts = {
                "effect_slot": effect,
                "effect_feature": effect,
                "effect_center": effect,
            }
        endpoint_error = (
            (predicted_actions - target_actions).square().mean(dim=-1)
        )
        endpoint_mse = (
            (endpoint_error * action_weight).sum()
            / action_weight.sum().clamp_min(1e-6)
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
                + model.config.language_effect_weight * language_effect
                + self.instruction_anchor_weight * anchor
                + self.instruction_rank_weight * rank
                + self.paraphrase_positive_weight * paraphrase
                + self.task_semantic_contrast_weight * task_semantic
            ),
            "flow": flow,
            "effect": effect,
            "language_effect": language_effect,
            "instruction_anchor": anchor,
            "instruction_rank": rank,
            "instruction_paraphrase": paraphrase,
            "task_semantic_contrast": task_semantic,
            "endpoint_mse": endpoint_mse,
            "canonical_endpoint_mse": canonical_mse,
            "context_rms": context.square().mean().sqrt(),
            "action_weight_mean": action_weight.mean(),
            "prior_token_gate": (
                torch.tanh(model.latent_actions.prior_token_conditioner.gate)
                if model.latent_actions.prior_token_conditioner is not None
                else flow.new_zeros(())
            ),
            **effect_parts,
            **language_parts,
            **contrast_parts,
            **task_semantic_parts,
        }
