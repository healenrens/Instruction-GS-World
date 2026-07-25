"""Joint non-causal prediction of future object latents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .action_embedding import bounded_action_embedding, gate_canonical_center
from .config import AdaptiveGaussianWMConfig
from .scale import ScaleModulatedBlock, inverse_signed_gap_scale


@dataclass
class JointDynamicsOutput:
    future_slots: torch.Tensor
    history_slots: torch.Tensor
    future_centers: torch.Tensor | None
    history_centers: torch.Tensor | None

class JointObjectLatentDynamics(nn.Module):
    """Predict all requested future object states in one Transformer pass."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.model_dim
        self.config = config
        self.object_aligned_actions = config.object_aligned_actions
        self.decoupled_jepa_slots = config.decoupled_jepa_slots
        self.action_query_modulation = config.action_query_modulation
        self.bounded_residual_action = config.bounded_residual_action
        self.action_residual_gate = config.action_residual_gate
        self.action_film_modulation = config.action_film_modulation
        self.kinematic_action_modulation = config.kinematic_action_modulation
        self.learned_velocity_baseline = config.learned_velocity_baseline
        self.slot_input = nn.Linear(config.object_dim, dim)
        self.identity_input = nn.Linear(config.object_dim, dim)
        self.activity_input = nn.Linear(1, dim)
        self.action_input = nn.Linear(
            6 if self.bounded_residual_action else config.action_dim,
            dim,
            bias=not self.bounded_residual_action,
        )
        self.residual_action_input = (
            nn.Linear(config.action_residual_dim, dim, bias=False)
            if self.bounded_residual_action and config.action_residual_dim > 0
            else None
        )
        self.condition_input = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
            if config.condition_dim > 0
            else None
        )
        self.center_input = (
            nn.Linear(2, dim)
            if self.decoupled_jepa_slots
            else None
        )
        self.history_type = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.future_type = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.action_type = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.history_mask_token = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.future_query = nn.Parameter(torch.randn(dim) / dim**0.5)
        self.blocks = nn.ModuleList(
            [
                ScaleModulatedBlock(dim, config.heads, config.dropout)
                for _ in range(config.dynamics_layers)
            ]
        )
        self.action_modulations = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * 6),
                    )
                    for _ in range(config.dynamics_layers)
                ]
            )
            if self.action_film_modulation
            else None
        )
        if self.action_modulations is not None:
            for modulation in self.action_modulations:
                nn.init.zeros_(modulation[-1].weight)
                nn.init.zeros_(modulation[-1].bias)
        self.condition_modulations = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * 6),
                    )
                    for _ in range(config.dynamics_layers)
                ]
            )
            if config.condition_dim > 0
            else None
        )
        if self.condition_modulations is not None:
            for modulation in self.condition_modulations:
                nn.init.zeros_(modulation[-1].weight)
                nn.init.zeros_(modulation[-1].bias)
        self.output_norm = nn.LayerNorm(dim)
        self.slot_output = nn.Linear(dim, config.object_dim)
        nn.init.normal_(self.slot_output.weight, std=1e-3)
        nn.init.zeros_(self.slot_output.bias)
        self.center_output = (
            nn.Linear(dim, 2)
            if self.decoupled_jepa_slots
            else None
        )
        if self.center_output is not None:
            nn.init.zeros_(self.center_output.weight)
            nn.init.zeros_(self.center_output.bias)
        if self.kinematic_action_modulation:
            state_dim = config.object_dim + 2
            center_state_dim = state_dim + int(self.learned_velocity_baseline) * 2
            action_dim = config.action_tokens * config.action_dim + 1
            def residual_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(input_dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, output_dim),
                )
            self.slot_motion_basis = residual_mlp(state_dim, config.object_dim)
            self.slot_action_gate = residual_mlp(action_dim, config.object_dim)
            self.center_motion_basis = residual_mlp(center_state_dim, 2)
            self.center_action_gate = residual_mlp(action_dim, 2)
            for gate in (self.slot_action_gate, self.center_action_gate):
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
            self.velocity_scale = (
                nn.Parameter(torch.zeros(()))
                if self.learned_velocity_baseline
                else None
            )
        else:
            self.slot_motion_basis = None
            self.slot_action_gate = None
            self.center_motion_basis = None
            self.center_action_gate = None
            self.velocity_scale = None

    def _kinematic_residuals(
        self,
        history_slots: torch.Tensor,
        history_scale: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
        history_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(
            module is None
            for module in (
                self.slot_motion_basis,
                self.slot_action_gate,
                self.center_motion_basis,
                self.center_action_gate,
            )
        ):
            raise ValueError("kinematic residual modules are not initialized")
        history_time = inverse_signed_gap_scale(
            history_scale,
            self.config.gap_reference,
        )
        future_time = inverse_signed_gap_scale(
            future_scale,
            self.config.gap_reference,
        )
        if history_slots.shape[1] > 1:
            reference_index = 0 if self.learned_velocity_baseline else -2
            time_delta = (
                history_time[:, -1] - history_time[:, reference_index]
            ).clamp_min(1e-6)
            velocity = (
                history_centers[:, -1]
                - history_centers[:, reference_index]
            ) / time_delta[:, None, None]
        else:
            velocity = torch.zeros_like(history_centers[:, -1])
        current_state = torch.cat(
            (history_slots[:, -1], history_centers[:, -1]),
            dim=-1,
        )
        center_state = (
            torch.cat((current_state, velocity), dim=-1)
            if self.learned_velocity_baseline
            else current_state
        )
        action_state = torch.cat(
            (actions.flatten(-2), future_scale[..., None]),
            dim=-1,
        )
        slot_basis = torch.tanh(self.slot_motion_basis(current_state))
        slot_gate = torch.tanh(self.slot_action_gate(action_state))
        slot_residual = slot_gate[:, :, None] * slot_basis[:, None]
        center_basis = torch.tanh(self.center_motion_basis(center_state))
        center_gate = torch.tanh(self.center_action_gate(action_state))
        center_residual = (
            0.5 * center_gate[:, :, None] * center_basis[:, None]
        )

        prediction_gap = future_time - history_time[:, -1, None]
        if self.learned_velocity_baseline:
            if self.velocity_scale is None:
                raise ValueError("learned velocity scale is not initialized")
            velocity_weight = torch.tanh(self.velocity_scale)
        else:
            velocity_weight = prediction_gap.new_tensor(1.0)
        center_base = (
            history_centers[:, -1, None]
            + velocity_weight
            * prediction_gap[..., None, None]
            * velocity[:, None]
        )
        return slot_residual, center_base, center_residual

    def forward(
        self,
        history_slots: torch.Tensor,
        history_activity: torch.Tensor,
        history_scale: torch.Tensor,
        future_scale: torch.Tensor,
        actions: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        history_centers: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> JointDynamicsOutput:
        batch, history_count, object_count, _ = history_slots.shape
        future_count = future_scale.shape[1]
        expected_action = (
            batch,
            future_count,
            self.config.action_tokens,
            self.config.action_dim,
        )
        if actions.shape != expected_action:
            raise ValueError(
                f"actions must have shape {expected_action}, got {actions.shape}"
            )
        if history_scale.shape != (batch, history_count):
            raise ValueError("history_scale must have shape [B,T]")
        if history_activity.shape != history_slots.shape[:3]:
            raise ValueError("history_activity must have shape [B,T,K]")
        if history_mask is None:
            history_mask = torch.zeros(
                history_slots.shape[:3],
                device=history_slots.device,
                dtype=torch.bool,
            )
        if history_mask.shape != history_slots.shape[:3]:
            raise ValueError("history_mask must have shape [B,T,K]")
        condition_token = None
        if condition is not None:
            if self.condition_input is None:
                raise ValueError("condition was provided to an unconditioned Dynamics")
            if condition.shape != (batch, self.config.model_dim):
                raise ValueError("condition must have shape [B,D]")
            condition_token = self.condition_input(condition)

        anchor = self.identity_input(history_slots[:, 0])
        identity = anchor[:, None].expand(-1, history_count, -1, -1)
        observed = (
            self.slot_input(history_slots)
            + identity
            + self.activity_input(history_activity[..., None])
            + self.history_type
        )
        if self.center_input is not None:
            if history_centers is None:
                raise ValueError("decoupled Dynamics requires history centers")
            if history_centers.shape != (*history_slots.shape[:3], 2):
                raise ValueError("history_centers must have shape [B,T,K,2]")
            observed = observed + self.center_input(history_centers)
        masked = self.history_mask_token + identity + self.history_type
        history_tokens = torch.where(history_mask[..., None], masked, observed)

        if self.bounded_residual_action:
            canonical_actions = gate_canonical_center(
                actions[..., :6], self.config.canonical_center_gate
            )
            canonical_condition = bounded_action_embedding(
                self.action_input,
                canonical_actions,
            )
            action_condition = canonical_condition
            if self.residual_action_input is not None:
                residual_condition = bounded_action_embedding(
                    self.residual_action_input, actions[..., 6:]
                )
                action_condition = action_condition + (
                    self.action_residual_gate * residual_condition
                )
        else:
            action_condition = self.action_input(
                gate_canonical_center(actions, self.config.canonical_center_gate)
            )
        future_tokens = (
            self.future_query
            + self.future_type
            + anchor[:, None].expand(-1, future_count, -1, -1)
        )
        if self.action_query_modulation:
            query_action = (
                action_condition
                if self.object_aligned_actions
                else action_condition.mean(dim=2)[:, :, None]
            )
            future_tokens = future_tokens + query_action
        action_tokens = action_condition + self.action_type
        if condition_token is not None:
            future_tokens = future_tokens + condition_token[:, None, None]
            action_tokens = action_tokens + condition_token[:, None, None]
        if self.object_aligned_actions:
            action_tokens = action_tokens + anchor[:, None]

        history_tokens = history_tokens.reshape(
            batch,
            history_count * object_count,
            -1,
        )
        future_tokens = future_tokens.reshape(
            batch,
            future_count * object_count,
            -1,
        )
        action_tokens = action_tokens.reshape(
            batch,
            future_count * self.config.action_tokens,
            -1,
        )
        tokens = torch.cat((history_tokens, future_tokens, action_tokens), dim=1)

        history_token_scale = history_scale[:, :, None].expand(
            -1,
            -1,
            object_count,
        )
        future_token_scale = future_scale[:, :, None].expand(
            -1,
            -1,
            object_count,
        )
        action_token_scale = future_scale[:, :, None].expand(
            -1,
            -1,
            self.config.action_tokens,
        )
        scale = torch.cat(
            (
                history_token_scale.reshape(batch, -1),
                future_token_scale.reshape(batch, -1),
                action_token_scale.reshape(batch, -1),
            ),
            dim=1,
        )
        film_condition = None
        if self.action_modulations is not None:
            future_condition = (
                action_condition
                if self.object_aligned_actions
                else action_condition.mean(dim=2)[:, :, None].expand(
                    -1,
                    -1,
                    object_count,
                    -1,
                )
            ).reshape(batch, future_count * object_count, -1)
            film_condition = torch.cat(
                (
                    torch.zeros_like(history_tokens),
                    future_condition,
                    torch.zeros_like(action_tokens),
                ),
                dim=1,
            )
        condition_film = None
        if condition_token is not None:
            future_condition = condition_token[:, None, None].expand(
                -1,
                future_count,
                object_count,
                -1,
            ).reshape(batch, future_count * object_count, -1)
            action_language = condition_token[:, None, None].expand(
                -1,
                future_count,
                self.config.action_tokens,
                -1,
            ).reshape(batch, future_count * self.config.action_tokens, -1)
            condition_film = torch.cat(
                (
                    torch.zeros_like(history_tokens),
                    future_condition,
                    action_language,
                ),
                dim=1,
            )
        for index, block in enumerate(self.blocks):
            action_modulation = (
                self.action_modulations[index](film_condition)
                if self.action_modulations is not None
                and film_condition is not None
                else None
            )
            language_modulation = (
                self.condition_modulations[index](condition_film)
                if self.condition_modulations is not None
                and condition_film is not None
                else None
            )
            if action_modulation is None:
                extra_modulation = language_modulation
            elif language_modulation is None:
                extra_modulation = action_modulation
            else:
                extra_modulation = action_modulation + language_modulation
            tokens = block(
                tokens,
                scale,
                extra_modulation=extra_modulation,
            )
        normalized_tokens = self.output_norm(tokens)
        token_delta = self.slot_output(normalized_tokens)

        history_end = history_count * object_count
        future_end = history_end + future_count * object_count
        history_delta = token_delta[:, :history_end].reshape(
            batch,
            history_count,
            object_count,
            -1,
        )
        future_delta = token_delta[:, history_end:future_end].reshape(
            batch,
            future_count,
            object_count,
            -1,
        )
        future_center_base = (
            history_centers[:, -1, None]
            if history_centers is not None
            else None
        )
        modulated_center_delta = None
        if self.kinematic_action_modulation:
            if history_centers is None:
                raise ValueError("kinematic modulation requires history centers")
            (
                modulated_slot_delta,
                future_center_base,
                modulated_center_delta,
            ) = self._kinematic_residuals(
                history_slots,
                history_scale,
                future_scale,
                actions,
                history_centers,
            )
            future_delta = future_delta + modulated_slot_delta
        predicted_history = history_slots + history_delta
        predicted_future = history_slots[:, -1, None] + future_delta
        predicted_history_centers = None
        predicted_future_centers = None
        if self.center_output is not None:
            raw_center_delta = 0.5 * torch.tanh(
                self.center_output(normalized_tokens)
            )
            history_center_delta = raw_center_delta[:, :history_end].reshape(
                batch,
                history_count,
                object_count,
                2,
            )
            future_center_delta = raw_center_delta[
                :, history_end:future_end
            ].reshape(
                batch,
                future_count,
                object_count,
                2,
            )
            if modulated_center_delta is not None:
                future_center_delta = (
                    future_center_delta + modulated_center_delta
                )
            predicted_history_centers = (
                history_centers + history_center_delta
            )
            predicted_future_centers = future_center_base + future_center_delta
        return JointDynamicsOutput(
            future_slots=predicted_future,
            history_slots=predicted_history,
            future_centers=predicted_future_centers,
            history_centers=predicted_history_centers,
        )
