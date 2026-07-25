"""Strictly separated online, target, posterior, and prior world-model paths."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
from .action_embedding import residual_action_dropout
from .conditioning import LanguageConditionProjector
from .config import AdaptiveGaussianWMConfig
from .decoder import GaussianReadout
from .dynamics import JointObjectLatentDynamics
from .gpstoken import GPSTokenState, LearnableGPSTokenAllocator
from .latent_action import LatentActionModel
from .language_effect import LanguageEffectAlignment
from .object_slots import ObjectSlotAggregator, ObjectSlotState
from .observed_action import posterior_from_targets
from .readout_runtime import decode_gaussian_readout, residual_future_features
from .rgb_supervision import residual_future_rgb, render_current_rgb, render_future_rgb
from .scale import signed_gap_scale

class AdaptiveGaussianObjectWorldModel(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.config = config
        self.allocator = LearnableGPSTokenAllocator(config)
        self.object_aggregator = ObjectSlotAggregator(config)
        self.target_allocator = copy.deepcopy(self.allocator)
        self.target_object_aggregator = copy.deepcopy(self.object_aggregator)
        for parameter in self.target_allocator.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_object_aggregator.parameters():
            parameter.requires_grad_(False)
        self.latent_actions = LatentActionModel(config)
        self.dynamics = JointObjectLatentDynamics(config)
        self.gaussian_readout = GaussianReadout(config)
        self.language_condition = (
            LanguageConditionProjector(config)
            if config.condition_dim > 0
            else None
        )
        self.language_effect_alignment = (
            LanguageEffectAlignment(config)
            if config.language_effect_weight > 0.0
            else None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_allocator.eval()
        self.target_object_aggregator.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        value = self.config.target_momentum if momentum is None else momentum
        if not 0.0 <= value < 1.0:
            raise ValueError("target momentum must be in [0, 1)")
        pairs = (
            (self.target_allocator, self.allocator),
            (self.target_object_aggregator, self.object_aggregator),
        )
        for target, online in pairs:
            for target_parameter, online_parameter in zip(
                target.parameters(),
                online.parameters(),
                strict=True,
            ):
                target_parameter.lerp_(online_parameter, 1.0 - value)
            for target_buffer, online_buffer in zip(
                target.buffers(),
                online.buffers(),
                strict=True,
            ):
                target_buffer.copy_(online_buffer)

    @staticmethod
    def _validate_sequence(
        features: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,T,N,C]")
        if coordinates.shape != (*features.shape[:3], 2):
            raise ValueError("coordinates must have shape [B,T,N,2]")
        if valid.shape != features.shape[:3]:
            raise ValueError("valid must have shape [B,T,N]")

    @staticmethod
    def _encode_sequence(
        features: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        allocator: LearnableGPSTokenAllocator,
        aggregator: ObjectSlotAggregator,
    ) -> dict:
        AdaptiveGaussianObjectWorldModel._validate_sequence(
            features,
            coordinates,
            valid,
        )
        token_states: list[GPSTokenState] = []
        slot_states: list[ObjectSlotState] = []
        anchor_slots = None
        anchor_centers = None
        for index in range(features.shape[1]):
            token_state = allocator(
                features[:, index],
                coordinates[:, index],
                valid[:, index],
            )
            slot_state = aggregator(
                token_state,
                anchor_slots,
                anchor_centers,
            )
            if anchor_slots is None:
                anchor_slots = slot_state.tracking_slots
                anchor_centers = slot_state.center
            token_states.append(token_state)
            slot_states.append(slot_state)
        return {
            "token_states": token_states,
            "slot_states": slot_states,
            "slots": torch.stack([state.slots for state in slot_states], dim=1),
            "tracking_slots": torch.stack(
                [state.tracking_slots for state in slot_states],
                dim=1,
            ),
            "activity": torch.stack(
                [state.activity for state in slot_states],
                dim=1,
            ),
            "center": torch.stack([state.center for state in slot_states], dim=1),
        }

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict:
        """Deployment path: this method reads history fields only."""
        return self._encode_sequence(
            batch["history_features"],
            batch["history_coordinates"],
            batch["history_valid"],
            self.allocator,
            self.object_aggregator,
        )

    def encode_condition(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if self.language_condition is None:
            return None
        if "condition_feature" not in batch:
            raise ValueError("language-conditioned model requires condition_feature")
        return self.language_condition(batch["condition_feature"])

    @torch.no_grad()
    def encode_targets(self, batch: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        """EMA target path; future information never reaches the online encoder."""
        target_history = self._encode_sequence(
            batch["history_features"],
            batch["history_coordinates"],
            batch["history_valid"],
            self.target_allocator,
            self.target_object_aggregator,
        )
        target_history["feature"] = torch.stack(
            [
                state.decoded_feature
                for state in target_history["slot_states"]
            ],
            dim=1,
        )
        anchor_slots = target_history["slot_states"][0].tracking_slots
        anchor_centers = target_history["slot_states"][0].center
        token_states = []
        slot_states = []
        for index in range(batch["future_features"].shape[1]):
            token_state = self.target_allocator(
                batch["future_features"][:, index],
                batch["future_coordinates"][:, index],
                batch["future_valid"][:, index],
            )
            slot_state = self.target_object_aggregator(
                token_state,
                anchor_slots,
                anchor_centers,
            )
            token_states.append(token_state)
            slot_states.append(slot_state)
        target_future = {
            "token_states": token_states,
            "slot_states": slot_states,
            "slots": torch.stack([state.slots for state in slot_states], dim=1),
            "tracking_slots": torch.stack(
                [state.tracking_slots for state in slot_states],
                dim=1,
            ),
            "activity": torch.stack(
                [state.activity for state in slot_states],
                dim=1,
            ),
            "center": torch.stack([state.center for state in slot_states], dim=1),
            "feature": torch.stack(
                [state.decoded_feature for state in slot_states],
                dim=1,
            ),
        }
        return target_history, target_future

    def make_history_mask(self, history_slots: torch.Tensor) -> torch.Tensor:
        shape = history_slots.shape[:3]
        mask = torch.zeros(shape, device=history_slots.device, dtype=torch.bool)
        if shape[1] == 1 or self.config.history_mask_ratio == 0.0:
            return mask
        mask = torch.rand(shape, device=history_slots.device) < (
            self.config.history_mask_ratio
        )
        mask[:, 0] = False
        missing = ~mask.flatten(1).any(dim=1)
        mask[missing, -1, 0] = True
        return mask

    def prior_context(
        self,
        history: dict,
        future_scale: torch.Tensor,
        history_scale: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        condition_tokens: torch.Tensor | None = None,
        condition_token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.latent_actions.prior_context(
            history["slots"],
            history["activity"],
            future_scale,
            history["center"],
            history_scale,
            condition,
            condition_tokens,
            condition_token_valid,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        history_mask: torch.Tensor | None = None,
        use_posterior: bool = True,
        phase: str = "joint",
        loss_weights=None,
        actions_override: torch.Tensor | None = None,
    ) -> dict:
        if phase == "representation":
            from .training import representation_pretrain_loss

            loss, parts = representation_pretrain_loss(self, batch)
            return {"loss": loss, "parts": parts}
        compute_joint_loss = phase in ("joint_loss", "posterior_dynamics_loss")
        posterior_dynamics_loss = phase == "posterior_dynamics_loss"
        if phase not in ("joint", "joint_loss", "posterior_dynamics_loss"):
            raise ValueError(f"unknown training phase: {phase}")
        history = self.encode_history(batch)
        condition = self.encode_condition(batch)
        target_history, target_future = self.encode_targets(batch)
        history_scale = signed_gap_scale(
            batch["history_times"],
            self.config.gap_reference,
        )
        future_scale = signed_gap_scale(
            batch["future_times"],
            self.config.gap_reference,
        )
        posterior_actions, action_rgb = posterior_from_targets(
            self,
            batch,
            history,
            target_future,
            future_scale,
            condition,
        )
        prior_context = self.prior_context(
            history,
            future_scale,
            history_scale,
            condition,
            batch.get("condition_tokens"),
            batch.get("condition_token_valid"),
        )
        if actions_override is not None:
            if actions_override.shape != posterior_actions.shape:
                raise ValueError("actions_override must match posterior actions")
            actions = actions_override
        elif use_posterior:
            actions = posterior_actions
        else:
            actions = self.latent_actions.prior.sample(
                prior_context,
                sample_count=1,
                stochastic=True,
            )[0]
        actions = residual_action_dropout(actions, self.config.action_residual_dropout, self.training)
        if history_mask is None:
            history_mask = self.make_history_mask(history["slots"])
        future_output = self.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions,
            history_mask,
            history["center"],
            condition,
        )
        history_output = self.dynamics(
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            torch.zeros_like(actions),
            history_mask,
            history["center"],
            condition,
        )
        predicted_future_centers = (
            future_output.future_centers
            if future_output.future_centers is not None
            else self.object_aggregator.decode_center(
                future_output.future_slots
            )
        )
        predicted_history_centers = (
            history_output.history_centers
            if history_output.history_centers is not None
            else self.object_aggregator.decode_center(
                history_output.history_slots
            )
        )
        zero_action_future_centers = (
            history_output.future_centers
            if history_output.future_centers is not None
            else self.object_aggregator.decode_center(
                history_output.future_slots
            )
        )
        current_tokens = history["token_states"][-1]
        current_slot_state = history["slot_states"][-1]
        readout, readout_context = decode_gaussian_readout(
            self,
            batch,
            current_tokens,
            current_slot_state,
            future_output.future_slots,
            predicted_future_centers,
        )
        future_count = future_output.future_slots.shape[1]
        current_readout, _ = decode_gaussian_readout(
            self,
            batch,
            current_tokens,
            current_slot_state,
            current_slot_state.slots[:, None].expand(
                -1, future_count, -1, -1
            ),
            current_slot_state.center[:, None].expand(
                -1, future_count, -1, -1
            ),
            readout_context.micro_rgb,
        )
        direct_rendered, coverage = self.gaussian_readout.splat_features(
            readout,
            batch["future_coordinates"],
        )
        residual_reference_features = self.gaussian_readout.splat_features(
            current_readout,
            batch["future_coordinates"],
        )[0]
        rendered = residual_future_features(
            direct_rendered,
            residual_reference_features,
            batch,
        )
        rendered_rgb = None
        rgb_coverage = None
        rendered_current_rgb = None
        current_rgb_coverage = None
        residual_reference_rgb = None
        if self.config.rgb_supervision:
            direct_rendered_rgb, rgb_coverage = render_future_rgb(
                readout,
                batch,
                self.config.rgb_render_chunk,
            )
            residual_reference_rgb, reference_coverage = render_current_rgb(
                current_readout,
                batch,
                self.config.rgb_render_chunk,
            )
            rendered_rgb = residual_future_rgb(
                direct_rendered_rgb,
                residual_reference_rgb,
                batch,
            )
            if not posterior_dynamics_loss:
                rendered_current_rgb = residual_reference_rgb[:, :1]
                current_rgb_coverage = reference_coverage[:, :1]
        result = {
            "predicted_future_slots": future_output.future_slots,
            "predicted_future_centers": predicted_future_centers,
            "predicted_future_object_features": (
                self.object_aggregator.decode_feature(
                    future_output.future_slots
                )
            ),
            "zero_action_future_slots": history_output.future_slots,
            "zero_action_future_centers": zero_action_future_centers,
            "predicted_history_slots": history_output.history_slots,
            "predicted_history_centers": predicted_history_centers,
            "target_future_slots": target_future["slots"],
            "target_future_activity": target_future["activity"],
            "target_future_centers": target_future["center"],
            "target_future_object_features": target_future["feature"],
            "current_object_rgb": action_rgb[0],
            "target_future_object_rgb": action_rgb[1],
            "target_history_slots": target_history["slots"],
            "target_history_activity": target_history["activity"],
            "target_history_centers": target_history["center"],
            "target_history_object_features": target_history["feature"],
            "online_history_slots": history["slots"],
            "online_history_centers": history["center"],
            "online_history_object_features": torch.stack(
                [
                    state.decoded_feature
                    for state in history["slot_states"]
                ],
                dim=1,
            ),
            "posterior_actions": posterior_actions,
            "dynamics_actions": actions,
            "prior_context": prior_context,
            "history_mask": history_mask,
            "gaussian_readout": readout,
            "rendered_future_features": rendered,
            "residual_reference_features": residual_reference_features,
            "render_coverage": coverage,
            "rendered_future_rgb": rendered_rgb,
            "residual_reference_rgb": residual_reference_rgb,
            "rgb_render_coverage": rgb_coverage,
            "rendered_current_rgb": rendered_current_rgb,
            "current_rgb_render_coverage": current_rgb_coverage,
            "language_condition": condition,
            "history_token_states": history["token_states"],
            "history_slot_states": history["slot_states"],
        }
        if compute_joint_loss:
            if loss_weights is None:
                raise ValueError("joint_loss phase requires loss_weights")
            from .losses import adaptive_world_model_loss

            loss, parts = adaptive_world_model_loss(
                self,
                batch,
                result,
                loss_weights,
            )
            result["loss"] = loss
            result["parts"] = parts
        return result

    def predict_prior_states(
        self,
        batch: dict[str, torch.Tensor],
        sample_count: int,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .inference import predict_prior_states

        return predict_prior_states(self, batch, sample_count, stochastic)

    def predict_prior(
        self,
        batch: dict[str, torch.Tensor],
        sample_count: int,
        stochastic: bool = True,
    ) -> torch.Tensor:
        return self.predict_prior_states(
            batch,
            sample_count,
            stochastic,
        )[0]

    def predict_prior_features(
        self,
        batch: dict[str, torch.Tensor],
        sample_count: int,
        stochastic: bool = True,
    ) -> torch.Tensor:
        from .inference import predict_prior_features

        return predict_prior_features(self, batch, sample_count, stochastic)
