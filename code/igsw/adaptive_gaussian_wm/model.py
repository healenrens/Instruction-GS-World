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
from .dynamics_runtime import factorized_result_fields, run_object_dynamics
from .factorized_dynamics import FactorizedObjectDynamics
from .gpstoken import LearnableGPSTokenAllocator
from .latent_action import LatentActionModel
from .language_effect import LanguageEffectAlignment
from .model_phases import joint_phase_flags, select_dynamics_actions
from .object_memory import ObjectMemoryTransition
from .object_slots import ObjectSlotAggregator
from .observed_action import posterior_from_targets
from .readout_runtime import decode_gaussian_readout, residual_future_features
from .rgb_supervision import residual_future_rgb, render_current_rgb, render_future_rgb
from .scale import signed_gap_scale
from .sequence_encoding import encode_visual_sequence
class AdaptiveGaussianObjectWorldModel(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.config = config
        self.allocator = LearnableGPSTokenAllocator(config)
        self.object_aggregator = ObjectSlotAggregator(config)
        self.target_allocator = copy.deepcopy(self.allocator)
        self.target_object_aggregator = copy.deepcopy(self.object_aggregator)
        self.object_memory = (
            ObjectMemoryTransition(config)
            if config.persistent_object_memory
            else None
        )
        self.target_object_memory = (
            copy.deepcopy(self.object_memory)
            if self.object_memory is not None
            else None
        )
        for parameter in self.target_allocator.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_object_aggregator.parameters():
            parameter.requires_grad_(False)
        if self.target_object_memory is not None:
            for parameter in self.target_object_memory.parameters():
                parameter.requires_grad_(False)
        self.latent_actions = LatentActionModel(config)
        self.dynamics = (
            FactorizedObjectDynamics(config)
            if config.factorized_dynamics
            else JointObjectLatentDynamics(config)
        )
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
        if self.target_object_memory is not None:
            self.target_object_memory.eval()
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
        if self.object_memory is not None and self.target_object_memory is not None:
            pairs = pairs + ((self.target_object_memory, self.object_memory),)
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

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict:
        """Deployment path: this method reads history fields only."""
        return encode_visual_sequence(
            batch["history_features"],
            batch["history_coordinates"],
            batch["history_valid"],
            batch["history_times"],
            self.allocator,
            self.object_aggregator,
            self.object_memory,
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
        target_history = encode_visual_sequence(
            batch["history_features"],
            batch["history_coordinates"],
            batch["history_valid"],
            batch["history_times"],
            self.target_allocator,
            self.target_object_aggregator,
            self.target_object_memory,
        )
        target_future = encode_visual_sequence(
            batch["future_features"],
            batch["future_coordinates"],
            batch["future_valid"],
            batch["future_times"],
            self.target_allocator,
            self.target_object_aggregator,
            self.target_object_memory,
            initial_memory=target_history["last_memory"],
            previous_time=batch["history_times"][:, -1],
            initial_anchor_slots=target_history.get("legacy_anchor_slots"),
            initial_anchor_centers=target_history.get("legacy_anchor_centers"),
        )
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
        compute_joint_loss, posterior_dynamics_loss, action_free = (
            joint_phase_flags(phase, self.config.architecture)
        )
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
        actions = select_dynamics_actions(
            self,
            posterior_actions,
            prior_context,
            use_posterior,
            actions_override,
            action_free,
        )
        actions = residual_action_dropout(
            actions,
            self.config.action_residual_dropout,
            self.training,
            self.config.canonical_action_dim,
        )
        if history_mask is None:
            history_mask = self.make_history_mask(history["slots"])
        dynamics_memory = {
            "history_relative_scale": history.get("relative_scale"),
            "history_relative_disparity": history.get("relative_disparity"),
            "history_relations": history.get("relations"),
            "history_existence": history.get("existence"),
        }
        future_output = run_object_dynamics(
            self,
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            actions,
            history_mask,
            history["center"],
            condition,
            **dynamics_memory,
        )
        history_output = run_object_dynamics(
            self,
            history["slots"],
            history["activity"],
            history_scale,
            future_scale,
            torch.zeros_like(actions),
            history_mask,
            history["center"],
            condition,
            **dynamics_memory,
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
            predicted_relative_scale=getattr(
                future_output, "future_relative_scale", None
            ),
            predicted_relative_disparity=getattr(
                future_output, "future_relative_disparity", None
            ),
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
            "target_future_visibility": target_future.get(
                "visibility", target_future["activity"]
            ),
            "target_future_existence": target_future.get(
                "existence", target_future["activity"]
            ),
            "target_future_in_frame": target_future.get(
                "in_frame", target_future["activity"]
            ),
            "target_future_relative_scale": target_future.get("relative_scale"),
            "target_future_relative_disparity": target_future.get(
                "relative_disparity"
            ),
            "target_future_relations": target_future.get("relations"),
            "target_future_centers": target_future["center"],
            "target_future_object_features": target_future["feature"],
            "current_object_rgb": action_rgb[0],
            "target_future_object_rgb": action_rgb[1],
            "target_history_slots": target_history["slots"],
            "target_history_activity": target_history["activity"],
            "target_history_visibility": target_history.get(
                "visibility", target_history["activity"]
            ),
            "target_history_existence": target_history.get(
                "existence", target_history["activity"]
            ),
            "target_history_in_frame": target_history.get(
                "in_frame", target_history["activity"]
            ),
            "target_history_relative_scale": target_history.get("relative_scale"),
            "target_history_relative_disparity": target_history.get(
                "relative_disparity"
            ),
            "target_history_relations": target_history.get("relations"),
            "target_history_centers": target_history["center"],
            "target_history_object_features": target_history["feature"],
            "online_history_slots": history["slots"],
            "online_history_centers": history["center"],
            "online_history_visibility": history.get(
                "visibility", history["activity"]
            ),
            "online_history_existence": history.get(
                "existence", history["activity"]
            ),
            "online_history_in_frame": history.get(
                "in_frame", history["activity"]
            ),
            "online_history_relative_scale": history.get("relative_scale"),
            "online_history_relative_disparity": history.get(
                "relative_disparity"
            ),
            "online_history_relations": history.get("relations"),
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
        result.update(factorized_result_fields(future_output, history_output, target_future))
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
