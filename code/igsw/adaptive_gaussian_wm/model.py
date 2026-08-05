"""Strictly separated online, target, posterior, and prior world-model paths."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .causal_region_transformer import CausalRegionTransformer
from .change_residual_readout import ChangeResidualReadout
from .conditioning import LanguageConditionProjector
from .config import AdaptiveGaussianWMConfig
from .decoder import GaussianReadout
from .dense_object_readout import DenseObjectReadout
from .dynamics import JointObjectLatentDynamics
from .dual_horizon_objective import dual_horizon_loss
from .dual_horizon_runtime import (
    mask_horizon_supervision,
    prepare_transition_effects,
    rollout_goal_prediction,
)
from .dynamics_runtime import run_object_dynamics
from .factorized_dynamics import FactorizedObjectDynamics
from .feature_readout_runtime import decode_feature_readouts
from .gpstoken import LearnableGPSTokenAllocator
from .latent_action import LatentActionModel
from .language_effect import LanguageEffectAlignment
from .latent_effect_composer import LatentEffectComposer
from .model_output_runtime import assemble_model_output
from .model_phases import joint_phase_flags
from .object_memory import ObjectMemoryTransition
from .object_slots import ObjectSlotAggregator
from .rgb_supervision import residual_future_rgb, render_current_rgb, render_future_rgb
from .scale import signed_gap_scale
from .sequence_encoding import encode_visual_sequence
from .hierarchical_region_dynamics import HierarchicalRegionDynamics
from .object_region_memory import ObjectRegionMemory
from .region_effect_posterior import RegionEffectPosterior
from .trainable_dino_encoder import TrainableDinoRegionEncoder


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
        self.latent_actions = (
            None if config.object_region_memory else LatentActionModel(config)
        )
        self.effect_composer = (
            LatentEffectComposer(config) if config.dual_horizon_dynamics else None
        )
        self.dynamics = (
            FactorizedObjectDynamics(config)
            if config.factorized_dynamics
            else JointObjectLatentDynamics(config)
        )
        self.gaussian_readout = (
            None
            if config.change_residual_readout or config.object_region_memory
            else GaussianReadout(config)
        )
        self.dense_readout = (
            DenseObjectReadout(config) if config.dense_object_readout else None
        )
        self.change_readout = (
            ChangeResidualReadout(config) if config.change_residual_readout else None
        )
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
        self.online_dino = None
        self.target_dino = None
        self.region_transformer = None
        self.target_region_transformer = None
        self.region_memory = None
        self.target_region_memory = None
        self.region_dynamics = None
        self.region_effect_posterior = None
        if config.object_region_memory:
            self.online_dino = TrainableDinoRegionEncoder(
                config, config.dino_frame_batch
            )
            self.target_dino = copy.deepcopy(self.online_dino)
            self.target_dino.freeze_as_target()
            self.region_transformer = CausalRegionTransformer(config)
            self.target_region_transformer = copy.deepcopy(self.region_transformer)
            self.target_region_transformer.requires_grad_(False)
            self.region_memory = ObjectRegionMemory(config)
            self.target_region_memory = copy.deepcopy(self.region_memory)
            self.target_region_memory.requires_grad_(False)
            self.region_dynamics = HierarchicalRegionDynamics(config)
            self.region_effect_posterior = RegionEffectPosterior(config)
            self.gaussian_readout = None
            self.dense_readout = None
            self.change_readout = None
            self.register_buffer(
                "curriculum_step", torch.zeros((), dtype=torch.long), persistent=True
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_allocator.eval()
        self.target_object_aggregator.eval()
        if self.target_object_memory is not None:
            self.target_object_memory.eval()
        if self.target_dino is not None:
            self.target_dino.eval()
            self.target_region_transformer.eval()
            self.target_region_memory.eval()
        return self

    def set_curriculum_step(self, step: int) -> None:
        if not self.config.object_region_memory:
            return
        if step < 0:
            raise ValueError("curriculum step must be non-negative")
        self.curriculum_step.fill_(step)

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
        if self.config.object_region_memory:
            pairs = pairs + (
                (self.target_dino, self.online_dino),
                (self.target_region_transformer, self.region_transformer),
                (self.target_region_memory, self.region_memory),
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
        if self.latent_actions is None:
            raise ValueError("v43 has no history-only latent-effect prior")
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
        collect_diagnostics: bool = False,
    ) -> dict:
        if self.config.object_region_memory:
            if actions_override is not None or not use_posterior:
                raise ValueError("v43 uses its internal curriculum posterior contract")
            from .v43_model_runtime import forward_v43

            return forward_v43(self, batch, collect_diagnostics)
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
        effects = prepare_transition_effects(
            self,
            batch,
            history,
            target_future,
            history_scale,
            future_scale,
            condition,
            use_posterior,
            actions_override,
            action_free,
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
            effects.dynamics,
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
            torch.zeros_like(effects.dynamics),
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
        readout_fields, readout_context = decode_feature_readouts(
            self,
            batch,
            current_tokens,
            current_slot_state,
            future_output,
            predicted_future_centers,
        )
        readout = readout_fields["gaussian_readout"]
        current_readout = readout_fields["current_gaussian_readout"]
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
        rgb_fields = {
            "rendered_future_rgb": rendered_rgb,
            "residual_reference_rgb": residual_reference_rgb,
            "rgb_render_coverage": rgb_coverage,
            "rendered_current_rgb": rendered_current_rgb,
            "current_rgb_render_coverage": current_rgb_coverage,
        }
        result = assemble_model_output(
            self,
            history,
            target_history,
            target_future,
            future_output,
            history_output,
            predicted_future_centers,
            predicted_history_centers,
            zero_action_future_centers,
            effects,
            history_mask,
            readout_fields,
            rgb_fields,
            condition,
        )
        if self.config.dual_horizon_dynamics and not action_free and phase != "history_prior_loss":
            result.update(
                rollout_goal_prediction(
                    self,
                    batch,
                    history,
                    current_tokens,
                    current_slot_state,
                    future_output,
                    effects,
                    condition,
                )
            )
        loss_batch = mask_horizon_supervision(batch, result, action_free)
        if compute_joint_loss:
            if loss_weights is None:
                raise ValueError("joint_loss phase requires loss_weights")
            from .losses import adaptive_world_model_loss

            loss, parts = adaptive_world_model_loss(
                self,
                loss_batch,
                result,
                loss_weights,
                collect_diagnostics,
            )
            dual_loss, dual_parts = dual_horizon_loss(self, loss_batch, result)
            loss = loss + dual_loss
            parts.update(dual_parts)
            parts["total"] = loss
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
