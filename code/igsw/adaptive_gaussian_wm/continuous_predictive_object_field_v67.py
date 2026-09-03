"""End-to-end continuous predictive object field model v67."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .continuous_object_dynamics_v67 import (
    ContinuousEffectPosteriorV67,
    ObjectFieldOperatorV67,
    shuffled_effect_v67,
    zero_effect_v67,
)
from .continuous_predictive_teacher_v67 import teacher_relation_evidence_v67
from .continuous_scale_field_v67 import (
    CausalSpatiotemporalFieldMixerV67,
    ContinuousScaleFieldEncoderV67,
)
from .predictive_rate_distortion_v67 import (
    posterior_dynamics_objective_v67,
    predictive_state_objective_v67,
)
from .query_predictive_state_v67 import (
    PointObservableDecoderV67,
    PredictiveObjectCodeNetworkV67,
    QueryRelationFieldNetworkV67,
)
from .v67_config import DYNAMICS_STAGE, STATE_STAGE


class ContinuousPredictiveObjectFieldV67(nn.Module):
    """Learn a predictive object function without fixed slots or object masks."""

    def __init__(self, config, stage: str):
        super().__init__()
        config.validate()
        if stage not in (STATE_STAGE, DYNAMICS_STAGE):
            raise ValueError(f"unsupported v67 stage: {stage}")
        self.config = config
        self.stage = stage
        self.online_field = ContinuousScaleFieldEncoderV67(config)
        self.history_mixer = CausalSpatiotemporalFieldMixerV67(config)
        self.relation_field = QueryRelationFieldNetworkV67(config)
        self.object_code = PredictiveObjectCodeNetworkV67(config)
        self.point_decoder = PointObservableDecoderV67(config)
        self.effect_posterior = ContinuousEffectPosteriorV67(config)
        self.operator = ObjectFieldOperatorV67(config)
        self.target_field = copy.deepcopy(self.online_field)
        self.target_code = copy.deepcopy(self.object_code)
        self.target_field.requires_grad_(False)
        self.target_code.requires_grad_(False)
        self.configure_stage(stage)

    def configure_stage(self, stage: str) -> None:
        self.stage = stage
        self.requires_grad_(False)
        if stage == STATE_STAGE:
            modules = (
                self.online_field,
                self.history_mixer,
                self.relation_field,
                self.object_code,
                self.point_decoder,
                self.operator.decoder,
            )
        else:
            modules = (self.effect_posterior, self.operator)
        for module in modules:
            module.requires_grad_(True)
        if stage == DYNAMICS_STAGE:
            # E0 defines the state semantics. E1 learns transition, not a new decoder.
            self.operator.decoder.requires_grad_(False)
        self.target_field.requires_grad_(False)
        self.target_code.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_field.eval()
        self.target_code.eval()
        if self.stage == DYNAMICS_STAGE:
            self.online_field.eval()
            self.history_mixer.eval()
            self.relation_field.eval()
            self.object_code.eval()
            self.point_decoder.eval()
        return self

    @torch.no_grad()
    def update_target_ema(self) -> None:
        momentum = self.config.target_ema_momentum
        for target, online in zip(
            self.target_field.parameters(), self.online_field.parameters()
        ):
            target.lerp_(online.detach(), 1.0 - momentum)
        for target, online in zip(
            self.target_code.parameters(), self.object_code.parameters()
        ):
            target.lerp_(online.detach(), 1.0 - momentum)

    def _source_field(self, batch, target):
        stop = self.config.source_frame + 1
        coordinates = target.anchor_coordinates[:, None].expand(-1, stop, -1, -1)
        scales = target.scales[:, None].expand(-1, stop, -1)
        sampled = self.online_field(
            batch["video_rgb"][:, :stop],
            batch["video_pixel_valid"][:, :stop],
            batch["native_image_hw"],
            coordinates,
            scales,
            batch["frame_times"][:, :stop],
        )
        return sampled, self.history_mixer(sampled)

    def encode_source(self, batch, target, sample: bool):
        sampled, history = self._source_field(batch, target)
        current_valid = history.valid[:, -1]
        relation = self.relation_field(
            history.current,
            target.anchor_coordinates,
            target.scales,
            current_valid,
            target.query_indices,
        )
        code = self.object_code(
            history.current,
            relation,
            current_valid,
            target.context_mask,
            target.query_indices,
            sample,
        )
        return sampled, history, relation, code

    @torch.no_grad()
    def encode_target(self, batch, target, frame: int):
        coordinates = target.track_coordinates[:, frame : frame + 1]
        scales = target.scales[:, None]
        sampled = self.target_field(
            batch["video_rgb"][:, frame : frame + 1],
            batch["video_pixel_valid"][:, frame : frame + 1],
            batch["native_image_hw"],
            coordinates,
            scales,
            batch["frame_times"][:, frame : frame + 1],
        )
        relation, _ = teacher_relation_evidence_v67(target, frame, self.config)
        valid = sampled.valid[:, 0] & target.visibility[:, frame]
        code = self.target_code.from_external_support(
            sampled.features[:, 0],
            relation,
            valid,
            target.context_mask,
            target.query_indices,
        )
        return sampled, code

    def _decode_source(self, target, code):
        anchor = target.anchor_coordinates.index_select(1, target.query_indices)
        return self.operator.decoder(
            code.sample,
            anchor,
            target.anchor_coordinates,
            target.scales,
        )

    def _continuity_error(self, target, code, decoded):
        point = torch.arange(
            target.anchor_coordinates.shape[1],
            device=target.anchor_coordinates.device,
            dtype=torch.float32,
        )
        perturbation = 0.005 * torch.stack((point.sin(), point.cos()), dim=-1)
        coordinates = (target.anchor_coordinates + perturbation[None]).clamp(-1.0, 1.0)
        scales = (target.scales * 1.02).clamp(
            self.config.minimum_scale, self.config.maximum_scale
        )
        anchor = target.anchor_coordinates.index_select(1, target.query_indices)
        perturbed = self.operator.decoder(code.sample, anchor, coordinates, scales)
        semantic = 0.5 * (
            (decoded.dino.float() - perturbed.dino.float()).square().mean()
            + (decoded.siglip.float() - perturbed.siglip.float()).square().mean()
        )
        support = (
            decoded.support_logits.float() - perturbed.support_logits.float()
        ).square().mean()
        return semantic + 0.1 * support

    def _forward_state(self, batch, target):
        sampled, history, relation, code = self.encode_source(
            batch, target, self.training
        )
        decoded = self._decode_source(target, code)
        point_dino, point_siglip = self.point_decoder(code.point_mean)
        with torch.no_grad():
            _, target_code = self.encode_target(batch, target, self.config.target_frame)
        output = {
            "source_samples": sampled,
            "source_history": history,
            "source_relation": relation,
            "source_code": code,
            "source_decoded": decoded,
            "point_dino": point_dino,
            "point_siglip": point_siglip,
            "target_code": target_code,
            "continuity_error": self._continuity_error(target, code, decoded),
        }
        loss, parts = predictive_state_objective_v67(
            self, target, output, self.config
        )
        output.update(loss=loss, parts=parts)
        return output

    def _operator(self, source, effect, delta, target, frame):
        anchor = target.anchor_coordinates.index_select(1, target.query_indices)
        return self.operator(
            source,
            effect,
            delta,
            anchor,
            target.track_coordinates[:, frame],
            target.scales,
        )

    def _forward_dynamics(self, batch, target):
        with torch.no_grad():
            _, _, relation, source = self.encode_source(batch, target, False)
            _, midpoint = self.encode_target(batch, target, self.config.midpoint_frame)
            _, goal = self.encode_target(batch, target, self.config.target_frame)
        source_time = batch["frame_times"][:, self.config.source_frame]
        midpoint_time = batch["frame_times"][:, self.config.midpoint_frame]
        goal_time = batch["frame_times"][:, self.config.target_frame]
        short_delta = midpoint_time - source_time
        tail_delta = goal_time - midpoint_time
        goal_delta = goal_time - source_time
        short_effect = self.effect_posterior(
            source, midpoint, short_delta, sample=self.training
        )
        tail_effect = self.effect_posterior(
            midpoint, goal, tail_delta, sample=self.training
        )
        goal_effect = self.effect_posterior(
            source, goal, goal_delta, sample=self.training
        )
        composed_effect = self.effect_posterior.compose(short_effect, tail_effect)

        short = self._operator(
            source, short_effect, short_delta, target, self.config.midpoint_frame
        )
        correct = self._operator(
            source, goal_effect, goal_delta, target, self.config.target_frame
        )
        zero = self._operator(
            source, zero_effect_v67(goal_effect), goal_delta, target, self.config.target_frame
        )
        shuffled = self._operator(
            source, shuffled_effect_v67(goal_effect), goal_delta, target, self.config.target_frame
        )
        direct = self._operator(
            source, composed_effect, goal_delta, target, self.config.target_frame
        )
        rollout = self._operator(
            short.code, tail_effect, tail_delta, target, self.config.target_frame
        )
        anchor = target.anchor_coordinates.index_select(1, target.query_indices)
        persistence = self.operator.decoder(
            source.mean,
            anchor,
            target.track_coordinates[:, self.config.target_frame],
            target.scales,
        )
        with torch.no_grad():
            goal_anchor = target.track_coordinates[:, self.config.target_frame].index_select(
                1, target.query_indices
            )
            goal_target_field = self.operator.decoder(
                goal.mean,
                goal_anchor,
                target.track_coordinates[:, self.config.target_frame],
                target.scales,
            )
        output = {
            "source_relation": relation,
            "source_code": source,
            "midpoint_target_code": midpoint,
            "goal_target_code": goal,
            "short_effect": short_effect,
            "tail_effect": tail_effect,
            "goal_effect": goal_effect,
            "short_correct": short,
            "goal_correct": correct,
            "goal_zero": zero,
            "goal_shuffled": shuffled,
            "goal_direct": direct,
            "goal_rollout": rollout,
            "goal_persistence": persistence,
            "goal_target_field": goal_target_field,
        }
        loss, parts = posterior_dynamics_objective_v67(
            self, target, output, self.config
        )
        output.update(loss=loss, parts=parts)
        return output

    def forward(self, batch, target):
        if self.stage == STATE_STAGE:
            return self._forward_state(batch, target)
        return self._forward_dynamics(batch, target)
