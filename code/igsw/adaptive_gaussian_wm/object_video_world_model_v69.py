"""Pretrained-feature Object Memory and posterior-conditioned future sequences."""

from copy import deepcopy

import torch
from torch import nn

from .pretrained_visual_encoder_v69 import sample_perception_v69
from .query_object_video_encoder_v69 import QueryObjectVideoEncoderV69, ObjectQueriesV69, history_queries_v69
from .object_sequence_dynamics_v69 import ObjectSequencePosteriorV69, ObjectSequenceDynamicsV69
from .object_sequence_readout_v69 import ObjectSequenceReadoutV69, appearance_binding_target_v69, binding_probability_v69
from .object_sequence_objective_v69 import object_sequence_loss_v69


class ObjectVideoWorldModelV69(nn.Module):
    def __init__(self, config, stage="state"):
        super().__init__()
        self.config, self.stage = config, stage
        self.encoder = QueryObjectVideoEncoderV69(config)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.readout = ObjectSequenceReadoutV69(config)
        self.posterior = ObjectSequencePosteriorV69(config)
        self.dynamics = ObjectSequenceDynamicsV69(config)
        if stage == "state":
            self.posterior.requires_grad_(False)
            self.dynamics.requires_grad_(False)
        else:
            self.encoder.requires_grad_(False)
            self.readout.requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        if self.stage == "state":
            for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target.lerp_(online, 1-self.config.ema)

    def encode_history(self, perception, supplied_queries=None):
        history = perception.prefix(self.config.history_frames)
        queries = supplied_queries if isinstance(supplied_queries, ObjectQueriesV69) else history_queries_v69(history, self.config.object_queries, supplied_queries)
        return queries, self.encoder(history, queries)

    def reference_readout(self, states, reference, indices, features):
        b, p = reference.shape[:2]
        ownership = reference.new_zeros((b, p, states[0].tokens.shape[1]+1))
        position = torch.zeros_like(reference)
        local = reference.new_zeros((b, p, states[0].tokens.shape[1], 2))
        for frame, state in enumerate(states):
            weights = binding_probability_v69(self.readout.binding(state, reference, features), state.query_valid)
            decoded = self.readout(state, reference, weights)
            selected = (indices == frame)[..., None]
            ownership = torch.where(selected, weights, ownership)
            position = torch.where(selected, decoded["position"], position)
            coordinates = reference[:, :, None]-state.centers[:, None, :, 0]
            local = torch.where(selected[..., None], coordinates, local)
        return ownership, position, local

    def render_sequence(self, states, reference, ownership, offset, local_coordinates):
        rows = [self.readout(state, reference, ownership, local_coordinates=local_coordinates) for state in states]
        return {"positions": reference[:, None] + torch.stack([row["position"] for row in rows], 1) - offset[:, None],
                "visibility": torch.stack([row["observation"][..., 0] for row in rows], 1)}

    def forward(self, perception, batch, supplied_queries=None, deterministic_effect=False):
        th = self.config.history_frames
        queries, history = self.encode_history(perception, supplied_queries)
        teacher = batch["teacher"]
        b, t, p = teacher["xy"].shape[:3]
        frame_index = torch.arange(t, device=teacher["xy"].device)[None].expand(b, -1)
        measured_features, measured_valid = sample_perception_v69(perception, teacher["xy"], frame_index)
        rows, columns = torch.arange(b, device=frame_index.device)[:, None], torch.arange(p, device=frame_index.device)[None]
        reference_feature = measured_features[rows, teacher["reference_index"], columns]
        ownership, offset, local = self.reference_readout(history, teacher["reference_xy"], teacher["reference_index"], reference_feature)
        output = {"queries": queries, "history_states": history, "source": history[-1],
                  "measurement_features": measured_features, "measurement_feature_valid": measured_valid,
                  "reference_ownership": ownership, "reference_offset": offset, "reference_local_xy": local}
        if self.stage == "state":
            state, future = history[-1], []
            for frame in range(th, t):
                state = self.encoder.observe(state, perception.features[:, frame], perception.coordinates, perception.valid[:, frame], perception.times[:, frame])
                future.append(state)
            states = history + future
            logits, decoded, targets, evidence = [], [], [], []
            for frame, state in enumerate(states):
                logit = self.readout.binding(state, teacher["xy"][:, frame], measured_features[:, frame])
                fields = self.readout(state, teacher["xy"][:, frame], binding_probability_v69(logit, queries.valid))
                target, confidence = appearance_binding_target_v69(queries, measured_features[:, frame], perception.features[:, frame], perception.valid[:, frame])
                logits.append(logit)
                decoded.append(fields["appearance"])
                targets.append(target)
                evidence.append(confidence)
            observed = self.render_sequence(states, teacher["reference_xy"], ownership, offset, local)
            output.update(binding_logits=torch.stack(logits, 1), binding_target=torch.stack(targets, 1),
                          binding_evidence=torch.stack(evidence, 1), observed_appearance=torch.stack(decoded, 1),
                          observed_positions=observed["positions"], observed_visibility=observed["visibility"], observed_states=states)
        else:
            with torch.no_grad():
                state = history[-1].detach()
                targets = []
                for frame in range(th, t):
                    state = self.target_encoder.observe(state, perception.features[:, frame], perception.coordinates, perception.valid[:, frame], perception.times[:, frame])
                    targets.append(state)
            effect = self.posterior(history[-1], targets, deterministic_effect or not self.training, batch["frame_valid"][:, th:])
            times = perception.times[:, th:]
            direct_states = self.dynamics(history[-1], effect["value"], times, rollout=False)
            rollout_states = self.dynamics(history[-1], effect["value"], times, rollout=True)
            shuffled_effect = effect["value"].roll(1, 0) if b > 1 else effect["value"].roll(1, 1)
            direct = self.render_sequence(direct_states, teacher["reference_xy"], ownership, offset, local)
            rollout = self.render_sequence(rollout_states, teacher["reference_xy"], ownership, offset, local)
            with torch.no_grad():
                shuffled_states = self.dynamics(history[-1], shuffled_effect, times, rollout=False)
                shuffled = self.render_sequence(shuffled_states, teacher["reference_xy"], ownership, offset, local)
                zero_states = self.dynamics(history[-1], torch.zeros_like(effect["value"]), times, rollout=False)
                zero = self.render_sequence(zero_states, teacher["reference_xy"], ownership, offset, local)
                observed_target = self.render_sequence(targets, teacher["reference_xy"], ownership, offset, local)
                current_copy = self.render_sequence([history[-1]], teacher["reference_xy"], ownership, offset, local)
            output.update(effect=effect,
                          target_states=targets, direct_states=direct_states, rollout_states=rollout_states,
                          direct_positions=direct["positions"], rollout_positions=rollout["positions"],
                          shuffled_positions=shuffled["positions"], zero_positions=zero["positions"],
                          observed_target_positions=observed_target["positions"],
                          current_state_copy_positions=current_copy["positions"].expand_as(observed_target["positions"]),
                          rollout_visibility=rollout["visibility"])
        loss, parts, epe = object_sequence_loss_v69(output, batch, self.config, self.stage)
        output.update(loss=loss, parts=parts, epe_px=epe)
        return output

    def forecast(self, history_perception, queries, effect, times, measurement_xy, measurement_features):
        """Deployable path: all arguments are observations, requested times, or externally selected effect."""
        _, states = self.encode_history(history_perception, queries)
        source = states[-1]
        ownership = binding_probability_v69(self.readout.binding(source, measurement_xy, measurement_features), source.query_valid)
        offset = self.readout(source, measurement_xy, ownership)["position"]
        local = measurement_xy[:, :, None]-source.centers[:, None, :, 0]
        future = self.dynamics(source, effect, times, rollout=True)
        return self.render_sequence(future, measurement_xy, ownership, offset, local), future
