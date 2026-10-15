"""Joint observable reconstruction and posterior-conditioned object rollout."""

import torch
from torch import nn
from torch.nn import functional as F

from .codec import letterbox_video
from .dynamics import EffectPosterior, ObjectDynamics
from .state import ObjectFeatureRenderer, ObjectStateEncoder, coordinates
from .scene_state import SceneFeatureRenderer, SceneStateEncoder


def weighted_mean(value, weight):
    weight = weight.expand_as(value).float()
    return (value.float()*weight).sum()/weight.sum().clamp_min(1)


class ObjectEventWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder, renderer = {"object": (ObjectStateEncoder, ObjectFeatureRenderer),
                             "scene": (SceneStateEncoder, SceneFeatureRenderer)}[config.state_layout]
        self.encoder = encoder(config)
        self.renderer = renderer(config)
        self.posterior = EffectPosterior(config)
        self.dynamics = ObjectDynamics(config)

    def observe(self, features, valid, times):
        xy = coordinates(round(features.shape[2]**.5), features.device)
        return self.encoder(features, xy, valid, times)

    def render_features(self, states, xy):
        rows = [self.renderer(state, xy) for state in states]
        return torch.stack([row[0] for row in rows], 1), torch.stack([row[1] for row in rows], 1)

    def forward(self, batch, codec, deterministic=False, rollout_weight=1., render_all=False):
        images, pixel_valid = letterbox_video(batch, codec.image_size)
        features = codec.encode(images)
        patch_valid = codec.patch_valid(pixel_valid)
        xy = coordinates(codec.grid_size, features.device)
        states = self.encoder(features, xy, patch_valid, batch["times"])
        observed_features, observed_masks = self.render_features(states, xy)
        th = batch["history_frames"]
        source = states[th-1]
        previous = features.new_zeros((len(features), self.config.slots, self.config.effect_dim))
        predicted, effects = [], []
        for index in range(th, len(states)):
            effect = self.posterior(states[index-1], states[index], previous,
                                    first=index == th, deterministic=deterministic)
            usable = batch["frame_valid"][:, index]
            previous = torch.where(usable[:, None, None], effect["value"], previous)
            effect["value"] = previous
            effect["boundary"] = effect["boundary"]*usable[:, None]
            source = self.dynamics(source, previous, batch["times"][:, index])
            predicted.append(source)
            effects.append(effect)
        predicted_features, predicted_masks = self.render_features(predicted, xy)
        future_valid = patch_valid[:, th:]
        observed_loss = weighted_mean((observed_features-features).square(), patch_valid[..., None])
        predicted_loss = weighted_mean((predicted_features-features[:, th:]).square(), future_valid[..., None])
        predicted_delta = torch.diff(torch.cat((observed_features[:, th-1:th], predicted_features), 1), dim=1)
        target_delta = torch.diff(features[:, th-1:], dim=1)
        paired_valid = patch_valid[:, th:] & patch_valid[:, th-1:-1]
        change_loss = weighted_mean((predicted_delta-target_delta).abs(), paired_valid[..., None])
        effect_kl = torch.stack([e["kl"] for e in effects], 1)
        boundaries = torch.stack([e["boundary"] for e in effects], 1)
        probabilities = torch.stack([e["boundary_probability"] for e in effects], 1)
        frame_valid = batch["frame_valid"][:, th:]
        kl_loss = weighted_mean(effect_kl*boundaries, frame_valid[..., None])
        update_loss = weighted_mean(probabilities[:, 1:], frame_valid[:, 1:, None])
        chosen = (torch.arange(len(predicted), device=features.device) if render_all else
                  torch.linspace(0, len(predicted)-1, self.config.rgb_loss_frames,
                                 device=features.device).round().long().unique())
        observed_rgb = codec.decode(observed_features[:, th+chosen])
        predicted_rgb = codec.decode(predicted_features[:, chosen])
        image_target = images[:, th+chosen]
        image_valid = pixel_valid[:, th+chosen]
        rgb_observed = weighted_mean((observed_rgb-image_target).abs(), image_valid)
        rgb_predicted = weighted_mean((predicted_rgb-image_target).abs(), image_valid)
        loss = self.config.feature_weight*observed_loss + self.config.rgb_weight*rgb_observed
        dynamic = (self.config.feature_weight*predicted_loss + self.config.rgb_weight*rgb_predicted
                   + self.config.change_weight*change_loss + self.config.effect_kl_weight*kl_loss)
        if self.config.effect_mode == "object_segment":
            dynamic = dynamic + self.config.update_weight*update_loss
        loss = loss + rollout_weight*dynamic
        rms = lambda value: value.detach().float().square().mean().sqrt()
        parts = {"observed_feature_mse": observed_loss, "posterior_feature_mse": predicted_loss,
                 "observed_rgb_l1": rgb_observed, "posterior_rgb_l1": rgb_predicted,
                 "feature_change_l1": change_loss, "effect_kl": kl_loss,
                 "effect_update_fraction": boundaries.float().mean(), "update_probability": update_loss,
                 "observed_state_rms_first": rms(states[0].tokens),
                 "observed_state_rms_last": rms(states[-1].tokens),
                 "rollout_state_rms_last": rms(predicted[-1].tokens),
                 "posterior_mean_abs_max": torch.stack([e["mean"].detach().abs().amax() for e in effects]).amax(),
                 "rollout_weight": features.new_tensor(rollout_weight)}
        return {"loss": loss, "parts": parts, "source": states[th-1],
                "observed_states": states, "predicted_states": predicted,
                "observed_masks": observed_masks, "predicted_masks": predicted_masks,
                "effects": effects, "target_rgb": image_target, "pixel_valid": image_valid,
                "observed_rgb": observed_rgb, "posterior_rgb": predicted_rgb,
                "rendered_future_indices": chosen, "target_features": features[:, th:],
                "predicted_features": predicted_features}

    def rollout(self, source, effect_sequence, times):
        states, state = [], source
        for effect, time in zip(effect_sequence.unbind(1), times.unbind(1)):
            state = self.dynamics(state, effect, time)
            states.append(state)
        return states
