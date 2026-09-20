"""RGB-only persistent slots and source-coordinate object transport.

Teacher queries are readout/loss locations, never encoder inputs. A slot is a
learned component hypothesis; its name alone does not establish object identity.
"""

from copy import deepcopy
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .continuous_scale_field_v67 import ContinuousScaleFieldEncoderV67, fourier_features_v67
from .v67_config import ContinuousPredictiveObjectFieldConfigV67


@dataclass(frozen=True)
class ObjectTransportConfigV68:
    width: int = 256
    objects: int = 16
    effect_dim: int = 32
    layers: int = 4
    heads: int = 8
    dino_dim: int = 1024
    siglip_dim: int = 768
    ema: float = .996

    def to_dict(self):
        return asdict(self)


class RGBObjectEncoderV68(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, count = config.width, config.objects + 2
        field_config = ContinuousPredictiveObjectFieldConfigV67(field_dim=width)
        # Reuse the native RGB frontend, without its spatial averaging layer.
        self.image = ContinuousScaleFieldEncoderV67(field_config).local_encoder[:-1]
        self.position = nn.Linear(18, width)
        self.initial = nn.Parameter(torch.randn(count, width) / math.sqrt(width))
        self.key, self.value, self.query = (nn.Linear(width, width) for _ in range(3))
        self.norm = nn.LayerNorm(width)
        self.update = nn.GRUCell(width, width)
        self.residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, rgb, pixel_valid, history_valid, native_hw, initial=None):
        batch, times = rgb.shape[:2]
        slots = self.initial[None].expand(batch, -1, -1) if initial is None else initial
        center = torch.zeros((batch, self.config.objects + 2, 2), device=rgb.device)
        for frame in range(times):
            feature_map = self.image(rgb[:, frame].float() / 255.0)
            height, width = feature_map.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(height, device=rgb.device), torch.arange(width, device=rgb.device), indexing="ij")
            xy = torch.stack((xx, yy), -1).flatten(0, 1).float()[None] * 4
            xy = xy / (native_hw[:, [1, 0]].float() - 1).clamp_min(1)[:, None] * 2 - 1
            valid = F.interpolate(pixel_valid[:, frame, None].float(), size=(height, width), mode="nearest")[:, 0].flatten(1) > .5
            features = feature_map.flatten(2).transpose(1, 2) + self.position(fourier_features_v67(xy, 4))
            keys, values = self.key(features), self.value(features)
            for _ in range(3):
                logits = torch.einsum("bkd,bnd->bkn", self.query(self.norm(slots)), keys) / math.sqrt(self.config.width)
                assignment = logits.float().softmax(1) * valid[:, None]
                pooling = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-6)
                update = torch.einsum("bkn,bnd->bkd", pooling.to(values.dtype), values)
                corrected = self.update(update.flatten(0, 1), slots.to(update.dtype).flatten(0, 1)).reshape_as(slots)
                corrected = corrected + self.residual(corrected)
                slots = torch.where(history_valid[:, frame, None, None], corrected, slots)
            measured_center = torch.einsum("bkn,bnd->bkd", pooling, xy)
            center = torch.where(history_valid[:, frame, None, None], measured_center, center)
        return {"slots": slots, "center": center, "feature_map": feature_map, "features": features,
                "keys": keys, "native_hw": native_hw, "valid": valid}

    def assignment(self, state, coordinates):
        height, width = state["feature_map"].shape[-2:]
        pixels = (coordinates.float() + 1) * .5 * (state["native_hw"][:, [1, 0]].float() - 1)[:, None]
        grid = pixels / torch.tensor([max(1, (width-1)*4), max(1, (height-1)*4)], device=pixels.device) * 2 - 1
        sampled = F.grid_sample(state["feature_map"].float(), grid[:, None], align_corners=True)[:, :, 0].transpose(1, 2)
        sampled = sampled + self.position(fourier_features_v67(coordinates.float(), 4))
        scores = torch.einsum("bnd,bkd->bnk", self.key(sampled), self.query(self.norm(state["slots"]))) / math.sqrt(self.config.width)
        return scores.float().softmax(-1)


class GroundedObjectTransportV68(nn.Module):
    def __init__(self, config=ObjectTransportConfigV68(), stage="state"):
        super().__init__()
        self.config, self.stage = config, stage
        width = config.width
        self.encoder = RGBObjectEncoderV68(config)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.appearance = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, config.dino_dim + config.siglip_dim))
        self.posterior = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, config.effect_dim * 2))
        self.effect_input = nn.Linear(config.effect_dim, width, bias=False)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, config.heads, width * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.dynamics = nn.TransformerEncoder(layer, config.layers)
        self.transport = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 2))
        self.visibility = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.composer = nn.Sequential(nn.Linear(config.effect_dim * 2, width), nn.GELU(), nn.Linear(width, config.effect_dim))
        if stage == "state":
            for module in (self.posterior, self.effect_input, self.time, self.dynamics, self.transport, self.visibility, self.composer):
                module.requires_grad_(False)
        else:
            self.encoder.requires_grad_(False)
            self.appearance.requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        if self.stage == "state":
            for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target.lerp_(online, 1 - self.config.ema)

    def encode_history(self, batch):
        return self.encoder(batch["video_rgb"][:, :4], batch["video_pixel_valid"][:, :4], batch["history_valid"], batch["native_image_hw"])

    def observed_future(self, batch, source, position, target=False):
        encoder = self.target_encoder if target else self.encoder
        return encoder(batch["video_rgb"][:, position:position+1], batch["video_pixel_valid"][:, position:position+1],
                       torch.ones_like(batch["history_valid"][:, :1]), batch["native_image_hw"], initial=source["slots"].detach())

    def decode_appearance(self, state, coordinates, ownership):
        relative = coordinates[:, :, None] - state["center"][:, None]
        slots = state["slots"][:, None].expand(-1, coordinates.shape[1], -1, -1)
        fields = self.appearance(torch.cat((slots.float(), relative), -1))
        # Independent fields, mixed only by the current-frame mask competition.
        return (fields * ownership[..., None]).sum(2)

    def extract_effect(self, source, target):
        mean, logvar = self.posterior(torch.cat((source["slots"].float(), target["slots"].float() - source["slots"].float()), -1)).chunk(2, -1)
        logvar = logvar.float().clamp(-6, 1)
        code = mean.float() + torch.randn_like(mean) * (.5 * logvar).exp() if self.training else mean.float()
        return code.tanh(), .5 * (mean.float().square() + logvar.exp() - logvar - 1).mean()

    def predict_transport(self, source, effect, seconds, source_coordinates, ownership):
        base = source["slots"] + self.time(torch.log1p(seconds.float())[:, None])[:, None]
        delta = self.dynamics(base + self.effect_input(effect)) - self.dynamics(base)
        relative = source_coordinates[:, :, None] - source["center"][:, None]
        changed = delta[:, None].expand(-1, source_coordinates.shape[1], -1, -1)
        zero = torch.zeros_like(changed)
        displacement = self.transport(torch.cat((changed.float(), relative), -1)) - self.transport(torch.cat((zero.float(), relative), -1))
        predicted = source_coordinates + (displacement.float() * ownership[..., None]).sum(2)
        center_zero = torch.zeros_like(source["center"])
        center_delta = self.transport(torch.cat((delta.float(), center_zero), -1)) - self.transport(torch.cat((torch.zeros_like(delta).float(), center_zero), -1))
        next_state = {**source, "slots": source["slots"] + delta, "center": source["center"] + center_delta.float()}
        visible_logits = (self.visibility(next_state["slots"]).float().transpose(1, 2) * ownership).sum(-1)
        return predicted, next_state, visible_logits

    @staticmethod
    def weighted(value, mask):
        return (value.float() * mask.float()).sum() / mask.float().sum().clamp_min(1)

    def forward(self, batch, teacher=None):
        source = self.encode_history(batch)
        coordinates = batch["coordinates"][:, 3]
        ownership = self.encoder.assignment(source, coordinates)
        if self.stage == "state":
            decoded = self.decode_appearance(source, coordinates, ownership)
            target_feature = teacher["features"][:, 0]
            dino_error = 1 - F.cosine_similarity(decoded[..., :self.config.dino_dim].float(), target_feature[..., :self.config.dino_dim], dim=-1)
            siglip_error = 1 - F.cosine_similarity(decoded[..., self.config.dino_dim:].float(), target_feature[..., self.config.dino_dim:], dim=-1)
            appearance = self.weighted((dino_error + siglip_error) * .5, batch["point_valid"][:, 3] & teacher["valid"][:, 0])
            future = self.observed_future(batch, source, 4)
            future_owner = self.encoder.assignment(future, batch["coordinates"][:, 4])
            mixture = .5 * (ownership + future_owner)
            js = .5 * ((ownership * (ownership.clamp_min(1e-6).log() - mixture.clamp_min(1e-6).log())).sum(-1) +
                       (future_owner * (future_owner.clamp_min(1e-6).log() - mixture.clamp_min(1e-6).log())).sum(-1))
            correspondence = self.weighted(js, batch["target_valid"][:, 0])
            roles = batch["roles"]
            grouped = torch.stack((ownership[..., :self.config.objects].sum(-1), ownership[..., self.config.objects], ownership[..., self.config.objects+1]), -1)
            role_loss = self.weighted(-grouped.clamp_min(1e-6).log().gather(-1, roles.clamp_min(0)[..., None])[..., 0], (roles >= 0) & batch["point_valid"][:, 3])
            # Same anchor-local region is positive support evidence, not an object ID.
            same = (batch["region_ids"][:, :, None] == batch["region_ids"][:, None]) & batch["motion_mask"][:, :, None] & batch["motion_mask"][:, None]
            same &= batch["point_valid"][:, 3, :, None] & batch["point_valid"][:, 3, None, :]
            off_diagonal = ~torch.eye(ownership.shape[1], dtype=torch.bool, device=ownership.device)[None]
            binding = self.weighted(1 - torch.einsum("bnk,bmk->bnm", ownership, ownership), same & off_diagonal)
            separation = self.weighted(torch.einsum("bnk,bmk->bnm", ownership, ownership), batch["different_motion_evidence"])
            loss = appearance + correspondence + .25 * role_loss + .1 * (binding + separation)
            return {"loss": loss, "parts": {"appearance_aux": appearance, "track_correspondence_js": correspondence,
                    "role_evidence_ce": role_loss, "local_region_binding": binding, "different_motion_separation": separation}, "source": source, "ownership": ownership}
        with torch.no_grad():
            short_target = self.observed_future(batch, source, 4, target=True)
            long_target = self.observed_future(batch, short_target, 5, target=True)
        short_effect, rate_s = self.extract_effect(source, short_target)
        tail_effect, rate_l = self.extract_effect(short_target, long_target)
        long_effect = self.composer(torch.cat((short_effect, tail_effect), -1)).tanh()
        ds = batch["frame_times"][:, 4] - batch["frame_times"][:, 3]
        dg = batch["frame_times"][:, 5] - batch["frame_times"][:, 3]
        short, short_state, vs = self.predict_transport(source, short_effect, ds, coordinates, ownership)
        direct, _, vg = self.predict_transport(source, long_effect, dg, coordinates, ownership)
        rollout, _, vr = self.predict_transport(short_state, tail_effect, dg-ds, short, ownership)
        shuffled_effect = long_effect.roll(1, 0) if len(long_effect) > 1 else long_effect.roll(1, 1)
        shuffled, _, _ = self.predict_transport(source, shuffled_effect, dg, coordinates, ownership)
        target_short, target_long = batch["coordinates"][:, 4], batch["coordinates"][:, 5]
        short_mask, long_mask = batch["target_valid"].unbind(1)
        def distance(pred, target, mask):
            return self.weighted(F.smooth_l1_loss(pred.float(), target.float(), beta=.01, reduction="none").sum(-1), mask)
        short_loss = distance(short, target_short, short_mask)
        long_loss = distance(direct, target_long, long_mask)
        roll_loss = distance(rollout, target_long, long_mask & short_mask)
        path_loss = distance(rollout, direct.detach(), long_mask & short_mask)
        persistence = distance(coordinates, target_long, long_mask)
        shuffled_loss = distance(shuffled, target_long, long_mask)
        intervention = F.relu(long_loss - shuffled_loss.detach() + .1 * persistence.detach())
        observation_known = batch["point_valid"][:, 3:4].expand(-1, 2, -1) & batch["motion_mask"][:, None]
        visibility = self.weighted(F.binary_cross_entropy_with_logits(torch.stack((vs, vg), 1), batch["point_valid"][:, 4:6].float(), reduction="none"), observation_known)
        loss = short_loss + long_loss + roll_loss + .25 * path_loss + .1 * intervention + .1 * visibility + .001 * (rate_s + rate_l)
        pixel_scale = (batch["native_image_hw"][:, [1, 0]].float()-1)[:, None] * .5
        epe = ((direct - target_long).float() * pixel_scale).norm(dim=-1)
        return {"loss": loss, "parts": {"short_transport": short_loss, "long_transport": long_loss, "rollout_transport": roll_loss,
                "path_consistency": path_loss, "persistence_transport": persistence, "shuffled_transport": shuffled_loss,
                "tracker_visibility_aux": visibility, "effect_rate": rate_s+rate_l, "intervention": intervention},
                "source": source, "ownership": ownership, "predicted_coordinates": direct,
                "epe_px": epe, "epe_valid": long_mask}
