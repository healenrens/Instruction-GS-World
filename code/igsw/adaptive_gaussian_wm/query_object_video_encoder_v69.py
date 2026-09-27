"""History-only visual queries and persistent root/carrier attention memory."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .object_video_attention_v69 import AttentionBlockV69, position_features
from .pretrained_visual_encoder_v69 import sample_perception_v69


@dataclass
class ObjectQueriesV69:
    xy: torch.Tensor
    frame_index: torch.Tensor
    features: torch.Tensor
    valid: torch.Tensor


@dataclass
class ObjectVideoStateV69:
    tokens: torch.Tensor
    centers: torch.Tensor
    time: torch.Tensor
    query_valid: torch.Tensor

    def detach(self):
        return ObjectVideoStateV69(self.tokens.detach(), self.centers.detach(), self.time.detach(), self.query_valid)


@torch.no_grad()
def history_queries_v69(perception, count, supplied=None):
    b, th, n = perception.features.shape[:3]
    if supplied is not None:
        xy, frames, valid = supplied["xy"], supplied["frame_index"], supplied["valid"]
        all_xy = xy[:, None].expand(-1, th, -1, -1)
        samples, sampled_valid = sample_perception_v69(perception, all_xy, torch.arange(th, device=xy.device)[None].expand(b, -1))
        rows = torch.arange(b, device=xy.device)[:, None]
        columns = torch.arange(xy.shape[1], device=xy.device)[None]
        return ObjectQueriesV69(xy, frames, samples[rows, frames, columns], valid & sampled_valid[rows, frames, columns])
    query_xy, query_frames, query_features, query_valid = [], [], [], []
    # Select appearance/spatially diverse observed tokens. These are region proposals, not object labels.
    for item in range(b):
        per_xy, per_frame, per_feature = [], [], []
        observed = torch.where(perception.valid[item].any(-1))[0]
        first = int(observed[0]) if len(observed) else 0
        last = int(observed[-1]) if len(observed) else th-1
        for frame, budget in ((first, count//2), (last, count-count//2)):
            ids = torch.where(perception.valid[item, frame])[0]
            value = F.normalize(perception.features[item, frame, ids].float(), dim=-1)
            xy = perception.coordinates[item, ids]
            center = F.normalize(value.mean(0), dim=-1)
            chosen = []
            score = 1 - value @ center
            for _ in range(min(budget, len(ids))):
                selected = int(score.argmax())
                chosen.append(selected)
                diversity = (1 - value @ value[selected]).clamp_min(0) + .25 * (xy - xy[selected]).square().sum(-1)
                score = diversity if len(chosen) == 1 else torch.minimum(score, diversity)
                score[chosen] = -1
            chosen = torch.tensor(chosen, device=value.device, dtype=torch.long)
            per_xy.append(xy[chosen])
            per_frame.append(torch.full((len(chosen),), frame, device=value.device, dtype=torch.long))
            per_feature.append(value[chosen])
        xy, frames, value = torch.cat(per_xy), torch.cat(per_frame), torch.cat(per_feature)
        active = len(xy)
        query_xy.append(F.pad(xy, (0, 0, 0, count-active)))
        query_frames.append(F.pad(frames, (0, count-active)))
        query_features.append(F.pad(value, (0, 0, 0, count-active)))
        query_valid.append(torch.arange(count, device=xy.device) < active)
    return ObjectQueriesV69(torch.stack(query_xy), torch.stack(query_frames), torch.stack(query_features), torch.stack(query_valid))


class QueryObjectVideoEncoderV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, r = config.width, config.tokens_per_object
        self.input = nn.Sequential(nn.LayerNorm(config.perception_dim), nn.Linear(config.perception_dim, d))
        self.query_input = nn.Linear(config.perception_dim, d)
        self.spatial = nn.Linear(18, d)
        self.time = nn.Sequential(nn.Linear(18, d), nn.SiLU(), nn.Linear(d, d))
        self.carriers = nn.Parameter(torch.randn(r, d) / math.sqrt(d))
        self.radius = nn.Parameter(torch.full((r,), -1.5))
        self.observation = nn.ModuleList(AttentionBlockV69(d, config.heads, cross=True) for _ in range(config.observation_layers))
        self.memory = nn.ModuleList(AttentionBlockV69(d, config.heads) for _ in range(config.memory_layers))
        self.mix = nn.Sequential(nn.LayerNorm(d*2), nn.Linear(d*2, d), nn.Sigmoid())

    def initial(self, queries, time):
        tokens = self.query_input(queries.features.float())[:, :, None] + self.carriers[None, None]
        centers = queries.xy[:, :, None].expand(-1, -1, self.config.tokens_per_object, -1).clone()
        return ObjectVideoStateV69(tokens, centers, time, queries.valid)

    def observe(self, state, feature, coordinates, valid, time):
        b, k, r, d = state.tokens.shape
        context = self.input(feature.float()) + self.spatial(position_features(coordinates))
        present = valid.any(-1)
        usable = valid.clone()
        usable[~present, 0] = True
        context = torch.where(present[:, None, None], context, torch.zeros_like(context))
        interval = torch.stack((time, time-state.time), -1)
        value = state.tokens + self.time(position_features(interval))[:, None, None]
        distance = (state.centers.flatten(1, 2)[:, :, None] - coordinates[:, None]).square().sum(-1)
        radius = (F.softplus(self.radius.float())+1e-4)[None, None].expand(b, k, -1).flatten(1)
        bias = -distance / (2 * radius[:, :, None].square())
        bias = bias.masked_fill(~usable[:, None], -torch.inf)
        for block in self.observation:
            flat, attention = block(value.flatten(1, 2), context, bias, return_attention=True)
            value = flat.reshape(b, k, r, d)
        centers = torch.einsum("bqn,bnd->bqd", attention.float(), coordinates.float()).reshape(b, k, r, 2)
        for block in self.memory:
            # Object-local memory tokens; interactions between objects belong to Dynamics.
            value, _ = block(value.reshape(b*k, r, d))
            value = value.reshape(b, k, r, d)
        gate = self.mix(torch.cat((state.tokens.float(), value.float()), -1))
        updated = state.tokens.float() + gate * (value.float() - state.tokens.float())
        # Preserve the observed query anchor; local carriers, not the whole state, absorb change.
        updated = torch.cat((state.tokens[:, :, :1].float(), updated[:, :, 1:]), 2)
        centers = state.centers.float() + gate.float().mean(-1, keepdim=True) * (centers-state.centers.float())
        active = present[:, None, None, None] & state.query_valid[:, :, None, None]
        return ObjectVideoStateV69(torch.where(active, updated, state.tokens), torch.where(active, centers, state.centers), time, state.query_valid)

    def forward(self, perception, queries, initial=None):
        state = initial if initial is not None else self.initial(queries, perception.times[:, 0])
        states = []
        for frame in range(perception.features.shape[1]):
            state = self.observe(state, perception.features[:, frame], perception.coordinates, perception.valid[:, frame], perception.times[:, frame])
            states.append(state)
        return states
