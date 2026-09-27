"""Compositional fields: measurements query compact states, never construct them."""

import torch
from torch import nn
from torch.nn import functional as F

from .object_video_attention_v69 import position_features, masked_probability_v69


class ObjectSequenceReadoutV69(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.width, config.readout_width
        self.local_radius = nn.Parameter(torch.tensor(-1.5))
        self.field = nn.Sequential(nn.LayerNorm(d*2+18), nn.Linear(d*2+18, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.appearance = nn.Linear(h, config.perception_dim)
        self.position = nn.Linear(h, 2)
        self.observation = nn.Linear(h, 1)
        self.binding_query = nn.Linear(d, d)
        self.binding_feature = nn.Linear(config.perception_dim, d)
        self.binding_geometry = nn.Sequential(nn.Linear(18, h), nn.GELU(), nn.Linear(h, 1))
        self.unbound = nn.Parameter(torch.zeros(()))

    def local(self, state, coordinates, local_coordinates=None):
        relative = coordinates[:, :, None].float() - state.centers[:, None, :, 0].float() if local_coordinates is None else local_coordinates.float()
        carrier_relative = state.centers.float()-state.centers[:, :, :1].float()
        distance = (relative[:, :, :, None] - carrier_relative[:, None]).square().sum(-1)
        attention = (-distance / (2*(F.softplus(self.local_radius.float())+1e-4).square())).softmax(-1)
        local = torch.einsum("bpkr,bkrd->bpkd", attention.to(state.tokens.dtype), state.tokens)
        root = state.tokens[:, None, :, 0].expand(-1, coordinates.shape[1], -1, -1)
        return local, root, relative

    def binding(self, state, coordinates, point_features):
        local, root, relative = self.local(state, coordinates)
        query = F.normalize(self.binding_query((local+root).float()).float(), dim=-1)
        measured = F.normalize(self.binding_feature(point_features.float()).float(), dim=-1)
        logits = 8 * (query * measured[:, :, None]).sum(-1) + self.binding_geometry(position_features(relative)).squeeze(-1)
        logits = logits.masked_fill(~state.query_valid[:, None], -torch.inf)
        return torch.cat((logits, self.unbound.float().expand(*logits.shape[:-1], 1)), -1)

    def forward(self, state, coordinates, ownership, object_keep=None, local_coordinates=None):
        local, root, relative = self.local(state, coordinates, local_coordinates)
        hidden = self.field(torch.cat((root.float(), local.float(), position_features(relative)), -1))
        fields = {"appearance": self.appearance(hidden).float(),
                  "position": state.centers[:, None, :, 0].float() + relative + self.position(hidden).float(),
                  "observation": self.observation(hidden).float()}
        # Last binding channel is unbound evidence, not a claimed background object.
        mixture = ownership[..., :state.tokens.shape[1]].float()
        if object_keep is not None:
            # Delete a component without renormalizing or asking surviving slots to replace it.
            mixture = mixture * object_keep[:, None].float()
        return {name: (value * mixture[..., None]).sum(2) for name, value in fields.items()}


@torch.no_grad()
def appearance_binding_target_v69(queries, measured, image_features, image_valid):
    affinity = torch.einsum("bpc,bkc->bpk", F.normalize(measured.float(), dim=-1), F.normalize(queries.features.float(), dim=-1))
    image_affinity = torch.einsum("bnc,bkc->bnk", F.normalize(image_features.float(), dim=-1), F.normalize(queries.features.float(), dim=-1))
    weight = image_valid[..., None].float()
    count = weight.sum(1).clamp_min(1)
    mean = (image_affinity * weight).sum(1) / count
    variance = ((image_affinity-mean[:, None]).square()*weight).sum(1) / count
    spread = variance.sqrt()
    standardized = ((affinity-mean[:, None])/spread[:, None].clamp_min(1e-4)).clamp(-8, 8)
    logits = torch.cat((standardized, standardized.new_zeros((*standardized.shape[:-1], 1))), -1)
    allowed = torch.cat((queries.valid, queries.valid.new_ones((len(queries.valid), 1))), -1)
    evidence = (standardized.abs() * queries.valid[:, None]).amax(-1).clamp(max=4) / 4
    return masked_probability_v69(logits, allowed[:, None]), evidence


def binding_probability_v69(logits, query_valid):
    valid = torch.cat((query_valid, query_valid.new_ones((len(query_valid), 1))), -1)
    while valid.ndim < logits.ndim:
        valid = valid.unsqueeze(1)
    return masked_probability_v69(logits, valid)
