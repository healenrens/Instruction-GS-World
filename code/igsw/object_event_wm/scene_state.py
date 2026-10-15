"""Unpartitioned scene-token control with the same state-token budget."""

import torch
from torch.nn import functional as F

from .state import ObjectFeatureRenderer, ObjectStateEncoder, position_features


class SceneStateEncoder(ObjectStateEncoder):
    def observation_weights(self, scores, valid):
        weights = scores.softmax(-1) * valid[:, None, None].float()
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

    def mix_tokens(self, tokens):
        return self.output_norm(self.memory(tokens.flatten(1, 2))).reshape_as(tokens)


class SceneFeatureRenderer(ObjectFeatureRenderer):
    def __init__(self, config):
        super().__init__(config, output_mask=False)

    def forward(self, state, xy):
        # K/R remain storage axes; no group has its own field or object mask.
        tokens, centers = state.tokens.flatten(1, 2), state.centers.flatten(1, 2)
        root = tokens.mean(1)[:, None].expand(-1, len(xy), -1)
        relative = xy[None] - centers.mean(1)[:, None]
        distance = (xy[None, :, None]-centers[:, None]).square().sum(-1)
        weights = (-distance/(F.softplus(self.local_radius.float()).square()+1e-4)).softmax(-1)
        local = torch.einsum("bnr,brd->bnd", weights.to(tokens.dtype), tokens)
        features = self.field(torch.cat((root.float(), local.float(), position_features(relative)), -1))
        return features.float(), features.new_ones((len(tokens), 1, len(xy)), dtype=torch.float32)
