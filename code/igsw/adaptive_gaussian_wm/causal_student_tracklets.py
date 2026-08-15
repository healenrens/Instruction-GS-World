"""Deployable causal tracklet observations built only from frozen DINO history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StudentTracklets:
    features: torch.Tensor
    residual_flow: torch.Tensor
    confidence: torch.Tensor


class CausalStudentTrackletEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = 2 * config.patch_dim + 3
        self.update = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.patch_dim),
            nn.GELU(),
            nn.Linear(config.patch_dim, config.patch_dim),
        )

    @torch.no_grad()
    def _transport(self, current, previous, current_xy, previous_xy, current_valid, previous_valid):
        similarity = torch.einsum(
            "bnd,bmd->bnm",
            F.normalize(current.float(), dim=-1, eps=1e-6),
            F.normalize(previous.float(), dim=-1, eps=1e-6),
        )
        distance = (current_xy[:, :, None].float() - previous_xy[:, None].float()).square().sum(-1)
        logits = similarity / self.config.student_tracklet_temperature
        logits = logits - distance / (2.0 * self.config.student_tracklet_spatial_sigma**2)
        pair_valid = current_valid[:, :, None] & previous_valid[:, None]
        weights = logits.masked_fill(~pair_valid, -1e4).softmax(dim=-1)
        weights = weights * current_valid[..., None].float()
        transported_feature = torch.einsum("bnm,bmd->bnd", weights, previous.float())
        transported_position = torch.einsum("bnm,bmd->bnd", weights, previous_xy.float())
        confidence = weights.amax(dim=-1) * current_valid.float()
        return transported_feature, transported_position, confidence

    def forward(self, patches, coordinates, valid) -> StudentTracklets:
        if patches.ndim != 4 or coordinates.shape != (*patches.shape[:3], 2):
            raise ValueError("student tracklet patch/coordinate shapes differ")
        features, flows, confidences = [], [], []
        for index in range(patches.shape[1]):
            current = patches[:, index]
            if index == 0:
                transported = current.detach().float()
                flow = torch.zeros_like(coordinates[:, index].float())
                confidence = valid[:, index].float()
            else:
                transported, transported_position, confidence = self._transport(
                    current,
                    patches[:, index - 1],
                    coordinates[:, index],
                    coordinates[:, index - 1],
                    valid[:, index],
                    valid[:, index - 1],
                )
                flow = coordinates[:, index].float() - transported_position
            inputs = torch.cat(
                (
                    current.float(),
                    current.float() - transported,
                    flow,
                    confidence[..., None],
                ),
                dim=-1,
            )
            update = self.update(inputs)
            tracklet = F.normalize(
                current.float() + 0.25 * torch.tanh(update.float()), dim=-1, eps=1e-6
            )
            features.append(tracklet.to(current.dtype))
            flows.append(flow)
            confidences.append(confidence)
        return StudentTracklets(
            features=torch.stack(features, dim=1),
            residual_flow=torch.stack(flows, dim=1),
            confidence=torch.stack(confidences, dim=1),
        )
