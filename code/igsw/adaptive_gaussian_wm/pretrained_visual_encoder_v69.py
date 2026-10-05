"""Local-weight perception adapters with native geometry and causal video prefixes."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PerceptionSequenceV69:
    features: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    native_hw: torch.Tensor
    grid_hw: torch.Tensor

    def prefix(self, count):
        return PerceptionSequenceV69(self.features[:, :count], self.coordinates, self.valid[:, :count],
                                     self.times[:, :count], self.native_hw, self.grid_hw)


class PretrainedVisualEncoderV69(nn.Module):
    def __init__(self, kind, repository, weights, frame_batch=2, dtype=torch.bfloat16, history_seconds=3.0, saved_backbone=None):
        super().__init__()
        self.kind, self.frame_batch, self.dtype = kind, frame_batch, dtype
        self.batch_across_samples = False
        self.history_seconds = history_seconds
        self.provenance = {"kind": kind, "repository": str(Path(repository).resolve()),
                           "weights": str(Path(weights).resolve()), "pretrained": True,
                           "frozen": True, "resize": "none; native image padded to multiples of 16"}
        # Architecture factories never download. Both branches load required local weights strictly.
        if kind == "dinov3_vitl16":
            self.backbone = torch.hub.load(repository, "dinov3_vitl16", source="local", pretrained=False)
            state = torch.load(weights, map_location="cpu", weights_only=False) if saved_backbone is None else saved_backbone
        else:
            self.backbone, predictor = torch.hub.load(repository, "vjepa2_1_vit_large_384", source="local", pretrained=False)
            del predictor
            if saved_backbone is None:
                saved = torch.load(weights, map_location="cpu", weights_only=False)
                state = {key.replace("module.", "").replace("backbone.", ""): value for key, value in saved["ema_encoder"].items()}
            else:
                state = saved_backbone
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)
        self.provenance["parameters"] = sum(p.numel() for p in self.backbone.parameters())

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb, pixel_valid, times, native_hw):
        if self.kind == "dinov3_vitl16" and self.batch_across_samples:
            return self._forward_dino_batch(rgb, pixel_valid, times, native_hw)
        b, t = rgb.shape[:2]
        grids = torch.div(native_hw + 15, 16, rounding_mode="floor")
        max_tokens = int(grids.prod(-1).max())
        features = torch.zeros((b, t, max_tokens, 1024), device=rgb.device, dtype=self.dtype)
        valid = torch.zeros((b, t, max_tokens), device=rgb.device, dtype=torch.bool)
        coordinates = torch.zeros((b, max_tokens, 2), device=rgb.device)
        for item in range(b):
            h, w = native_hw[item].tolist()
            gh, gw = grids[item].tolist()
            n = gh * gw
            # Do not let another image's batch padding change this image's encoder context.
            images = rgb[item, :, :, :h, :w].float() / 255.0
            images = F.pad((images - self.mean) / self.std, (0, gw * 16 - w, 0, gh * 16 - h))
            pixels = F.pad(pixel_valid[item, :, None, :h, :w].float(), (0, gw * 16 - w, 0, gh * 16 - h))
            valid[item, :, :n] = F.avg_pool2d(pixels, 16, 16)[:, 0].flatten(1) > .5
            yy, xx = torch.meshgrid(torch.arange(gh, device=rgb.device), torch.arange(gw, device=rgb.device), indexing="ij")
            xy = torch.stack((xx, yy), -1).flatten(0, 1).float() * 16 + 7.5
            coordinates[item, :n] = xy / xy.new_tensor([w - 1, h - 1]).clamp_min(1) * 2 - 1
            with torch.autocast(rgb.device.type, dtype=self.dtype, enabled=rgb.is_cuda):
                if self.kind == "dinov3_vitl16":
                    for first in range(0, t, self.frame_batch):
                        stop = min(t, first + self.frame_batch)
                        encoded = self.backbone.forward_features(images[first:stop])["x_norm_patchtokens"]
                        features[item, first:stop, :n] = encoded.to(self.dtype)
                else:
                    for frame in range(t):
                        if not bool(pixel_valid[item, frame].any()):
                            continue
                        observed = torch.where((times[item, :frame+1] >= times[item, frame] - self.history_seconds)
                                               & pixel_valid[item, :frame+1].flatten(1).any(-1))[0]
                        clip = images[observed]
                        if len(clip) > 1 and len(clip) % 2:
                            clip = torch.cat((clip[:1], clip), 0)
                        encoded = self.backbone(clip.permute(1, 0, 2, 3)[None])
                        features[item, frame, :n] = encoded[0, -n:].to(self.dtype)
        return PerceptionSequenceV69(features, coordinates, valid, times, native_hw, grids)

    def _forward_dino_batch(self, rgb, pixel_valid, times, native_hw):
        b, t = rgb.shape[:2]
        shapes = native_hw.tolist()
        grids = torch.div(native_hw + 15, 16, rounding_mode="floor")
        max_tokens = int(grids.prod(-1).max())
        features = torch.zeros((b, t, max_tokens, 1024), device=rgb.device, dtype=self.dtype)
        valid = torch.zeros((b, t, max_tokens), device=rgb.device, dtype=torch.bool)
        coordinates = torch.zeros((b, max_tokens, 2), device=rgb.device)
        groups = {}
        for item, (h, w) in enumerate(shapes):
            groups.setdefault((h, w), []).append(item)
        for (h, w), members in groups.items():
            gh, gw = (h + 15) // 16, (w + 15) // 16
            n = gh * gw
            yy, xx = torch.meshgrid(torch.arange(gh, device=rgb.device), torch.arange(gw, device=rgb.device), indexing="ij")
            xy = torch.stack((xx, yy), -1).flatten(0, 1).float() * 16 + 7.5
            coordinates[members, :n] = xy / xy.new_tensor([w - 1, h - 1]).clamp_min(1) * 2 - 1
            # Normalize only the current frame microbatch, not the whole RGB batch.
            items = torch.tensor(members, device=rgb.device).repeat_interleave(t)
            frames = torch.arange(t, device=rgb.device).repeat(len(members))
            for first in range(0, len(items), self.frame_batch):
                ii, tt = items[first:first+self.frame_batch], frames[first:first+self.frame_batch]
                images = rgb[ii, tt, :, :h, :w].float() / 255.0
                images = F.pad((images - self.mean) / self.std, (0, gw * 16 - w, 0, gh * 16 - h))
                pixels = F.pad(pixel_valid[ii, tt, None, :h, :w].float(), (0, gw * 16 - w, 0, gh * 16 - h))
                valid[ii, tt, :n] = F.avg_pool2d(pixels, 16, 16)[:, 0].flatten(1) > .5
                with torch.autocast(rgb.device.type, dtype=self.dtype, enabled=rgb.is_cuda):
                    encoded = self.backbone.forward_features(images)["x_norm_patchtokens"]
                features[ii, tt, :n] = encoded.to(self.dtype)
        return PerceptionSequenceV69(features, coordinates, valid, times, native_hw, grids)


def sample_perception_v69(perception, coordinates, frame_indices):
    """Bilinear readout at native measurement coordinates; never defines object identity."""
    b, t, p = coordinates.shape[:3]
    output, validity = [], []
    for item in range(b):
        gh, gw = perception.grid_hw[item].tolist()
        h, w = perception.native_hw[item].tolist()
        values = perception.features[item, frame_indices[item], :gh*gw].float().reshape(t, gh, gw, -1).permute(0, 3, 1, 2)
        xy = (coordinates[item].float() + 1) * .5 * coordinates.new_tensor([w - 1, h - 1])
        xy = (xy - 7.5) / xy.new_tensor([max(1, (gw-1)*16), max(1, (gh-1)*16)]) * 2 - 1
        sampled = F.grid_sample(values, xy[:, None], align_corners=True, padding_mode="border")[:, :, 0].transpose(1, 2)
        mask = perception.valid[item, frame_indices[item], :gh*gw].reshape(t, 1, gh, gw).float()
        measured = F.grid_sample(mask, xy[:, None], align_corners=True, padding_mode="border")[:, 0, 0] > .5
        output.append(sampled)
        validity.append(measured & (coordinates[item].abs().amax(-1) <= 1))
    return F.normalize(torch.stack(output), dim=-1), torch.stack(validity)
