"""Frozen, matched DINOv2/RAE assets with gradients through spatial decoding."""

import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


def letterbox_video(batch, size):
    images, masks = [], []
    for frames, hw in zip(batch["rgb"], batch["native_hw"]):
        h, w = hw.tolist()
        scale = size / max(h, w)
        rh, rw = round(h * scale), round(w * scale)
        resized = F.interpolate(frames[:, :, :h, :w].float()/255,
                                size=(rh, rw), mode="bicubic", align_corners=False).clamp(0, 1)
        left, top = (size-rw)//2, (size-rh)//2
        image = F.pad(resized, (left, size-rw-left, top, size-rh-top))
        mask = frames.new_zeros((len(frames), 1, size, size), dtype=torch.bool)
        mask[:, :, top:top+rh, left:left+rw] = True
        images.append(image)
        masks.append(mask)
    return torch.stack(images), torch.stack(masks) & batch["frame_valid"][:, :, None, None, None]


class FrozenVisualCodec(nn.Module):
    def __init__(self, assets, frame_batch=4, decode_batch=2):
        super().__init__()
        config = json.loads(Path(assets).read_text())
        sys.path.insert(0, str(Path(config["rae_repository"]) / "src"))
        from stage1.decoders import GeneralDecoder
        from stage1.decoders.utils import ViTMAEConfig

        self.asset_config = config
        self.image_size = config["image_size"]
        self.encoder_size = self.image_size * 14 // 16
        self.grid_size = self.image_size // 16
        self.feature_dim = 768
        self.frame_batch, self.decode_batch = frame_batch, decode_batch
        self.encoder = Dinov2WithRegistersModel.from_pretrained(
            config["encoder"], local_files_only=True)
        # The released RAE decoder is matched to a non-affine final DINO norm.
        self.encoder.layernorm.elementwise_affine = False
        self.encoder.layernorm.weight = None
        self.encoder.layernorm.bias = None
        processor = AutoImageProcessor.from_pretrained(config["encoder"], local_files_only=True)
        self.register_buffer("image_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(processor.image_std).view(1, 3, 1, 1))
        decoder_config = ViTMAEConfig.from_pretrained(config["decoder_config"], local_files_only=True)
        decoder_config.hidden_size = self.feature_dim
        decoder_config.patch_size = 16
        decoder_config.image_size = self.image_size
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.grid_size**2)
        self.decoder.load_state_dict(torch.load(config["decoder"], map_location="cpu", weights_only=True))
        stats = torch.load(config["statistics"], map_location="cpu", weights_only=True)
        self.register_buffer("latent_mean", stats["mean"])
        self.register_buffer("latent_scale", (stats["var"] + 1e-5).sqrt())
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.decoder.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        b, t = images.shape[:2]
        rows = []
        for frames in images.flatten(0, 1).split(self.frame_batch):
            resized = F.interpolate(frames, (self.encoder_size, self.encoder_size),
                                    mode="bicubic", align_corners=False)
            value = (resized - self.image_mean) / self.image_std
            tokens = self.encoder(value).last_hidden_state[:, 5:]
            latent = tokens.transpose(1, 2).reshape(-1, self.feature_dim, self.grid_size, self.grid_size)
            latent = (latent.float() - self.latent_mean) / self.latent_scale
            rows.append(latent.flatten(2).transpose(1, 2))
        return torch.cat(rows).reshape(b, t, self.grid_size**2, self.feature_dim)

    def _decode(self, tokens):
        latent = tokens.transpose(1, 2).reshape(-1, self.feature_dim, self.grid_size, self.grid_size)
        latent = latent * self.latent_scale + self.latent_mean
        tokens = latent.flatten(2).transpose(1, 2)
        pixels = self.decoder(tokens, drop_cls_token=False).logits
        return self.decoder.unpatchify(pixels) * self.image_std + self.image_mean

    def decode(self, tokens):
        shape = tokens.shape[:-2]
        rows = []
        for value in tokens.reshape(-1, *tokens.shape[-2:]).split(self.decode_batch):
            if torch.is_grad_enabled() and value.requires_grad:
                rows.append(checkpoint(self._decode, value, use_reentrant=False))
            else:
                rows.append(self._decode(value))
        return torch.cat(rows).reshape(*shape, 3, self.image_size, self.image_size)

    def patch_valid(self, pixel_valid):
        shape = pixel_valid.shape[:2]
        return F.avg_pool2d(pixel_valid.flatten(0, 1).float(), 16).flatten(1).reshape(*shape, -1) > 0
