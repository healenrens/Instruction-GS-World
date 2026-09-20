"""Frozen native-tile appearance targets, retaining original DINO/SigLIP widths."""

import torch
from torch.nn import functional as F

from .native_local_feature_field_v65 import (
    NativeTiledPerceptionRuntimeV65, NativeTokenFieldV65, _token_coordinates,
    pool_native_local_features_v65,
)


class GroundedAppearanceTeacherV68(NativeTiledPerceptionRuntimeV65):
    @torch.no_grad()
    def _encode_dino(self, tiles, validity, starts, native_hw, batch, frames):
        outputs, masks = [], []
        for start in range(0, len(tiles), self.dino.frame_batch):
            stop = start + self.dino.frame_batch
            images, valid = tiles[start:stop].float() / 255.0, validity[start:stop, None].float()
            encoded = self.dino.backbone.forward_features((((images - self.dino.mean) / self.dino.std) * valid).to(self.dino.dtype))
            outputs.append(F.normalize(encoded[:, self.dino.prefix_tokens:].float(), dim=-1))
            masks.append(F.avg_pool2d(valid, self.dino.patch_size, self.dino.patch_size).flatten(1).ge(.5))
        features = torch.cat(outputs).reshape(batch, frames, -1, self.config.dino_dim)
        valid = torch.cat(masks).reshape(batch, frames, -1)
        grid = self.config.native_tile_size // self.dino.patch_size
        coordinates = _token_coordinates(starts, grid, self.dino.patch_size, native_hw, features.device)
        return NativeTokenFieldV65(features, coordinates[:, None].expand(batch, frames, -1, -1), valid, native_hw)

    @torch.no_grad()
    def _encode_siglip(self, tiles, validity, starts, native_hw, batch, frames):
        outputs, masks = [], []
        for start in range(0, len(tiles), self.siglip.frame_batch):
            stop = start + self.siglip.frame_batch
            images, valid = tiles[start:stop].float() / 255.0, validity[start:stop, None].float()
            encoded = self.siglip.model(pixel_values=(images - .5) / .5)
            outputs.append(F.normalize(encoded.last_hidden_state.float(), dim=-1))
            masks.append(F.avg_pool2d(valid, self.siglip.patch_size, self.siglip.patch_size).flatten(1).ge(.5))
        features = torch.cat(outputs).reshape(batch, frames, -1, self.config.siglip_dim)
        valid = torch.cat(masks).reshape(batch, frames, -1)
        grid = self.config.native_tile_size // self.siglip.patch_size
        coordinates = _token_coordinates(starts, grid, self.siglip.patch_size, native_hw, features.device)
        return NativeTokenFieldV65(features, coordinates[:, None].expand(batch, frames, -1, -1), valid, native_hw)

    @torch.no_grad()
    def __call__(self, batch):
        current = {**batch, "video_rgb": batch["video_rgb"][:, 3:4], "video_pixel_valid": batch["video_pixel_valid"][:, 3:4]}
        fields = super().__call__(current)
        coordinates = batch["coordinates"][:, 3:4]
        dino = pool_native_local_features_v65(fields.dino, coordinates, self.config.local_radii_pixels, self.config.local_tokens_per_scale)
        siglip = pool_native_local_features_v65(fields.siglip, coordinates, self.config.local_radii_pixels, self.config.local_tokens_per_scale)
        return {"features": torch.cat((dino.features, siglip.features), -1), "valid": dino.valid & siglip.valid}
