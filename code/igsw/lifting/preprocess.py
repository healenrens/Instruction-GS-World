"""Image preprocessing for the feed-forward geometry models.

Replicates Pi3's `load_images_as_tensor` resize logic EXACTLY (no simplification):
constrain total pixels to PIXEL_LIMIT, snap both sides to multiples of the ViT
patch size (14), LANCZOS resize, return float tensor in [0,1] as [N,3,H,W].
ImageNet normalization is applied *inside* the Pi3/VGGT models, not here.

Reference: third_party/Pi3/pi3/utils/basic.py::load_images_as_tensor
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


def compute_target_size(w_orig: int, h_orig: int, pixel_limit: int = 255000, patch: int = 14):
    """Pi3's exact target-size rule for an input of size (w_orig, h_orig)."""
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / patch), round(h_target / patch)
    while (k * patch) * (m * patch) > pixel_limit:
        if (k / m) > (w_target / h_target):
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch  # (TARGET_W, TARGET_H)


def preprocess_frames(
    frames: np.ndarray,
    pixel_limit: int = 255000,
    patch: int = 14,
    resample: int = Image.Resampling.LANCZOS,
) -> torch.Tensor:
    """uint8 [N,H,W,3] RGB -> float32 [N,3,Ht,Wt] in [0,1], Pi3-faithful resize.

    The target size is computed from the FIRST frame (matches Pi3, which assumes
    a uniform stream). All frames are resized to that common size for stacking.
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"expect [N,H,W,3], got {frames.shape}"
    n, h0, w0, _ = frames.shape
    tw, th = compute_target_size(w0, h0, pixel_limit, patch)
    out = torch.empty((n, 3, th, tw), dtype=torch.float32)
    for i in range(n):
        img = Image.fromarray(frames[i]).convert("RGB").resize((tw, th), resample)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        out[i] = torch.from_numpy(arr).permute(2, 0, 1)
    return out
