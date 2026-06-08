"""Pi3 feed-forward geometry wrapper (faithful to third_party/Pi3).

Loads the permutation-equivariant Pi3 model and lifts a set of RGB frames into:
    points        [N,H,W,3]  global (world == affine-invariant gauge) point map
    local_points  [N,H,W,3]  per-camera-frame point map
    conf          [N,H,W]    confidence in (0,1)  (model emits logits)
    camera_poses  [N,4,4]    camera-to-world (OpenCV)
    images        [N,3,H,W]  the preprocessed frames in [0,1]
    mask          [N,H,W]    conf>thr AND not a depth/normal edge

Matches the official example.py inference exactly (bf16 autocast, sigmoid conf
threshold 0.1, depth_normal_edge rtol 0.03).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

DEFAULT_PI3_REPO = "/mnt/pfs/public/xuhaoming/instruct_gs_world/third_party/Pi3"
DEFAULT_PI3_CKPT = "/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/Pi3/model.safetensors"


def _ensure_repo_on_path(repo: str):
    if repo not in sys.path:
        sys.path.insert(0, repo)


class Pi3Lifter:
    def __init__(
        self,
        ckpt: str | None = DEFAULT_PI3_CKPT,
        repo: str = DEFAULT_PI3_REPO,
        device: str = "cuda",
        pixel_limit: int = 255000,
    ):
        _ensure_repo_on_path(repo)
        from pi3.models.pi3 import Pi3  # noqa: E402

        self.device = torch.device(device)
        self.pixel_limit = pixel_limit
        if ckpt and os.path.isfile(ckpt):
            self.model = Pi3().to(self.device).eval()
            if ckpt.endswith(".safetensors"):
                from safetensors.torch import load_file

                self.model.load_state_dict(load_file(ckpt))
            else:
                self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
        else:
            # falls back to HF hub (requires proxy env set by the caller)
            self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()
        self._amp_dtype = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    @torch.no_grad()
    def lift(
        self,
        frames_uint8: np.ndarray,
        conf_thr: float = 0.1,
        edge_rtol: float = 0.03,
    ) -> dict:
        """frames_uint8: [N,H,W,3] RGB uint8 -> dict of tensors on cpu (float32)."""
        from pi3.utils.geometry import depth_normal_edge  # noqa: E402

        from .preprocess import preprocess_frames

        imgs = preprocess_frames(frames_uint8, pixel_limit=self.pixel_limit).to(self.device)  # [N,3,H,W]
        with torch.amp.autocast("cuda", dtype=self._amp_dtype):
            res = self.model(imgs[None])  # add batch dim

        conf_logits = res["conf"]                  # [1,N,H,W,1]
        conf = torch.sigmoid(conf_logits[..., 0])  # [1,N,H,W] in (0,1)
        masks = conf > conf_thr
        if edge_rtol and edge_rtol > 0:            # edge_rtol<=0 -> keep depth edges
            non_edge = ~depth_normal_edge(res["local_points"], rtol=edge_rtol, mask=masks)
            masks = torch.logical_and(masks, non_edge)
        masks = masks[0]  # [N,H,W]

        return {
            "points": res["points"][0].float().cpu(),          # [N,H,W,3]
            "local_points": res["local_points"][0].float().cpu(),
            "conf": conf[0].float().cpu(),                      # [N,H,W]
            "camera_poses": res["camera_poses"][0].float().cpu(),  # [N,4,4]
            "images": imgs.float().cpu(),                       # [N,3,H,W] in [0,1]
            "mask": masks.cpu(),                                # [N,H,W] bool
        }
