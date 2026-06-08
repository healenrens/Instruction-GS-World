"""CoTracker3 wrapper + Pi3-pointmap sampling for ground-truth Gaussian trajectories.

Our Gaussians are born 1:1 from anchor-frame pixels, so a Gaussian's correspondence
over time = the 2D track of its anchor pixel. CoTracker3 (frozen) produces those
tracks + visibility; sampling Pi3's per-frame point maps (joint gauge) at the tracked
locations yields the GROUND-TRUTH 3D trajectory each Gaussian should follow — the
direct, geometry-level supervision (vs the indirect 2D render loss).
"""

from __future__ import annotations

import torch

DEFAULT_COTRACKER_CKPT = (
    "/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/cotracker/scaled_offline.pth"
)


class CoTrackerTracker:
    def __init__(self, ckpt: str = DEFAULT_COTRACKER_CKPT, device: str = "cuda"):
        from cotracker.predictor import CoTrackerPredictor
        self.device = torch.device(device)
        self.model = CoTrackerPredictor(checkpoint=ckpt, v2=False, offline=True).to(self.device).eval()

    @torch.no_grad()
    def track(self, frames_uint8or01: torch.Tensor, queries_xy: torch.Tensor):
        """frames: [T,3,H,W] in [0,1] (or [0,255]); queries_xy: [M,2] (x,y) at frame 0.
        Returns (tracks [T,M,2] in pixel coords, vis [T,M] bool)."""
        vid = frames_uint8or01
        if vid.dtype != torch.float32:
            vid = vid.float()
        if vid.max() <= 1.5:
            vid = vid * 255.0
        vid = vid[None].to(self.device)                       # [1,T,3,H,W]
        m = queries_xy.shape[0]
        q = torch.zeros(1, m, 3, device=self.device)
        q[0, :, 0] = 0.0                                       # query at frame 0
        q[0, :, 1:] = queries_xy.to(self.device)
        tracks, vis = self.model(vid, queries=q)              # [1,T,M,2], [1,T,M]
        return tracks[0], vis[0].bool()


def sample_pointmaps_at(points_all: torch.Tensor, tracks_xy: torch.Tensor) -> torch.Tensor:
    """points_all [T,H,W,3] (joint-gauge per-frame point maps), tracks_xy [T,M,2] (x,y px)
    -> sampled 3D positions [T,M,3] via bilinear grid_sample."""
    import torch.nn.functional as F
    T, H, W, _ = points_all.shape
    M = tracks_xy.shape[1]
    pm = points_all.permute(0, 3, 1, 2)                       # [T,3,H,W]
    # normalize pixel coords to [-1,1]
    gx = tracks_xy[..., 0] / max(W - 1, 1) * 2 - 1
    gy = tracks_xy[..., 1] / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[:, :, None, :]      # [T,M,1,2]
    out = F.grid_sample(pm, grid, mode="bilinear", align_corners=True)  # [T,3,M,1]
    return out[..., 0].permute(0, 2, 1)                       # [T,M,3]
