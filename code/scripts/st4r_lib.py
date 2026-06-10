"""St4RTrack thin wrapper for pure-video data-gen on LIBERO (agent.md §46, notes/research_pure_video_4d.md §P3).

St4RTrack (third_party/St4RTrack, weights checkpoints/model.safetensors) is a DUST3R-derived dual-branch net:
with anchor_view=0 it takes the RGB clip and, for every (frame0, frame_i) pair, returns
  - pred1['pts3d']               : frame-0 geometry IN frame-0 world  (reconstruction branch; ~constant over pairs)
  - pred2['pts3d_in_other_view'] : WHERE each frame-0 pixel is at frame i, IN frame-0 world (tracking branch)
  - pred1/2['conf']              : per-pixel confidence (occlusion / reliability proxy; St4RTrack has no explicit p_vis)
so the per-pixel 3D WORLD TRAJECTORY of frame-0's pixels is exactly stack_i(pred2_i['pts3d_in_other_view']).
Camera intrinsics (focal) come from estimate_focal_knowing_depth on the frame-0 pointmap (pp=centre); per-frame
camera pose from PnP of the frame-0 pixels vs their pred2 3D positions (== oneref_viewer's intrinsics/pair solver).

This module ONLY runs the net + recovers {pointmap0, tracks[T,H,W,3], conf[T,H,W], focal, poses_c2w[T]} and resizing
maps. It is import-safe in the main venv (torch 2.8); no gradio/no global align needed. Used by st4r_stage1.py
(validation) and video_gt.py (clip emission)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ST4R = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "St4RTrack")
_ST4R = os.path.abspath(_ST4R)
if _ST4R not in sys.path:
    sys.path.insert(0, _ST4R)


def load_model(weights: str | None = None, device: str = "cuda"):
    from dust3r.model import AsymmetricCroCo3DStereo
    weights = weights or os.path.join(_ST4R, "checkpoints")
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device).eval()
    return model


def _make_view_from_array(rgb_uint8: np.ndarray, idx: int, size: int = 512,
                          square_ok: bool = True, crop: bool = False):
    """Replicate dust3r.utils.image.load_images for a single in-memory RGB frame (square LIBERO 256²).
    square_ok=True + crop=False => keep the FULL square frame, just RESIZE the long side to `size`
    (no vertical 4:3 crop), so every source pixel survives and the model_px->source_px map is uniform.
    Returns the view dict + (W,H) resized true_shape + source size."""
    import PIL.Image
    from dust3r.utils.image import crop_img, ImgNorm
    H0, W0 = rgb_uint8.shape[:2]
    pil = PIL.Image.fromarray(rgb_uint8).convert("RGB")
    pil = crop_img(pil, size, square_ok=square_ok, crop=crop)
    W, H = pil.size
    img = ImgNorm(pil)[None]
    view = dict(img=img, true_shape=np.int32([[H, W]]),
                idx=idx, instance=str(idx))
    return view, (W, H), (W0, H0)


@torch.no_grad()
def run_st4rtrack(model, frames_rgb: list[np.ndarray], device: str = "cuda",
                  size: int = 512, batch_size: int = 64, square_ok: bool = True, crop: bool = False):
    """frames_rgb: list of [H0,W0,3] uint8 RGB (frame 0 first = the anchor).
    Returns dict:
      pts0     [H,W,3]   frame-0 geometry in frame-0 world (reconstruction branch, from the (0,0)/first pair)
      tracks   [T,H,W,3] per-pixel 3D world track of frame-0 pixels (T=len(frames); tracks[0]==pts0-ish)
      conf     [T,H,W]   tracking-branch confidence per frame
      colors0  [H,W,3]   frame-0 RGB in [0,1] at model resolution
      rgb_t    [T,H,W,3] each frame's RGB in [0,1] at model resolution (for render validation)
      H,W                model resolution (e.g. 512)
      src_hw   (H0,W0)   source resolution
      scale_xy (sx,sy)   model_px -> source_px multipliers (source = model/scale)
    """
    from dust3r.inference import inference
    from dust3r.utils.image import rgb as _rgb
    views = []
    res_wh = src_wh = None
    for i, fr in enumerate(frames_rgb):
        v, res_wh, src_wh = _make_view_from_array(fr, i, size=size, square_ok=square_ok, crop=crop)
        views.append(v)
    if len(views) == 1:
        import copy
        views = [views[0], copy.deepcopy(views[0])]
        views[1]["idx"] = 1
        views[1]["instance"] = "1"
    out = inference(views, model, device, batch_size=batch_size, verbose=False, anchor_view=0)
    # out is a flat list of pair results; with anchor_view=0 pair k pairs (anchor, frame k)
    W, H = res_wh
    W0, H0 = src_wh
    T = len(frames_rgb)
    tracks = np.zeros((T, H, W, 3), np.float32)
    conf = np.zeros((T, H, W), np.float32)
    pts0 = None
    colors0 = None
    rgb_t = np.zeros((T, H, W, 3), np.float32)
    seen = 0
    for entry in out:
        pred1, pred2 = entry["pred1"], entry["pred2"]
        view2 = entry["view2"]
        p1 = pred1["pts3d"].detach().cpu().numpy()                  # [B,H,W,3] frame-0 geom
        p2 = pred2["pts3d_in_other_view"].detach().cpu().numpy()    # [B,H,W,3] frame-0 px @ frame j
        c2 = pred2["conf"].detach().cpu().numpy()                   # [B,H,W]
        B = p2.shape[0]
        for b in range(B):
            j = seen
            if j >= T:
                break
            tracks[j] = p2[b]
            conf[j] = c2[b]
            rgb_t[j] = _rgb(view2["img"][b])
            if pts0 is None:
                pts0 = p1[b]
                # colors0 from the anchor view1 img
                colors0 = _rgb(entry["view1"]["img"][b])
            seen += 1
    if pts0 is None:                                                # degenerate
        pts0 = tracks[0]
        colors0 = rgb_t[0]
    return dict(pts0=pts0.astype(np.float32), tracks=tracks, conf=conf,
                colors0=colors0.astype(np.float32), rgb_t=rgb_t,
                H=H, W=W, src_hw=(H0, W0),
                scale_xy=(W / float(W0), H / float(H0)))


def estimate_focal(pts0: np.ndarray) -> float:
    """Focal (pixels, at model resolution) from the frame-0 pointmap, pp=centre. == oneref_viewer.intrinsics_solver."""
    from dust3r.post_process import estimate_focal_knowing_depth
    H, W = pts0.shape[:2]
    pp = torch.tensor([W / 2.0, H / 2.0])
    t = torch.from_numpy(pts0)[None].float()
    f = float(estimate_focal_knowing_depth(t, pp, focal_mode="weiszfeld"))
    return f


def solve_pose_c2w(pts3d_world: np.ndarray, focal: float, conf: np.ndarray | None = None,
                   conf_thr: float = 1.5):
    """PnP: given frame-0 pixels (grid) whose 3D WORLD positions at frame j are pts3d_world[H,W,3] (the track),
    recover the camera-from-world (w2c) and its inverse c2w for frame j. Mirrors oneref_viewer.pair_solver.
    Returns (c2w[4,4], ok)."""
    import cv2
    H, W = pts3d_world.shape[:2]
    pp = np.array([W / 2.0, H / 2.0], np.float32)
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([uu, vv], axis=-1).astype(np.float32)        # [H,W,2]
    K = np.float32([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]])
    msk = pts3d_world[..., 2] > 1e-5
    if conf is not None:
        msk &= (conf > conf_thr)
    if msk.sum() < 16:
        msk = pts3d_world[..., 2] > 1e-5
    obj = pts3d_world[msk].astype(np.float32)
    img = pixels[msk].astype(np.float32)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, K, None, iterationsCount=100,
            reprojectionError=5.0, flags=cv2.SOLVEPNP_SQPNP)
    except Exception:
        ok = False
    if not ok:
        return np.eye(4, dtype=np.float32), False
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec[:, 0]
    c2w = np.linalg.inv(w2c)
    return c2w.astype(np.float32), True
