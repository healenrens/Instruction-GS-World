"""GPSToken-style spatially-adaptive token placement (paper Algorithm 1, arXiv 2509.01109) +
motion/task-saliency boost + lift-to-control-index — the single source of truth shared by the viz
script (scripts/gpstoken_init.py) and the trainer (scripts/train_sim.py).

Core idea (§93): place a small set of tokens on INFORMATION-RICH regions via entropy-driven recursive
partition, then (for the world model) BOOST the placement by motion/task saliency so tokens land on the
MOVERS, and bind each token to its nearest dense Gaussian -> a smarter ctrl_idx (keypoint selection),
leaving the per-control translation field untouched. Registration across frames is then free: a token IS
a persistent dense index whose 3D trajectory the SC-GS rollout already carries.

Training-free (pure numpy/cv2 for placement). cv2 imported lazily so importing this module is light.
"""
from __future__ import annotations

import numpy as np
import torch


def grad_mag(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a [H,W] float grayscale image."""
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def complexity(E: np.ndarray, r, lam: float, gmax: float, sal=None, beta: float = 0.0, nbins: int = 512) -> float:
    """Region complexity m = h*w*H^lam (H = gradient-magnitude histogram entropy). Optional motion/task
    saliency boost: m *= (1 + beta*mean_saliency_in_region) so movers get split deeper -> more tokens."""
    x, y, w, h = r
    sub = E[y:y + h, x:x + w]
    if sub.size == 0:
        return 0.0
    hist, _ = np.histogram(sub, bins=nbins, range=(0.0, gmax))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    nz = p[p > 0]
    Hh = float(-(nz * np.log(nz)).sum())
    m = float(h) * float(w) * (Hh ** lam)
    if sal is not None and beta > 0.0:
        m = m * (1.0 + beta * float(sal[y:y + h, x:x + w].mean()))
    return m


def gpstoken_init(E: np.ndarray, l: int, lam: float = 2.5, smin: int = 4, sal=None, beta: float = 0.0):
    """Paper Algorithm 1: recursively split the MOST-complex region until l regions. Rect -> split longer
    side; square -> try width/height split, keep the one with smaller min-complexity. Returns
    (regions [(x,y,w,h)], gaussians [(mux,muy,sigx,sigy)]). sal/beta add the motion-saliency boost.

    Complexity is CACHED per region (only the 2 new sub-regions are recomputed each split) -> O(l) total
    histograms instead of O(l^2) (the naive recompute-all was ~16s/clip at l=256; cached ~0.5s)."""
    H, W = E.shape
    gmax = float(E.max()) + 1e-6
    regions = [(0, 0, W, H)]
    comps = [complexity(E, regions[0], lam, gmax, sal, beta)]   # cached complexity per region
    while len(regions) < l:
        imax, mmax = -1, -1.0
        for i, (x, y, w, h) in enumerate(regions):
            if (w > smin or h > smin) and comps[i] > mmax:
                imax, mmax = i, comps[i]
        if imax < 0:
            break
        x, y, w, h = regions[imax]
        if w != h:
            if w > h:
                w1 = w // 2
                subs = [(x, y, w1, h), (x + w1, y, w - w1, h)]
            else:
                h1 = h // 2
                subs = [(x, y, w, h1), (x, y + h1, w, h - h1)]
            sub_c = [complexity(E, subs[0], lam, gmax, sal, beta), complexity(E, subs[1], lam, gmax, sal, beta)]
        else:
            w1, h1 = w // 2, h // 2
            I1, I2 = (x, y, w1, h), (x + w1, y, w - w1, h)
            I3, I4 = (x, y, w, h1), (x, y + h1, w, h - h1)
            m1, m2 = complexity(E, I1, lam, gmax, sal, beta), complexity(E, I2, lam, gmax, sal, beta)
            m3, m4 = complexity(E, I3, lam, gmax, sal, beta), complexity(E, I4, lam, gmax, sal, beta)
            if min(m1, m2) <= min(m3, m4):
                subs, sub_c = [I1, I2], [m1, m2]
            else:
                subs, sub_c = [I3, I4], [m3, m4]
        regions = regions[:imax] + subs + regions[imax + 1:]
        comps = comps[:imax] + sub_c + comps[imax + 1:]
    gauss = [(x + w / 2.0, y + h / 2.0, w / 6.0, h / 6.0) for (x, y, w, h) in regions]
    return regions, gauss


def mover_saliency(uv: torch.Tensor, disp: torch.Tensor, n_keep: int, H: int, W: int,
                   thresh: float = 0.01, blur: int = 31) -> np.ndarray:
    """GT-motion saliency map [H,W] in [0,1]: splat the movers (disp>thresh) at their frame-0 uv, blur,
    normalize. This is the TRAINING-time oracle saliency (§93-T2: lifts cluttered-scene concentration
    1.08x->8.14x). At INFERENCE this map is replaced by the frozen-Qwen relevance grid (no GT)."""
    import cv2
    mv = (disp[:n_keep] > thresh).float().cpu().numpy()
    uvn = uv[:n_keep].cpu().numpy()
    img = np.zeros((H, W), np.float32)
    xs = np.clip(uvn[:, 0].astype(int), 0, W - 1)
    ys = np.clip(uvn[:, 1].astype(int), 0, H - 1)
    np.maximum.at(img, (ys, xs), mv)
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img / max(float(img.max()), 1e-6)


def relevance_saliency(rel, ghw, H: int, W: int, blur: int = 31):
    """INFERENCE-AVAILABLE placement saliency [H,W] in [0,1] (drop-in for mover_saliency, which used GT
    future motion). `rel` is the [gh,gw] instruction<->image-patch relevance map from
    QwenVLEncoder.relevance_grid; upsample to [H,W], blur, renorm. rel=None -> None (caller then falls back
    to pure-entropy placement; we NEVER fall back to the GT mover_saliency at inference)."""
    if rel is None:
        return None
    import cv2
    r = rel.detach().float().cpu().numpy() if torch.is_tensor(rel) else np.asarray(rel, dtype=np.float32)
    r = cv2.resize(r, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(blur) | 1)                                  # force odd kernel
    r = cv2.GaussianBlur(r, (k, k), 0)
    r = r - float(r.min())
    return r / max(float(r.max()), 1e-6)


def gpstoken_ctrl_idx(rgb_uint8: np.ndarray, uv: torch.Tensor, n_keep: int, L: int, dev,
                      sal=None, beta: float = 0.0):
    """Full keypoint-selection: frame-0 RGB -> entropy-partition (+saliency) -> L token centers ->
    nearest dense Gaussian per token (in uv pixel space) -> deduped ctrl_idx (a smarter control subset).
    Returns a LongTensor on `dev`. Effective M may be < L after dedup (logged by the caller)."""
    import cv2
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    E = grad_mag(gray)
    _, gauss = gpstoken_init(E, min(L, n_keep), sal=sal, beta=beta)
    cen = torch.tensor([[g[0], g[1]] for g in gauss], device=dev, dtype=torch.float32)   # [L,2]
    uvk = uv[:n_keep].to(dev).float()                                                     # [n_keep,2]
    nn = torch.cdist(cen, uvk).argmin(dim=1)                                              # [L]
    return torch.unique(nn)                                                               # dedup collisions
