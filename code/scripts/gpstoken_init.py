"""GPSToken spatially-adaptive token INITIALIZATION (paper Algorithm 1, training-free) — applied to
our world-model frames to test the §9.3 keypoint-selection premise:

  Do information-adaptive (entropy-partition) tokens land on the MOVERS (manipulated object + gripper),
  unlike the current RANDOM control points whose object coverage is luck?

Algorithm 1 (arXiv 2509.01109, appendix p.13): Sobel gradient magnitude -> 512-bin histogram entropy H
-> region complexity m = h*w*H^lambda (lambda=2.5) -> recursively split the MOST-complex region (rect:
along longer side; square: try width/height split, keep the one with smaller min-complexity) until l
regions -> init g={sigma_x=w/6, sigma_y=h/6, rho=0, mu=center}. s_min=4.

Outputs a montage [RGB | grad-entropy | tokens-on-RGB | tokens-vs-mover-mask] + prints the coverage
metric (token-on-mover fraction vs uniform-random baseline = concentration ratio).
Usage: python code/scripts/gpstoken_init.py <out_tag> <l> <clip1.pt> [clip2.pt ...]
"""
import os
import sys

import numpy as np
import torch
import cv2
import imageio.v2 as iio

OUT = "outputs/gpstoken"
os.makedirs(OUT, exist_ok=True)
TAG = sys.argv[1]
L = int(sys.argv[2])
CLIPS = sys.argv[3:]


def grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def complexity(E, r, lam, gmax, sal=None, beta=0.0, nbins=512):
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
        # MOTION/TASK-aware boost: regions containing more saliency get a higher complexity, so the
        # splitter subdivides them deeper -> more tokens land there. (training oracle = GT mover mask;
        # at inference this map = language-relevance / learned motion-saliency, NOT GT.)
        m = m * (1.0 + beta * float(sal[y:y + h, x:x + w].mean()))
    return m


def gpstoken_init(E, l, lam=2.5, smin=4, sal=None, beta=0.0):
    """Return regions [(x,y,w,h)] and gaussians [(mux,muy,sx,sy)] per Algorithm 1 (+optional saliency)."""
    H, W = E.shape
    gmax = float(E.max()) + 1e-6
    regions = [(0, 0, W, H)]
    while len(regions) < l:
        cand = [i for i, (x, y, w, h) in enumerate(regions) if w > smin or h > smin]
        if not cand:
            break
        comps = [complexity(E, regions[i], lam, gmax, sal, beta) for i in cand]
        imax = cand[int(np.argmax(comps))]
        x, y, w, h = regions[imax]
        if w != h:
            if w > h:
                w1 = w // 2
                subs = [(x, y, w1, h), (x + w1, y, w - w1, h)]
            else:
                h1 = h // 2
                subs = [(x, y, w, h1), (x, y + h1, w, h - h1)]
        else:
            w1, h1 = w // 2, h // 2
            I1, I2 = (x, y, w1, h), (x + w1, y, w - w1, h)
            I3, I4 = (x, y, w, h1), (x, y + h1, w, h - h1)
            m1, m2 = complexity(E, I1, lam, gmax, sal, beta), complexity(E, I2, lam, gmax, sal, beta)
            m3, m4 = complexity(E, I3, lam, gmax, sal, beta), complexity(E, I4, lam, gmax, sal, beta)
            subs = [I1, I2] if min(m1, m2) <= min(m3, m4) else [I3, I4]
        regions = regions[:imax] + subs + regions[imax + 1:]
    gauss = [(x + w / 2.0, y + h / 2.0, w / 6.0, h / 6.0) for (x, y, w, h) in regions]
    return regions, gauss


def build_mover_mask(clip, H, W, disp_thresh=0.01, dilate=7):
    """2D mask of MOVERS (frame0->K displacement > thresh) splatted at their uv. Also object-only mask."""
    uv = clip["uv"].cpu().numpy()
    tr = clip["traj"].cpu().numpy()
    disp = np.linalg.norm(tr[-1] - tr[0], axis=-1)
    is_obj = clip["is_obj"].cpu().numpy() if "is_obj" in clip else np.zeros(len(uv), bool)
    mover = disp > disp_thresh
    masks = {}
    for name, sel in [("mover", mover), ("object", is_obj)]:
        m = np.zeros((H, W), np.uint8)
        p = uv[sel]
        xs = np.clip(p[:, 0].astype(int), 0, W - 1)
        ys = np.clip(p[:, 1].astype(int), 0, H - 1)
        m[ys, xs] = 1
        if dilate > 0:
            m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
        masks[name] = m
    return masks, float(mover.mean()), float(is_obj.mean())


for ci, p in enumerate(CLIPS):
    c = torch.load(p, map_location="cpu", weights_only=False)
    rgb = c["gt_rgb"][0].cpu().numpy()  # [H,W,3] uint8 frame-0
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    E = grad_mag(gray)

    masks, mover_area, obj_area = build_mover_mask(c, H, W)
    # optional MOTION-aware saliency boost (env GPS_MOTION_BETA>0): soft mover map drives token splitting
    beta = float(os.environ.get("GPS_MOTION_BETA", "0"))
    sal = None
    if beta > 0:
        sal = cv2.GaussianBlur(masks["mover"].astype(np.float32), (31, 31), 0)
        sal = sal / max(sal.max(), 1e-6)
    regions, gauss = gpstoken_init(E, L, sal=sal, beta=beta)

    # coverage: fraction of token centers landing in the mover/object mask
    cen = np.array([(g[0], g[1]) for g in gauss])
    cx = np.clip(cen[:, 0].astype(int), 0, W - 1)
    cy = np.clip(cen[:, 1].astype(int), 0, H - 1)
    tok_mover = float(masks["mover"][cy, cx].mean())
    tok_obj = float(masks["object"][cy, cx].mean())
    # uniform-random baseline (= expected mask area) over many draws
    rng = np.random.default_rng(0)
    rxy = rng.integers([0, 0], [W, H], size=(20000, 2))
    rand_mover = float(masks["mover"][rxy[:, 1], rxy[:, 0]].mean())
    rand_obj = float(masks["object"][rxy[:, 1], rxy[:, 0]].mean())

    tagc = os.path.basename(p).replace(".pt", "")
    print(f"[{ci}] {tagc} | {c.get('backend','?')} | {c.get('instruction','')[:45]!r} | L={len(regions)}")
    print(f"    mover-mask area={mover_area*100:.1f}% obj-mask area={obj_area*100:.1f}%")
    print(f"    token-on-MOVER  = {tok_mover*100:.1f}%  | random {rand_mover*100:.1f}%  -> concentration {tok_mover/max(rand_mover,1e-6):.2f}x")
    print(f"    token-on-OBJECT = {tok_obj*100:.1f}%  | random {rand_obj*100:.1f}%  -> concentration {tok_obj/max(rand_obj,1e-6):.2f}x")

    # ---- visualization ----
    En = (np.log1p(E) / np.log1p(E.max() + 1e-6) * 255).astype(np.uint8)
    En = cv2.applyColorMap(En, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]  # to RGB
    tok_rgb = rgb.copy()
    for (mux, muy, sx, sy) in gauss:
        cv2.ellipse(tok_rgb, (int(mux), int(muy)), (max(int(sx * 2), 1), max(int(sy * 2), 1)),
                    0, 0, 360, (0, 255, 0), 1)
        cv2.circle(tok_rgb, (int(mux), int(muy)), 1, (255, 60, 60), -1)
    # tokens over mover mask (mover=red tint), token center green if on mover else yellow
    movv = rgb.copy()
    movv[masks["mover"] > 0] = (0.5 * movv[masks["mover"] > 0] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
    for (mux, muy, sx, sy) in gauss:
        on = masks["mover"][min(int(muy), H - 1), min(int(mux), W - 1)] > 0
        cv2.circle(movv, (int(mux), int(muy)), 2, (40, 255, 40) if on else (255, 230, 40), -1)

    def lab(im, t):
        im = im.copy()
        cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return im

    montage = np.concatenate([
        lab(rgb, "frame0 RGB"),
        lab(En, "grad magnitude"),
        lab(tok_rgb, f"{len(regions)} GPS-tokens"),
        lab(movv, f"tokens vs MOVER ({tok_mover*100:.0f}% on, rand {rand_mover*100:.0f}%)"),
    ], axis=1)
    iio.imwrite(f"{OUT}/{TAG}_{tagc}_L{L}.png", montage)
    print(f"    -> {OUT}/{TAG}_{tagc}_L{L}.png")

print("DONE")
