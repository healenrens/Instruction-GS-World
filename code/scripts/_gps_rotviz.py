"""Adversarial VERIFICATION of the 'model learns rotation' claim (rot@1500 predicts ~99deg). Two checks
the bare angle can't give: (1) AXIS alignment — is the predicted rotation about the SAME axis as GT
(genuine rotation), or a noisy Kabsch that happens to read ~99deg? (2) a top-down QUIVER render of the
cube tokens: GT displacement (green) vs PRED displacement (red). A genuine learned rotation => the two
swirl the same way around the same center. Saves PNGs + prints per-clip axis-err.
Usage: _gps_rotviz.py --ckpt checkpoints/gpswm_rot/wm_001500.pt --data data/rot_v1 --split held --thresh 60
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.losses import kabsch_R  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def axis_angle(R):
    """Rotation axis (unit) + angle(deg) from a 3x3 rotation matrix."""
    ang = float(torch.arccos(torch.clamp((R.diagonal().sum() - 1) / 2, -1, 1)) * 180 / np.pi)
    ax = torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], device=R.device)
    n = ax.norm()
    ax = ax / n if n > 1e-6 else torch.tensor([0., 0., 1.], device=R.device)
    return ax, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="held")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--thresh", type=float, default=60.0)
    ap.add_argument("--outdir", default="outputs/rotviz")
    ap.add_argument("--nviz", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.outdir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128),
                       feat_source=cargs.get("feat_source", "qwen"),
                       dino_imgsize=cargs.get("dino_imgsize", 518)).to(dev)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc = model.encoder

    def mv_in(inp):
        return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inp.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    axerrs, angerrs, gtangs, prangs = [], [], [], []
    nplotted = 0
    for cp in clips:
        c = torch.load(cp, map_location=dev, weights_only=False)
        means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
        traj = c["traj"].to(dev).float(); N = means.shape[0]
        H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
        Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
        seg = c["seg_per_g"].to(dev).long()
        instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
        disp = (traj[K] - traj[0]).norm(dim=-1)
        sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
        rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
        cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
        tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]
        sig_n = (sig / float(max(H, W))).clamp(0, 1)
        center = means[:n_keep].mean(0, keepdim=True)
        radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
        vlm0 = mv_in(enc.build_inputs(instr, rgb0))
        with torch.no_grad(), amp:
            ctx, ctxm, cond, text_feats = model.encode_cond(vlm0)
            grid0, ghw0 = (model.dino.grid(rgb0) if model.dino is not None else enc.image_grid_features(vlm0))
            tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
            x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctx, ctxm, cond)
            xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
        xyz1_pred = xyz1_pred.float()
        segk = seg[idx]; mv = disp[idx] > 0.01
        # pick the single highest-GT-rotation entity in this clip = the spinning cube
        best = None
        for s in torch.unique(segk).tolist():
            if s == 0:
                continue
            sel = (segk == s) & mv
            if sel.sum() < 8:
                continue
            Rg = kabsch_R(tok_xyz0[sel], xyz1_gt[sel]); ga = axis_angle(Rg)[1]
            if ga < args.thresh:
                continue
            if best is None or ga > best[1]:
                best = (s, ga, sel)
        if best is None:
            continue
        s, ga, sel = best
        Rg = kabsch_R(tok_xyz0[sel], xyz1_gt[sel]); Rp = kabsch_R(tok_xyz0[sel], xyz1_pred[sel])
        axg, angg = axis_angle(Rg); axp, angp = axis_angle(Rp)
        axerr = float(torch.arccos(torch.clamp((axg * axp).sum().abs(), -1, 1)) * 180 / np.pi)
        relerr = float(torch.arccos(torch.clamp(((Rp.T @ Rg).diagonal().sum() - 1) / 2, -1, 1)) * 180 / np.pi)
        axerrs.append(axerr); angerrs.append(relerr); gtangs.append(angg); prangs.append(angp)
        # quiver render: top-down (world x,y) of the cube tokens; GT vs PRED displacement
        if nplotted < args.nviz:
            p0 = tok_xyz0[sel].cpu().numpy(); pg = xyz1_gt[sel].cpu().numpy(); pp = xyz1_pred[sel].cpu().numpy()
            ctr = p0.mean(0)
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
            for a, dst, ttl, col in ((ax[0], pg, f"GT  rot={angg:.0f}deg", "g"), (ax[1], pp, f"PRED rot={angp:.0f}deg", "r")):
                a.scatter((p0[:, 0] - ctr[0]) * 100, (p0[:, 1] - ctr[1]) * 100, s=8, c="k", alpha=0.4, label="t0")
                a.quiver((p0[:, 0] - ctr[0]) * 100, (p0[:, 1] - ctr[1]) * 100,
                         (dst[:, 0] - p0[:, 0]) * 100, (dst[:, 1] - p0[:, 1]) * 100,
                         angles="xy", scale_units="xy", scale=1, color=col, width=0.005, alpha=0.7)
                a.set_title(ttl); a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
                a.set_aspect("equal"); a.grid(True, alpha=0.3)
            fig.suptitle(f"{os.path.basename(cp)}  cube tokens (top-down)  axis-err={axerr:.0f}deg  rel-rot-err={relerr:.0f}deg")
            fig.tight_layout()
            fn = os.path.join(args.outdir, f"rotviz_{nplotted:02d}_{os.path.basename(cp).replace('.pt','')}.png")
            fig.savefig(fn, dpi=90); plt.close(fig)
            nplotted += 1

    def med(x):
        return float(np.median(x)) if x else float("nan")
    print(f"[{os.path.basename(args.ckpt)}] spinning-cube entities n={len(gtangs)} over {len(clips)} clips")
    print(f"  GT rotation:    median {med(gtangs):.0f}deg")
    print(f"  PRED rotation:  median {med(prangs):.0f}deg")
    print(f"  AXIS error:     median {med(axerrs):.0f}deg   <== small => predicted rotation is about the RIGHT axis (genuine)")
    print(f"  rel rot-err:    median {med(angerrs):.0f}deg")
    print(f"  saved {nplotted} quiver PNGs -> {args.outdir}/")


if __name__ == "__main__":
    main()
