"""Overlay viz: frame0 RGB with per-token GT (green) vs PRED (red) displacement arrows for the movers,
zoomed to the moving object. Arrows overlapping = the model's predicted 2D-Gaussian motion matches GT.
Usage: _gps_overlayviz.py --ckpt <wm.pt> --data data/trans_v3 --split heldseed --clips 0,1,2 --out o.png
"""
import argparse, glob, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.tokens import project_to_uv  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def mv_in(inp, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inp.items()}


def predict_clip(model, enc, cargs, c, dev, beta=30.0, L=1024):
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float(); traj = c["traj"].to(dev).float()
    N = means.shape[0]; H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    disp = (traj[K] - traj[0]).norm(dim=-1)
    sal = mover_saliency(uv, disp, n_keep, H, W)
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, L, dev, sal=sal, beta=beta)
    tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]; sig_n = (sig / float(max(H, W))).clamp(0, 1)
    center = means[:n_keep].mean(0, keepdim=True); radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
    vlm0 = mv_in(enc.build_inputs(instr, rgb0), dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctxc, ctxm, cond, text_feats = model.encode_cond(vlm0)
        grid0, ghw0 = (model.dino.grid(rgb0) if model.dino is not None else enc.image_grid_features(vlm0))
        tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
        cam_tok = None
        if cargs.get("cam_cond", False):
            cg, cam_tok = model.cam_cond_signals(tok_xyz0.float(), center, radius, Ki.float(), vm.float())
            cond = cond + cg
        x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctxc, ctxm, cond, cam_tok=cam_tok)
        if cargs.get("fuse", False):
            gate = torch.sigmoid(model.relevance(tok_feat, text_feats.mean(0)))
            g = gate[:, None] * model.geom_head(x[0]).float()
            xyz1_pred = model.geom_to_xyz(g, tok_xyz0.float(), Ki.float(), vm.float())
        else:
            xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
    xyz1_pred = xyz1_pred.float()
    uv0 = project_to_uv(tok_xyz0, Ki, vm); uv1g = project_to_uv(xyz1_gt, Ki, vm); uv1p = project_to_uv(xyz1_pred, Ki, vm)
    dcos = float(torch.cosine_similarity((xyz1_pred - tok_xyz0), (xyz1_gt - tok_xyz0), dim=-1)[disp[idx] > 0.01].mean())
    return rgb0, uv0.cpu().numpy(), uv1g.cpu().numpy(), uv1p.cpu().numpy(), disp[idx].cpu().numpy(), instr, dcos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="heldseed"); ap.add_argument("--clips", default="0,1,2")
    ap.add_argument("--out", default="outputs/overlay.png"); ap.add_argument("--maxarrows", type=int, default=45)
    args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cargs = ck.get("args", {})
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128),
                       feat_source=cargs.get("feat_source", "qwen"), dino_imgsize=cargs.get("dino_imgsize", 518)).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    enc = model.encoder
    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    cis = [int(x) for x in args.clips.split(",")]
    fig, axs = plt.subplots(1, len(cis), figsize=(5.6 * len(cis), 5.8))
    if len(cis) == 1: axs = [axs]
    for ax, ci in zip(axs, cis):
        c = torch.load(clips[ci % len(clips)], map_location=dev, weights_only=False)
        rgb0, uv0, uv1g, uv1p, disp, instr, dcos = predict_clip(model, enc, cargs, c, dev)
        H, W = rgb0.shape[:2]
        ax.imshow(rgb0)
        mv = np.where(disp > 0.02)[0]
        if len(mv) > args.maxarrows:                                   # subsample for legibility
            mv = mv[np.argsort(-disp[mv])[:args.maxarrows]]
        for i in mv:
            ax.annotate("", xy=(uv1g[i, 0], uv1g[i, 1]), xytext=(uv0[i, 0], uv0[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="#00ff44", lw=1.3, alpha=0.85))   # GT green
            ax.annotate("", xy=(uv1p[i, 0], uv1p[i, 1]), xytext=(uv0[i, 0], uv0[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="#ff2222", lw=1.3, alpha=0.7))     # PRED red
        # zoom to the mover bbox (+ margin)
        au = np.concatenate([uv0[mv], uv1g[mv], uv1p[mv]], 0)
        x0, y0 = au[:, 0].min(), au[:, 1].min(); x1, y1 = au[:, 0].max(), au[:, 1].max()
        mxx = 0.35 * (x1 - x0) + 12; myy = 0.35 * (y1 - y0) + 12
        ax.set_xlim(max(0, x0 - mxx), min(W, x1 + mxx)); ax.set_ylim(min(H, y1 + myy), max(0, y0 - myy))
        ax.set_title(f"{os.path.basename(clips[ci%len(clips)])[:22]}\n3D dir-cos={dcos:+.2f}  | green=GT  red=PRED", fontsize=10)
        ax.axis("off")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.suptitle(f"{instr[:70]}", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(args.out, dpi=145, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
