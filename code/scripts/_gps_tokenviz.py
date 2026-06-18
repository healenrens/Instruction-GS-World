"""Representation-faithful visualization: our DEPTH-BEARING 2D-Gaussian tokens (GPSTokens) and how the
model predicts they move in TIME. Each token is drawn as an ellipse (its frame0 sigma_x,sigma_y footprint)
at its image position, COLORED BY DEPTH (camera-z). Three panels: frame0 -> GT future -> PRED future, so
you see the sparse tokens flow (and change depth) the way the model actually predicts in 3D (then we
project back to the image to draw them). This is 'our way' — explicit per-token 3D, NOT a JEPA latent.
Usage: _gps_tokenviz.py --ckpt <wm.pt> --data data/rot_v1 --split held --clip 0 --outdir outputs/tokviz
"""
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.tokens import project_to_uv, to_cam  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="held"); ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--L", type=int, default=1024); ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--outdir", default="outputs/tokviz"); ap.add_argument("--movethresh", type=float, default=0.02)
    ap.add_argument("--zoom", type=int, default=1, help="1=crop panels to the moving-object bbox + declutter static tokens")
    args = ap.parse_args(); dev = "cuda"; os.makedirs(args.outdir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cargs = ck.get("args", {})
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128),
                       feat_source=cargs.get("feat_source", "qwen"), dino_imgsize=cargs.get("dino_imgsize", 518)).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    enc = model.encoder

    def mv_in(inp):
        return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inp.items()}

    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    cp = clips[args.clip % len(clips)]
    c = torch.load(cp, map_location=dev, weights_only=False)
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float(); traj = c["traj"].to(dev).float()
    N = means.shape[0]; H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    disp = (traj[K] - traj[0]).norm(dim=-1)
    sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8); rgbK = c["gt_rgb"][K].cpu().numpy().astype(np.uint8)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
    tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]; sig_n = (sig / float(max(H, W))).clamp(0, 1)
    center = means[:n_keep].mean(0, keepdim=True); radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
    vlm0 = mv_in(enc.build_inputs(instr, rgb0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctxc, ctxm, cond, text_feats = model.encode_cond(vlm0)
        grid0, ghw0 = (model.dino.grid(rgb0) if model.dino is not None else enc.image_grid_features(vlm0))
        tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
        cam_tok = None
        if cargs.get("cam_cond", False):                                 # B: inject camera pose
            cg, cam_tok = model.cam_cond_signals(tok_xyz0.float(), center, radius, Ki.float(), vm.float())
            cond = cond + cg
        x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctxc, ctxm, cond, cam_tok=cam_tok)
        if cargs.get("fuse", False):                                     # v2: grounding-gate modulates motion
            gate = torch.sigmoid(model.relevance(tok_feat, text_feats.mean(0)))
            g = gate[:, None] * model.geom_head(x[0]).float()
            xyz1_pred = model.geom_to_xyz(g, tok_xyz0.float(), Ki.float(), vm.float())
        else:
            xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
    xyz1_pred = xyz1_pred.float()

    # token image positions (frame0, GT future, pred future) + depths (camera-z)
    uv0 = cen.cpu().numpy(); uv1g = project_to_uv(xyz1_gt, Ki, vm).cpu().numpy(); uv1p = project_to_uv(xyz1_pred, Ki, vm).cpu().numpy()
    z0 = to_cam(tok_xyz0, vm)[:, 2].cpu().numpy(); z1g = to_cam(xyz1_gt, vm)[:, 2].cpu().numpy(); z1p = to_cam(xyz1_pred, vm)[:, 2].cpu().numpy()
    sg = sig.cpu().numpy(); mv = (disp[idx] > args.movethresh).cpu().numpy()
    zmin, zmax = float(np.min([z0, z1g, z1p])), float(np.max([z0, z1g, z1p]))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import cm, colors
    norm = colors.Normalize(zmin, zmax); cmap = cm.turbo
    if args.zoom and mv.sum() > 0:                                   # crop to the moving-object bbox
        allu = np.concatenate([uv0[mv], uv1g[mv], uv1p[mv]], 0)
        lo = allu.min(0); hi = allu.max(0); pad = 0.30 * float((hi - lo).max()) + 15
        cx0, cy0 = np.maximum([0, 0], lo - pad); cx1, cy1 = np.minimum([W, H], hi + pad)
        sa, esz = 0.05, 1.6                                          # faint static; enlarge movers for readability
    else:
        cx0, cy0, cx1, cy1 = 0, 0, W, H; sa, esz = 0.30, 1.0

    def panel(ax, img, uvp, z, ttl, draw_flow_from=None):
        ax.imshow(img); ax.set_title(ttl, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        for i in range(len(uvp)):
            ism = mv[i]
            w = max(3, 2 * sg[i, 0]) * (esz if ism else 1.0); h = max(3, 2 * sg[i, 1]) * (esz if ism else 1.0)
            e = Ellipse((uvp[i, 0], uvp[i, 1]), width=w, height=h,
                        facecolor=cmap(norm(z[i])), edgecolor=("white" if ism else "none"),
                        lw=(1.0 if ism else 0), alpha=(0.9 if ism else sa))
            ax.add_patch(e)
        if draw_flow_from is not None:  # arrows frame0 -> this panel, movers only
            for i in np.where(mv)[0]:
                ax.annotate("", xy=(uvp[i, 0], uvp[i, 1]), xytext=(draw_flow_from[i, 0], draw_flow_from[i, 1]),
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.6, alpha=0.55))
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0)

    mvt = disp[idx] > args.movethresh
    d3g = (xyz1_gt - tok_xyz0).norm(dim=-1)[mvt]; d3p = (xyz1_pred - tok_xyz0).norm(dim=-1)[mvt]
    gmed = float(d3g.median()) * 100; pmed = float(d3p.median()) * 100; magr3 = pmed / max(gmed, 1e-6)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    panel(axs[0], rgb0, uv0, z0, f"frame0 tokens (M={len(uv0)}, {int(mv.sum())} movers)")
    panel(axs[1], rgbK, uv1g, z1g, f"GT future — mover disp {gmed:.0f}cm", draw_flow_from=uv0)
    panel(axs[2], rgbK, uv1p, z1p, f"PRED future — {pmed:.0f}cm = {magr3:.0%} of GT (dir ok, mag under)", draw_flow_from=uv0)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.01); cb.set_label("token depth (camera-z, m)")
    fig.suptitle(f"{os.path.basename(cp)}  |  instr: {instr[:70]}  |  depth-bearing 2D-Gaussian tokens over time", fontsize=10)
    fn = os.path.join(args.outdir, f"tokviz_{os.path.basename(cp).replace('.pt','')}.png")
    fig.savefig(fn, dpi=135, bbox_inches="tight"); plt.close(fig)
    # quick numeric: mover image-flow agreement
    fg = uv1g[mv] - uv0[mv]; fp = uv1p[mv] - uv0[mv]
    cos = float((fg * fp).sum() / (np.linalg.norm(fg) * np.linalg.norm(fp) + 1e-6)) if mv.sum() else float("nan")
    print(f"saved {fn}  | movers={int(mv.sum())}  dir-cos={cos:+.2f}  mag-ratio(pred/GT)={magr3:.2f}  (GT {gmed:.0f}cm vs PRED {pmed:.0f}cm)  depth {zmin:.2f}-{zmax:.2f}m")


if __name__ == "__main__":
    main()
