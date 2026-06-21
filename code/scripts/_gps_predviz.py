"""Prediction viz: for a trained ckpt, draw GT (green) vs PREDICTED (red) image-flow arrows on frame0
for the real movers (GT image-flow > thr). Shows seen(heldseed) vs unseen(heldtask) prediction quality.
Green and red overlapping = good prediction (right direction + magnitude).
Usage: _gps_predviz.py --ckpt ckpt.pt --data data/rtvid_multi_v2 --split heldseed --out out.png --n 4
"""
import argparse, glob, os, sys
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.tokens import project_to_uv  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="heldseed"); ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--beta", type=float, default=30.0); ap.add_argument("--gt_flow_thr", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--out", required=True)
    args = ap.parse_args(); dev = "cuda"
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
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    picked = []
    for cp in clips:
        c = torch.load(cp, map_location=dev, weights_only=False)
        means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float(); traj = c["traj"].to(dev).float()
        N = means.shape[0]; H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
        Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
        instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
        disp = (traj[K] - traj[0]).norm(dim=-1)
        sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
        rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
        cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
        tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]; sig_n = (sig / float(max(H, W))).clamp(0, 1)
        center = means[:n_keep].mean(0, keepdim=True); radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
        vlm0 = mv_in(enc.build_inputs(instr, rgb0))
        with torch.no_grad(), amp:
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
        Wn = xyz1_pred.new_tensor([float(W), float(H)])
        uv0 = project_to_uv(tok_xyz0, Ki, vm); uv1p = project_to_uv(xyz1_pred, Ki, vm); uv1g = project_to_uv(xyz1_gt, Ki, vm)
        fg = (uv1g - uv0) / Wn; mv = fg.norm(dim=-1) > args.gt_flow_thr
        if mv.sum() < 5: continue
        dcos = float(F.cosine_similarity(((uv1p - uv0) / Wn)[mv], fg[mv], dim=-1).mean())
        picked.append((rgb0, uv0[mv].cpu().numpy(), uv1g[mv].cpu().numpy(), uv1p[mv].cpu().numpy(),
                       dcos, int(mv.sum()), instr[:26]))
        if len(picked) >= args.n: break

    n = len(picked)
    if n == 0:
        print(f"[predviz] no clips with movers for {args.split}", flush=True); return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]
    for ax, (rgb0, u0, u1g, u1p, dcos, nm, instr) in zip(axes, picked):
        ax.imshow(rgb0)
        for i in range(len(u0)):
            ax.annotate("", xy=(u1g[i, 0], u1g[i, 1]), xytext=(u0[i, 0], u0[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="lime", lw=1.1, alpha=0.8))      # GT green
            ax.annotate("", xy=(u1p[i, 0], u1p[i, 1]), xytext=(u0[i, 0], u0[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.1, alpha=0.7))        # pred red
        ax.set_title(f"{instr}\n{nm} movers  dcos={dcos:+.2f}", fontsize=9); ax.axis("off")
    fig.suptitle(f"{args.split}: GT=green  PRED=red   ({os.path.basename(args.ckpt)})", fontsize=11)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[predviz] {args.out} n={n} mean_dcos={np.mean([p[4] for p in picked]):+.2f}", flush=True)


if __name__ == "__main__":
    main()
