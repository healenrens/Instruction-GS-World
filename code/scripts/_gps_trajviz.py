"""Trajectory viz (curve task): draw the PREDICTED full trajectory polyline (red) vs the GT full
trajectory polyline (green) on frame0, for mover tokens. Works for --traj_pred ckpts (per-frame
prediction) AND straight ckpts (the pred 'trajectory' = the straight chord toward the endpoint, for a
fair visual comparison of curve vs straight). Frame0 token = blue dot.
Usage: _gps_trajviz.py --ckpt ckpt.pt --data data/rtvid_multi_v2 --split heldseed --out out.png --n 4
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
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--max_tok", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cargs = ck.get("args", {})
    tp = bool(cargs.get("traj_pred", 0))
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128),
                       feat_source=cargs.get("feat_source", "qwen"), dino_imgsize=cargs.get("dino_imgsize", 518),
                       traj_pred=tp, Kf=12).to(dev)
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
        tok_xyz0 = means[idx]; sig_n = (sig / float(max(H, W))).clamp(0, 1)
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
            else:
                g = model.geom_head(x[0]).float()
            if tp:
                traj_p = model.geom_to_traj(g, tok_xyz0.float(), Ki.float(), vm.float())  # [Kf,M,3]
                traj_p = torch.cat([tok_xyz0[None].float(), traj_p], 0)                   # prepend t=0 -> [Kf+1,M,3]
            else:
                if cargs.get("fuse", False):
                    xyz1 = model.geom_to_xyz(g, tok_xyz0.float(), Ki.float(), vm.float())
                else:
                    xyz1, _ = model.heads(x, tok_xyz0, Ki, vm)
                # straight pred 'trajectory' = linear interp toward the endpoint
                ts = torch.linspace(0, 1, K + 1, device=dev).view(-1, 1, 1)
                traj_p = tok_xyz0[None].float() + (xyz1[None].float() - tok_xyz0[None].float()) * ts
        # GT trajectory uv [Kf+1, M, 2], pred trajectory uv
        uv_g = torch.stack([project_to_uv(traj[t][idx], Ki, vm) for t in range(K + 1)], 0)   # [Kf+1,M,2]
        uv_p = torch.stack([project_to_uv(traj_p[t], Ki, vm) for t in range(K + 1)], 0)
        Wn = torch.tensor([float(W), float(H)], device=dev)
        fg = (uv_g[K] - uv_g[0]) / Wn; mv = fg.norm(dim=-1) > args.gt_flow_thr
        if mv.sum() < 5:
            continue
        mvi = torch.where(mv)[0]
        if mvi.numel() > args.max_tok:                                # subsample for legibility
            mvi = mvi[torch.linspace(0, mvi.numel() - 1, args.max_tok).long()]
        dcos = float(F.cosine_similarity(((uv_p[K] - uv_p[0]) / Wn)[mv], fg[mv], dim=-1).mean())
        picked.append((rgb0, uv_g[:, mvi].cpu().numpy(), uv_p[:, mvi].cpu().numpy(),
                       dcos, int(mv.sum()), instr[:26]))
        if len(picked) >= args.n:
            break

    n = len(picked)
    if n == 0:
        print(f"[trajviz] no clips with movers for {args.split}", flush=True); return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (rgb0, ug, up, dcos, nm, instr) in zip(axes, picked):
        ax.imshow(rgb0)
        M = ug.shape[1]
        for i in range(M):
            ax.plot(ug[:, i, 0], ug[:, i, 1], "-", color="lime", lw=1.2, alpha=0.85)   # GT curve
            ax.plot(up[:, i, 0], up[:, i, 1], "-", color="red", lw=1.0, alpha=0.7)      # pred curve
            ax.plot(ug[0, i, 0], ug[0, i, 1], ".", color="cyan", ms=3)                  # start dot
        ax.set_title(f"{instr}\n{nm} movers  endpt-dcos={dcos:+.2f}", fontsize=9); ax.axis("off")
    mode = "TRAJ(per-frame)" if tp else "STRAIGHT(lerp)"
    fig.suptitle(f"{os.path.basename(args.ckpt)}  [{mode}]  {args.split}   green=GT  red=pred  cyan=start", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[trajviz] saved {args.out}  ({n} clips, mode={mode})", flush=True)


if __name__ == "__main__":
    main()
