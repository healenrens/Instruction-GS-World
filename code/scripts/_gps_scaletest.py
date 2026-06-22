"""Tests the user's hypothesis: is the under-prediction just a single missing GLOBAL SCALE, with the
RELATIVE displacement field already correct? For each held clip, take the model's predicted mover
displacement field and find the single scalar alpha that best rescales it to GT (least squares
alpha* = <pred,GT>/<pred,pred>). Report EPE & mag-ratio BEFORE vs AFTER that one-scalar rescale, plus
dcos (scale-invariant). If EPE collapses and magR->1 after rescale, the field is right up to one global
scale (= the user's decomposition: motion = predictable relative field x aleatoric global scale).
Usage: _gps_scaletest.py --ckpt <wm.pt> --data data/mix_v15 --split heldseed
"""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="heldseed"); ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=30.0)
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
    rows = []
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
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
            ctxc, ctxm, cond, _ = model.encode_cond(vlm0)
            grid0, ghw0 = (model.dino.grid(rgb0) if model.dino is not None else enc.image_grid_features(vlm0))
            tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
            x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctxc, ctxm, cond)
            xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
        xyz1_pred = xyz1_pred.float()
        mv = disp[idx] > 0.01
        if mv.sum() < 5:
            continue
        pd = (xyz1_pred - tok_xyz0)[mv]; gd = (xyz1_gt - tok_xyz0)[mv]          # mover displacement fields
        alpha = float((pd * gd).sum() / (pd * pd).sum().clamp_min(1e-9))        # best single global scale
        epe_b = float((xyz1_pred[mv] - xyz1_gt[mv]).norm(dim=-1).median()) * 100
        epe_a = float(((tok_xyz0[mv] + alpha * pd) - xyz1_gt[mv]).norm(dim=-1).median()) * 100
        magr_b = float(pd.norm(dim=-1).median() / gd.norm(dim=-1).median().clamp_min(1e-9))
        dcos = float(F.cosine_similarity(pd, gd, dim=-1).mean())
        rows.append((alpha, epe_b, epe_a, magr_b, alpha * magr_b, dcos))

    A = np.array(rows)
    al, eb, ea, mb, ma, dc = A.T
    def md(x): return float(np.median(x))
    print(f"[{os.path.basename(args.ckpt)} on {args.split}]  n_clip={len(A)}")
    print(f"  best global scale alpha*: median {md(al):.2f}  (pred field is {md(al):.2f}x too small -> needs x{md(al):.1f})")
    print(f"  dcos (direction): {md(dc):+.2f}")
    print(f"  EPE3D median:  BEFORE {md(eb):.1f}cm  ->  AFTER 1-scalar rescale {md(ea):.1f}cm   ({100*(1-md(ea)/max(md(eb),1e-6)):.0f}% drop)")
    print(f"  mag-ratio:     BEFORE {md(mb):.2f}      ->  AFTER rescale {md(ma):.2f}")
    verdict = "FIELD IS RIGHT UP TO ONE GLOBAL SCALE (user's hypothesis holds)" if md(ea) < 0.6 * md(eb) else "rescale doesn't fix it -> relative field itself is off"
    print(f"  >>> {verdict}")


if __name__ == "__main__":
    main()
