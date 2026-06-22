"""Cube/hand-specific rotation readout (Track B make-or-break). The standard eval's median rot-err/
GT-rot is over ALL moving entities, dominated by the 8 panda arm LINKS (small rotation) which wash out
the genuinely-spinning CUBE/HAND (~124deg). This loads a trained model, predicts on held clips, and
reports — for entities with GT-rot > thresh (the spinning ones) — GT rotation vs the model's PREDICTED
rotation (Kabsch on predicted token positions). Honest test: does the per-token translation field
EXPRESS the rotation (pred-rot >> 0, aligned with GT), or still predict ~0 rotation (the old failure)?
Usage: _gps_rotread.py --ckpt checkpoints/gpswm_rot/wm_000750.pt --data data/rot_v1 --split held --thresh 60
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


def rot_deg(R):
    c = (R.diagonal().sum() - 1) / 2
    return float(torch.arccos(torch.clamp(c, -1, 1)) * 180 / np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="held")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--thresh", type=float, default=60.0)
    args = ap.parse_args()
    dev = "cuda"
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

    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    rows = []
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
        for s in torch.unique(segk).tolist():
            if s == 0:
                continue
            sel = (segk == s) & mv
            if sel.sum() < 6:
                continue
            Rg = kabsch_R(tok_xyz0[sel], xyz1_gt[sel])
            gt_rot = rot_deg(Rg)
            if gt_rot < args.thresh:
                continue
            Rp = kabsch_R(tok_xyz0[sel], xyz1_pred[sel])
            rerr = rot_deg(Rp.T @ Rg)
            gt_tr = (xyz1_gt[sel] - tok_xyz0[sel]).mean(0).norm().item() * 100
            pred_tr = (xyz1_pred[sel] - tok_xyz0[sel]).mean(0).norm().item() * 100
            rows.append((gt_rot, rot_deg(Rp), rerr, gt_tr, pred_tr, int(sel.sum())))

    A = np.array(rows)
    if len(A) == 0:
        print(f"no entities with GT-rot>{args.thresh}deg (try lower --thresh)")
        return
    gtr, pr, rerr, gtt, prt, npts = A.T
    print(f"[{os.path.basename(args.ckpt)}] high-rot entities (GT-rot>{args.thresh:.0f}deg, the spinning cube/hand): "
          f"n={len(A)} over {len(clips)} clips, median {int(np.median(npts))} tok/entity")
    print(f"  GT rotation:    median {np.median(gtr):.0f}deg   (the real spin)")
    print(f"  PRED rotation:  median {np.median(pr):.0f}deg   <== ~0 => model predicts NO rotation (old failure); >>0 => expresses rotation")
    print(f"  rel rot-err:    median {np.median(rerr):.0f}deg   (vs GT {np.median(gtr):.0f}deg; < GT => learning rotation)")
    print(f"  pred-rot>30deg: {100*np.mean(pr > 30):.0f}% of entities   |  pred-rot>60deg: {100*np.mean(pr > 60):.0f}%")
    print(f"  translation:    GT {np.median(gtt):.1f}cm  PRED {np.median(prt):.1f}cm")


if __name__ == "__main__":
    main()
