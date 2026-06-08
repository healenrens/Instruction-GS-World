"""Feasibility gate 3 (SC-GS): control-set dynamics + dense LBS, render full density.

Same clip/lift/conditioning as overfit_clip.py, but the dynamics runs on M=2048
control gaussians and deforms the full ~120k dense set via LBS, which is rendered.
Expect substantially higher absolute PSNR than the downsampled-control version,
and the conditioned rollout should beat the static-dense baseline.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data import AgiBotLeRobotTask, list_tasks  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.gaussians import (  # noqa: E402
    render_gaussianset, psnr, intrinsics_from_local_points, viewmat_from_pose,
)
from igsw.dynamics import GaussianDynamics, DynamicsConfig  # noqa: E402
from igsw.dynamics.scgs import SCGSRollout  # noqa: E402
from igsw.dynamics.conditioning import QwenVLEncoder  # noqa: E402
from igsw.training import photometric_loss, delta_reg, velocity_smoothness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None); ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--f0", type=int, default=400); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--no_lang", action="store_true"); ap.add_argument("--out", default="outputs/overfit_scgs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    dev = "cuda"
    task = args.task or list_tasks()[0]
    t = AgiBotLeRobotTask(task)
    lang_str = t.language(args.ep)
    idx = [args.f0 + i * args.stride for i in range(args.K + 1)]
    print(f"[clip] ep={args.ep} frames={idx} span={args.K*args.stride/t.fps:.2f}s instr={lang_str!r}")
    frames = t.decode_frames(args.ep, "observation.images.head", idx)

    lifter = Pi3Lifter(device=dev)
    tic = time.time()
    res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.03)
    print(f"[lift] {time.time()-tic:.2f}s points={tuple(res['points'].shape)}")
    pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
    local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
    Kf1, _, H, W = imgs.shape
    Ks = torch.stack([intrinsics_from_local_points(local[i]) for i in range(Kf1)], 0)
    viewmats = torch.stack([viewmat_from_pose(poses[i]) for i in range(Kf1)], 0)
    gt = imgs.permute(0, 2, 3, 1)

    dense0 = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0)
    print(f"[dense] {len(dense0)} gaussians, control M={args.M}")

    with torch.no_grad():
        base = [psnr(render_gaussianset(dense0, viewmats[tt], Ks[tt], W, H)[0][0].clamp(0, 1), gt[tt])
                for tt in range(1, Kf1)]
        base_psnr = float(np.mean(base))
    print(f"[baseline] static-dense future PSNR={base_psnr:.3f} per-step={[f'{p:.1f}' for p in base]}")

    if args.no_lang:
        hidden = torch.randn(1, 16, 2048, device=dev, dtype=torch.bfloat16)
        lmask = torch.ones(1, 16, dtype=torch.bool, device=dev)
        print("[cond] ABLATION random conditioning")
    else:
        enc = QwenVLEncoder(device=dev)
        h, m = enc.encode(lang_str, image=frames[0]); hidden, lmask = h[None], m[None]
        del enc; torch.cuda.empty_cache()
        print(f"[cond] Qwen3-VL hidden={tuple(hidden.shape)}")

    cfg = DynamicsConfig(d_model=args.dim, n_layers=args.layers, lang_dim=2048, use_checkpoint=True)
    model = GaussianDynamics(cfg).to(dev)
    print(f"[model] {model.num_params()/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    best = -1e9
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True); model.train()
        roll = SCGSRollout(model, dense0, n_control=args.M)  # rebind each iter (dense0 fixed)
        with amp:
            dense_states, deltas, ctrl_traj = roll.rollout(hidden, lmask, args.K)
        loss = 0.0; ps = []
        for k in range(args.K):
            tt = k + 1
            gsk = dense_states[k]
            gsk = type(gsk)(gsk.means.float(), gsk.quats.float(), gsk.scales.float(),
                            gsk.opacities.float(), gsk.colors.float(), None)
            colors, _, _ = render_gaussianset(gsk, viewmats[tt], Ks[tt], W, H)
            pl, _, _ = photometric_loss(colors[0], gt[tt]); loss = loss + pl
            ps.append(psnr(colors[0].clamp(0, 1).detach(), gt[tt]))
        loss = loss / args.K
        reg = sum(delta_reg(v, om, dls) for (v, om, dls) in deltas) / args.K
        vel = velocity_smoothness(ctrl_traj)
        total = loss + 1e-3 * reg + 1e-2 * vel
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mp = float(np.mean(ps)); best = max(best, mp)
        if it % 20 == 0 or it == args.iters - 1:
            print(f"  it{it:4d} loss={loss.item():.4f} PSNR={mp:.3f} "
                  f"(base {base_psnr:.3f} Δ={mp-base_psnr:+.3f}) best={best:.3f}")

    model.eval()
    roll = SCGSRollout(model, dense0, n_control=args.M)
    with torch.no_grad(), amp:
        dense_states, _, _ = roll.rollout(hidden, lmask, args.K)
    for k in [0, args.K // 2, args.K - 1]:
        tt = k + 1; gsk = dense_states[k]
        gsk = type(gsk)(gsk.means.float(), gsk.quats.float(), gsk.scales.float(),
                        gsk.opacities.float(), gsk.colors.float(), None)
        c, _, _ = render_gaussianset(gsk, viewmats[tt], Ks[tt], W, H)
        cb, _, _ = render_gaussianset(dense0, viewmats[tt], Ks[tt], W, H)
        row = torch.cat([gt[tt], cb[0].clamp(0, 1), c[0].clamp(0, 1)], 1).cpu().numpy()
        Image.fromarray((row * 255).astype(np.uint8)).save(os.path.join(args.out, f"step{tt}_gt_base_pred.png"))
    verdict = "PASS" if best > base_psnr + 0.3 else "INCONCLUSIVE"
    print(f"[RESULT] best dense rollout PSNR={best:.3f} vs static {base_psnr:.3f} => Δ={best-base_psnr:+.3f} dB ({verdict})")


if __name__ == "__main__":
    main()
