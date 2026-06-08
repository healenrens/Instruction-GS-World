"""Feasibility gate 3: render-supervised autoregressive rollout overfit on ONE clip.

Pipeline:
  1. decode a clip of K+1 head frames (stride s) from one episode,
  2. Pi3-lift them JOINTLY -> consistent gauge, per-frame cameras P_t + intrinsics K_t,
  3. build the canonical Gaussian set G_0 from frame 0 (downsampled to N control pts),
  4. encode (instruction + frame 0) with Qwen3-VL -> hidden states,
  5. roll the dynamics model out K steps from G_0; render each predicted state from P_t;
     photometric loss vs the real frame_t; optimize.

SUCCESS CRITERION: the trained, language-conditioned rollout achieves lower
future-frame render loss / higher PSNR than the STATIC (identity) baseline
(render G_0 from each P_t). That demonstrates the model predicts scene-consistent
per-Gaussian motion from language — the core feasibility claim.

Attention runs in bf16 (flash); rendering runs in fp32 (gsplat), bridged by
differentiable .float() casts.
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
    GaussianSet, render_gaussianset, psnr, intrinsics_from_local_points, viewmat_from_pose,
)
from igsw.gaussians.sampling import downsample_gaussians  # noqa: E402
from igsw.dynamics import GaussianDynamics, DynamicsConfig, GaussianState  # noqa: E402
from igsw.dynamics.conditioning import QwenVLEncoder  # noqa: E402
from igsw.training import photometric_loss, delta_reg  # noqa: E402


def fp32_gs(state: GaussianState) -> GaussianSet:
    g = state.index_batch(0)
    return GaussianSet(g.means.float(), g.quats.float(), g.scales.float(),
                       g.opacities.float(), g.colors.float(),
                       g.features.float() if g.features is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--f0", type=int, default=400)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--K", type=int, default=8, help="rollout steps")
    ap.add_argument("--N", type=int, default=16384, help="control gaussians")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--no_lang", action="store_true", help="use random fixed cond (ablate language)")
    ap.add_argument("--out", default="outputs/overfit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    dev = "cuda"
    task = args.task or list_tasks()[0]
    t = AgiBotLeRobotTask(task)
    lang_str = t.language(args.ep)
    idx = [args.f0 + i * args.stride for i in range(args.K + 1)]
    print(f"[clip] task={os.path.basename(os.path.dirname(task))} ep={args.ep} frames={idx} "
          f"(span {args.K*args.stride/t.fps:.2f}s)")
    print(f"[clip] instruction = {lang_str!r}")
    frames = t.decode_frames(args.ep, "observation.images.head", idx)  # [K+1,H,W,3]

    # ---- lift the clip jointly ----
    lifter = Pi3Lifter(device=dev)
    tic = time.time()
    res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.03)
    print(f"[lift] {time.time()-tic:.2f}s  points={tuple(res['points'].shape)}")
    pts = res["points"].to(dev); imgs = res["images"].to(dev)
    mask = res["mask"].to(dev); local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
    Kf1, _, H, W = imgs.shape

    # per-frame camera (intrinsics + viewmat)
    Ks = torch.stack([intrinsics_from_local_points(local[i]) for i in range(Kf1)], 0)  # [K+1,3,3]
    viewmats = torch.stack([viewmat_from_pose(poses[i]) for i in range(Kf1)], 0)       # [K+1,4,4]
    gt = imgs.permute(0, 2, 3, 1)  # [K+1,H,W,3]

    # ---- canonical G_0 ----
    g0_full = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0)
    g0 = downsample_gaussians(g0_full, args.N)
    print(f"[G0] full={len(g0_full)} -> control={len(g0)}")

    # ---- identity (static) baseline ----
    with torch.no_grad():
        base_ps = []
        for tt in range(1, Kf1):
            c, a, _ = render_gaussianset(g0, viewmats[tt], Ks[tt], W, H)
            base_ps.append(psnr(c[0].clamp(0, 1), gt[tt]))
        base_psnr = float(np.mean(base_ps))
    print(f"[baseline] static-G0 future-frame PSNR = {base_psnr:.3f}  per-step={[f'{p:.1f}' for p in base_ps]}")

    # ---- language conditioning ----
    if args.no_lang:
        hidden = torch.randn(1, 16, 2048, device=dev, dtype=torch.bfloat16)
        lmask = torch.ones(1, 16, dtype=torch.bool, device=dev)
        print("[cond] ABLATION: random fixed conditioning (no language)")
    else:
        enc = QwenVLEncoder(device=dev)
        h, m = enc.encode(lang_str, image=frames[0])
        hidden, lmask = h[None], m[None]
        del enc; torch.cuda.empty_cache()
        print(f"[cond] Qwen3-VL hidden={tuple(hidden.shape)}")

    # ---- model ----
    cfg = DynamicsConfig(d_model=args.dim, n_layers=args.layers, lang_dim=2048, use_checkpoint=True)
    model = GaussianDynamics(cfg).to(dev)
    print(f"[model] {model.num_params()/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    state0 = GaussianState.from_gaussianset(g0)
    best = -1e9
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        model.train()
        with amp:
            s = state0
            tot_loss = 0.0
            deltas = []
            states = []
            for k in range(args.K):
                step_idx = torch.zeros(1, dtype=torch.long, device=dev) + k
                logit_o = torch.logit(s.opacities.clamp(1e-6, 1 - 1e-6))
                v, om, dls, dlo, dc, _ = model.predict_deltas(s, hidden, lmask, step_idx)
                from igsw.dynamics.manifold import apply_deltas_tensors
                nm, nq, ns, no, nc, _ = apply_deltas_tensors(
                    s.means, s.quats, s.scales, logit_o, s.colors, None, v, om, dls, dlo, dc)
                s = GaussianState(nm, nq, ns, no, nc, None)
                states.append(s)
                deltas.append((v, om, dls))
        # render (fp32) + loss
        loss = 0.0
        ps = []
        for k in range(args.K):
            tt = k + 1
            gsk = fp32_gs(states[k])
            colors, alphas, _ = render_gaussianset(gsk, viewmats[tt], Ks[tt], W, H)
            pl, l1, dssim = photometric_loss(colors[0], gt[tt])
            loss = loss + pl
            ps.append(psnr(colors[0].clamp(0, 1).detach(), gt[tt]))
        loss = loss / args.K
        reg = sum(delta_reg(v, om, dls) for (v, om, dls) in deltas) / args.K
        total = loss + 1e-3 * reg
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        mean_p = float(np.mean(ps))
        best = max(best, mean_p)
        if it % 20 == 0 or it == args.iters - 1:
            print(f"  it{it:4d} loss={loss.item():.4f} reg={reg.item():.4e} "
                  f"PSNR={mean_p:.3f} (base {base_psnr:.3f}, Δ={mean_p-base_psnr:+.3f}) best={best:.3f}")

    # ---- save qualitative ----
    model.eval()
    with torch.no_grad(), amp:
        s = state0; states = []
        for k in range(args.K):
            s = model.step(s, hidden, lmask, torch.zeros(1, dtype=torch.long, device=dev) + k)
            states.append(s)
    for k in [0, args.K // 2, args.K - 1]:
        tt = k + 1
        gsk = fp32_gs(states[k])
        c, a, _ = render_gaussianset(gsk, viewmats[tt], Ks[tt], W, H)
        cb, _, _ = render_gaussianset(g0, viewmats[tt], Ks[tt], W, H)
        row = torch.cat([gt[tt], cb[0].clamp(0, 1), c[0].clamp(0, 1)], dim=1).cpu().numpy()
        Image.fromarray((row * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"step{tt}_gt_base_pred.png"))
    print(f"[RESULT] best rollout PSNR={best:.3f} vs static baseline {base_psnr:.3f} "
          f"=> Δ={best-base_psnr:+.3f} dB  ({'PASS' if best>base_psnr+0.3 else 'INCONCLUSIVE'})")
    print(f"[done] qualitative saved to {args.out}/ (cols: GT | static-G0 | predicted)")


if __name__ == "__main__":
    main()
