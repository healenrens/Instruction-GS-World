"""Long-horizon rollout evaluation (v2 model) — the >=10 s demonstration.

Loads an InstructGSWorldModel checkpoint, lifts frame f0 of an episode -> G0,
then autoregressively rolls out N steps conditioned on the instruction (all-layer
Qwen3-VL features). Renders each predicted dense state from the (static) frame-0
camera, compares to the real future frame, and writes a side-by-side mp4
(GT | static baseline | predicted) + PSNR-vs-horizon curve.

  python scripts/eval_longhorizon.py --ckpt checkpoints/run2/ckpt_last.pt --ep 0 --f0 60 --N 100 --stride 3
Test language control with a different --instruction.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data import AgiBotLeRobotTask, list_tasks  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.gaussians import (  # noqa: E402
    GaussianSet, render_gaussianset, psnr, intrinsics_from_local_points, viewmat_from_pose,
)
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default=None); ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--f0", type=int, default=60); ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--instruction", default=None, help="override instruction (tests language control)")
    ap.add_argument("--instruction2", default=None,
                    help="second instruction: roll out the SAME scene under both and measure divergence "
                         "(language-control / generalization test)")
    ap.add_argument("--vlm_image", type=int, default=0, help="0=text-only cond (match text-only training), 1=image+text")
    ap.add_argument("--cond_mode", default="aggregator", choices=["aggregator", "metaquery"], help="must match the checkpoint")
    ap.add_argument("--out", default="outputs/longhorizon")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image

    dev = "cuda"
    task = args.task or list_tasks()[0]
    t = AgiBotLeRobotTask(task)
    lang = args.instruction or t.language(args.ep)
    print(f"[eval] ep={args.ep} f0={args.f0} N={args.N} stride={args.stride} "
          f"span={args.N*args.stride/t.fps:.1f}s instr={lang!r}")

    idx = [args.f0] + [args.f0 + (k + 1) * args.stride for k in range(args.N)]
    idx = [i for i in idx if i < t.episode_meta(args.ep).length]
    N = len(idx) - 1
    frames = t.decode_frames(args.ep, "observation.images.head", idx)

    lifter = Pi3Lifter(device=dev)
    res = lifter.lift(frames[:1], conf_thr=0.1, edge_rtol=0.03)
    pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
    local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
    _, _, H, W = imgs.shape
    K0 = intrinsics_from_local_points(local[0]); view0 = viewmat_from_pose(poses[0])
    g0 = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0)
    print(f"[eval] G0={len(g0)} gaussians  H={H} W={W}")

    def to_lift_res(fr):
        return np.asarray(Image.fromarray(fr).resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    gt_future = [to_lift_res(frames[k + 1]) for k in range(N)]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # CPU: avoid pinning ~28GB optimizer on GPU (OOM next to training)
    cfg = DynamicsConfig(**ck["cfg"]) if "cfg" in ck else DynamicsConfig()
    n_query = ck.get("n_query", 16); M = ck.get("M", args.M)
    model = InstructGSWorldModel(cfg, n_control=M, n_query=n_query, cond_mode=args.cond_mode).to(dev).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[eval] loaded {args.ckpt} step {ck.get('step','?')} (missing {len(missing)} unexpected {len(unexpected)})")

    frame0 = (imgs[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() if args.vlm_image else None
    vlm_inputs = model.encoder.build_inputs(lang, frame0)
    vlm_inputs = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                      else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vlm_inputs.items()}

    def build_vlm(text):
        vi = model.encoder.build_inputs(text, frame0)
        return {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                    else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.no_grad(), amp:
        out = model(vlm_inputs, g0, N)

    # ---- DRIFT DIAGNOSTIC: what blows up over the rollout? scale (multiplicative) vs position drift ----
    with torch.no_grad():
        radius = (g0.means - g0.means.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
        s0 = float(g0.scales.max()); sm0 = float(g0.scales.mean())
        print(f"[drift] G0: maxScale={s0:.3f} meanScale={sm0:.4f} radius={float(radius):.3f}")
        print("[drift] step  maxScale  meanScale  meanDisp/r  maxDisp/r")
        for k in range(0, N, max(1, N // 12)):
            sc = out["scales"][k].float(); disp = (out["means"][k].float() - g0.means).norm(dim=-1)
            print(f"[drift] {k+1:3d}  {sc.max().item():8.3f}  {sc.mean().item():8.4f}  "
                  f"{(disp.mean()/radius).item():8.3f}  {(disp.max()/radius).item():8.3f}")

    # ---- language-control / generalization test: same scene, different instruction ----
    if args.instruction2:
        with torch.no_grad(), amp:
            out2 = model(build_vlm(args.instruction2), g0, N)
        radius = (g0.means - g0.means.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
        per_step = [((out["means"][k] - out2["means"][k]).norm(dim=-1).mean() / radius).item()
                    for k in range(N)]
        print(f"[lang-control] instr1={lang!r}\n               instr2={args.instruction2!r}")
        print(f"[lang-control] rollout divergence (frac of scene radius): "
              f"mean={np.mean(per_step):.4f} end={per_step[-1]:.4f} "
              f"(higher => language more strongly controls the predicted dynamics)")
        np.savez(os.path.join(args.out, "lang_divergence.npz"), per_step=np.array(per_step))

    writer = imageio.get_writer(os.path.join(args.out, "rollout.mp4"), fps=max(1, int(t.fps / args.stride)))
    ps_pred, ps_base = [], []
    base_render = render_gaussianset(g0, view0, K0, W, H)[0][0].clamp(0, 1)
    for k in range(N):
        s = GaussianSet(out["means"][k].float(), out["quats"][k].float(), out["scales"][k].float(),
                        out["opacities"][k].float(), out["colors"][k].float(), None)
        with torch.no_grad():
            c = render_gaussianset(s, view0, K0, W, H)[0][0].clamp(0, 1)
        gtt = torch.from_numpy(gt_future[k]).to(dev)
        ps_pred.append(psnr(c, gtt)); ps_base.append(psnr(base_render, gtt))
        row = torch.cat([gtt, base_render, c], dim=1).cpu().numpy()
        writer.append_data((row * 255).astype(np.uint8))
    writer.close()

    ps_pred = np.array(ps_pred); ps_base = np.array(ps_base)
    np.savez(os.path.join(args.out, "psnr_curve.npz"), pred=ps_pred, base=ps_base)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tt = np.arange(1, N + 1) * args.stride / t.fps
        plt.plot(tt, ps_pred, label="rollout (predicted)"); plt.plot(tt, ps_base, "--", label="static G0")
        plt.xlabel("time (s)"); plt.ylabel("PSNR (dB)"); plt.legend(); plt.title("Long-horizon render PSNR")
        plt.savefig(os.path.join(args.out, "psnr_curve.png"), dpi=120, bbox_inches="tight")
    except Exception as e:
        print("[warn] plot failed:", e)

    print(f"[eval] horizon {N*args.stride/t.fps:.1f}s | mean PSNR pred={ps_pred.mean():.2f} "
          f"base={ps_base.mean():.2f} | Δ_mean={ps_pred.mean()-ps_base.mean():+.2f}")
    print(f"[eval] saved video + curve to {args.out}/")


if __name__ == "__main__":
    main()
