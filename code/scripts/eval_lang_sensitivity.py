"""Language-sensitivity eval — the QUANTITATIVE language-control metric (research_F Δ).

For a checkpoint, measures how much the predicted motion depends on the instruction:

  Δ_null  = ‖ v(ℓ) − v(∅) ‖ / ‖ v(ℓ) ‖     (step-0 control velocity: correct vs EMPTY instruction)
  Δ_wrong = ‖ v(ℓ) − v(ℓ') ‖ / ‖ v(ℓ) ‖    (correct vs a DIFFERENT clip's instruction)

v(·) = out["v"][0] = the model's step-0 per-control velocity [M,3]. The control SET is
held FIXED across the three conditions (same ctrl_idx) so v is directly comparable; the
ONLY thing that changes is the text fed to the frozen Qwen. This isolates the immediate
language effect on the dynamics (no autoregressive-drift confound).

Interpretation:
  Δ ≈ 0      -> language is IGNORED (posterior collapse; the old hinge sat here).
  Δ >> 0     -> the dynamics genuinely USES the instruction.
This scalar is (a) the confirmation that InfoNCE fixed language-ignoring (stream11 vs the
hinge baseline stream10), and (b) the A/B metric for MetaQuery vs the implicit cross-attn
conditioning. We average over several eval clips (fixed seed -> SAME clips across models,
fair A/B). A downstream rollout-means divergence (frac of scene radius) is also reported.

  python scripts/eval_lang_sensitivity.py --ckpt checkpoints/stream11_infonce/ckpt_0006000.pt \
      --vlm_image 1 --n_clips 16 --K 8 --stride 8
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data.streaming import StreamingClipDataset  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402


def _to_dev(vi, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_clips", type=int, default=16, help="eval clips (fixed seed -> reproducible across models)")
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--vlm_image", type=int, default=1, help="match training (stream11 = 1)")
    ap.add_argument("--seed", type=int, default=20260606, help="fixed -> SAME clips for a fair A/B")
    ap.add_argument("--boundary_ratio", type=float, default=4.0, help="sample onset clips (where language matters most)")
    ap.add_argument("--out", default="outputs/lang_sensitivity")
    ap.add_argument("--cond_mode", default="aggregator", choices=["aggregator", "metaquery"],
                    help="must match the checkpoint's conditioning (stream14=metaquery)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    # ---- model ----
    # load to CPU: map_location=dev would pin the whole ckpt (incl. ~28GB optimizer state) on the
    # GPU and OOM when running alongside training. We only need ck["model"].
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = DynamicsConfig(**ck["cfg"]) if "cfg" in ck else DynamicsConfig()
    n_query = ck.get("n_query", 16); M = ck.get("M", 2048)
    model = InstructGSWorldModel(cfg, n_control=M, n_query=n_query, cond_mode=args.cond_mode).to(dev).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[eval] {args.ckpt} step {ck.get('step','?')} | n_query={n_query} M={M} vlm_image={args.vlm_image} "
          f"(missing {len(missing)} unexpected {len(unexpected)})", flush=True)

    lifter = Pi3Lifter(device=dev)

    # ---- fixed eval clips (boundary-biased: onset is where language is most necessary) ----
    ds = StreamingClipDataset(K=args.K, stride=args.stride, seed=args.seed,
                              boundary_ratio=args.boundary_ratio, load_actions=False)
    it = iter(ds)
    clips = []
    while len(clips) < args.n_clips:
        c = next(it)
        if (c.get("instruction") or "").strip():
            clips.append(c)
    instrs = [c["instruction"] for c in clips]

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    rel_null, rel_wrong, abs_null, mag, roll_null, roll_wrong = [], [], [], [], [], []
    for i, c in enumerate(clips):
        frame0 = c["frames"][0]                                  # uint8 HWC
        res = lifter.lift(frame0[None], conf_thr=0.1, edge_rtol=0.03)
        pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
        g0 = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0)
        if len(g0) < M:
            continue
        # FIX the control set across the 3 conditions (so v is directly comparable)
        gen = torch.Generator(device=dev).manual_seed(args.seed + i)
        ctrl_idx = torch.randperm(len(g0), generator=gen, device=dev)[:M]
        img0 = (imgs[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() if args.vlm_image else None

        def run(text):
            vi = _to_dev(model.encoder.build_inputs(text, img0), dev)
            with torch.no_grad(), amp:
                return model(vi, g0, args.K, ctrl_idx=ctrl_idx)

        wrong = instrs[(i + 1) % len(instrs)]                    # a DIFFERENT clip's instruction
        o_c = run(instrs[i]); o_0 = run(""); o_w = run(wrong)
        v_c = o_c["v"][0].float(); v_0 = o_0["v"][0].float(); v_w = o_w["v"][0].float()   # [M,3] step-0
        nc = v_c.norm() + 1e-8
        rel_null.append(((v_c - v_0).norm() / nc).item())
        rel_wrong.append(((v_c - v_w).norm() / nc).item())
        abs_null.append((v_c - v_0).norm().item()); mag.append(v_c.norm().item())
        # downstream: rollout dense-means divergence as frac of scene radius
        radius = (g0.means - g0.means.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
        roll_null.append(((o_c["means"] - o_0["means"]).norm(dim=-1).mean() / radius).item())
        roll_wrong.append(((o_c["means"] - o_w["means"]).norm(dim=-1).mean() / radius).item())
        print(f"  clip {i:2d} Δnull={rel_null[-1]:.3f} Δwrong={rel_wrong[-1]:.3f} "
              f"roll_null={roll_null[-1]:.4f} | {instrs[i][:42]!r}", flush=True)

    def stat(x):
        a = np.array(x); return f"mean={a.mean():.4f} med={np.median(a):.4f} std={a.std():.4f}"
    print("\n========== LANGUAGE SENSITIVITY ==========")
    print(f"clips={len(rel_null)} K={args.K} stride={args.stride} vlm_image={args.vlm_image}")
    print(f"Δ_null  (v(ℓ) vs v(∅),   step0 ctrl-vel, frac):  {stat(rel_null)}")
    print(f"Δ_wrong (v(ℓ) vs v(ℓ'),  step0 ctrl-vel, frac):  {stat(rel_wrong)}")
    print(f"rollout-means div vs null (frac scene radius):    {stat(roll_null)}")
    print(f"rollout-means div vs wrong(frac scene radius):    {stat(roll_wrong)}")
    print(f"|v(ℓ)| step0 magnitude (context):                 {stat(mag)}")
    print("Δ≈0 => language IGNORED (collapse); Δ>>0 => dynamics USES the instruction.")
    np.savez(os.path.join(args.out, "lang_sensitivity.npz"),
             rel_null=np.array(rel_null), rel_wrong=np.array(rel_wrong),
             roll_null=np.array(roll_null), roll_wrong=np.array(roll_wrong), mag=np.array(mag),
             step=ck.get("step", -1))
    print(f"[eval] saved {args.out}/lang_sensitivity.npz")


if __name__ == "__main__":
    main()
