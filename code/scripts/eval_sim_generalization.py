"""GENERALIZATION eval on CLEAN held-out sim clips (agent.md §39 — the deliverable).

The overfit proved CAPACITY (corr 0.95 memorizing one clip). This measures GENERALIZATION: on
HELD-OUT clips (never trained on — either a held-out seed of a trained task, or an ENTIRELY held-out
task), does the model produce LOCALIZED, language-responsive motion?

For each held-out clip we compute (with the SAME spatial-grounding model call + mover-biased control
sampling as training):
  1. MOTION localization:
       corr(GT_disp, PRED_disp)              -> high = the model puts motion on the right controls
       top-mover ratio = PRED/GT magnitude on the GT top-5% movers
  2. LANGUAGE sensitivity (control set + image FIXED, only the TEXT to frozen Qwen changes):
       Δ_null  = ‖v(ℓ) − v(∅)‖ / ‖v(ℓ)‖      (correct vs EMPTY instruction)
       Δ_wrong = ‖v(ℓ) − v(ℓ')‖ / ‖v(ℓ)‖     (correct vs a DIFFERENT task's instruction)
  3. (optional) render a GT|static|pred rollout video for a couple of held-out clips.

We also run it on a few TRAIN clips so the report can contrast train-corr vs held-out-corr (the
overfitting-vs-generalization verdict). Reports per-split means.

  python code/scripts/eval_sim_generalization.py --ckpt checkpoints/sim_gen/ckpt_last.pt \
      --data data/maniskill --video 2 --out outputs/sim_gen_eval
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data.sim_clips import list_clips  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset, psnr  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402

# a few alternate-task instructions for the Δ_wrong language test
WRONG_INSTRS = [
    "Push the cube to the goal region.",
    "Pick up the red cube and stack it on top of the green cube.",
    "Pick up the red cube and move it to the goal position.",
    "Pull the cube to the goal region.",
]


def _to_dev(vi, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}


def sample_controls(g0_means, disp_all, M, gen, mover_thresh=0.01):
    N = g0_means.shape[0]; M = min(M, N)
    mover_g = (disp_all > mover_thresh).nonzero(as_tuple=True)[0]
    static_g = (disp_all <= mover_thresh).nonzero(as_tuple=True)[0]
    n_mover = int(min(mover_g.numel(), M // 2)); n_static = M - n_mover
    n_static = int(min(n_static, static_g.numel())); n_mover = M - n_static
    n_mover = int(min(n_mover, mover_g.numel()))
    sel_m = mover_g[torch.randperm(mover_g.numel(), device=g0_means.device, generator=gen)[:n_mover]]
    sel_s = static_g[torch.randperm(static_g.numel(), device=g0_means.device, generator=gen)[:n_static]]
    idx = torch.cat([sel_m, sel_s])
    return idx[torch.randperm(idx.numel(), device=g0_means.device, generator=gen)]


@torch.no_grad()
def eval_clip(model, clip, dev, M, K_max, spatial, video_path=None):
    g0 = GaussianSet(clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
                     clip["opacities"].to(dev), clip["colors"].to(dev), None).to(dev)
    N = len(g0)
    uv = clip["uv"].to(dev)
    traj_full = clip["traj"].to(dev)
    Kf = int(clip["Kf"]); K = min(K_max, Kf)
    H, W = int(clip["H"]), int(clip["W"])
    K_intr = clip["K_intr"].to(dev).float(); viewmat = clip["viewmat"].to(dev).float()
    gt_rgb = clip["gt_rgb"].to(dev).float() / 255.0
    instruction = clip["instruction"]

    gen = torch.Generator(device=dev).manual_seed(0)   # fixed -> reproducible control set
    disp_all = (traj_full[K] - traj_full[0]).norm(dim=-1)
    ctrl_idx = sample_controls(g0.means, disp_all, M, gen, 0.01)
    M_ = ctrl_idx.numel()
    control_uv = uv[ctrl_idx]
    gt_pos = traj_full[:, ctrl_idx, :]
    init = g0.means[ctrl_idx]
    radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius

    cu = control_uv if spatial else None
    cuhw = (H, W) if spatial else None
    img0 = (gt_rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    seg_pg = clip["seg_per_g"].to(dev) if "seg_per_g" in clip else None    # §49/§54: entity-LBS/gate/relevance
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(model.encoder.build_inputs(instruction, img0), dev)
        out = model(vi, g0, K, ctrl_idx=ctrl_idx, control_uv=cu, control_uv_hw=cuhw, seg_per_g=seg_pg)
        v_correct = out["v"][0].float()                                    # [M,3] step-0 velocity

        # language sensitivity: same control set + image, only the TEXT changes
        vi_null = _to_dev(model.encoder.build_inputs("", img0), dev)
        out_null = model(vi_null, g0, K, ctrl_idx=ctrl_idx, control_uv=cu, control_uv_hw=cuhw)
        v_null = out_null["v"][0].float()
        wrong = next((w for w in WRONG_INSTRS if w.strip() != (instruction or "").strip()), WRONG_INSTRS[0])
        vi_wrong = _to_dev(model.encoder.build_inputs(wrong, img0), dev)
        out_wrong = model(vi_wrong, g0, K, ctrl_idx=ctrl_idx, control_uv=cu, control_uv_hw=cuhw)
        v_wrong = out_wrong["v"][0].float()

    pred_disp = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pred_disp.float()]))[0, 1].item()
    topk = gt_disp.topk(max(1, M_ // 20)).indices
    ratio = (pred_disp[topk].mean() / gt_disp[topk].mean().clamp_min(1e-6)).item()
    vnorm = v_correct.norm(dim=-1).mean().clamp_min(1e-9)
    d_null = ((v_correct - v_null).norm(dim=-1).mean() / vnorm).item()
    d_wrong = ((v_correct - v_wrong).norm(dim=-1).mean() / vnorm).item()
    frac_pred = (pred_disp > 0.02).float().mean().item()
    frac_gt = (gt_disp > 0.02).float().mean().item()
    # Exp-1 metrics: static-leakage = mean PRED disp (normalized) of GT-static controls (raw disp<thresh
    # -> the table) -> should ->0 with the gate; mover precision/recall of sigmoid(p_dyn)>0.5.
    mover_label = (gt_pos[K] - gt_pos[0]).norm(dim=-1) > 0.01                  # [M] raw-metric mover
    stat_m = ~mover_label
    leak = pred_disp[stat_m].mean().item() if stat_m.any() else float("nan")
    m_prec = m_rec = float("nan")
    if "p_dyn" in out:
        pred_mv = torch.sigmoid(out["p_dyn"].float()) > 0.5
        tp = (pred_mv & mover_label).sum().float()
        m_prec = (tp / pred_mv.sum().clamp_min(1)).item()
        m_rec = (tp / mover_label.sum().clamp_min(1)).item()

    if video_path is not None:
        _render_video(model, clip, g0, ctrl_idx, control_uv, out, traj_full, K_intr,
                      viewmat, gt_rgb, H, W, K, dev, video_path)

    return {"env": clip.get("env"), "seed": clip.get("seed"), "split": clip.get("split"),
            "corr": corr, "ratio": ratio, "d_null": d_null, "d_wrong": d_wrong,
            "frac_pred": frac_pred, "frac_gt": frac_gt, "leak": leak,
            "m_prec": m_prec, "m_rec": m_rec, "instr": (instruction or "")[:40]}


@torch.no_grad()
def _render_video(model, clip, g0, ctrl_idx, control_uv, out, traj_full, K_intr, viewmat,
                  gt_rgb, H, W, K, dev, path):
    """Render GT | static(frame0) | predicted rollout, side by side, to an mp4.
    GT = the analytic-moved dense gaussians (the clean target); static = frame-0 gaussians held;
    pred = the model's rolled-out dense gaussians."""
    import imageio.v3 as iio
    from scripts.maniskill_gt import apply_traj_to_gaussians
    seg_per_g = clip["seg_per_g"].to(dev); poses = clip["poses"] if "poses" in clip else None
    frames = []
    bg = gt_rgb[0]
    for t in range(K + 1):
        # GT (analytic) — if poses saved use full rotation; else just move means
        if poses is not None:
            g_gt = apply_traj_to_gaussians(g0, traj_full[t], seg_per_g, poses[t], poses[0], dev)
        else:
            g_gt = g0.clone(); g_gt.means = traj_full[t]
        c_gt, a_gt, _ = render_gaussianset(g_gt, viewmat[None], K_intr[None], W, H)
        gt_im = (c_gt[0] + (1 - a_gt[0]) * bg).clamp(0, 1)
        # static
        c_s, a_s, _ = render_gaussianset(g0, viewmat[None], K_intr[None], W, H)
        st_im = (c_s[0] + (1 - a_s[0]) * bg).clamp(0, 1)
        # predicted (t=0 is g0; t>=1 from out["means"][t-1])
        if t == 0:
            g_p = g0
        else:
            g_p = GaussianSet(out["means"][t - 1].float(), out["quats"][t - 1].float(),
                              out["scales"][t - 1].float(), out["opacities"][t - 1].float(),
                              out["colors"][t - 1].float(), None)
        c_p, a_p, _ = render_gaussianset(g_p, viewmat[None], K_intr[None], W, H)
        pr_im = (c_p[0] + (1 - a_p[0]) * bg).clamp(0, 1)
        row = torch.cat([gt_im, st_im, pr_im], dim=1)                      # [H, 3W, 3]
        frames.append((row * 255).to(torch.uint8).cpu().numpy())
    iio.imwrite(path, np.stack(frames, 0), fps=4, codec="libx264")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/data/maniskill")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--n_heldseed", type=int, default=30)
    ap.add_argument("--n_heldtask", type=int, default=30)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--video", type=int, default=2, help="# held-out rollout videos to render")
    ap.add_argument("--out", default="outputs/sim_gen_eval")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = DynamicsConfig(**ck["cfg"]); M = ck.get("M", 2048)
    spatial = bool(ck.get("spatial_ground", 1))
    cond_mode = ck.get("cond_mode", "aggregator")
    dyn_gate = bool(ck.get("dyn_gate", 0)); sem_dim = int(ck.get("sem_dim", 0))  # Exp-1
    gate_uses_sem = bool(ck.get("gate_uses_sem", 1))  # §44h (default on); must match the gate's dyn_head width
    model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16),
                                 cond_mode=cond_mode, spatial_ground=spatial,
                                 dyn_gate=dyn_gate, sem_dim=sem_dim, gate_uses_sem=gate_uses_sem,
                                 gate_entity_pool=bool(ck.get("gate_entity_pool", 0)),  # §49/§54: were SILENTLY
                                 entity_lbs=bool(ck.get("entity_lbs", 0)),              # OFF -> §49 ckpts mis-evaluated
                                 rel_head=bool(ck.get("rel_head", 0)),
                                 entity_head=bool(ck.get("entity_head", 0)),
                                 rigid_agg=bool(ck.get("rigid_agg", 0))).to(dev).eval()
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    print(f"[eval] ckpt={args.ckpt} step={ck.get('step')} spatial={spatial} cond={cond_mode} "
          f"dyn_gate={dyn_gate} sem_dim={sem_dim} gate_uses_sem={gate_uses_sem} M={M} | "
          f"load missing={len(miss)} unexpected={len(unexp)}", flush=True)

    splits = {"train": (("train",), args.n_train),
              "heldseed": (("heldseed",), args.n_heldseed),
              "heldtask": (("heldtask",), args.n_heldtask)}
    results = {}
    vids_done = 0
    for name, (flt, n) in splits.items():
        paths = list_clips(args.data, flt)[:n]
        if not paths:
            print(f"[eval] split {name}: NO clips", flush=True)
            continue
        rows = []
        for i, p in enumerate(paths):
            clip = torch.load(p, map_location="cpu", weights_only=False)
            vpath = None
            if name in ("heldseed", "heldtask") and vids_done < args.video:
                vpath = os.path.join(args.out, f"rollout_{name}_{os.path.basename(p)[:-3]}.mp4")
            r = eval_clip(model, clip, dev, M, args.K, spatial, video_path=vpath)
            rows.append(r)
            if vpath:
                vids_done += 1
                print(f"[eval] video -> {vpath}", flush=True)
        results[name] = rows
        corr = np.array([x["corr"] for x in rows]); ratio = np.array([x["ratio"] for x in rows])
        dn = np.array([x["d_null"] for x in rows]); dw = np.array([x["d_wrong"] for x in rows])
        fp = np.array([x["frac_pred"] for x in rows]); fg = np.array([x["frac_gt"] for x in rows])
        lk = np.array([x["leak"] for x in rows])
        mpr = np.array([x["m_prec"] for x in rows]); mre = np.array([x["m_rec"] for x in rows])
        print(f"\n=== SPLIT {name} (n={len(rows)}) ===", flush=True)
        print(f"  corr(GT,PRED):  mean={np.nanmean(corr):.3f}  median={np.nanmedian(corr):.3f}  "
              f"min={np.nanmin(corr):.3f}  max={np.nanmax(corr):.3f}", flush=True)
        print(f"  top-mover ratio: mean={np.nanmean(ratio):.3f}  median={np.nanmedian(ratio):.3f}", flush=True)
        print(f"  static-leakage:  mean={np.nanmean(lk):.4f}  median={np.nanmedian(lk):.4f}  (->0 with gate)", flush=True)
        if np.isfinite(mpr).any():
            print(f"  mover P/R(p_dyn): precision={np.nanmean(mpr):.3f}  recall={np.nanmean(mre):.3f}", flush=True)
        print(f"  frac>0.02r:      PRED mean={np.nanmean(fp):.3f}  GT mean={np.nanmean(fg):.3f}", flush=True)
        print(f"  language Δ_null:  mean={np.nanmean(dn):.3f}  median={np.nanmedian(dn):.3f}", flush=True)
        print(f"  language Δ_wrong: mean={np.nanmean(dw):.3f}  median={np.nanmedian(dw):.3f}", flush=True)

    # save raw rows
    import json
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2, default=float)
    print(f"\n[eval] wrote {os.path.join(args.out, 'results.json')}", flush=True)
    # the generalization verdict in one line
    if "heldseed" in results or "heldtask" in results:
        tr = np.nanmean([x["corr"] for x in results.get("train", [])]) if results.get("train") else float("nan")
        hs = np.nanmean([x["corr"] for x in results.get("heldseed", [])]) if results.get("heldseed") else float("nan")
        ht = np.nanmean([x["corr"] for x in results.get("heldtask", [])]) if results.get("heldtask") else float("nan")
        print(f"\n[VERDICT] train corr={tr:.3f} | held-seed corr={hs:.3f} | held-task corr={ht:.3f}", flush=True)


if __name__ == "__main__":
    main()
