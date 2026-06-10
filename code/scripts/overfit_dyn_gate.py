"""Exp-1 ACCEPTANCE TEST — overfit ONE PickCube clip WITH vs WITHOUT the dynamics gate (--dyn_gate).

Reuses overfit_motion_sim.py's setup (clean sim clip, mover-biased control sampling, resume dynamics
from stream11c, spatial-grounding on) and adds the Exp-1 mover-BCE supervision on the per-control gate.
It runs the SAME clip in BOTH modes and prints the before/after numbers the acceptance gate needs:

  static-leakage = mean PRED disp (normalized) of GT-STATIC controls (the table) -> should ->0 WITH gate
  cube-move      = mean PRED disp on GT-cube/mover controls vs GT (PASS: > 0.5x GT)
  mover P/R      = precision/recall of sigmoid(p_dyn)>0.5 vs the free mover label (PASS: >0.85)
  corr           = corr(GT_disp, PRED_disp) (PASS: gate corr >= ungated corr)

PASS (notes Exp-1 §3): with the gate the table stops moving (leak~0), the cube moves (>0.5x GT),
mover-P/R > 0.85, and corr >= the ungated run.

Usage (server, GPU 0):
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python code/scripts/overfit_dyn_gate.py \
      data/maniskill_fused/pickcube_s1002_train.pt 500
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
from igsw.gaussians.types import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from igsw.training.losses import trajectory_loss, rotation_loss, mover_bce_loss

dev = "cuda"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill_fused/pickcube_s1002_train.pt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
RESUME = "checkpoints/stream11c_infonce/ckpt_0006000.pt"
M = 2048
MOVER_THRESH = 0.01

# ---------------------------------------------------------------------------- #
# load the CLEAN sim clip (identical prep to overfit_motion_sim.py)
# ---------------------------------------------------------------------------- #
clip = torch.load(CLIP, map_location="cpu", weights_only=False)
g0 = GaussianSet(clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
                 clip["opacities"].to(dev), clip["colors"].to(dev)).to(dev)
N = len(g0)
uv = clip["uv"].to(dev)                                    # [N,2] frame-0 pixels
seg_per_g = clip["seg_per_g"].to(dev)                      # [N] entity id
traj_full = clip["traj"].to(dev)                           # [Kf+1,N,3] EXACT GT
K = int(clip["Kf"])
H, W = int(clip["H"]), int(clip["W"])
img_hw = (H, W)
instruction = clip["instruction"]
gt_rgb0 = clip["gt_rgb"][0].numpy()                        # [H,W,3] uint8

# mover-biased control sampling (shared by BOTH runs via fixed seed -> identical control set)
gen = torch.Generator(device=dev).manual_seed(0)
disp_all = (traj_full[K] - traj_full[0]).norm(dim=-1)      # [N]
mover_g = (disp_all > MOVER_THRESH).nonzero(as_tuple=True)[0]
static_g = (disp_all <= MOVER_THRESH).nonzero(as_tuple=True)[0]
n_mover = int(min(mover_g.numel(), M // 2))
n_static = M - n_mover
sel_m = mover_g[torch.randperm(mover_g.numel(), device=dev, generator=gen)[:n_mover]]
sel_s = static_g[torch.randperm(static_g.numel(), device=dev, generator=gen)[:n_static]]
ctrl_idx = torch.cat([sel_m, sel_s])
ctrl_idx = ctrl_idx[torch.randperm(ctrl_idx.numel(), device=dev, generator=gen)]
M = ctrl_idx.numel()

control_uv = uv[ctrl_idx]                                  # [M,2]
gt_pos = traj_full[:, ctrl_idx, :]                         # [K+1,M,3]
vis = torch.ones(K + 1, M, dtype=torch.bool, device=dev)
gt_traj = gt_pos[1:K + 1]                                  # [K,M,3]
vis_traj = vis[1:K + 1]
init = g0.means[ctrl_idx]                                  # [M,3]
gt_full = torch.cat([init[None], gt_traj], 0)

radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius    # [M] normalized GT displacement
mover_label = (disp_all[ctrl_idx] > MOVER_THRESH).float()  # [M] FREE mover label (the BCE target)
is_mover = mover_label > 0.5
is_static = ~is_mover
gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
knn_idx = torch.cdist(init, init).topk(9, largest=False).indices[:, 1:]

print(f"clip={os.path.basename(CLIP)}  N={N} M={M} K={K}  instruction={instruction!r}", flush=True)
print(f"controls: movers={int(is_mover.sum())} static={int(is_static.sum())}  "
      f"GT mover-disp(norm) mean={gt_disp[is_mover].mean():.3f}  static mean={gt_disp[is_static].mean():.4f}",
      flush=True)

ck = torch.load(RESUME, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])


@torch.no_grad()
def metrics(out):
    pd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius              # [M] normalized PRED disp
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pd.float()]))[0, 1].item()
    leak = pd[is_static].mean().item()                                         # static-leakage -> 0
    cube = pd[is_mover].mean().item()                                          # mover PRED disp
    cube_gt = gt_disp[is_mover].mean().item()
    out_d = {"corr": corr, "leak": leak, "cube": cube, "cube_gt": cube_gt,
             "cube_ratio": cube / max(cube_gt, 1e-6)}
    if "p_dyn" in out:
        pred_mv = torch.sigmoid(out["p_dyn"].float()) > 0.5
        tp = (pred_mv & is_mover).sum().float()
        out_d["prec"] = (tp / pred_mv.sum().clamp_min(1)).item()
        out_d["rec"] = (tp / is_mover.sum().clamp_min(1)).item()
    return out_d


def run(dyn_gate: bool, w_dyn: float = 1.0):
    torch.manual_seed(0)
    model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16),
                                 cond_mode="aggregator", spatial_ground=True,
                                 dyn_gate=dyn_gate).to(dev)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
              else (v.to(dev) if torch.is_tensor(v) else v))
          for k, v in model.encoder.build_inputs(instruction, gt_rgb0).items()}
    sg = ("vis_tok", "vis_film", "vis_norm", "vis_vhead", "dyn_head", "sem_head")
    sgp = [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sg)]
    base = [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sg)]
    opt = torch.optim.AdamW([{"params": base, "lr": 3e-4}, {"params": sgp, "lr": 1e-3}])
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    tag = "GATE" if dyn_gate else "BASE"
    print(f"\n===== {tag} (dyn_gate={dyn_gate}, w_dyn={w_dyn if dyn_gate else 0}) | "
          f"load missing={len(miss)} unexpected={len(unexp)} =====", flush=True)
    last = None
    for s in range(STEPS + 1):
        with amp:
            out = model(vi, g0, K, ctrl_idx=ctrl_idx, control_uv=control_uv, control_uv_hw=img_hw)
        pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                       relevance=rel, obj_focus=0.0)
        rot_l = rotation_loss(out["om"].float(), gt_full.float(), knn_idx,
                              torch.cat([vis[:1], vis_traj], 0))
        loss = pos_l + vel_l + 0.2 * rot_l
        dyn_l = out["v"].new_zeros(())
        if dyn_gate and "p_dyn" in out:
            dyn_l = mover_bce_loss(out["p_dyn"].float(), mover_label)
            loss = loss + w_dyn * dyn_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if s % 50 == 0 or s == STEPS:
            md = metrics(out)
            last = md
            extra = (f"  P{md.get('prec', float('nan')):.2f} R{md.get('rec', float('nan')):.2f} "
                     f"dyn{dyn_l.item():.3f}") if dyn_gate else ""
            print(f"  step {s:4d}  pos{pos_l.item():.4f}  corr={md['corr']:.3f}  "
                  f"leak={md['leak']:.4f}  cube={md['cube']:.3f}(gt {md['cube_gt']:.3f}, "
                  f"{md['cube_ratio']:.2f}x){extra}", flush=True)
    return last


base = run(dyn_gate=False)
gate = run(dyn_gate=True, w_dyn=1.0)

print("\n================ ACCEPTANCE SUMMARY (final step) ================", flush=True)
print(f"               {'BASE(no gate)':>16s}  {'GATE(--dyn_gate)':>16s}", flush=True)
print(f"corr           {base['corr']:>16.3f}  {gate['corr']:>16.3f}", flush=True)
print(f"static-leakage {base['leak']:>16.4f}  {gate['leak']:>16.4f}", flush=True)
print(f"cube PRED/GT   {base['cube_ratio']:>15.2f}x  {gate['cube_ratio']:>15.2f}x", flush=True)
print(f"mover precision {'  n/a':>15s}  {gate.get('prec', float('nan')):>16.3f}", flush=True)
print(f"mover recall    {'  n/a':>15s}  {gate.get('rec', float('nan')):>16.3f}", flush=True)
# PASS checks
p_leak = gate["leak"] <= max(0.02, 0.25 * base["leak"])         # leak dropped toward 0
p_cube = gate["cube_ratio"] > 0.5                               # cube moves > 0.5x GT
p_pr = gate.get("prec", 0) > 0.85 and gate.get("rec", 0) > 0.85
p_corr = gate["corr"] >= base["corr"] - 1e-3
ok = p_leak and p_cube and p_pr and p_corr
print(f"\nPASS: leak↓={p_leak}  cube>0.5x={p_cube}  moverP/R>0.85={p_pr}  corr>=base={p_corr}  "
      f"=> {'PASS' if ok else 'FAIL'}", flush=True)
