"""§44h ACCEPTANCE — overfit ONE PickCube clip with the SEM head ON, A/B-ing whether the dynamics GATE
CONSUMES the occlusion-robust 3D identity e_sem (`--gate_uses_sem`).

The gate's per-control feature is the Qwen 2D image patch. At a pixel where the ARM OCCLUDES the TABLE
the patch is the ARM's feature (overlap-ambiguous) -> that table-control can leak (table "sinks" under the
arm = the user's complaint). §44h feeds the per-control object-identity embedding e_sem (supervised by the
3D per-Gaussian seg id, OVERLAP-INVARIANT) INTO the gate (concat with the patch), so the gate can know
"this 3D point is table (static)" even under the arm.

This script runs the SAME clip in BOTH modes (gate_uses_sem 0 vs 1), with the SAME sem head + losses
(--sem_dim 16 --w_seg 0.3 --obj_focus 1.5), resuming the SAME dynamics ckpt, and prints:

  seg            = the object-semantic (seg_per_g prototype-CE + 3D-NN) loss -> must DROP (head learns)
  static-leakage = mean PRED disp (normalized) of GT-STATIC controls (the table) -> should ->0
  leak_occ       = static-leakage RESTRICTED to table controls whose frame-0 2D pixel is OCCLUDED by a
                   MOVER (the arm) -> the user's specific overlap metric; vs leak_unocc (un-occluded table)
  cube_ratio     = mean PRED disp on GT-mover controls / GT (the cube magnitude)
  corr           = corr(GT_disp, PRED_disp)
  mover P/R      = precision/recall of sigmoid(p_dyn)>0.5 vs the free mover label

PASS (§44h): trains w/o NaN, seg drops, gate works (leak low, corr>=0.85), and ideally
leak(gate_uses_sem=1) <= leak(gate_uses_sem=0) — especially leak_occ (the overlap region the 2D patch
cannot disambiguate but e_sem can).

Usage (server, GPU 0):
  CUDA_VISIBLE_DEVICES=0 HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ./.venv/bin/python code/scripts/overfit_gate_sem.py data/maniskill_fused/pickcube_s1002_train.pt 200
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
from igsw.gaussians.types import GaussianSet
from igsw.gaussians.render import render_gaussianset
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from igsw.training.losses import (trajectory_loss, rotation_loss, mover_bce_loss,
                                  semantic_id_loss)

dev = "cuda"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill_fused/pickcube_s1002_train.pt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
RESUME = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/sim_gen/ckpt_last.pt"
M = 2048
MOVER_THRESH = 0.01
SEM_DIM = 16
W_SEG = 0.3
W_DYN = 1.0
OBJ_FOCUS = 1.5

# ---------------------------------------------------------------------------- #
# load the CLEAN sim clip
# ---------------------------------------------------------------------------- #
clip = torch.load(CLIP, map_location="cpu", weights_only=False)
g0 = GaussianSet(clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
                 clip["opacities"].to(dev), clip["colors"].to(dev)).to(dev)
N = len(g0)
uv = clip["uv"].to(dev)                                    # [N,2] frame-0 pixels (x=col,y=row)
seg_per_g = clip["seg_per_g"].to(dev)                      # [N] entity id
traj_full = clip["traj"].to(dev)                           # [Kf+1,N,3] EXACT GT
K = int(clip["Kf"])
H, W = int(clip["H"]), int(clip["W"])
img_hw = (H, W)
instruction = clip["instruction"]
gt_rgb0 = clip["gt_rgb"][0].numpy()                        # [H,W,3] uint8
K_intr = clip["K_intr"].to(dev).float()                   # [3,3]
viewmat = clip["viewmat"].to(dev).float()                 # [4,4] world->cam

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
seg_ctrl = seg_per_g[ctrl_idx]                             # [M] entity id per control

radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius    # [M] normalized GT displacement
mover_label = (disp_all[ctrl_idx] > MOVER_THRESH).float()  # [M] FREE mover label (the BCE target)
is_mover = mover_label > 0.5
is_static = ~is_mover
gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
knn_idx = torch.cdist(init, init).topk(9, largest=False).indices[:, 1:]

# ---------------------------------------------------------------------------- #
# OCCLUSION MASK (the user's overlap metric): a STATIC (table) control is "occluded by the arm" iff the
# NEAREST RENDERED SURFACE at its frame-0 2D pixel is a MOVING entity. We render frame-0 with each
# Gaussian colored by its mover/static label (mover=1, static=0) broadcast to RGB; the rasterizer's
# front-to-back compositing makes the rendered value at a pixel ~the front surface's label. We read it
# back at each static control's uv: value>0.5 => the front surface there is a mover (arm) => occluded.
# ---------------------------------------------------------------------------- #
with torch.no_grad():
    mover_g_mask = (disp_all > MOVER_THRESH).float()                    # [N] per-GAUSSIAN mover label
    seg_color = mover_g_mask[:, None].repeat(1, 3)                      # [N,3] (1=mover surface)
    g_id = GaussianSet(g0.means, g0.quats, g0.scales, g0.opacities, seg_color)
    surf, alpha, _ = render_gaussianset(g_id, viewmat[None], K_intr[None], W, H)  # [1,H,W,3]
    surf = surf[0, :, :, 0]                                             # [H,W] front-surface mover frac
    # sample at each control's frame-0 pixel (x=col,y=row); clamp to the image
    cx = control_uv[:, 0].round().long().clamp(0, W - 1)
    cy = control_uv[:, 1].round().long().clamp(0, H - 1)
    front_mover = surf[cy, cx] > 0.5                                    # [M] front surface is a mover
    occ_static = is_static & front_mover                               # static control occluded by a mover
    unocc_static = is_static & (~front_mover)

print(f"clip={os.path.basename(CLIP)}  N={N} M={M} K={K}  instruction={instruction!r}", flush=True)
print(f"controls: movers={int(is_mover.sum())} static={int(is_static.sum())} | "
      f"OCCLUDED-static(table-under-arm)={int(occ_static.sum())} unocc-static={int(unocc_static.sum())}",
      flush=True)
print(f"GT mover-disp(norm) mean={gt_disp[is_mover].mean():.3f}  static mean={gt_disp[is_static].mean():.4f}",
      flush=True)

ck = torch.load(RESUME, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])


@torch.no_grad()
def metrics(out):
    pd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius              # [M] normalized PRED disp
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pd.float()]))[0, 1].item()
    leak = pd[is_static].mean().item()                                         # global static-leakage
    leak_occ = pd[occ_static].mean().item() if occ_static.any() else float("nan")
    leak_unocc = pd[unocc_static].mean().item() if unocc_static.any() else float("nan")
    cube = pd[is_mover].mean().item()
    cube_gt = gt_disp[is_mover].mean().item()
    d = {"corr": corr, "leak": leak, "leak_occ": leak_occ, "leak_unocc": leak_unocc,
         "cube_ratio": cube / max(cube_gt, 1e-6), "nan": bool(torch.isnan(pd).any())}
    if "p_dyn" in out:
        pred_mv = torch.sigmoid(out["p_dyn"].float()) > 0.5
        tp = (pred_mv & is_mover).sum().float()
        d["prec"] = (tp / pred_mv.sum().clamp_min(1)).item()
        d["rec"] = (tp / is_mover.sum().clamp_min(1)).item()
    return d


def run(gate_uses_sem: int):
    torch.manual_seed(0)
    model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16),
                                 cond_mode="aggregator", spatial_ground=True,
                                 dyn_gate=True, sem_dim=SEM_DIM,
                                 gate_uses_sem=bool(gate_uses_sem)).to(dev)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
              else (v.to(dev) if torch.is_tensor(v) else v))
          for k, v in model.encoder.build_inputs(instruction, gt_rgb0).items()}
    sg = ("vis_tok", "vis_film", "vis_norm", "vis_vhead", "dyn_head", "sem_head", "sem_proto")
    sgp = [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sg)]
    base = [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sg)]
    opt = torch.optim.AdamW([{"params": base, "lr": 3e-4}, {"params": sgp, "lr": 1e-3}])
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    # cosine LR -> 0 by STEPS (the §44c "settled" schedule that locks the gate-working state)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, STEPS))
    tag = f"gate_uses_sem={gate_uses_sem}"
    miss_dh = [m for m in miss if "dyn_head" in m or "sem_head" in m or "sem_proto" in m]
    print(f"\n===== {tag} | load missing={len(miss)} unexpected={len(unexp)} "
          f"(reinit gate/sem e.g. {miss_dh[:3]}) =====", flush=True)
    last = None
    for s in range(STEPS + 1):
        with amp:
            out = model(vi, g0, K, ctrl_idx=ctrl_idx, control_uv=control_uv, control_uv_hw=img_hw)
        pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                       relevance=rel, obj_focus=OBJ_FOCUS)
        rot_l = rotation_loss(out["om"].float(), gt_full.float(), knn_idx,
                              torch.cat([vis[:1], vis_traj], 0))
        loss = pos_l + vel_l + 0.2 * rot_l
        dyn_l = mover_bce_loss(out["p_dyn"].float(), mover_label)
        seg_l = semantic_id_loss(out["e_sem"], seg_ctrl, knn_idx, sem_proto=out.get("sem_proto"))
        loss = loss + W_DYN * dyn_l + W_SEG * seg_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if s < STEPS:
            sched.step()
        if s % 25 == 0 or s == STEPS:
            md = metrics(out)
            last = md
            print(f"  step {s:4d}  pos{pos_l.item():.4f}  seg{seg_l.item():.3f}  dyn{dyn_l.item():.3f}  "
                  f"corr={md['corr']:.3f}  leak={md['leak']:.4f}  leak_occ={md['leak_occ']:.4f}  "
                  f"cube={md['cube_ratio']:.2f}x  P{md.get('prec', float('nan')):.2f} "
                  f"R{md.get('rec', float('nan')):.2f}  nan={md['nan']}", flush=True)
    last["seg_final"] = seg_l.item()
    return last


off = run(gate_uses_sem=0)   # sem head ON, but the GATE ignores e_sem (the 2D-only baseline)
on = run(gate_uses_sem=1)    # §44h: the GATE consumes e_sem (occlusion-robust identity)

print("\n================ §44h ACCEPTANCE SUMMARY (final step) ================", flush=True)
print(f"                    {'gate_uses_sem=0':>16s}  {'gate_uses_sem=1':>16s}", flush=True)
print(f"seg (final)         {off['seg_final']:>16.3f}  {on['seg_final']:>16.3f}", flush=True)
print(f"corr                {off['corr']:>16.3f}  {on['corr']:>16.3f}", flush=True)
print(f"static-leakage      {off['leak']:>16.4f}  {on['leak']:>16.4f}", flush=True)
print(f"leak OCCLUDED-table {off['leak_occ']:>16.4f}  {on['leak_occ']:>16.4f}", flush=True)
print(f"leak unocc-table    {off['leak_unocc']:>16.4f}  {on['leak_unocc']:>16.4f}", flush=True)
print(f"cube PRED/GT        {off['cube_ratio']:>15.2f}x  {on['cube_ratio']:>15.2f}x", flush=True)
print(f"mover precision     {off.get('prec', float('nan')):>16.3f}  {on.get('prec', float('nan')):>16.3f}", flush=True)
print(f"mover recall        {off.get('rec', float('nan')):>16.3f}  {on.get('rec', float('nan')):>16.3f}", flush=True)
# PASS checks (§44h)
p_nan = (not off["nan"]) and (not on["nan"])
p_seg = on["seg_final"] < 1.0                                   # sem head learned (drops from ~7)
p_corr = on["corr"] >= 0.85
p_leak = on["leak"] <= 0.03
# the user's specific overlap win: occluded-table leak no worse with the sem-fed gate (ideally lower)
p_occ = (not np.isnan(on["leak_occ"])) and (np.isnan(off["leak_occ"]) or on["leak_occ"] <= off["leak_occ"] + 1e-4)
ok = p_nan and p_seg and p_corr and p_leak
print(f"\nPASS: noNaN={p_nan}  segDrops={p_seg}  corr>=0.85={p_corr}  leak<=0.03={p_leak}  "
      f"=> {'PASS' if ok else 'FAIL'}", flush=True)
print(f"overlap win (leak_occ[sem-gate] <= leak_occ[2D-gate]): {p_occ}", flush=True)
