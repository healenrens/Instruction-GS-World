"""DECISIVE capacity test on CLEAN ManiSkill-simulation GT (agent.md section 38 data pivot).

Adapts overfit_motion.py: the ONLY change is the clip source. Instead of Pi3+CoTracker
(occlusion-noisy real-data GT, on which the model could NOT localize motion even overfitting
one clip), we load a CLEAN sim clip from maniskill_gt.py — EXACT rendered-depth 3DGS + EXACT,
occlusion-free per-Gaussian 3D trajectory (vis = all ones).

The decisive question: on perfect GT, does the dynamics LOCALIZE motion?
  corr(GT_disp, PRED_disp) -> 0.7+  AND  topmover_ratio -> 0.5+   => architecture is CAPABLE
                                                                     (real failure was GT noise)
  still uniform / global drift                                    => genuine architecture limit

Usage:
  python code/scripts/overfit_motion_sim.py <clip.pt> [obj_focus=0] [spatial=0] [steps=600]

The 1.76B InstructGSWorldModel resumes its dynamics weights from stream11c (strict=False).
Model call (unchanged): model(vlm_inputs, g0, K, ctrl_idx=ctrl_idx, control_uv=, control_uv_hw=)
  -> out["ctrl"] [K,M,3] predicted control trajectory; out["v"]/["om"] deltas.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, "code")
from igsw.gaussians.types import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from igsw.training.losses import trajectory_loss, rotation_loss

dev = "cuda"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill/pickcube_s0.pt"
OBJF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0     # 0 = plain L1 (clean capacity test)
SPATIAL = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 600
M = 2048

# ---------------------------------------------------------------------------- #
# load the CLEAN sim clip (replaces the entire Pi3 + CoTracker prep)
# ---------------------------------------------------------------------------- #
clip = torch.load(CLIP, map_location="cpu", weights_only=False)
g0 = GaussianSet(
    clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
    clip["opacities"].to(dev), clip["colors"].to(dev),
).to(dev)
N = len(g0)
uv = clip["uv"].to(dev)                                    # [N,2] (x=col, y=row) frame-0 pixels
seg_per_g = clip["seg_per_g"].to(dev)                      # [N] entity id per gaussian
traj_full = clip["traj"].to(dev)                           # [Kf+1, N, 3] EXACT per-gaussian world traj
K = int(clip["Kf"])                                        # actual horizon used
H, W = int(clip["H"]), int(clip["W"])
img_hw = (H, W)
instruction = clip["instruction"]
gt_rgb0 = clip["gt_rgb"][0].numpy()                        # [H,W,3] uint8 frame-0 RGB for the VLM

# control set: sample M controls, BIASED to include the movers so the test exercises localization
# (a uniform sample of 2048 from ~200k would get ~10 cube gaussians — too few to measure ratio).
gen = torch.Generator(device=dev).manual_seed(0)
disp_all = (traj_full[K] - traj_full[0]).norm(dim=-1)      # [N]
mover_g = (disp_all > 0.01).nonzero(as_tuple=True)[0]      # gaussians that actually move
static_g = (disp_all <= 0.01).nonzero(as_tuple=True)[0]
n_mover = int(min(mover_g.numel(), M // 2))                # up to half the controls are movers
n_static = M - n_mover
sel_m = mover_g[torch.randperm(mover_g.numel(), device=dev, generator=gen)[:n_mover]]
sel_s = static_g[torch.randperm(static_g.numel(), device=dev, generator=gen)[:n_static]]
ctrl_idx = torch.cat([sel_m, sel_s])
ctrl_idx = ctrl_idx[torch.randperm(ctrl_idx.numel(), device=dev, generator=gen)]   # shuffle
M = ctrl_idx.numel()

control_uv = uv[ctrl_idx]                                  # [M,2]
gt_pos = traj_full[:, ctrl_idx, :]                         # [K+1, M, 3] EXACT control trajectory
vis = torch.ones(K + 1, M, dtype=torch.bool, device=dev)  # SIM = fully observed (no occlusion noise)
gt_traj = gt_pos[1:K + 1]                                  # [K,M,3]
vis_traj = vis[1:K + 1]
init = g0.means[ctrl_idx]                                  # [M,3] frame-0 control positions

# normalization radius over the controls (workspace scale), used by corr/topmover diagnostics
radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius    # [M] per-control GT displacement
# per-control task relevance (from GT motion) for obj_focus, as in the real-data overfit
gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
knn_idx = torch.cdist(init, init).topk(9, largest=False).indices[:, 1:]

# ---------------------------------------------------------------------------- #
# model (resume dynamics from stream11c, strict=False) — identical to overfit_motion.py
# ---------------------------------------------------------------------------- #
ck = torch.load("checkpoints/stream11c_infonce/ckpt_0006000.pt", map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16), cond_mode="aggregator",
                             spatial_ground=SPATIAL).to(dev)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
print("clip=%s  N=%d M=%d K=%d  instruction=%r" % (os.path.basename(CLIP), N, M, K, instruction), flush=True)
print("spatial_ground=%s obj_focus=%.1f | load missing=%d unexpected=%d" % (
    SPATIAL, OBJF, len(missing), len(unexpected)), flush=True)
print("GT: frac>0.02r=%.3f  top5%%disp=%.3f  movers_in_controls=%d/%d" % (
    (gt_disp > 0.02).float().mean().item(), gt_disp.topk(M // 20).values.mean().item(),
    int((gt_disp > 0.02).sum()), M), flush=True)

vi = model.encoder.build_inputs(instruction, gt_rgb0)
vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
          else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}

# higher LR on freshly-initialized spatial-grounding params (as in overfit_motion.py)
sg_names = ("vis_tok", "vis_film", "vis_norm", "vis_vhead")
sg_params = [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sg_names)]
base_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sg_names)]
if SPATIAL and sg_params:
    opt = torch.optim.AdamW([{"params": base_params, "lr": 3e-4}, {"params": sg_params, "lr": 1e-3}])
else:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
amp = torch.autocast("cuda", dtype=torch.bfloat16)
gt_full = torch.cat([init[None], gt_traj], 0)

for s in range(STEPS + 1):
    with amp:
        out = model(vi, g0, K, ctrl_idx=ctrl_idx, control_uv=control_uv, control_uv_hw=img_hw)
    pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                   relevance=rel, obj_focus=OBJF)
    rot_l = rotation_loss(out["om"].float(), gt_full.float(), knn_idx, torch.cat([vis[:1], vis_traj], 0))
    loss = pos_l + vel_l + 0.2 * rot_l
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(gn):
        opt.step()
    if s % 50 == 0:
        with torch.no_grad():
            pd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius
            corr = torch.corrcoef(torch.stack([gt_disp.float(), pd.float()]))[0, 1]
            topk = gt_disp.topk(M // 20).indices
            ratio = (pd[topk].mean() / gt_disp[topk].mean().clamp_min(1e-6)).item()
            print("step %3d  pos%.4f  PRED p50=%.3f p90=%.3f max=%.3f  corr=%.3f  topmover_ratio=%.2f" % (
                s, pos_l.item(), pd.median().item(), pd.quantile(0.9).item(), pd.max().item(),
                corr.item(), ratio), flush=True)
