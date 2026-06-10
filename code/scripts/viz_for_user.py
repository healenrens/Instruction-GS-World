"""Visualizations for the user:
  (1) TRAINING DATA  — the clean whole-video-FUSED GT rollout of train clips (what the model learns from).
  (2) TEST RESULTS   — on HELD-OUT clips: GT | static | model-PREDICTION rollout (how the model does on
      unseen data), with the held-out corr printed.
All rendered with FROZEN g0 appearance (isolates MOTION; sidesteps the appearance-drift artifact).
Usage: python code/scripts/viz_for_user.py [CKPT]"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
import imageio.v2 as iio  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from scripts.eval_sim_generalization import sample_controls, _to_dev  # noqa: E402

dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sim_fused_ft/ckpt_0000300.pt"
DATA = "data/maniskill_fused"
TS = [0, 4, 8, 12, 16]
os.makedirs("outputs/viz_user", exist_ok=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
model = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                             cond_mode="aggregator", spatial_ground=True).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
print(f"model ckpt={CKPT} step={ck.get('step')}", flush=True)


def ds(im):  # 512 -> 256 downscale for a manageable montage
    return im[::2, ::2]


def rend(g, vm, K, W, H):
    c, _, _ = render_gaussianset(g, vm[None], K[None], W, H)
    return ds(c[0].clamp(0, 1).cpu().numpy())


def gmove(g0, m):
    return GaussianSet(m.float(), g0.quats, g0.scales, g0.opacities, g0.colors, None)


def bar(row, rgb):  # colored left-border to label a montage row
    row = row.copy(); row[:, :6] = np.array(rgb, np.float32); return row


def load(p):
    c = torch.load(p, map_location=dev, weights_only=False)
    g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
    return c, g0


# ---------------- (1) TRAINING DATA: fused GT rollout ----------------
trs = ([p for p in sorted(glob.glob(f"{DATA}/*_train.pt")) if "pickcube" in p][:1] +
       [p for p in sorted(glob.glob(f"{DATA}/*_train.pt")) if "pushcube" in p][:1])
rows = []
for p in trs:
    c, g0 = load(p)
    traj = c["traj"].to(dev); K = min(16, int(c["Kf"])); H, W = int(c["H"]), int(c["W"])
    Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
    row = np.concatenate([rend(gmove(g0, traj[t]), vm, Ki, W, H) for t in TS if t <= K], axis=1)
    rows.append(bar(row, (0.1, 0.9, 0.1)))
    print(f"TRAIN {os.path.basename(p)}  N={len(g0)}  instr='{c['instruction'][:55]}'", flush=True)
iio.imwrite("outputs/viz_user/1_training_data_fusedGT.png", (np.concatenate(rows, 0) * 255).astype(np.uint8))

# ---------------- (2) TEST RESULTS: GT | static | PRED on held-out ----------------
held = sorted(glob.glob(f"{DATA}/*_heldseed.pt"))[:1] + sorted(glob.glob(f"{DATA}/*_heldtask.pt"))[:1]
for p in held:
    c, g0 = load(p)
    traj = c["traj"].to(dev); K = min(16, int(c["Kf"])); H, W = int(c["H"]), int(c["W"])
    Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
    uv = c["uv"].to(dev); disp = (traj[K] - traj[0]).norm(dim=-1)
    gen = torch.Generator(device=dev).manual_seed(0)
    ci = sample_controls(g0.means, disp, ck.get("M", 2048), gen, 0.01)
    cu = uv[ci]
    img0 = (c["gt_rgb"][0].to(dev).float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(model.encoder.build_inputs(c["instruction"], img0), dev)
        out = model(vi, g0, K, ctrl_idx=ci, control_uv=cu, control_uv_hw=(H, W))
    init = g0.means[ci]
    radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    gtd = (traj[K, ci] - traj[0, ci]).norm(dim=-1) / radius
    prd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius
    corr = torch.corrcoef(torch.stack([gtd.float(), prd.float()]))[0, 1].item()
    gt = bar(np.concatenate([rend(gmove(g0, traj[t]), vm, Ki, W, H) for t in TS if t <= K], 1), (0.1, 0.9, 0.1))
    st = bar(np.concatenate([rend(g0, vm, Ki, W, H) for t in TS if t <= K], 1), (0.6, 0.6, 0.6))
    pr = bar(np.concatenate([rend(gmove(g0, out["means"][t - 1] if t > 0 else g0.means), vm, Ki, W, H)
                             for t in TS if t <= K], 1), (0.95, 0.15, 0.15))
    split = "heldseed" if "heldseed" in p else "heldtask"
    iio.imwrite(f"outputs/viz_user/2_test_{split}_GT_static_PRED.png",
                (np.concatenate([gt, st, pr], 0) * 255).astype(np.uint8))
    print(f"TEST [{split}] {os.path.basename(p)}  instr='{c['instruction'][:50]}'  corr={corr:.3f}", flush=True)
print("done -> outputs/viz_user/", flush=True)
