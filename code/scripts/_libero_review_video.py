"""Render the REVIEW video for a LIBERO pure-video clip: per frame t=0..Kf three columns
   [ REAL video | GT-motion render | MODEL-prediction render ]  (renders at x4 splat scales).
Usage: _libero_review_video.py <clip.pt> <ckpt.pt> <out.mp4>"""
import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import sample_controls, _to_dev

clip_p, ckpt_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
c = torch.load(clip_p, map_location="cuda", weights_only=False)
ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
mdl = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                           cond_mode="aggregator", spatial_ground=True, dyn_gate=bool(ck.get("dyn_gate", 0)),
                           sem_dim=ck.get("sem_dim", 0), gate_uses_sem=bool(ck.get("gate_uses_sem", 1)),
                           gate_entity_pool=bool(ck.get("gate_entity_pool", 0)),
                           entity_lbs=bool(ck.get("entity_lbs", 0))).cuda().eval()
mdl.load_state_dict(ck["model"], strict=False)

g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
tr = c["traj"].cuda(); K = int(c["Kf"]); uv = c["uv"].cuda()
disp = (tr[K] - tr[0]).norm(dim=-1)
gen = torch.Generator(device="cuda").manual_seed(0)
ci = sample_controls(g0.means, disp, ck.get("M", 2048), gen, 0.01); cu = uv[ci]
img0 = (c["gt_rgb"][0].cuda().float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
seg_g = c["seg_per_g"].cuda().long() if (ck.get("entity_lbs", 0) or ck.get("gate_entity_pool", 0)) else None
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vi = _to_dev(mdl.encoder.build_inputs(c["instruction"], img0), "cuda")
    out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=cu, control_uv_hw=(int(c["H"]), int(c["W"])),
              seg_per_g=seg_g)
pred = out["means"].float()                       # [K,N,3]

pe = (pred[K - 1] - tr[K]).norm(dim=-1)
mv = disp > 0.01
print(f"clip={clip_p}  instr={c['instruction']!r}")
print(f"endpoint err: ALL med {float(pe.median())*100:.1f}cm | MOVERS med {float(pe[mv].median())*100:.1f}cm "
      f"p90 {float(pe[mv].quantile(0.9))*100:.1f}cm  (n_mov={int(mv.sum())}, GT mover disp med {float(disp[mv].median())*100:.1f}cm)")

sc = c["scales"] * 4.0; vm = c["viewmat"][None].float(); Ki = c["K_intr"][None].float()
W = int(c["W"]); H = int(c["H"])


def R(mn):
    g = GaussianSet(mn.float(), c["quats"], sc, c["opacities"], c["colors"], None)
    col, _, _ = render_gaussianset(g, vm, Ki, W, H)
    return (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def label(im, txt):
    im = im.copy(); cv2.putText(im, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


frames = []
for t in range(K + 1):
    real = c["gt_rgb"][t].cpu().numpy()
    gtr = R(tr[t])
    prr = R(pred[t - 1] if t > 0 else g0.means)
    row = np.concatenate([label(real, f"REAL t={t}"), label(gtr, "GT-motion"), label(prr, "MODEL pred")], 1)
    frames.append(row)
# 2 fps + hold first/last so the eye can compare
frames = [frames[0]] * 2 + frames + [frames[-1]] * 3
iio.mimwrite(out_p, frames, fps=2, quality=8)
print(f"saved {out_p}  ({len(frames)} frames, {W*3}x{H})")
