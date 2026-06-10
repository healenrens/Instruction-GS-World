"""Export the GATED MODEL'S PREDICTION as 3DGS .ply for 3D REVIEW (not video): load a dyn_gate ckpt +
a held-out clip, roll out, and write the predicted dense Gaussians at t=0/8/K as standard 3DGS .ply
(frozen g0 appearance = isolate motion), alongside the GT at the same timesteps for side-by-side.
Open in SuperSplat to judge whether the TABLE stays static (no sink) + the CUBE moves.
Usage: python code/scripts/export_pred_3dgs_ply.py CKPT [clip.pt]"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
from igsw.gaussians import GaussianSet  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from scripts.eval_sim_generalization import sample_controls, _to_dev  # noqa: E402

C0 = 0.28209479177387814  # SH band-0


def write_3dgs_ply(path, means, scales, quats, opac, colors):
    means = means.detach().cpu().float().numpy()
    scales = scales.detach().cpu().float().numpy()
    if scales.ndim == 1:
        scales = np.repeat(scales[:, None], 3, 1)
    quats = quats.detach().cpu().float().numpy()
    opac = opac.detach().cpu().float().numpy().reshape(-1)
    colors = colors.detach().cpu().float().clamp(0, 1).numpy()
    n = means.shape[0]
    f_dc = (colors - 0.5) / C0
    log_scale = np.log(np.clip(scales, 1e-8, None))
    o = np.clip(opac, 1e-6, 1 - 1e-6)
    opac_logit = np.log(o / (1 - o))
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n +
           "".join("property float %s\n" % p for p in props) + "end_header\n")
    d = np.zeros((n, 17), dtype=np.float32)
    d[:, 0:3] = means; d[:, 6:9] = f_dc; d[:, 9] = opac_logit
    d[:, 10:13] = log_scale; d[:, 13:17] = q
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(d.tobytes())
    print(f"wrote {path}  ({n} gaussians)", flush=True)

dev = "cuda"
CKPT = sys.argv[1]
CLIP = sys.argv[2] if len(sys.argv) > 2 else "data/maniskill_fused/pickcube_s1000_heldseed.pt"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
model = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                             cond_mode=ck.get("cond_mode", "aggregator"),
                             spatial_ground=bool(ck.get("spatial_ground", 1)),
                             dyn_gate=bool(ck.get("dyn_gate", 0)), sem_dim=ck.get("sem_dim", 0),
                             gate_uses_sem=bool(ck.get("gate_uses_sem", 1))).to(dev).eval()  # §44h
miss, unexp = model.load_state_dict(ck["model"], strict=False)
print(f"ckpt={CKPT} step={ck.get('step')} dyn_gate={ck.get('dyn_gate')} sem_dim={ck.get('sem_dim')} "
      f"gate_uses_sem={ck.get('gate_uses_sem', 1)} | load miss={len(miss)} unexp={len(unexp)}", flush=True)

c = torch.load(CLIP, map_location=dev, weights_only=False)
g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
traj = c["traj"].to(dev); K = min(16, int(c["Kf"]))
H, W = int(c["H"]), int(c["W"]); uv = c["uv"].to(dev)
disp = (traj[K] - traj[0]).norm(dim=-1)
gen = torch.Generator(device=dev).manual_seed(0)
ci = sample_controls(g0.means, disp, ck.get("M", 2048), gen, 0.01); cu = uv[ci]
img0 = (c["gt_rgb"][0].to(dev).float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vi = _to_dev(model.encoder.build_inputs(c["instruction"], img0), dev)
    out = model(vi, g0, K, ctrl_idx=ci, control_uv=cu, control_uv_hw=(H, W))

os.makedirs("outputs/ply_pred", exist_ok=True)
base = os.path.basename(CLIP).replace(".pt", "") + f"_s{ck.get('step')}"
A = (g0.scales, g0.quats, g0.opacities, g0.colors)  # frozen appearance (isolate motion)
write_3dgs_ply(f"outputs/ply_pred/{base}_t00.ply", g0.means, *A)
for t in [8, K]:
    write_3dgs_ply(f"outputs/ply_pred/{base}_PRED_t{t:02d}.ply", out["means"][t - 1].float(), *A)
    write_3dgs_ply(f"outputs/ply_pred/{base}_GT_t{t:02d}.ply", traj[t], *A)
print(f"done -> outputs/ply_pred/ (PRED vs GT at t=8,{K})", flush=True)
