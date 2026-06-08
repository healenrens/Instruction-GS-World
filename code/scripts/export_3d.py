"""Export for inspection: 3D point-cloud .ply (g0 / GT-final / pred-final) + a CLEAN
(no background-composite) rollout video + a LONGER free rollout (beyond training horizon).
Usage: python code/scripts/export_3d.py [clip.pt] [ckpt.pt] [N_long]"""
import torch, numpy as np, sys, os
sys.path.insert(0, "code")
from igsw.gaussians import GaussianSet, render_gaussianset
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import sample_controls, _to_dev
import imageio.v2 as iio

dev = "cuda"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/maniskill/pickcube_s1000_heldseed.pt"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/sim_gen/ckpt_last.pt"
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 40
tag = os.path.basename(CLIP).replace(".pt", "")
clip = torch.load(CLIP, map_location=dev, weights_only=False)
g0 = GaussianSet(clip["means"], clip["quats"], clip["scales"], clip["opacities"], clip["colors"], None)
traj = clip["traj"].to(dev); Kf = int(clip["Kf"]); K = min(16, Kf); H, W = int(clip["H"]), int(clip["W"])
Ki = clip["K_intr"].to(dev).float(); vm = clip["viewmat"].to(dev).float()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
model = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                             cond_mode="aggregator", spatial_ground=True).to(dev).eval()
miss, unexp = model.load_state_dict(ck["model"], strict=False)
gen = torch.Generator(device=dev).manual_seed(0)
disp = (traj[K] - traj[0]).norm(dim=-1)
ci = sample_controls(g0.means, disp, ck.get("M", 2048), gen, 0.01)
cu = clip["uv"].to(dev)[ci]
img0 = (clip["gt_rgb"][0].to(dev).float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
print(f"clip={tag} N={len(g0)} traj={tuple(traj.shape)} K={K} NL={NL} step={ck.get('step')} instr={clip['instruction'][:50]!r}", flush=True)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vi = _to_dev(model.encoder.build_inputs(clip["instruction"], img0), dev)
    out = model(vi, g0, NL, ctrl_idx=ci, control_uv=cu, control_uv_hw=(H, W))

os.makedirs("outputs/ply", exist_ok=True); os.makedirs("outputs/clean", exist_ok=True)

def wply(p, xyz, rgb):
    xyz = xyz.detach().cpu().numpy().astype(np.float32); rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
           "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(xyz)).encode()
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(p, "wb") as f:
        f.write(hdr); f.write(rec.tobytes())

wply(f"outputs/ply/{tag}_g0.ply", g0.means, g0.colors)
wply(f"outputs/ply/{tag}_gt_final.ply", traj[K], g0.colors)
wply(f"outputs/ply/{tag}_pred_final.ply", out["means"][K - 1].float(), g0.colors)
print("PLY written: g0 / gt_final / pred_final", flush=True)

def rend(g):  # render on BLACK (no gt_rgb composite -> no double-image ghost)
    c, _, _ = render_gaussianset(g, vm[None], Ki[None], W, H)
    return c[0].clamp(0, 1).cpu().numpy()

def gmove(means_t):
    return GaussianSet(means_t.float(), g0.quats, g0.scales, g0.opacities, g0.colors, None)

# clean GT | static | pred over the trained K horizon
w = iio.get_writer(f"outputs/clean/{tag}_clean16.mp4", fps=6)
for t in range(K + 1):
    gt_t = gmove(traj[t])
    pr_t = gmove(out["means"][t - 1])  # CLEAN: pred POSITIONS + g0 appearance (isolate motion)
    row = np.concatenate([rend(gt_t), rend(g0), rend(pr_t)], axis=1)
    w.append_data((row * 255).astype(np.uint8))
w.close()
# longer free rollout (pred only) — beyond the K=16 training horizon
wl = iio.get_writer(f"outputs/clean/{tag}_long{NL}.mp4", fps=6)
for t in range(NL):
    pr = gmove(out["means"][t])  # CLEAN: pred POSITIONS + g0 appearance
    wl.append_data((rend(pr) * 255).astype(np.uint8))
wl.close()
print("VIDEOS written: clean16 + long%d" % NL, flush=True)
