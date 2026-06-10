import sys; sys.path.insert(0, "code")
import torch, imageio.v2 as iio, numpy as np
from igsw.gaussians import GaussianSet, render_gaussianset
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import sample_controls, _to_dev

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/libero_v1/ckpt_0000200.pt"
c = torch.load("data/libero_video/epi000400_heldtask.pt", map_location="cuda", weights_only=False)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])
mdl = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                           cond_mode="aggregator", spatial_ground=True, dyn_gate=bool(ck.get("dyn_gate", 0)),
                           sem_dim=ck.get("sem_dim", 0), gate_uses_sem=bool(ck.get("gate_uses_sem", 1)),
                           gate_entity_pool=bool(ck.get("gate_entity_pool", 0)),
                           entity_lbs=bool(ck.get("entity_lbs", 0))).cuda().eval()
mdl.load_state_dict(ck["model"], strict=False)
g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
tr = c["traj"].cuda(); K = min(12, int(c["Kf"]))
uv = c["uv"].cuda(); disp = (tr[K] - tr[0]).norm(dim=-1)
gen = torch.Generator(device="cuda").manual_seed(0)
ci = sample_controls(g0.means, disp, ck.get("M", 2048), gen, 0.01); cu = uv[ci]
img0 = (c["gt_rgb"][0].cuda().float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
seg_g = c["seg_per_g"].cuda().long() if (ck.get("entity_lbs", 0) or ck.get("gate_entity_pool", 0)) else None
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vi = _to_dev(mdl.encoder.build_inputs(c["instruction"], img0), "cuda")
    out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=cu, control_uv_hw=(int(c["H"]), int(c["W"])),
              seg_per_g=seg_g)

# also report: are the PREDICTED dense positions scattered vs GT? per-Gaussian endpoint error
pe = (out["means"][K - 1].float() - tr[K].float()).norm(dim=-1)
print(f"pred-vs-GT endpoint err (m): med {float(pe.median()):.3f} p90 {float(pe.quantile(0.9)):.3f} max {float(pe.max()):.3f}")
print(f"GT disp max {float((tr[K]-tr[0]).norm(dim=-1).max()):.3f}  PRED disp max {float((out['means'][K-1]-g0.means).norm(dim=-1).max()):.3f}")

sc = c["scales"] * 4.0; vm = c["viewmat"][None].float(); Ki = c["K_intr"][None].float()


def R(mn):
    col, _, _ = render_gaussianset(GaussianSet(mn.float(), c["quats"], sc, c["opacities"], c["colors"], None),
                                   vm, Ki, int(c["W"]), int(c["H"]))
    return col[0].clamp(0, 1).cpu().numpy()


rows = [np.concatenate([R(tr[t]), R(out["means"][t - 1] if t > 0 else g0.means)], 1) for t in [0, 6, K]]
iio.imwrite("outputs/clean/_libero_pred_x4.png", (np.concatenate(rows, 0)[::2, ::2] * 255).astype(np.uint8))
print("saved _libero_pred_x4.png (rows t0/6/12, cols GT|PRED, x4 scales)")
