"""Control-level failure analysis for epi440 (model barely moved the object, err 34cm ~= GT 40cm).
Questions: (a) is the GATE closed on the movers? (b) is the velocity head under-firing? (c) does the
mid-rollout diverge? Usage: _libero_440debug.py <ckpt> [clip]"""
import sys; sys.path.insert(0, "code")
import torch
from igsw.gaussians import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import sample_controls, _to_dev

ckpt_p = sys.argv[1]
clip_p = sys.argv[2] if len(sys.argv) > 2 else "data/libero_video/epi000440_heldtask.pt"
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

gt_d = disp[ci]; mv = gt_d > 0.01
pred_ctrl = out["ctrl"].float()                 # [K,M,3]
init = g0.means[ci]
pd = (pred_ctrl[K - 1] - init).norm(dim=-1)
print(f"clip={clip_p}")
print(f"controls: M={ci.numel()} movers={int(mv.sum())}  GT mover disp med {float(gt_d[mv].median()):.3f}m")
print(f"PRED mover endpoint disp: med {float(pd[mv].median()):.3f} p10 {float(pd[mv].quantile(0.1)):.3f} "
      f"p90 {float(pd[mv].quantile(0.9)):.3f}   (ratio med {float((pd[mv]/gt_d[mv]).median()):.2f})")
print(f"PRED static endpoint disp: med {float(pd[~mv].median()):.4f}")
if "p_dyn" in out:
    gate = torch.sigmoid(out["p_dyn"].float())
    print(f"GATE on movers: med {float(gate[mv].median()):.3f} p10 {float(gate[mv].quantile(0.1)):.3f} "
          f"frac<0.5 {float((gate[mv] < 0.5).float().mean()):.2f}")
    print(f"GATE on statics: med {float(gate[~mv].median()):.3f} frac>0.5 {float((gate[~mv] > 0.5).float().mean()):.2f}")
# per-step mover speed: is the motion just slow (uniform undershoot) or does it stall mid-rollout?
sp = (pred_ctrl[1:] - pred_ctrl[:-1]).norm(dim=-1)[:, mv].mean(1)   # [K-1]
gt_sp = (tr[2:, ci][:, mv] - tr[1:-1, ci][:, mv]).norm(dim=-1).mean(1)
print("per-step mover speed PRED:", " ".join(f"{float(s)*100:.1f}" for s in sp), "cm")
print("per-step mover speed   GT:", " ".join(f"{float(s)*100:.1f}" for s in gt_sp), "cm")
# direction agreement at endpoint
dirp = (pred_ctrl[K - 1] - init)[mv]; dirg = (tr[K, ci] - init)[mv]
cos = torch.nn.functional.cosine_similarity(dirp, dirg, dim=-1)
print(f"endpoint direction cos: med {float(cos.median()):.2f} frac>0.5 {float((cos > 0.5).float().mean()):.2f}")
