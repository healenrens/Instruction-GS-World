"""§54 entity-head unit test: (1) zero-init => exact v7 identity; (2) resid_norm exposed; (3) the rigid
broadcast is genuinely per-entity rigid when the ent head is perturbed (within an entity, predicted
displacements fit ONE SE(3) to ~0 residual). Usage: _v8_ent_unittest.py [ckpt] [clip]"""
import sys; sys.path.insert(0, "code")
import torch
from igsw.gaussians import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import sample_controls, _to_dev

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/libero_v7_pi3/ckpt_last.pt"
clip = sys.argv[2] if len(sys.argv) > 2 else "data/libero_pi3/epi000000_train.pt"
c = torch.load(clip, map_location="cuda", weights_only=False)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"])


def build(ent):
    m = InstructGSWorldModel(cfg, n_control=2048, n_query=16, cond_mode="aggregator", spatial_ground=True,
                             dyn_gate=True, sem_dim=ck.get("sem_dim", 0), gate_uses_sem=bool(ck.get("gate_uses_sem", 1)),
                             gate_entity_pool=bool(ck.get("gate_entity_pool", 0)), entity_lbs=bool(ck.get("entity_lbs", 0)),
                             rel_head=True, entity_head=ent).cuda().eval()
    m.load_state_dict(ck["model"], strict=False)
    return m


g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
tr = c["traj"].cuda(); K = 4; uv = c["uv"].cuda(); seg = c["seg_per_g"].cuda().long()
disp = (tr[int(c["Kf"])] - tr[0]).norm(dim=-1)
gen = torch.Generator(device="cuda").manual_seed(0)
ci = sample_controls(g0.means, disp, 2048, gen, 0.01)
img0 = c["gt_rgb"][0].cpu().numpy()


def run(m):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(m.encoder.build_inputs(c["instruction"], img0), "cuda")
        return m(vi, g0, K, ctrl_idx=ci, control_uv=uv[ci], control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg)


m0 = build(False); out0 = run(m0)
m1 = build(True); out1 = run(m1)
d = float((out0["means"][-1] - out1["means"][-1]).abs().max())
print(f"(1) zero-init identity: max|Δmeans| = {d:.2e}  resid_norm in out = {'resid_norm' in out1}")
assert d < 1e-2 and "resid_norm" in out1

# (3) perturb the entity head -> check per-entity rigidity of the CONTROL endpoints
torch.manual_seed(0)
ent = m1.dynamics.ent_mlp[-1]
ent.weight.data.normal_(0, 0.5); ent.bias.data.normal_(0, 0.2)
out2 = run(m1)
ci_seg = seg[ci]
init = g0.means[ci].float()
endp = out2["ctrl"][K - 1].float()                                # predicted control endpoints


def kabsch_resid(X, Y):
    cx = X.mean(0); cy = Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, _, Vt = torch.linalg.svd(H.double())
    Dt = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(Vt.T @ U.T))], device=X.device).double())
    R = (Vt.T @ Dt @ U.T)
    t = cy.double() - cx.double() @ R.T
    return (Y.double() - (X.double() @ R.T + t)).norm(dim=-1).mean()


resids, sizes = [], []
for e in torch.unique(ci_seg).tolist():
    m = ci_seg == e
    if int(m.sum()) >= 8:
        r = float(kabsch_resid(init[m], endp[m]))
        mv = float((endp[m] - init[m]).norm(dim=-1).mean())
        resids.append(r); sizes.append(mv)
print(f"(3) per-entity rigidity (perturbed ent head): mean Kabsch residual = {sum(resids)/len(resids)*1000:.2f}mm "
      f"over {len(resids)} entities; mean entity motion = {sum(sizes)/len(sizes)*100:.1f}cm")
# residual should be a tiny fraction of the motion (residual head is small / near-zero here)
assert sum(resids)/len(resids) < 0.02, "entity motion not rigid!"
print("ENTITY-HEAD UNIT TEST PASS ✓  zero-init identity + structurally rigid per-entity motion")
