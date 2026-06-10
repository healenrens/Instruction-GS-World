"""§52 LANGUAGE-SWAP test — the load-bearing claim check. Same scene/g0, two instructions:
(a) the TRUE one (names the data's mover), (b) a SWAPPED one naming a DIFFERENT object in the scene.
If the model is truly language-conditioned, the moved entity must FOLLOW the instruction; if it
ignores language, it moves the same entity both times (visual prior: object near the gripper).
Controls are sampled PER-ENTITY (uniform, no GT-mover bias) so every candidate object has controls.
Usage: _libero_langswap.py <ckpt> <clip> "<swap instruction>"
"""
import sys; sys.path.insert(0, "code")
import torch
from igsw.gaussians import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import _to_dev

ckpt_p, clip_p, swap_instr = sys.argv[1], sys.argv[2], sys.argv[3]
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
seg = c["seg_per_g"].cuda().long()
gt_disp = (tr[K] - tr[0]).norm(dim=-1)

# per-entity uniform control sampling (NO GT-mover bias): up to 256 controls per entity id
gen = torch.Generator(device="cuda").manual_seed(0)
parts = []
for e in torch.unique(seg).tolist():
    ii = (seg == e).nonzero(as_tuple=True)[0]
    take = min(256, ii.numel())
    sel = ii[torch.randperm(ii.numel(), device="cuda", generator=gen)[:take]]
    parts.append(sel)
ci = torch.cat(parts)
if ci.numel() > ck.get("M", 2048):
    ci = ci[torch.randperm(ci.numel(), device="cuda", generator=gen)[:ck.get("M", 2048)]]
cu = uv[ci]
seg_c = seg[ci]
img0 = (c["gt_rgb"][0].cuda().float()).clamp(0, 255).to(torch.uint8).cpu().numpy()
seg_g = seg if (ck.get("entity_lbs", 0) or ck.get("gate_entity_pool", 0)) else None


def run(instr):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(mdl.encoder.build_inputs(instr, img0), "cuda")
        out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=cu, control_uv_hw=(int(c["H"]), int(c["W"])),
                  seg_per_g=seg_g)
    pd = (out["ctrl"][K - 1].float() - g0.means[ci]).norm(dim=-1)        # control endpoint disp
    gate = torch.sigmoid(out["p_dyn"].float()) if "p_dyn" in out else torch.ones_like(pd)
    return pd, gate


true_instr = c["instruction"]
pd_t, gate_t = run(true_instr)
pd_s, gate_s = run(swap_instr)
print(f"clip={clip_p}")
print(f"TRUE: {true_instr!r}")
print(f"SWAP: {swap_instr!r}")
print(f"{'ent':>4} {'n':>4} {'GTdisp':>7} | {'pred(true)':>10} {'gate(true)':>10} | {'pred(swap)':>10} {'gate(swap)':>10}")
for e in torch.unique(seg_c).tolist():
    m = seg_c == e
    print(f"{e:>4} {int(m.sum()):>4} {float(gt_disp[ci][m].mean())*100:6.1f}cm |"
          f" {float(pd_t[m].mean())*100:8.1f}cm {float(gate_t[m].mean()):>10.3f} |"
          f" {float(pd_s[m].mean())*100:8.1f}cm {float(gate_s[m].mean()):>10.3f}")
# verdict: ratio of swap-vs-true motion on the TRUE mover entity (1.0 = language ignored)
obj_ids = [e for e in torch.unique(seg_c).tolist() if e not in (0, 2, 8, 10) and e < 50]
if obj_ids:
    mv_e = max(obj_ids, key=lambda e: float(gt_disp[ci][seg_c == e].mean()))
    m = seg_c == mv_e
    r = float(pd_s[m].mean()) / max(float(pd_t[m].mean()), 1e-6)
    print(f"[VERDICT] true-mover entity {mv_e}: swap/true motion ratio = {r:.2f} "
          f"(1.0 => instruction IGNORED; ~0 => language actually selects the object)")
