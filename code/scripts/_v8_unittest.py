"""§54 unit test: rel_head zero-init must preserve v7 behavior EXACTLY (warm-start intact),
and resume must leave only rel_* missing. Usage: _v8_unittest.py [ckpt] [clip]"""
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
print(f"clip keys: is_obj={'is_obj' in c}  n_fill={c.get('n_fill', 0)}")


def build(rel):
    m = InstructGSWorldModel(cfg, n_control=2048, n_query=16, cond_mode="aggregator", spatial_ground=True,
                             dyn_gate=True, sem_dim=ck.get("sem_dim", 0), gate_uses_sem=bool(ck.get("gate_uses_sem", 1)),
                             gate_entity_pool=bool(ck.get("gate_entity_pool", 0)), entity_lbs=bool(ck.get("entity_lbs", 0)),
                             rel_head=rel, entity_head=False).cuda().eval()
    miss = m.load_state_dict(ck["model"], strict=False)
    return m, miss


g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
tr = c["traj"].cuda(); K = 4; uv = c["uv"].cuda(); seg = c["seg_per_g"].cuda().long()
disp = (tr[int(c["Kf"])] - tr[0]).norm(dim=-1)
gen = torch.Generator(device="cuda").manual_seed(0)
ci = sample_controls(g0.means, disp, 2048, gen, 0.01)
img0 = c["gt_rgb"][0].cpu().numpy()
wrong = {k: v for k, v in c.items()}  # build a wrong instruction input to exercise the CF branch


def run(m, with_wrong=False):
    vw = None
    if with_wrong:
        vw = _to_dev(m.encoder.build_inputs("pick up the ketchup and place it in the basket", img0), "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(m.encoder.build_inputs(c["instruction"], img0), "cuda")
        return m(vi, g0, K, ctrl_idx=ci, vlm_inputs_wrong=vw, control_uv=uv[ci],
                 control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg)


m0, _ = build(0); out0 = run(m0)
m1, miss = build(1); out1 = run(m1, with_wrong=True)
nonrel_missing = [k for k in miss.missing_keys if not k.startswith(("rel_",))]
print(f"resume rel_head=1: {len(miss.missing_keys)} missing, of which NON-rel: {nonrel_missing[:6]} (n={len(nonrel_missing)})")
d = float((out0["means"][-1] - out1["means"][-1]).abs().max())
print(f"max |means(rel0) - means(rel1)| = {d:.2e}  (expect <1e-2: zero-init identity preserved)")
print(f"out1 has: p_rel={'p_rel' in out1} objmask={'objmask' in out1} p_dyn_wrong={'p_dyn_wrong' in out1} p_rel_wrong={'p_rel_wrong' in out1}")
assert len(nonrel_missing) == 0, f"resume drops non-rel weights! {nonrel_missing[:6]}"
assert d < 1e-2, f"rel_head=1 NOT identity to rel_head=0 ({d}) — warm-start broken!"
assert all(k in out1 for k in ("p_rel", "objmask", "p_dyn_wrong", "p_rel_wrong")), "missing v8 outputs"
print("UNIT TEST PASS ✓  rel_head zero-init = exact v7 behavior; CF branch live; resume clean")
