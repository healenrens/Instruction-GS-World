"""§71 R1 diagnosis: is the magnitude collapse the GATE closing (sigmoid(p_dyn)->0) or the HEAD
under-predicting (raw velocity small even with gate open)? Dump, per mover entity, on collapse clips
vs healthy clips, BOTH under the TRUE instruction and a WRONG one:
  gate = sigmoid(p_dyn).mean   (the multiplier on v; <1 shrinks motion)
  rel  = sigmoid(p_rel).mean   (relevance; added to gate logit for objects)
  predMag / gtMag              (the collapse itself)
If collapse clips show gate~0.1 -> gate mis-calibration. If gate~1 but predMag still 0.1x -> head under-predicts.
Also: if WRONG-instruction gate ~= TRUE-instruction gate on the mover -> relevance not discriminating (the
swap/true 0.3-0.44 signature) -> recalibrate relevance.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from igsw.gaussians import GaussianSet                                   # noqa: E402
from scripts.eval_langswap import build_model, uniform_controls, INSTR, NOUNS  # noqa: E402
from scripts.eval_sim_generalization import _to_dev                       # noqa: E402

CKPT = "checkpoints/libero_v9lang_rigid/ckpt_last.pt"
DATA = "data/libero_pi3_v2"
COLLAPSE = ["epi000100_c_train", "epi000130_c_train", "epi000160_c_train", "epi000110_c_train"]
HEALTHY = ["epi000330_c_heldseed", "epi000240_c_heldseed", "epi000000_c_train", "epi000030_c_train"]


@torch.no_grad()
def run_diag(mdl, c, ci, seg, instr, K):
    g0 = GaussianSet(c["means"].cuda(), c["quats"].cuda(), c["scales"].cuda(),
                     c["opacities"].cuda(), c["colors"].cuda(), None)
    img0 = c["gt_rgb"][0].cpu().numpy()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(mdl.encoder.build_inputs(instr, img0), "cuda")
        out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=c["uv"].cuda()[ci],
                  control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg)
    return out


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
mdl = build_model(ck)
print(f"{'clip':<26} {'GTmag':>6} {'predMag':>8} {'ratio':>6} {'gate(T)':>8} {'rel(T)':>7} {'gate(W)':>8} {'kind'}")
for kind, lst in [("COLLAPSE", COLLAPSE), ("HEALTHY", HEALTHY)]:
    for name in lst:
        c = torch.load(f"{DATA}/{name}.pt", map_location="cuda", weights_only=False)
        seg = c["seg_per_g"].cuda().long()
        N = len(seg)
        nkeep = N - int(c.get("n_fill", 0))
        K = int(c["Kf"])
        tr = c["traj"].cuda().float()
        ci = uniform_controls(seg, nkeep, ck.get("M", 2048))
        seg_c = seg[ci]
        init = tr[0][ci]
        gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
        obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
        mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
        mv = (seg_c == mv_e)
        instr = c["instruction"]
        wrong = INSTR(next(n for n in NOUNS if n not in instr))
        out_t = run_diag(mdl, c, ci, seg, instr, K)
        out_w = run_diag(mdl, c, ci, seg, wrong, K)
        gate_t = torch.sigmoid(out_t["p_dyn"][mv].float()).mean().item() if "p_dyn" in out_t else float("nan")
        rel_t = torch.sigmoid(out_t["p_rel"][mv].float()).mean().item() if "p_rel" in out_t else float("nan")
        gate_w = torch.sigmoid(out_w["p_dyn"][mv].float()).mean().item() if "p_dyn" in out_w else float("nan")
        pred_disp = (out_t["ctrl"][K - 1] - init).norm(dim=-1)[mv].mean().item()
        gtm = gt_disp[mv].mean().item()
        print(f"{name:<26} {gtm*100:5.1f}c {pred_disp*100:7.1f}c {pred_disp/max(gtm,1e-6):5.2f}x "
              f"{gate_t:8.3f} {rel_t:7.3f} {gate_w:8.3f} {kind}")
