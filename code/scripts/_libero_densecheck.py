"""Pinpoint the dense endpoint error per seg entity (control-level was fine; dense via LBS is off).
Usage: _libero_densecheck.py <ckpt> <clip>"""
import sys; sys.path.insert(0, "code")
import torch
from igsw.gaussians import GaussianSet
from igsw.model_full import InstructGSWorldModel
from igsw.dynamics.model import DynamicsConfig
from scripts.eval_sim_generalization import _to_dev
from scripts.eval_langswap import build_model, uniform_controls

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = torch.load(sys.argv[2], map_location="cuda", weights_only=False)
mdl = build_model(ck)
g0 = GaussianSet(c["means"].cuda(), c["quats"].cuda(), c["scales"].cuda(), c["opacities"].cuda(), c["colors"].cuda(), None)
seg = c["seg_per_g"].cuda().long(); K = int(c["Kf"]); N = len(seg); nf = int(c.get("n_fill", 0))
tr = c["traj"].cuda().float()
ci = uniform_controls(seg, N - nf, ck.get("M", 2048))
img0 = c["gt_rgb"][0].cpu().numpy()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    vi = _to_dev(mdl.encoder.build_inputs(c["instruction"], img0), "cuda")
    out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=c["uv"].cuda()[ci], control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg)
pred = out["means"][K - 1].float()                       # dense [N,3] last frame
pe = (pred - tr[K]).norm(dim=-1)                          # dense endpoint err
gtd = (tr[K] - tr[0]).norm(dim=-1)                        # GT dense disp
fill = torch.zeros(N, dtype=torch.bool, device="cuda"); fill[N - nf:] = True
print(f"dense N={N} (fill={nf})  control endpoints ok? ctrl err med "
      f"{float((out['ctrl'][K-1].float() - tr[K][ci]).norm(dim=-1).median())*100:.1f}cm")
print(f"{'entity':>8} {'N':>7} {'inCtrl':>6} {'GTdisp':>7} {'PREDerr':>8}")
for e in torch.unique(seg).tolist():
    m = (seg == e) & ~fill
    if int(m.sum()) < 10:
        continue
    in_ctrl = int((seg[ci] == e).sum())
    print(f"{e:>8} {int(m.sum()):>7} {in_ctrl:>6} {float(gtd[m].median())*100:>6.1f}cm {float(pe[m].median())*100:>7.1f}cm")
mf = (seg == 1) & ~fill                                   # the manipulated object
print(f"\nOBJECT id1 dense: GTdisp {float(gtd[mf].median())*100:.1f}cm  PREDerr {float(pe[mf].median())*100:.1f}cm "
      f"PREDdisp {float((pred[mf]-tr[0][mf]).norm(dim=-1).median())*100:.1f}cm")
