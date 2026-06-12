"""§70 R0 — 3D-FIRST evaluation suite (replaces the norm-only metrics that were blind to magnitude
collapse). For each clip, run the model under the TRUE instruction and compare the predicted per-control
3D trajectory to the analytic GT clip["traj"] [Kf+1,N,3]. All metrics are 3D; render/2D are NOT used.

Per mover entity (the object-class entity with the largest GT displacement):
  EPE3D          ‖pred(t)-gt(t)‖ median over the mover's controls — per-step curve + endpoint (cm)
  Acc3DS/Acc3DR  fraction of mover controls with endpoint EPE3D ≤ 5cm|5%  /  ≤ 10cm|10%  (Gojcic flow std)
  mag-ratio      ‖pred centroid disp‖ / ‖gt centroid disp‖  — the MAGNITUDE-COLLAPSE detector (median + P10)
  5deg5cm        entity Kabsch SE(3): rot err ≤5° ∧ trans err ≤5cm  — FIRST rotation measurement
  rot-err        geodesic angle(R_pred, R_gt) (deg)
  coherence      extent-ratio + rigid-residual (reused from eval_langswap)

Aggregate prints MEDIAN + P10 (not mean — magnitude collapse hides in the tail) and a per-clip table.

  python code/scripts/eval_3d.py --ckpt checkpoints/libero_v9lang_rigid/ckpt_last.pt --data data/libero_pi3_v2 --split heldseed
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from igsw.gaussians import GaussianSet                                   # noqa: E402
from scripts.eval_langswap import build_model, uniform_controls, _ext, _rigid_res, _median  # noqa: E402
from scripts.eval_sim_generalization import _to_dev                       # noqa: E402


def _p10(xs):
    s = sorted(xs)
    return 0.0 if not s else s[max(0, int(0.10 * (len(s) - 1)))]


def kabsch_Rt(P, Q):
    """Best rigid R,t mapping P->Q (both [M,3]). Returns (R[3,3], t[3])."""
    Pm, Qm = P.mean(0), Q.mean(0)
    U, S, Vt = torch.linalg.svd((P - Pm).T @ (Q - Qm))
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.eye(3, device=P.device, dtype=P.dtype)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    return R, Qm - R @ Pm


def rot_angle_deg(R1, R2):
    c = ((R1.T @ R2).diagonal().sum() - 1.0) * 0.5
    return float(torch.arccos(c.clamp(-1.0, 1.0)) * 180.0 / 3.14159265)


@torch.no_grad()
def run_full(mdl, c, ci, seg_g, instr, K):
    """Predicted control trajectory out["ctrl"] [K,M,3] (positions at steps 1..K) under `instr`."""
    g0 = GaussianSet(c["means"].cuda(), c["quats"].cuda(), c["scales"].cuda(),
                     c["opacities"].cuda(), c["colors"].cuda(), None)
    img0 = c["gt_rgb"][0].cpu().numpy()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(mdl.encoder.build_inputs(instr, img0), "cuda")
        out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=c["uv"].cuda()[ci],
                  control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg_g)
    return out["ctrl"].float()                                            # [K,M,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/libero_pi3_v2")
    ap.add_argument("--split", default="heldseed")
    ap.add_argument("--force_rigid_agg", type=int, default=0,
                    help="§66: force rigid projection ON (eval a base ckpt in the production config)")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mdl = build_model(ck, force_rigid_agg=bool(args.force_rigid_agg))
    clips = sorted(glob.glob(os.path.join(args.data, f"*_{args.split}.pt")))
    print(f"ckpt={args.ckpt} rigid_agg={ck.get('rigid_agg', 0)} | {len(clips)} {args.split} clips\n"
          f"{'clip':<26} {'GTdisp':>7} {'EPE3D':>7} {'magR':>6} {'rotErr':>7} {'5°5cm':>6} "
          f"{'Acc3DS':>7} {'Acc3DR':>7} {'rigRes':>7}")
    mags, epes, rots, accs, accr, ok5s, rigs, gtrots = [], [], [], [], [], [], [], []
    for cp in clips:
        c = torch.load(cp, map_location="cuda", weights_only=False)
        seg = c["seg_per_g"].cuda().long()
        N = len(seg)
        nkeep = N - int(c.get("n_fill", 0))
        K = int(c["Kf"])
        tr = c["traj"].cuda().float()                                     # [K+1,N,3]
        ci = uniform_controls(seg, nkeep, ck.get("M", 2048))
        seg_c = seg[ci]
        init = tr[0][ci]                                                  # [M,3]
        gt = tr[1:K + 1][:, ci]                                           # [K,M,3]
        gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
        obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
        if not obj_es:
            continue
        mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
        mv = seg_c == mv_e
        pred = run_full(mdl, c, ci, seg, c["instruction"], K)             # [K,M,3]
        pe, ge, ii = pred[:, mv], gt[:, mv], init[mv]                     # mover subset
        # EPE3D (per-step median, endpoint)
        epe_steps = [(pe[t] - ge[t]).norm(dim=-1).median().item() for t in range(K)]
        epe_end = epe_steps[-1]
        # magnitude ratio (entity centroid displacement) — the collapse detector
        pd = pe[K - 1].mean(0) - ii.mean(0)
        gd = ge[K - 1].mean(0) - ii.mean(0)
        magR = float(pd.norm() / gd.norm().clamp_min(1e-6))
        # 5deg5cm (entity rigid SE(3)) — first rotation measurement
        Rp, _ = kabsch_Rt(ii, pe[K - 1])
        Rg, _ = kabsch_Rt(ii, ge[K - 1])
        rot_err = rot_angle_deg(Rp, Rg)
        I3 = torch.eye(3, device=Rg.device, dtype=Rg.dtype)
        gt_rot = rot_angle_deg(Rg, I3)                                # GT's OWN rotation magnitude (is it real?)
        gtrots.append(gt_rot)
        trans_err = float((pd - gd).norm())
        ok5 = (rot_err <= 5.0) and (trans_err <= 0.05)
        # Acc3DS/Acc3DR (mover controls, endpoint)
        e_end = (pe[K - 1] - ge[K - 1]).norm(dim=-1)
        gmag = (ge[K - 1] - ii).norm(dim=-1)
        a3ds = float(((e_end <= 0.05) | (e_end <= 0.05 * gmag)).float().mean())
        a3dr = float(((e_end <= 0.10) | (e_end <= 0.10 * gmag)).float().mean())
        rig_res = _rigid_res(ii, pe[K - 1])
        gtd = float(gt_disp[mv].mean())
        mags.append(magR); epes.append(epe_end); rots.append(rot_err)
        accs.append(a3ds); accr.append(a3dr); ok5s.append(1.0 if ok5 else 0.0); rigs.append(rig_res)
        print(f"{os.path.basename(cp):<26} {gtd*100:6.1f}c {epe_end*100:6.1f}c {magR:5.2f}x "
              f"{rot_err:6.1f}° {'Y' if ok5 else 'n':>5} {a3ds:7.2f} {a3dr:7.2f} {rig_res*100:6.2f}c")
    n = max(1, len(mags))
    print(f"\n=== 3D SUMMARY ({n} clips, {args.split}) — magnitude looks at MEDIAN + P10 (tail) ===")
    print(f"  mag-ratio    median {_median(mags):.2f}x   P10 {_p10(mags):.2f}x   "
          f"(collapse = median<<1 or P10<0.5; target [0.85,1.15] & P10>=0.5)")
    print(f"  EPE3D        median {_median(epes)*100:.1f}cm   P90 {sorted(epes)[int(0.9*(n-1))]*100:.1f}cm")
    print(f"  5°5cm rate   {sum(ok5s)/n:.2f}    rot-err median {_median(rots):.1f}°    "
          f"GT-rot median {_median(gtrots):.1f}° (is GT actually rotating? if ~0 the pred rot is spurious)")
    print(f"  Acc3DS(5cm)  {sum(accs)/n:.2f}    Acc3DR(10cm) {sum(accr)/n:.2f}")
    print(f"  coherence    rigid-residual median {_median(rigs)*100:.2f}cm")


if __name__ == "__main__":
    main()
