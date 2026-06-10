"""Test the fix: rigid-project the per-control predicted motion (per-entity Kabsch SE(3)) -> does it
kill the scatter while KEEPING the (direction-correct) translation? Renders GT | raw pred | rigid pred,
and prints extent-ratio + direction before/after."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.eval_langswap import build_model, uniform_controls, run, INSTR, NOUNS

CKPT = "checkpoints/libero_v9lang/ckpt_last.pt"; DATA = "data/libero_pi3_v2"


def proj(P, K, vm):
    Pc = (vm[:3, :3] @ P.T + vm[:3, 3:4]).T; z = np.clip(Pc[:, 2], 1e-4, None)
    return np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], 1)


def kabsch(P, Q):                                  # best rigid R,t : R@P+t ~ Q
    Pm, Qm = P.mean(0), Q.mean(0); Pc, Qc = P - Pm, Q - Qm
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ P.T).T + (Qm - R @ Pm)


def ext(X):
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))


ck = torch.load(CKPT, map_location="cpu", weights_only=False); mdl = build_model(ck)
clips = sorted(glob.glob(f"{DATA}/*_c_heldseed.pt"))[:4]
fig, axes = plt.subplots(len(clips), 3, figsize=(9, 3.15 * len(clips)))
print(f"{'clip':<24} {'raw ext-ratio':<14} {'rigid ext-ratio':<16} {'raw dir':<9} {'rigid dir'}")
for r, cp in enumerate(clips):
    c = torch.load(cp, map_location="cuda", weights_only=False)
    seg = c["seg_per_g"].cuda().long(); N = len(seg); nkeep = N - int(c.get("n_fill", 0))
    K = int(c["Kf"]); tr = c["traj"].cuda().float()
    Ki = c["K_intr"].cpu().numpy(); vm = c["viewmat"].cpu().numpy()
    Hh, Ww = int(c["H"]), int(c["W"]); rgb0 = c["gt_rgb"][0].cpu().numpy()
    ci = uniform_controls(seg, nkeep, ck.get("M", 2048)); seg_c = seg[ci]
    gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
    obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
    mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean())); mv = (seg_c == mv_e).cpu().numpy()
    instr = c["instruction"]
    gt0 = tr[0][ci].cpu().numpy()[mv]; gtK = tr[K][ci].cpu().numpy()[mv]
    epT, init = run(mdl, c, ci, seg, instr, K); init = init.cpu().numpy()[mv]; epT = epT.cpu().numpy()[mv]
    rig = kabsch(init, epT)                          # rigid-project the predicted field
    draw = [(gt0, gtK, "GT object motion"),
            (init, epT, f"PRED raw  ext x{ext(epT)/ext(init):.2f}"),
            (init, rig, f"PRED rigid-fit  ext x{ext(rig)/ext(init):.2f}")]
    for cc, (p0, pK, t) in enumerate(draw):
        ax = axes[r, cc]; ax.imshow((rgb0.astype(np.float32) * 0.4).astype(np.uint8))
        a = proj(p0, Ki, vm); b = proj(pK, Ki, vm)
        ax.scatter(a[:, 0], a[:, 1], s=4, c="red", alpha=.6, linewidths=0)
        ax.scatter(b[:, 0], b[:, 1], s=4, c="cyan", alpha=.6, linewidths=0)
        ax.set_xlim(0, Ww); ax.set_ylim(Hh, 0); ax.axis("off"); ax.set_title(t, fontsize=8)
    gd = gtK.mean(0) - gt0.mean(0)
    rawd = epT.mean(0) - init.mean(0); rigd = rig.mean(0) - init.mean(0)
    dc = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"{os.path.basename(cp):<24} {ext(epT)/ext(init):<14.2f} {ext(rig)/ext(init):<16.2f} "
          f"{dc(rawd,gd):<+9.2f} {dc(rigd,gd):+.2f}")
plt.tight_layout(); plt.savefig("viz/ov_data/v9_rigidfix.png", dpi=115, bbox_inches="tight")
print("saved viz/ov_data/v9_rigidfix.png")
