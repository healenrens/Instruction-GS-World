"""Visualize the MODEL'S ACTUAL PREDICTION (not the data) on held-out clips, in Gaussian-motion space.
Per clip: RGB | GT object motion | model pred under CORRECT instr | model pred under WRONG instr.
Language causality is visible iff col-3 object moves like GT and col-4 object stays put."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.eval_langswap import build_model, uniform_controls, run, INSTR, NOUNS

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/libero_v9lang/ckpt_last.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/libero_pi3_v2"


def proj(P, K, vm):
    Pc = (vm[:3, :3] @ P.T + vm[:3, 3:4]).T
    z = np.clip(Pc[:, 2], 1e-4, None)
    return np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], 1)


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
mdl = build_model(ck)
clips = sorted(glob.glob(f"{DATA}/*_c_heldseed.pt"))[:4]
print(f"model={CKPT} | {len(clips)} heldseed clips")

fig, axes = plt.subplots(len(clips), 4, figsize=(12, 3.15 * len(clips)))
if len(clips) == 1:
    axes = axes[None]

for r, cp in enumerate(clips):
    c = torch.load(cp, map_location="cuda", weights_only=False)
    seg = c["seg_per_g"].cuda().long(); N = len(seg); nkeep = N - int(c.get("n_fill", 0))
    K = int(c["Kf"]); tr = c["traj"].cuda().float()
    Ki = c["K_intr"].cpu().numpy(); vm = c["viewmat"].cpu().numpy()
    Hh, Ww = int(c["H"]), int(c["W"]); rgb0 = c["gt_rgb"][0].cpu().numpy()
    ci = uniform_controls(seg, nkeep, ck.get("M", 2048)); seg_c = seg[ci]
    gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
    obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
    mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
    mv = (seg_c == mv_e).cpu().numpy()
    instr = c["instruction"]; wrong = INSTR(next(n for n in NOUNS if n not in instr))
    gt0 = tr[0][ci].cpu().numpy(); gtK = tr[K][ci].cpu().numpy()
    epT, init = run(mdl, c, ci, seg, instr, K); epT = epT.cpu().numpy(); init = init.cpu().numpy()
    epW, _ = run(mdl, c, ci, seg, wrong, K); epW = epW.cpu().numpy()

    def scat(ax, p0, pK, title):
        ax.imshow((rgb0.astype(np.float32) * 0.4).astype(np.uint8))
        a = proj(p0[mv], Ki, vm); b = proj(pK[mv], Ki, vm)
        ax.scatter(a[:, 0], a[:, 1], s=4, c="red", alpha=.6, linewidths=0, label="t0")
        ax.scatter(b[:, 0], b[:, 1], s=4, c="cyan", alpha=.6, linewidths=0, label="tK")
        ax.set_xlim(0, Ww); ax.set_ylim(Hh, 0); ax.axis("off"); ax.set_title(title, fontsize=8)

    pd = epT[mv].mean(0) - init[mv].mean(0); gd = gtK[mv].mean(0) - gt0[mv].mean(0)
    dcos = float(np.dot(pd, gd) / (np.linalg.norm(pd) * np.linalg.norm(gd) + 1e-9))
    dgt = np.linalg.norm(gtK[mv] - gt0[mv], axis=1).mean() * 100
    dT = np.linalg.norm(epT[mv] - init[mv], axis=1).mean() * 100
    dW = np.linalg.norm(epW[mv] - init[mv], axis=1).mean() * 100
    axes[r, 0].imshow(rgb0); axes[r, 0].axis("off")
    axes[r, 0].set_title(f"RGB t0\n{instr[:30]}", fontsize=8)
    scat(axes[r, 1], gt0, gtK, f"GT object motion\n{dgt:.0f}cm (red t0 -> cyan tK)")
    scat(axes[r, 2], init, epT, f"PRED | correct instr\nmoves {dT:.0f}cm  dir {dcos:+.2f}")
    scat(axes[r, 3], init, epW, f"PRED | WRONG instr ('{wrong.split('the ')[1].split(' and')[0]}')\nmoves {dW:.0f}cm (should be ~0)")
    print(f"  {os.path.basename(cp)}: GT {dgt:.0f}cm | pred-correct {dT:.0f}cm dir{dcos:+.2f} | pred-wrong {dW:.0f}cm")

plt.tight_layout(); os.makedirs("viz/ov_data", exist_ok=True)
plt.savefig("viz/ov_data/v9_predictions.png", dpi=115, bbox_inches="tight")
print("saved viz/ov_data/v9_predictions.png")
