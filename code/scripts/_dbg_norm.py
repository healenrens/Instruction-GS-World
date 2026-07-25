"""Debug #1: verify Δqpos normalization + data processing (the too-large-dq / slow-flailing root-cause).
Checks: norm_stats values, recompute-from-data match, round-trip, ckpt normalizer buffers, z~N(0,1),
and the ACTUAL GT dq magnitude (small per-step delta? or genuinely large?). No GPU needed."""
import glob, sys, torch, numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/rt2_act"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/vla_50k_v2/vla_040000.pt"
NAMES = [f"L_arm{i}" for i in range(6)] + ["L_grip"] + [f"R_arm{i}" for i in range(6)] + ["R_grip"]
GRIP = [6, 13]
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

ns = torch.load(f"{DATA}/norm_stats.pt", map_location="cpu", weights_only=False)
print(f"=== norm_stats.pt ({ns['n_windows']} win, {ns['n_steps']} steps) ===")
print(f"{'dim':<8}{'mean':>10}{'std':>10}{'abs_max':>10}{'nz_frac':>9}")
for i, nm in enumerate(NAMES):
    print(f"{nm:<8}{ns['mean'][i]:>10.5f}{ns['std'][i]:>10.5f}{ns['abs_max'][i]:>10.4f}"
          f"{ns['nonzero_frac'][i]:>9.4f}{'  <grip' if i in GRIP else ''}")

files = sorted(glob.glob(f"{DATA}/*_train.pt"))[:400]
dq = np.concatenate([torch.load(f, map_location="cpu", weights_only=False)["dq"].numpy().astype(np.float64)
                     for f in files], 0)
print(f"\n=== recomputed from {len(files)} shards, {dq.shape[0]} steps: |dq| percentiles (rad/step) ===")
print(f"{'dim':<8}{'std':>10}{'p50|x|':>10}{'p95|x|':>10}{'p99|x|':>10}{'max':>10}")
for i, nm in enumerate(NAMES):
    a = np.abs(dq[:, i])
    print(f"{nm:<8}{dq[:,i].std():>10.5f}{np.percentile(a,50):>10.5f}{np.percentile(a,95):>10.5f}"
          f"{np.percentile(a,99):>10.5f}{a.max():>10.4f}{'  <grip' if i in GRIP else ''}")

print(f"\nstd match norm_stats vs recomputed: max|Δ| = {np.abs(ns['std'].numpy() - dq.std(0)).max():.5f}")

mean, std = ns['mean'].numpy().copy(), ns['std'].numpy().copy()
mean[GRIP] = 0.0; std[GRIP] = 1.0                                 # the ActionNormalizer override
z = (dq - mean) / std
print(f"round-trip max err: {np.abs(z * std + mean - dq).max():.2e}")
print(f"z[arm] mean={z[:,ARM].mean():+.3f} std={z[:,ARM].std():.3f} (want ~0, ~1)")
print(f"z[arm] |p95|={np.percentile(np.abs(z[:,ARM]),95):.3f} |p99|={np.percentile(np.abs(z[:,ARM]),99):.3f}")

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
km = [k for k in sd if k.endswith("act_norm.mean")]
ks = [k for k in sd if k.endswith("act_norm.std")]
if km and ks:
    cm, cs = sd[km[0]].numpy(), sd[ks[0]].numpy()
    print(f"\nckpt act_norm.mean vs norm_stats(grip->0): max|Δ|={np.abs(cm - mean).max():.6f}")
    print(f"ckpt act_norm.std  vs norm_stats(grip->1): max|Δ|={np.abs(cs - std).max():.6f}")
    print(f"  ckpt std values: {np.array2string(cs, precision=4, max_line_width=200)}")
else:
    print(f"\n!! NO act_norm buffers in ckpt. norm-ish keys: {[k for k in sd if 'norm' in k.lower()][:6]}")

p95 = np.percentile(np.abs(dq[:, ARM]), 95)
print(f"\n=== INTERP: arm GT |dq| p95 = {p95:.4f} rad/step @16.67Hz -> {p95*16.67:.3f} rad/s ===")
print("  If the model's predicted |dq| is ~5x this, that's the slow-flailing (model over-predicts, not data).")
