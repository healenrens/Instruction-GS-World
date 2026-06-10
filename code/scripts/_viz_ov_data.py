"""CPU-only visualization of the OPEN-VOCAB training data (no GPU — training is running).
Per clip, a row of panels: RGB(t0) | RGB(tK) | open-vocab entity seg (what we train on) |
Gaussian point-cloud (the 3DGS the model holds) | motion arrows (the GT supervision target)."""
import sys, re, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import cv2
from video_gt import load_episode_full, find_object_id, pick_window   # GT mask for the honesty compare

# libero entity-id -> colour (0 bg, 1 target, 2 basket, 3-7 distractors, 8 arm, 10 gripper)
ID_COL = {0: (0.12, 0.12, 0.14), 1: (0.90, 0.16, 0.16), 2: (0.16, 0.78, 0.30),
          3: (0.95, 0.74, 0.10), 4: (0.78, 0.45, 0.90), 5: (0.95, 0.55, 0.20),
          6: (0.40, 0.85, 0.95), 7: (0.85, 0.40, 0.60), 8: (0.20, 0.45, 0.95),
          9: (0.55, 0.55, 0.55), 10: (0.10, 0.85, 0.85)}
ID_NAME = {1: "target", 2: "basket", 8: "arm", 10: "gripper"}


def proj(P, K, vm):
    Pc = (vm[:3, :3] @ P.T + vm[:3, 3:4]).T
    z = np.clip(Pc[:, 2], 1e-4, None)
    return np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], 1), Pc[:, 2]


def seg_image(uv, seg, z, H, W):
    img = np.zeros((H, W, 3), np.float32)
    order = np.argsort(-z)                                   # far first -> near paints over
    u = np.clip(uv[:, 0].round().astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].round().astype(int), 0, H - 1)
    for g in order:
        img[v[g], u[g]] = ID_COL.get(int(seg[g]), (1, 1, 1))
    return img


def pc_image(uv, col, z, H, W):
    img = np.zeros((H, W, 3), np.float32)
    order = np.argsort(-z)
    u = np.clip(uv[:, 0].round().astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].round().astype(int), 0, H - 1)
    img[v[order], u[order]] = np.clip(col[order], 0, 1)
    return img


CLIPS = sys.argv[1:] if len(sys.argv) > 1 else [
    "data/libero_pi3_v2_ov/epi000000_c_train.pt",
    "data/libero_pi3_v2_ov/epi000040_c_heldseed.pt",
    "data/libero_pi3_v2_ov/epi000140_c_heldseed.pt",   # the butter outlier (IoU 0.55) — honesty
    "data/libero_pi3_v2_ov/epi000410_c_heldtask.pt",
]
ncol = 5
fig, axes = plt.subplots(len(CLIPS), ncol, figsize=(3.0 * ncol, 3.1 * len(CLIPS)))
if len(CLIPS) == 1:
    axes = axes[None]

for r, path in enumerate(CLIPS):
    c = torch.load(path, map_location="cpu", weights_only=False)
    H, W, Kf = int(c["H"]), int(c["W"]), int(c["Kf"])
    rgb = c["gt_rgb"].numpy()
    seg = c["seg_per_g"].numpy()
    uv = c["uv"].numpy()
    col = c["colors"].numpy()
    K = c["K_intr"].numpy()
    vm = c["viewmat"].numpy()
    traj = c["traj"].numpy()                                # [Kf+1,N,3]
    is_obj = c["is_obj"].numpy()
    instr = c["instruction"]
    # depth at frame-0 for paint order (traj[0] is canonical == cam0 frame)
    _, z0 = proj(traj[0], K, vm)

    # re-derive the EXACT frame this clip used (pipeline is deterministic) -> GT mask for the honesty compare
    epi = int(re.search(r"epi0*(\d+)_", path).group(1))
    wmode = "early" if "_e_" in path else "center"
    gt_seg_img = None
    try:
        rgb_all, msk_all, _, _, n = load_episode_full(epi)
        oid, _ = find_object_id(msk_all, n)
        widx = pick_window((msk_all == oid).astype(np.uint8), n, 48, mode=wmode, rng=np.random.default_rng(epi))
        gtm = cv2.resize(msk_all[widx[0]], (W, H), interpolation=cv2.INTER_NEAREST)
        gt_seg_img = np.zeros((H, W, 3), np.float32)
        for i in np.unique(gtm):
            gt_seg_img[gtm == i] = ID_COL.get(int(i), (1, 1, 1))
    except Exception as ex:
        print(f"[warn] GT mask for epi{epi}: {type(ex).__name__}: {ex}")

    axes[r, 0].imshow(rgb[0]); axes[r, 0].set_title(f"RGB t0\n{instr[:34]}", fontsize=8)
    axes[r, 1].imshow(seg_image(uv, seg, z0, H, W))
    counts = {i: int((seg == i).sum()) for i in ID_NAME}
    axes[r, 1].set_title("OPEN-VOCAB seg (what we train on)\n" + " ".join(
        f"{ID_NAME[i]}={counts[i]}" for i in (1, 2, 8)), fontsize=7)
    if gt_seg_img is not None:
        axes[r, 2].imshow(gt_seg_img)
    axes[r, 2].set_title("GT seg (ground truth)\nsame target/basket/arm colours", fontsize=7)
    axes[r, 3].imshow(pc_image(uv, col, z0, H, W))
    axes[r, 3].set_title(f"3DGS point-cloud\nN={len(seg)} (PSNR {c['val_psnr'][0]:.1f})", fontsize=8)

    # motion: object Gaussians at t0 (red) vs tKf (cyan) over a dimmed RGB -> the shift IS the GT target
    axes[r, 4].imshow((rgb[0].astype(np.float32) * 0.4).astype(np.uint8))
    oi = np.where(is_obj)[0]
    p0, _ = proj(traj[0][oi], K, vm)
    pK, _ = proj(traj[Kf][oi], K, vm)
    disp3d = np.linalg.norm(traj[Kf][oi] - traj[0][oi], axis=1)
    axes[r, 4].scatter(p0[:, 0], p0[:, 1], s=2, c="red", alpha=0.55, linewidths=0, label="t0")
    axes[r, 4].scatter(pK[:, 0], pK[:, 1], s=2, c="cyan", alpha=0.55, linewidths=0, label=f"t{Kf}")
    axes[r, 4].set_xlim(0, W); axes[r, 4].set_ylim(H, 0)
    axes[r, 4].legend(fontsize=6, loc="upper right", framealpha=0.4)
    axes[r, 4].set_title(f"object motion  red t0 -> cyan t{Kf}\nmedian {np.median(disp3d) * 100:.0f}cm (3D)", fontsize=8)
    for cc in range(ncol):
        axes[r, cc].axis("off")

plt.tight_layout()
out = "viz/ov_data/ov_training_data.png"
import os; os.makedirs("viz/ov_data", exist_ok=True)
plt.savefig(out, dpi=115, bbox_inches="tight")
print("saved", out)
