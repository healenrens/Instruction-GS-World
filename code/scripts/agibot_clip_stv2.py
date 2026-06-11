"""§82 R3 clip assembly: REAL AgiBot video -> training-format clip with the StV2 backend.

One gauge end-to-end (canonical = frame-0 camera):
  VGGT4Track front  -> per-frame camera-frame pointmaps + intrinsics + c2w poses
  StV2 offline      -> world-frame 3D tracks (clean motion pseudo-GT, §81)
  openvocab (§64)   -> entity ids at frame-0 (robot lump id8; objects 1..7; container 2)
  motion arbitration-> target entity = the non-robot entity whose tracks move most (id -> 1)
  per-entity trimmed Kabsch per frame -> traj[Kf+1, N, 3] (robot k-means sub-parts 50+c)
  viewmats[t] = inv(c2w[t]) @ c2w[0]  (canonical -> cam_t; ego-ready schema §63)

Saves data/_agibot/clip_stv2_ep{EP}.pt + an audit visual (GT-moved points over real frames).
Run from the SpaTrackerV2 repo dir with the MAIN venv (§81 install):
  cd /mnt/pfs/public/xuhaoming/SpaTrackerV2 && CUDA_VISIBLE_DEVICES=0 \
  /mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python \
  /mnt/pfs/public/xuhaoming/instruct_gs_world/code/scripts/agibot_clip_stv2.py
"""
import os
import sys

WS = "/mnt/pfs/public/xuhaoming/instruct_gs_world"
sys.path.insert(0, os.path.join(WS, "code"))
sys.path.insert(0, os.path.join(WS, "code/scripts"))
sys.path.insert(0, "/mnt/pfs/public/xuhaoming/SpaTrackerV2")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians                   # noqa: E402

TASK = os.environ.get("AGIBOT_TASK", "task_327")
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WIN, K = 48, 12
HEAD = "observation.images.head"
ID_ARM, ID_OBJ = 8, 1


CONTAINERS = {"shelf", "bag", "cart", "basket", "table", "plastic", "shopping"}


def parse_target_noun(instr: str) -> str:
    """Object noun from AgiBot action_text. Handles 'Place the held X into...' and
    'Retrieve X from the shelf.' (v1.2's naive split grabbed 'shelf.'). Container words banned."""
    import re
    s = instr.lower().rstrip(".")
    for p in (r"held ([a-z]+)",
              r"(?:retrieve|pick up|pickup|grasp|take|pick|fetch|get)\s+(?:the\s+)?([a-z]+)",
              r"place\s+(?:the\s+)?([a-z]+)"):
        m = re.search(p, s)
        if m and m.group(1) not in CONTAINERS:
            return m.group(1)
    return "object"


def trimmed_kabsch(X, Y, trim=0.25):
    """Rigid R,t mapping X->Y with worst-`trim` residuals dropped + refit. [P,3] each."""
    def fit(A, B):
        Am, Bm = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((A - Am).T @ (B - Bm))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        return R, Bm - R @ Am
    R, t = fit(X, Y)
    res = np.linalg.norm((X @ R.T + t) - Y, axis=-1)
    keep = res <= np.quantile(res, 1.0 - trim)
    if keep.sum() >= 4:
        R, t = fit(X[keep], Y[keep])
    return R, t


def main():
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
    from openvocab_seg import segment_frame_amg

    dev = "cuda"
    root = next(r for r in list_tasks() if r.rstrip("/").endswith(TASK))
    t = AgiBotLeRobotTask(root)
    pq = t.read_parquet(EP, ["observation.states.end.position"])
    eef = pq["observation.states.end.position"].reshape(-1, 2, 3)
    Tt = eef.shape[0]
    segs = t.subtasks(EP)
    cand = [(float(np.linalg.norm(eef[min(int(s["end_frame"]), Tt - 1)] - eef[int(s["start_frame"])], axis=-1).max()),
             int(s["start_frame"]), s.get("action_text", "")) for s in segs
            if int(s.get("end_frame", 0)) - int(s.get("start_frame", 0)) >= WIN]
    _, a, instruction = sorted(cand, reverse=True)[0]
    a = max(0, min(a, Tt - WIN - 1))
    widx = np.clip(np.unique(np.linspace(a, a + WIN, K + 1).round().astype(int)), 0, Tt - 1)
    Kf = len(widx) - 1
    frames = np.asarray(t.decode_frames(EP, HEAD, widx.tolist()))
    H0, W0 = frames.shape[1:3]
    print(f"[clip] window [{a},{a+WIN}] Kf={Kf} instr={instruction[:50]!r}", flush=True)

    # ---- StV2 front: pointmaps (camera frame), intrinsics, c2w --------------------------------
    vt5 = preprocess_image(torch.from_numpy(frames).permute(0, 3, 1, 2).float())[None]
    front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pf = front(vt5.cuda() / 255)
    pts_cam = pf["points_map"].squeeze().float().cpu().numpy()              # [T,h,w,3] cam frame
    unc = pf["unc_metric"].squeeze().float().cpu().numpy()                  # [T,h,w]
    intrs = pf["intrs"].squeeze().float().cpu().numpy()                     # [T,3,3]
    vt = vt5.squeeze()
    h, w = vt.shape[2:]
    del front
    torch.cuda.empty_cache()

    # ---- noun boxes at frame-0 (BEFORE the tracker) -> seed extra queries inside them ---------
    # v1.3: the 27x27 grid can miss a SMALL held object entirely (ep2/3: 0-1 tracks on the
    # cucumber). Seeding a 5x6 query grid inside each noun box guarantees tracks ON the object.
    from openvocab_seg import load_models, _gd_detect
    from PIL import Image
    noun = parse_target_noun(instruction)
    models = load_models(dev)
    sx, sy = W0 / float(w), H0 / float(h)
    nboxes0 = [d["box"] for d in _gd_detect(models, Image.fromarray(frames[0]), [noun],
                                            box_thresh=0.2, text_thresh=0.2)][:3]
    seeds = []
    for b in nboxes0:
        x0b, y0b, x1b, y1b = [float(v) for v in b]                          # original px -> model px
        gy, gx = np.meshgrid(np.linspace(y0b / sy + 2, y1b / sy - 2, 5),
                             np.linspace(x0b / sx + 2, x1b / sx - 2, 6), indexing="ij")
        seeds.append(np.stack([gx.ravel(), gy.ravel()], -1))
    seeds = np.concatenate(seeds, 0) if seeds else np.zeros((0, 2))
    print(f"[clip] noun={noun!r} boxes@f0={len(nboxes0)} seeded queries={len(seeds)}", flush=True)

    # ---- StV2 tracker: world tracks + c2w -----------------------------------------------------
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").cuda().eval()
    model.spatrack.track_num = 756
    grid_pts = get_points_on_a_grid(27, (h, w), device="cpu")
    grid_all = np.concatenate([grid_pts[0].numpy(), seeds], 0)
    query_xyt = np.concatenate([np.zeros((len(grid_all), 1)), grid_all], 1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        (c2w_traj, intrs_o, point_map, conf_depth, track3d_pred, track2d_pred,
         vis_pred, conf_pred, video) = model.forward(
            vt, depth=pts_cam[..., 2], intrs=intrs, extrs=np.eye(4)[None].repeat(Kf + 1, 0),
            queries=query_xyt, fps=1, full_point=False, iters_track=4, query_no_BA=True,
            fixed_cam=False, stage=1, unc_metric=unc > 0.5, support_frame=Kf, replace_ratio=0.2)
    c2w = c2w_traj.squeeze().float().cpu().numpy()                          # [T,4,4]
    tr3w = track3d_pred.squeeze().float().cpu().numpy()[..., :3]            # [T,Q,3] world
    tr2 = track2d_pred.squeeze().float().cpu().numpy()[..., :2]             # [T,Q,2] model px
    vis = vis_pred.squeeze().cpu().numpy() > 0.5
    del model
    torch.cuda.empty_cache()

    # ---- canonical gauge = frame-0 camera ------------------------------------------------------
    A = np.linalg.inv(c2w[0])                                               # world -> cam0
    tr3 = tr3w @ A[:3, :3].T + A[:3, 3]                                     # [T,Q,3] canonical
    viewmats = np.stack([np.linalg.inv(c2w[tt]) @ c2w[0] for tt in range(Kf + 1)]).astype(np.float32)

    # ---- g0 from frame-0 camera-frame pointmap (== canonical) ---------------------------------
    z0 = pts_cam[0][..., 2]
    valid = np.isfinite(pts_cam[0]).all(-1) & (z0 > 1e-4) & (unc[0] > 0.5)
    g0, uv = points_to_gaussians(torch.from_numpy(pts_cam[0]).float()[None], (vt[0:1] / 255.0),
                                 torch.from_numpy(valid)[None], opacity_init=0.9,
                                 scale_factor=0.6, return_uv=True)
    N = len(g0)

    # ---- entities: openvocab seg @ frame-0; target by MOTION arbitration ----------------------
    idm = segment_frame_amg(frames[0], instruction, device=dev)             # [H0,W0]
    sx, sy = W0 / float(w), H0 / float(h)
    uv_np = uv.numpy()
    seg_g = idm[np.clip((uv_np[:, 1] * sy).round().astype(int), 0, H0 - 1),
                np.clip((uv_np[:, 0] * sx).round().astype(int), 0, W0 - 1)].astype(np.int64)
    tr_seg = idm[np.clip((tr2[0][:, 1] * sy).round().astype(int), 0, H0 - 1),
                 np.clip((tr2[0][:, 0] * sx).round().astype(int), 0, W0 - 1)].astype(np.int64)
    ok = vis.mean(0) > 0.6
    disp = np.linalg.norm(tr3[Kf] - tr3[0], axis=-1)
    movers = ok & (disp > max(float(np.percentile(disp[ok], 80)), 0.02))

    # ---- v1.1 TARGET = HELD-OBJECT arbitration: moving tracks inside the instruction-noun box --
    # (the held object rides INSIDE the gripper -> openvocab lumps it into robot id8; v1's
    #  per-openvocab-entity motion vote therefore picked a static shelf item. Fix: GD the noun
    #  (parse_target_noun, v1.3) on frame0 AND lastframe, target = mover ∩ noun-box. The seeded
    #  queries (v1.3) guarantee tracks ON the object even when the 27x27 grid misses it.)
    tgt_q = np.zeros(tr2.shape[1], bool)
    for fi in (0, Kf):
        dets = _gd_detect(models, Image.fromarray(frames[fi]), [noun], box_thresh=0.2, text_thresh=0.2)
        for d in dets:
            x0b, y0b, x1b, y1b = [float(v) for v in d["box"]]
            inb = ((tr2[fi][:, 0] * sx >= x0b) & (tr2[fi][:, 0] * sx <= x1b)
                   & (tr2[fi][:, 1] * sy >= y0b) & (tr2[fi][:, 1] * sy <= y1b))
            tgt_q |= (inb & movers)
    print(f"[clip] held-object arbitration: noun={noun!r} mover-tracks-in-box={int(tgt_q.sum())} "
          f"(total movers {int(movers.sum())})", flush=True)
    if tgt_q.sum() >= 4:
        tr_seg[tgt_q] = ID_OBJ
        # carve the target's GAUSSIANS out of the robot lump: pixels near target tracks @ frame0...
        d2t = ((uv_np[:, None, :] - tr2[0][tgt_q][None, :, :]) ** 2).sum(-1).min(1)
        near_t = d2t < (0.03 * max(h, w)) ** 2
        # ...v1.2 + MOTION-CONSISTENCY: the carved Gaussian's own local motion (3-NN IDW over ALL
        # tracks) must be a real fraction of the target's motion — static same-class instances
        # (shelf cucumbers under a noun box crossed by mover paths) have ~0 local motion -> dropped.
        okq = np.where(ok)[0]
        cand = np.where(near_t)[0]
        d2a = ((uv_np[cand][:, None, :] - tr2[0][okq][None, :, :]) ** 2).sum(-1)     # [C,Qok]
        nn3 = np.argsort(d2a, axis=1)[:, :3]
        w3 = 1.0 / np.clip(np.take_along_axis(d2a, nn3, axis=1), 1e-6, None)
        w3 = w3 / w3.sum(1, keepdims=True)
        dKf = np.linalg.norm(tr3[Kf][okq] - tr3[0][okq], axis=-1)                     # [Qok]
        loc_mov = (w3 * dKf[nn3]).sum(1)                                              # [C]
        tgt_med = float(np.median(disp[tgt_q]))
        keep = loc_mov > 0.4 * tgt_med
        carved = cand[keep]
        print(f"[clip] carve motion-filter: {len(cand)} candidates -> {int(keep.sum())} kept "
              f"(target med {tgt_med*100:.1f}, thresh {0.4*tgt_med*100:.1f})", flush=True)
        m_c = np.zeros(N, bool)
        m_c[carved] = True
        seg_g[m_c & (seg_g == ID_ARM)] = ID_OBJ
        seg_g[m_c & (seg_g == 0)] = ID_OBJ
    else:
        print("[clip] WARN: no moving noun box -> falling back to v1 entity-motion vote", flush=True)
        obj_ids = [int(i) for i in np.unique(tr_seg) if 1 <= i <= 7]
        if obj_ids:
            mov = {i: float(np.median(disp[ok & (tr_seg == i)])) if (ok & (tr_seg == i)).sum() >= 3 else 0.0
                   for i in obj_ids}
            tgt = max(mov, key=mov.get)
            tr_seg[tr_seg == tgt] = ID_OBJ
            seg_g[seg_g == tgt] = ID_OBJ

    # ---- v1.1 traj: target RIGID (objects are rigid); robot NON-RIGID via k-NN track transfer --
    # (v1's k=3 rigid sub-parts smeared the articulated arm; StV2's raw tracks are clean (§81) so
    #  transfer displacement directly: per robot Gaussian, IDW blend of its 3 nearest robot tracks.)
    traj = np.broadcast_to(g0.means.numpy()[None], (Kf + 1, N, 3)).copy()
    seg_final = seg_g.copy()
    g_np = g0.means.numpy()
    # target: trimmed Kabsch (rigid)
    qo = np.where(tr_seg == ID_OBJ)[0]
    if len(qo) >= 4:
        idx_o = np.where(seg_final == ID_OBJ)[0]
        X0 = tr3[0][qo]
        for tt in range(1, Kf + 1):
            R, tv = trimmed_kabsch(X0, tr3[tt][qo])
            traj[tt, idx_o] = g_np[idx_o] @ R.T + tv
    # robot: non-rigid k-NN inverse-distance transfer + k-means SUB-IDS (for entity-LBS binding only)
    arm_q = np.where(ok & (tr_seg == ID_ARM))[0]
    if len(arm_q) >= 8:
        idx_a = np.where(seg_final == ID_ARM)[0]
        d2 = ((uv_np[idx_a][:, None, :] - tr2[0][arm_q][None, :, :]) ** 2).sum(-1)   # [Na,Qa]
        nn = np.argsort(d2, axis=1)[:, :3]
        wgt = 1.0 / np.clip(np.take_along_axis(d2, nn, axis=1), 1e-6, None)
        wgt = wgt / wgt.sum(1, keepdims=True)                                        # [Na,3]
        for tt in range(1, Kf + 1):
            dtr = tr3[tt][arm_q] - tr3[0][arm_q]                                     # [Qa,3]
            traj[tt, idx_a] = g_np[idx_a] + (wgt[..., None] * dtr[nn]).sum(1)
        from sklearn.cluster import KMeans
        kk = min(4, len(arm_q) // 4)
        lab = KMeans(n_clusters=kk, n_init=4, random_state=0).fit_predict(tr3[Kf, arm_q] - tr3[0, arm_q])
        owner = lab[d2.argmin(1)]
        seg_final[idx_a] = 50 + owner
    is_obj = seg_final == ID_OBJ
    obj_disp = np.linalg.norm(traj[Kf][is_obj] - traj[0][is_obj], axis=-1)
    subids = sorted(int(i) for i in np.unique(seg_final) if i >= 50)
    print(f"[clip] N={N} obj_gauss={int(is_obj.sum())} obj disp median={np.median(obj_disp)*100:.1f} "
          f"(StV2 units x100) | arm sub-ids={subids}", flush=True)

    gt_rgb = (vt.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)        # [T,h,w,3]
    save = {"means": g0.means, "quats": g0.quats, "scales": g0.scales,
            "opacities": g0.opacities, "colors": g0.colors, "uv": uv,
            "seg_per_g": torch.from_numpy(seg_final), "traj": torch.from_numpy(traj),
            "K_intr": torch.from_numpy(intrs_o.squeeze().float().cpu().numpy()[0]),
            "viewmat": torch.eye(4), "viewmats": torch.from_numpy(viewmats),
            "H": h, "W": w, "Kf": Kf, "instruction": instruction,
            "gt_rgb": torch.from_numpy(gt_rgb),
            "is_obj": torch.from_numpy(is_obj), "n_fill": 0,
            "backend": "stv2", "epi": EP, "task": TASK, "split": "real"}
    os.makedirs(os.path.join(WS, "data/_agibot"), exist_ok=True)
    out_pt = os.path.join(WS, f"data/_agibot/clip_stv2_ep{EP}.pt")
    torch.save(save, out_pt)
    print(f"[clip] saved {out_pt}", flush=True)

    # ---- audit visual: GT-moved entity points projected over the real frames ------------------
    Ki = intrs_o.squeeze().float().cpu().numpy()[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, tt in zip(axes, [0, Kf // 2, Kf]):
        ax.imshow(gt_rgb[tt]); ax.axis("off")
        vm = viewmats[tt]
        for m, col in [(is_obj, "red"), (seg_final >= 50, "cyan")]:
            P = traj[tt][m] @ vm[:3, :3].T + vm[:3, 3]
            zc = np.clip(P[:, 2], 1e-4, None)
            ax.scatter(Ki[0, 0] * P[:, 0] / zc + Ki[0, 2], Ki[1, 1] * P[:, 1] / zc + Ki[1, 2],
                       s=1.5, c=col, alpha=0.5, linewidths=0)
        ax.set_title(f"GT traj @ t{tt} (red=target cyan=arm)", fontsize=8)
    plt.tight_layout()
    outp = os.path.join(WS, f"viz/agibot/r3_clip_stv2_ep{EP}.png")
    plt.savefig(outp, dpi=120, bbox_inches="tight")
    print(f"[clip] saved {outp}", flush=True)


if __name__ == "__main__":
    main()
