"""§78 R3 m4: METRIC SCALE calibration of the Pi3 reconstruction via the REAL EEF proprioception.
GroundingDINO finds the gripper box -> CoTracker tracks gripper points across the window -> Pi3
canonical pointmaps lift them to 3D (Pi3 gauge) -> Umeyama similarity fit (s,R,t) against the real
dual-arm EEF trajectory (meters, robot-base frame). Output: metric scale s (Pi3->meters), the
base->cam0 transform, and the residual (cm) = the calibration quality. Tries both arms, keeps the
better-fitting one. This makes ALL Pi3 predictions metrically evaluable (manipulator 5deg5cm, m5)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402
from igsw.lifting.pi3_lifter import Pi3Lifter                               # noqa: E402
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at     # noqa: E402
from openvocab_seg import load_models, _gd_detect                           # noqa: E402
from PIL import Image                                                       # noqa: E402

TASK, EP, WIN, K = "task_327", 0, 48, 12
HEAD = "observation.images.head"


def umeyama(X, Y):
    """Similarity fit Y ~ s*R@X + t.  X,Y [T,3]. Returns s, R, t, rms residual."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    cov = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (Xc ** 2).sum() / len(X)
    s = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    t = my - s * R @ mx
    res = float(np.sqrt(((s * (R @ X.T).T + t - Y) ** 2).sum(-1).mean()))
    return s, R, t, res


def main():
    dev = "cuda"
    root = next(r for r in list_tasks() if r.rstrip("/").endswith(TASK))
    t = AgiBotLeRobotTask(root)
    pq = t.read_parquet(EP, ["observation.states.end.position"])
    eef = pq["observation.states.end.position"].reshape(-1, 2, 3)        # [T,2,3] meters
    Tt = eef.shape[0]
    segs = t.subtasks(EP)
    cand = [(float(np.linalg.norm(eef[min(int(s["end_frame"]), Tt - 1)] - eef[int(s["start_frame"])], axis=-1).max()),
             int(s["start_frame"])) for s in segs if int(s.get("end_frame", 0)) - int(s.get("start_frame", 0)) >= WIN]
    a = max(0, min(sorted(cand, reverse=True)[0][1], Tt - WIN - 1)) if cand else 0
    widx = np.clip(np.unique(np.linspace(a, a + WIN, K + 1).round().astype(int)), 0, Tt - 1)
    Kf = len(widx) - 1
    frames = np.asarray(t.decode_frames(EP, HEAD, widx.tolist()))
    H0, W0 = frames.shape[1:3]
    print(f"[eefcal] window [{a},{a+WIN}] Kf={Kf}")

    lifter = Pi3Lifter(device=dev)
    res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.0)
    local = res["local_points"].numpy()
    poses = res["camera_poses"].numpy().astype(np.float64)
    Hm, Wm = local.shape[1], local.shape[2]
    rel = np.stack([np.linalg.inv(poses[0]) @ poses[i] for i in range(Kf + 1)]).astype(np.float32)
    canon = (np.einsum("tij,thwj->thwi", rel[:, :3, :3], local) + rel[:, None, None, :3, 3]).astype(np.float32)

    # gripper boxes on frame-0 (both arms if found)
    models = load_models(dev)
    dets = _gd_detect(models, Image.fromarray(frames[0]), ["robot gripper", "robotic arm"],
                      box_thresh=0.2, text_thresh=0.2)
    gboxes = [d["box"] for d in dets if "gripper" in d["label"]][:2] or [d["box"] for d in dets[:2]]
    print(f"[eefcal] gripper boxes: {[[int(v) for v in b] for b in gboxes]}")

    ct = CoTrackerTracker(device=dev)
    frT = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    results = []
    for bi, b in enumerate(gboxes):
        x0, y0, x1, y1 = [float(v) for v in b]
        gy, gx = torch.meshgrid(torch.linspace(y0 + 5, y1 - 5, 5), torch.linspace(x0 + 5, x1 - 5, 5), indexing="ij")
        q = torch.stack([gx.flatten(), gy.flatten()], -1)
        tracks, vis = ct.track(frT, q.to(dev))
        tr_m = tracks.cpu() * torch.tensor([Wm / float(W0), Hm / float(H0)])
        p3 = sample_pointmaps_at(torch.from_numpy(canon), tr_m).numpy()   # [T,Q,3] Pi3 gauge
        ok = np.isfinite(p3).all(-1).all(0) & (vis.cpu().numpy().mean(0) > 0.5)
        if ok.sum() < 3:
            print(f"[eefcal] box{bi}: only {int(ok.sum())} valid tracks -> skip")
            continue
        zmed0 = float(np.nanmedian(canon[0][..., 2]))
        # PER-TRACK Umeyama (consensus, RANSAC-flavored): a track rigidly attached to the EEF fits its
        # trajectory SHAPE with low residual; arm-body/background tracks can't fit at any scale.
        for arm in (0, 1):
            E = eef[widx, arm]                                             # [T,3] meters
            disp = float(np.linalg.norm(E[-1] - E[0]))
            if disp < 0.03:
                continue
            for qi in np.where(ok)[0]:
                s, R, tt, resid = umeyama(p3[:, qi], E)
                rel_r = resid / disp
                zmed = zmed0 * s
                sane = (rel_r < 0.15) and (0.3 < zmed < 5.0)
                results.append((rel_r, resid, s, arm, bi, disp, zmed, sane))
        n_sane = sum(1 for r in results if r[7] and r[4] == bi)
        print(f"[eefcal] box{bi}: {int(ok.sum())} tracks fitted, {n_sane} pass sanity")
    if not results:
        print("[eefcal] FAILED: no valid fit")
        return
    sane_r = sorted([r for r in results if r[7]])
    if sane_r:
        top = sane_r[:max(3, len(sane_r) // 4)]                            # consensus over best tracks
        ss = sorted(r[2] for r in top)
        s_con = ss[len(ss) // 2]
        relr, resid, s, arm, bi, disp, zmed, _ = sane_r[0]
        print(f"\n[eefcal] consensus over {len(top)} sane tracks: scale median={s_con:.3f} "
              f"(spread {ss[0]:.2f}..{ss[-1]:.2f})")
        print(f"[eefcal] BEST track: arm{arm} box{bi} scale={s:.3f} residual={resid*100:.1f}cm "
              f"({relr*100:.0f}% rel) scene-z*s={zmed:.2f}m -> PASS")
    else:
        relr, resid, s, arm, bi, disp, zmed, _ = sorted(results)[0]
        print(f"\n[eefcal] FAIL (no track passed rel<15% AND 0.3<z*s<5m). best: arm{arm} box{bi} "
              f"scale={s:.3f} rel={relr*100:.0f}% z*s={zmed:.2f}m")


if __name__ == "__main__":
    main()
