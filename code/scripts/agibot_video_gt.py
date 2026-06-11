"""§76 R3 milestone-1: turn ONE real AgiBot episode into a 3DGS clip foundation.
Read a sub-task window (head-cam RGB + REAL EEF 6-DOF proprioception + language) -> Pi3 lift to a
frame-0 GaussianSet + per-frame camera poses -> save a minimal clip + a validation visual
(RGB t0/tK | Pi3-lifted 3DGS point-cloud | dual-arm EEF 3D trajectory). This proves the real-video
-> 3DGS path works and the real EEF GT is in hand. (Tracking / openvocab / EEF-scale-calibration =
milestone 2+.)

  python code/scripts/agibot_video_gt.py --task task_327 --ep 0 --win 48 --K 12 --out data/_agibot/clip.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402
from igsw.lifting.pi3_lifter import Pi3Lifter                               # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians                   # noqa: E402

HEAD = "observation.images.head"
EEF_POS = "observation.states.end.position"          # [T,2,3] dual-arm xyz (m, world)
EEF_ORI = "observation.states.end.orientation"       # [T,2,4] dual-arm quat xyzw
GRIP = "observation.states.effector.position"        # [T,2]


def _attr(o, n):
    a = getattr(o, n)
    return a() if callable(a) else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_327")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--out", default="data/_agibot/clip.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    root = next(r for r in list_tasks() if r.rstrip("/").endswith(args.task))
    t = AgiBotLeRobotTask(root)
    print(f"[agibot] {args.task} ep{args.ep} | {root}", flush=True)

    # ---- pick a sub-task window with real motion (largest EEF displacement segment) ----
    pq = t.read_parquet(args.ep, [EEF_POS, EEF_ORI, GRIP])
    eef = pq[EEF_POS].reshape(-1, 2, 3)                  # [T,2,3]
    T = eef.shape[0]
    segs = _attr(t, "subtasks") if False else t.subtasks(args.ep)
    cand = []
    for s in segs:
        a, b = int(s.get("start_frame", 0)), int(s.get("end_frame", T - 1))
        if b - a >= args.win:
            disp = float(np.linalg.norm(eef[b] - eef[a], axis=-1).max())   # max over the 2 arms
            cand.append((disp, a, b, s.get("action_text", "")))
    cand.sort(reverse=True)
    if not cand:
        a, b, text = 0, min(args.win, T - 1), t.language(args.ep)
    else:
        _, a, b, text = cand[0]
    a = max(0, min(a, T - args.win - 1))
    widx = np.linspace(a, a + args.win, args.K + 1).round().astype(int)
    widx = np.clip(np.unique(widx), 0, T - 1)
    Kf = len(widx) - 1
    print(f"[agibot] window [{a},{a+args.win}] frames T={T} | instr: {text[:70]!r}", flush=True)

    # ---- decode head-cam RGB at the window frames ----
    frames = t.decode_frames(args.ep, HEAD, widx.tolist())       # [Kf+1,H,W,3] uint8
    frames = np.asarray(frames)
    H0, W0 = frames.shape[1:3]
    print(f"[agibot] head RGB {frames.shape} ({W0}x{H0})", flush=True)

    # ---- Pi3 lift the window -> per-frame pointmaps + camera poses ----
    lifter = Pi3Lifter(device=dev)
    res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.0)
    local = res["local_points"].numpy()                          # [T,Hm,Wm,3] cam frame
    confs = res["conf"].numpy()
    poses = res["camera_poses"].numpy().astype(np.float64)        # [T,4,4] cam2world
    imgs = res["images"]                                          # [T,3,Hm,Wm]
    Hm, Wm = local.shape[1], local.shape[2]
    cam_t = np.linalg.norm((np.linalg.inv(poses[0]) @ poses)[:, :3, 3], axis=-1)
    print(f"[agibot] Pi3 {Wm}x{Hm} | camera MOVES (ego): max||rel-t||={cam_t.max():.3f}m "
          f"(real ego video -> non-static cam, viewmats schema §63 applies)", flush=True)

    # ---- frame-0 GaussianSet (canonical = cam0) ----
    z0 = local[0][..., 2].astype(np.float32)
    valid = np.isfinite(local[0]).all(-1) & (z0 > 1e-4) & (confs[0] > 0.1)
    g0, uv = points_to_gaussians(torch.from_numpy(local[0]).float()[None], imgs[0:1],
                                 torch.from_numpy(valid)[None], opacity_init=0.9,
                                 scale_factor=0.6, return_uv=True)
    N = len(g0)
    print(f"[agibot] g0: N={N} Gaussians from real head-cam frame-0", flush=True)

    # ---- save minimal clip + the REAL EEF GT for the window ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save = dict(means=g0.means.cpu(), colors=g0.colors.cpu(), uv=uv.cpu(),
                eef_pos=torch.from_numpy(eef[widx]).float(),                # [Kf+1,2,3] REAL GT
                eef_ori=torch.from_numpy(pq[EEF_ORI].reshape(-1, 2, 4)[widx]).float(),
                grip=torch.from_numpy(pq[GRIP].reshape(-1, 2)[widx]).float(),
                instruction=text, task=args.task, ep=args.ep, widx=widx.tolist(),
                Kf=Kf, H=Hm, W=Wm, backend="pi3_agibot")
    torch.save(save, args.out)
    print(f"[agibot] saved {args.out}", flush=True)

    # ---- validation visual: RGB t0/tK | lifted 3DGS point-cloud | EEF 3D trajectory ----
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 4, 1); ax.imshow(frames[0]); ax.axis("off"); ax.set_title(f"head RGB t0\n{text[:34]}", fontsize=8)
    ax = fig.add_subplot(1, 4, 2); ax.imshow(frames[Kf]); ax.axis("off"); ax.set_title(f"head RGB t{Kf}", fontsize=8)
    # 3DGS point cloud (project means by frame-0 pinhole-ish: just (uv) colored)
    ax = fig.add_subplot(1, 4, 3)
    col = g0.colors.cpu().numpy().clip(0, 1)
    uvn = uv.cpu().numpy()
    ax.scatter(uvn[:, 0], uvn[:, 1], s=1, c=col, linewidths=0)
    ax.set_xlim(0, Wm); ax.set_ylim(Hm, 0); ax.axis("off"); ax.set_title(f"Pi3 3DGS N={N}", fontsize=8)
    # EEF 3D trajectory (dual-arm), top-down xy
    ax = fig.add_subplot(1, 4, 4)
    ew = eef[widx]
    for arm, c in [(0, "tab:blue"), (1, "tab:red")]:
        ax.plot(ew[:, arm, 0], ew[:, arm, 1], "-o", ms=2, c=c, label=f"arm{arm}")
        ax.scatter(ew[0, arm, 0], ew[0, arm, 1], c="k", s=20, zorder=3)
    ax.set_title(f"REAL EEF xy traj\nmaxdisp {np.linalg.norm(ew[-1]-ew[0],axis=-1).max()*100:.0f}cm",
                 fontsize=8); ax.legend(fontsize=6); ax.set_aspect("equal")
    plt.tight_layout()
    out_png = "viz/agibot/r3_m1.png"
    os.makedirs("viz/agibot", exist_ok=True)
    plt.savefig(out_png, dpi=115, bbox_inches="tight")
    print(f"[agibot] saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
