"""AgiBot Beta (real-world) SpaTrackerV2 GT EVALUATION harness.

Goal: HONESTLY assess whether the SIM-validated SpaTracker GT pipeline (robotwin_spatrack_clip.py)
works on REAL-WORLD AgiBot video. NOT for training — for a quality/failure-mode report.

It borrows the producer's VGGT4Track(depth+intrinsics) + Predictor(joint 2D+3D track) core verbatim,
but instead of a static-cam assumption it:
  * decodes head-camera frames straight from the AV1 mp4 (PyAV),
  * runs the Predictor with BOTH fixed_cam=True and fixed_cam=False,
  * inspects the returned per-frame c2w to decide if the real camera is static or moving,
  * computes tracking quality (mover count, 2D flow magnitude, vis fraction, reproj err, depth range),
  * draws the GT image-flow arrows (top movers, frame0->frameK) and a c2w-translation curve.

Usage (one episode, both cam modes):
  agibot_spatrack_eval.py --task task_327 --episode 0 --win_start 0.30 --win_frac 0.30 \
      --kf 12 --grid 40 --out_dir /mnt/pfs/public/xuhaoming/instruct_gs_world/outputs

--win_start/--win_frac select a sub-window (fraction of the full episode) so we land on a span with
clear object motion instead of idle approach frames. Default = middle 40% of the episode.
"""
import argparse, sys, os, json, glob
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)                                                 # never cache to the 30GB home (~/)
import numpy as np, torch, av
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor                          # noqa: E402
from models.SpaTrackV2.models.utils import get_points_on_a_grid                   # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track        # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image    # noqa: E402

VIDEO_ROOT = os.environ.get("COT_VIDEO_ROOT",
                            "/mnt/pfs/public/agibot-world-beta-lerobot/agibot-world-beta-lerobot")


def np3(x):
    x = x.float().cpu().numpy()
    while x.ndim > 3: x = x[0]
    return x                                                                      # [T,N,C]


def agibot_head_video(task, episode):
    """-> path to the per-episode head-camera mp4 (base offset 0; the simple AgiBot layout)."""
    ep = "episode_%06d.mp4" % episode
    cands = glob.glob(f"{VIDEO_ROOT}/{task}/{task}/videos/chunk-*/observation.images.head/{ep}")
    if not cands:
        raise FileNotFoundError(f"no head video for {task}/{ep} under {VIDEO_ROOT}")
    return cands[0]


def task_label(task):
    p = f"{VIDEO_ROOT}/{task}/{task}/meta/tasks.jsonl"
    try:
        with open(p) as f:
            return json.loads(f.readline()).get("task", "")
    except Exception:
        return ""


def decode_window(vid, kf, win_start, win_frac):
    """Decode kf+1 frames evenly across a sub-window [win_start, win_start+win_frac] of the episode.
    Returns frames[T,H,W,3] uint8 and the absolute source frame indices used."""
    c = av.open(vid); st = c.streams.video[0]
    total = st.frames or 0
    if total <= 0:                                                                # fallback: count by decode
        total = sum(1 for _ in c.decode(st)); c.close(); c = av.open(vid); st = c.streams.video[0]
    a = int(round(total * win_start)); b = int(round(total * min(1.0, win_start + win_frac)))
    a = max(0, min(a, total - 1)); b = max(a + kf, min(b, total - 1))
    want = sorted(set(np.linspace(a, b, kf + 1).astype(int).tolist()))
    frames, got = {}, set(want)
    for i, fr in enumerate(c.decode(st)):
        if i in got:
            frames[i] = fr.to_ndarray(format="rgb24")
            if len(frames) == len(got): break
    c.close()
    idx = sorted(frames.keys())
    return np.stack([frames[i] for i in idx]).astype(np.uint8), idx


def run_core(frames, vggt, model, grid, iters_track, fixed_cam):
    """The producer's VGGT+Predictor core. Returns dict of numpy arrays (track2d/3d/vis/intr/c2w/H/W)."""
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()                     # [T,3,H0,W0]
    vt_p = preprocess_image(vt)[None]                                             # [1,T,3,h,w]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
        intrs = pred["intrs"]; depth = pred["points_map"][..., 2]; unc = pred["unc_metric"]
    depth_t = depth.squeeze().cpu().numpy()
    vt_in = vt_p[0]; H, W = int(vt_in.shape[-2]), int(vt_in.shape[-1])
    grid_pts = get_points_on_a_grid(grid, (H, W), device="cpu")
    qxyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out_t = model.forward(vt_in, depth=depth_t,
                              intrs=intrs.squeeze(0) if intrs.dim() > 3 else intrs,
                              extrs=None, queries=qxyt, fps=1, full_point=False, iters_track=iters_track,
                              query_no_BA=True, fixed_cam=fixed_cam, stage=1, unc_metric=unc,
                              support_frame=len(vt_in) - 1, replace_ratio=0.2)
    c2w, intr2, pmap, confd, track3d, track2d, vis, conf, video = out_t
    t2 = np3(track2d)[..., :2]; t3 = np3(track3d)[..., :3]
    T, N = t2.shape[0], t2.shape[1]
    vv = np.squeeze(vis.float().cpu().numpy())
    if vv.ndim == 1 and vv.size == N: vv = np.broadcast_to(vv[None], (T, N)).copy()
    if vv.shape != (T, N): vv = np.ones((T, N), dtype=np.float32)
    I2 = intr2.float().cpu().numpy(); C2W = c2w.float().cpu().numpy()
    img0 = video[0].float().cpu().numpy() if hasattr(video, "float") else np.asarray(video)[0]
    if img0.shape[0] in (3, 4): img0 = img0.transpose(1, 2, 0)
    img0 = (img0[..., :3] * (255 if img0.max() <= 1.01 else 1)).clip(0, 255).astype(np.uint8)
    del out_t, track3d, track2d, vis; torch.cuda.empty_cache()
    return dict(t2=t2, t3=t3, vv=vv, I2=I2, C2W=C2W, H=H, W=W, img0=img0)


def cam_motion_stats(C2W):
    """How much does the returned camera pose move across frames? Returns (trans_mm_equiv, rot_deg, is_static)."""
    C = C2W
    if C.ndim == 2:                                                              # single pose -> static by construction
        return 0.0, 0.0, True
    t = C[:, :3, 3]                                                              # [T,3] translations
    tr = float(np.linalg.norm(t - t[0], axis=1).max())                          # max drift from frame0 (model units)
    R = C[:, :3, :3]
    rots = []
    for i in range(len(R)):
        Rr = R[0].T @ R[i]
        cos = (np.trace(Rr) - 1) / 2
        rots.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    rot = float(np.max(rots))
    # scene scale ref: median scene depth, so we can express translation as a FRACTION of scene depth
    return tr, rot, (tr < 1e-4 and rot < 1e-2)


def reproj_err(res):
    """Median per-frame reprojection error of track3d (camera-frame hypothesis) vs track2d."""
    t2, t3, I2 = res["t2"], res["t3"], res["I2"]
    errs = []
    for t in range(t2.shape[0]):
        Kt = I2[t] if I2.ndim == 3 else I2
        z = np.clip(t3[t][..., 2:3], 1e-4, None)
        uvh = t3[t] / z
        pA = (uvh @ Kt.T)[..., :2]
        errs.append(float(np.median(np.linalg.norm(pA - t2[t], axis=1))))
    return errs


def quality(res, mover_px=3.0):
    t2, t3, vv = res["t2"], res["t3"], res["vv"]
    T = t2.shape[0]; Kf = T - 1
    d2 = np.linalg.norm(t2[Kf] - t2[0], axis=1)                                  # 2D pixel flow per track
    d3 = np.linalg.norm(t3[Kf] - t3[0], axis=1)
    movers = int((d2 > mover_px).sum())
    mv_mask = d2 > mover_px
    z = t3[..., 2]
    return dict(N=t2.shape[1], Kf=Kf, med2d=float(np.median(d2)), p90_2d=float(np.percentile(d2, 90)),
                max2d=float(d2.max()), movers=movers, mover_frac=float(mv_mask.mean()),
                mover_med2d=float(np.median(d2[mv_mask])) if movers else 0.0,
                med3d=float(np.median(d3)), max3d=float(d3.max()),
                vis=float(vv.mean()), zmin=float(z.min()), zmax=float(z.max()), zmed=float(np.median(z)),
                d2=d2)


def draw_flow(ax, res, q, title):
    ax.imshow(res["img0"]); t2 = res["t2"]; Kf = t2.shape[0] - 1
    mv = np.argsort(-q["d2"])[:120]
    for i in mv:
        if q["d2"][i] < 2: continue
        ax.annotate("", xy=(t2[Kf, i, 0], t2[Kf, i, 1]), xytext=(t2[0, i, 0], t2[0, i, 1]),
                    arrowprops=dict(arrowstyle="->", color=plt.cm.turbo(min(q["d2"][i] / 80, 1)),
                                    lw=0.9, alpha=0.85))
    ax.set_title(title, fontsize=8); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--win_start", type=float, default=0.30)
    ap.add_argument("--win_frac", type=float, default=0.40)
    ap.add_argument("--kf", type=int, default=12)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--iters_track", type=int, default=4)
    ap.add_argument("--out_dir", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    vid = agibot_head_video(args.task, args.episode)
    lbl = task_label(args.task)
    frames, idx = decode_window(vid, args.kf, args.win_start, args.win_frac)
    print(f"[agibot] {args.task} ep{args.episode}: {vid}", flush=True)
    print(f"[agibot] label: {lbl[:90]}", flush=True)
    print(f"[agibot] decoded {frames.shape} src-frames {idx[0]}..{idx[-1]} (n={len(idx)})", flush=True)

    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()

    results, qs = {}, {}
    for fc in (True, False):
        res = run_core(frames, vggt, model, args.grid, args.iters_track, fixed_cam=fc)
        q = quality(res); tr, rot, static = cam_motion_stats(res["C2W"])
        re = reproj_err(res)
        zmed = q["zmed"]
        tr_frac = tr / max(zmed, 1e-6)                                           # translation as fraction of scene depth
        results[fc] = (res, q, dict(tr=tr, rot=rot, static=static, tr_frac=tr_frac, reproj=re))
        qs[fc] = q
        print(f"[agibot] fixed_cam={fc}: N={q['N']} Kf={q['Kf']} {res['W']}x{res['H']} "
              f"med2d={q['med2d']:.1f}px movers(>3px)={q['movers']} ({100*q['mover_frac']:.0f}%) "
              f"mover_med2d={q['mover_med2d']:.1f}px vis={q['vis']:.2f} "
              f"z[{q['zmin']:.2f},{q['zmax']:.2f}] med={zmed:.2f} | "
              f"cam: trans={tr:.4f}({100*tr_frac:.1f}% of depth) rot={rot:.2f}deg static={static} "
              f"reproj_med={np.median(re):.2f}px", flush=True)

    # ---- montage: frame0 flow for both cam modes + a c2w translation curve (fixed_cam=False) ----
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or f"{args.task}_ep{args.episode}"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    resT, qT, mT = results[True]; resF, qF, mF = results[False]
    draw_flow(axes[0], resT, qT,
              f"fixed_cam=True  movers>3px={qT['movers']} mv_med={qT['mover_med2d']:.0f}px vis={qT['vis']:.2f}")
    draw_flow(axes[1], resF, qF,
              f"fixed_cam=False  movers>3px={qF['movers']} mv_med={qF['mover_med2d']:.0f}px vis={qF['vis']:.2f}")
    C = resF["C2W"]
    if C.ndim == 3:
        t = C[:, :3, 3]; t = t - t[0]
        axes[2].plot(t[:, 0], label="x"); axes[2].plot(t[:, 1], label="y"); axes[2].plot(t[:, 2], label="z")
        axes[2].set_title(f"c2w translation drift (fixed_cam=False)\nmax={mF['tr']:.4f} "
                          f"({100*mF['tr_frac']:.1f}% of scene depth) rot={mF['rot']:.2f}deg", fontsize=8)
        axes[2].legend(fontsize=7); axes[2].set_xlabel("frame")
    else:
        axes[2].text(0.5, 0.5, "single c2w returned\n(model treated cam as static)", ha="center")
        axes[2].axis("off")
    fig.suptitle(f"{args.task} ep{args.episode} [{lbl[:70]}]  src{idx[0]}-{idx[-1]}", fontsize=10)
    out_png = os.path.join(args.out_dir, f"agibot_gt_{tag}.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)

    summary = dict(task=args.task, episode=args.episode, label=lbl, src_frames=[idx[0], idx[-1]],
                   HxW=[resF["H"], resF["W"]], png=out_png,
                   fixed_cam_true=dict(movers=qT["movers"], mover_frac=qT["mover_frac"],
                                       mover_med2d=qT["mover_med2d"], med2d=qT["med2d"], vis=qT["vis"],
                                       zmin=qT["zmin"], zmax=qT["zmax"], zmed=qT["zmed"],
                                       reproj_med=float(np.median(mT["reproj"]))),
                   fixed_cam_false=dict(movers=qF["movers"], mover_frac=qF["mover_frac"],
                                        mover_med2d=qF["mover_med2d"], med2d=qF["med2d"], vis=qF["vis"],
                                        zmin=qF["zmin"], zmax=qF["zmax"], zmed=qF["zmed"],
                                        reproj_med=float(np.median(mF["reproj"])),
                                        cam_trans=mF["tr"], cam_trans_frac=mF["tr_frac"],
                                        cam_rot_deg=mF["rot"], cam_static=mF["static"]))
    with open(os.path.join(args.out_dir, f"agibot_gt_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[agibot] SAVED {out_png}", flush=True)
    print("[agibot] SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "label"}), flush=True)


if __name__ == "__main__":
    main()
