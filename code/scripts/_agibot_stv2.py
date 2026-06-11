"""§80 R3 m2-redo: SpatialTrackerV2 on the AgiBot window — CLEAN object 3D motion pseudo-GT.
VGGT4Track front-end (depth+intrinsics+poses) -> StV2 offline tracker (world-space 3D tracks).
Compares against the noisy CoTracker+Pi3 m2 result (median disp / outliers / mover localization)
and saves the motion visual + npz for the clip builder.

Run with the MAIN venv (torch 2.8) from the SpaTrackerV2 repo dir (models/ package):
  cd /mnt/pfs/public/xuhaoming/SpaTrackerV2 && \
  /mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python \
  /mnt/pfs/public/xuhaoming/instruct_gs_world/code/scripts/_agibot_stv2.py
"""
import os
import sys

WS = "/mnt/pfs/public/xuhaoming/instruct_gs_world"
sys.path.insert(0, os.path.join(WS, "code"))
sys.path.insert(0, "/mnt/pfs/public/xuhaoming/SpaTrackerV2")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402

TASK, EP, WIN, K = "task_327", 0, 48, 12
HEAD = "observation.images.head"


def main():
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    root = next(r for r in list_tasks() if r.rstrip("/").endswith(TASK))
    t = AgiBotLeRobotTask(root)
    pq = t.read_parquet(EP, ["observation.states.end.position"])
    eef = pq["observation.states.end.position"].reshape(-1, 2, 3)
    Tt = eef.shape[0]
    segs = t.subtasks(EP)
    cand = [(float(np.linalg.norm(eef[min(int(s["end_frame"]), Tt - 1)] - eef[int(s["start_frame"])], axis=-1).max()),
             int(s["start_frame"])) for s in segs if int(s.get("end_frame", 0)) - int(s.get("start_frame", 0)) >= WIN]
    a = max(0, min(sorted(cand, reverse=True)[0][1], Tt - WIN - 1)) if cand else 0
    widx = np.clip(np.unique(np.linspace(a, a + WIN, K + 1).round().astype(int)), 0, Tt - 1)
    Kf = len(widx) - 1
    frames = np.asarray(t.decode_frames(EP, HEAD, widx.tolist()))           # [T,H,W,3] uint8
    H0, W0 = frames.shape[1:3]
    print(f"[stv2] window [{a},{a+WIN}] {W0}x{H0} Kf={Kf}", flush=True)

    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()     # [T,3,H,W] 0..255
    video_tensor = preprocess_image(video_tensor)[None]                     # [1,T,3,h,w]
    front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = front(video_tensor.cuda() / 255)
        extrinsic, intrinsic = pred["poses_pred"], pred["intrs"]
        depth_map, depth_conf = pred["points_map"][..., 2], pred["unc_metric"]
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    vt = video_tensor.squeeze()                                              # [T,3,h,w]
    h, w = vt.shape[2:]
    print(f"[stv2] front done: depth {depth_tensor.shape} (model res {w}x{h})", flush=True)
    del front
    torch.cuda.empty_cache()

    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").cuda().eval()
    model.spatrack.track_num = 756
    grid_pts = get_points_on_a_grid(27, (h, w), device="cpu")               # [1,Q,2]
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        (c2w_traj, intrs_o, point_map, conf_depth,
         track3d_pred, track2d_pred, vis_pred, conf_pred, video) = model.forward(
            vt, depth=depth_tensor, intrs=intrs, extrs=extrs, queries=query_xyt,
            fps=1, full_point=False, iters_track=4, query_no_BA=True, fixed_cam=False,
            stage=1, unc_metric=unc_metric, support_frame=Kf, replace_ratio=0.2)
    tr3 = track3d_pred.squeeze().float().cpu().numpy()                      # [T,Q,3+] world
    vis = vis_pred.squeeze().cpu().numpy() > 0.5                            # [T,Q]
    tr2 = track2d_pred.squeeze().float().cpu().numpy()                      # [T,Q,2+]
    print(f"[stv2] tracks: {tr3.shape} vis-rate {vis.mean():.2f}", flush=True)

    p3 = tr3[..., :3]
    disp = np.linalg.norm(p3[Kf] - p3[0], axis=-1)
    ok = vis.mean(0) > 0.6
    print(f"[stv2] Q={p3.shape[1]} ok={int(ok.sum())} | disp median={np.median(disp[ok])*100:.1f} "
          f"p90={np.percentile(disp[ok],90)*100:.1f} max={disp[ok].max()*100:.1f} (StV2 units)", flush=True)
    movers = ok & (disp > np.percentile(disp[ok], 85))

    np.savez(os.path.join(WS, "data/_agibot/stv2_tracks.npz"),
             track3d=p3, track2d=tr2[..., :2], vis=vis, widx=widx,
             intrs=intrs_o.squeeze().float().cpu().numpy(),
             c2w=c2w_traj.squeeze().float().cpu().numpy(), model_hw=(h, w))
    sx, sy = W0 / float(w), H0 / float(h)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].imshow(frames[0]); axes[0].axis("off")
    sc = axes[0].scatter(tr2[0, :, 0] * sx, tr2[0, :, 1] * sy, s=7, c=np.clip(disp * 100, 0, 30),
                         cmap="hot", linewidths=0)
    plt.colorbar(sc, ax=axes[0], fraction=0.03, label="3D disp (StV2 units x100)")
    axes[0].set_title("StV2 3D displacement (frame0)", fontsize=8)
    axes[1].imshow((frames[0].astype(np.float32) * 0.45).astype(np.uint8)); axes[1].axis("off")
    for q in np.where(movers)[0]:
        axes[1].plot(tr2[:, q, 0] * sx, tr2[:, q, 1] * sy, "-", lw=0.8, c="cyan", alpha=0.7)
    axes[1].set_title(f"top movers ({int(movers.sum())}) 2D paths", fontsize=8)
    plt.tight_layout()
    out = os.path.join(WS, "viz/agibot/r3_m2redo_stv2.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[stv2] saved {out}", flush=True)


if __name__ == "__main__":
    main()
