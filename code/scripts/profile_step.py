"""Profile the per-step phases of the streaming trainer to find the GPU bottleneck."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.data.streaming import StreamingClipDataset
from igsw.lifting import Pi3Lifter, points_to_gaussians
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at
from igsw.gaussians import GaussianSet, render_gaussianset, intrinsics_from_local_points, viewmat_from_pose
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from igsw.training.losses import trajectory_loss, photometric_loss

dev = "cuda"
def sync(): torch.cuda.synchronize()
def now(): sync(); return time.time()

K, M = 8, 2048
ckpt_flag = (len(sys.argv) < 2) or (sys.argv[1].lower() not in ("0", "false", "no"))
ce = int(sys.argv[2]) if len(sys.argv) > 2 else 1
lifter = Pi3Lifter(device=dev)
tracker = CoTrackerTracker(device=dev)
cfg = DynamicsConfig(d_model=1536, n_layers=28, n_heads=16, use_checkpoint=ckpt_flag, checkpoint_every=ce)
print(f"use_checkpoint={ckpt_flag} checkpoint_every={ce}")
model = InstructGSWorldModel(cfg, n_control=M, n_query=16).to(dev)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
ds = StreamingClipDataset(K=K, stride=3, seed=0)
it = iter(ds)
amp = torch.autocast("cuda", dtype=torch.bfloat16)

T = {k: [] for k in ["decode", "lift", "track", "encode", "rollout", "render", "backward"]}
for step in range(6):
    t = now(); clip = next(it); T["decode"].append(now() - t)
    frames = clip["frames"]
    t = now(); res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.03); T["lift"].append(now() - t)
    pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
    local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
    Kf1, _, H, W = imgs.shape
    Ks = torch.stack([intrinsics_from_local_points(local[i]) for i in range(Kf1)], 0)
    viewmats = torch.stack([viewmat_from_pose(poses[i]) for i in range(Kf1)], 0)
    gt_img = imgs.permute(0, 2, 3, 1)
    g0, uv = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, return_uv=True)
    N = len(g0); ci = torch.randperm(N, device=dev)[:M]
    t = now(); tracks, vis = tracker.track(imgs, uv[ci]); gt_pos = sample_pointmaps_at(pts, tracks); T["track"].append(now() - t)
    init = g0.means[ci]; gt_traj = gt_pos[1:K + 1]; vis_traj = vis[1:K + 1]
    frame0 = (gt_img[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    vlm = model.encoder.build_inputs(clip["instruction"], frame0)
    vlm = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vlm.items()}
    opt.zero_grad(set_to_none=True)
    with amp:
        t = now()
        ctx, m, cg = model.encode(vlm); T["encode"].append(now() - t)
        t = now(); out = model(vlm, g0, K, ctrl_idx=ci); T["rollout"].append(now() - t)
    pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float())
    t = now()
    rloss = out["ctrl"].new_zeros(())
    for k in [0, K - 1]:
        s = GaussianSet(out["means"][k].float(), out["quats"][k].float(), out["scales"][k].float(),
                        out["opacities"][k].float(), out["colors"][k].float(), None)
        c, _, _ = render_gaussianset(s, viewmats[k + 1], Ks[k + 1], W, H)
        pl, _, _ = photometric_loss(c[0], gt_img[k + 1]); rloss = rloss + pl
    T["render"].append(now() - t)
    total = pos_l + vel_l + 0.1 * rloss
    t = now(); total.backward(); opt.step(); T["backward"].append(now() - t)
    print(f"step {step} N={N} done")

print("\n=== mean phase times (s), skipping step0 (warmup) ===")
tot = 0
for k, v in T.items():
    m = float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))
    tot += m
    print(f"  {k:9s} {m:.3f}")
print(f"  {'TOTAL':9s} {tot:.3f}  => {1/tot:.2f} it/s/GPU")
print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
