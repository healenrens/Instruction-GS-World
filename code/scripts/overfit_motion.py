"""DECISIVE capacity test: can the dynamics produce LOCALIZED motion at all?
Overfit the model on a SINGLE fixed clip for N steps. If corr(GT,PRED) -> high and the
top-mover reproduction ratio -> ~1, the architecture IS capable of localized prediction
(so the full-training under-prediction is a GT-noise / optimization / generalization issue,
e.g. fix via cleaner GT). If it CANNOT overfit even one clip, the per-control pathway is
fundamentally too weak (-> architecture fix). 1.76B params on one clip should memorize easily."""
import numpy as np, torch, sys, math
sys.path.insert(0, "code")
from igsw.data.streaming import StreamingClipDataset
from igsw.lifting import Pi3Lifter, points_to_gaussians
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from igsw.training.losses import trajectory_loss, rotation_loss

dev = "cuda"; K = 16; stride = 8; STEPS = 600
SPATIAL = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False   # arg2: 1 = per-control spatial grounding (agent.md §37)
ck = torch.load("checkpoints/stream11c_infonce/ckpt_0006000.pt", map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"]); M = ck.get("M", 2048)
model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16), cond_mode="aggregator",
                             spatial_ground=SPATIAL).to(dev)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
print("spatial_ground=%s | load_state_dict missing=%d unexpected=%d (new spatial layers re-init)" % (
    SPATIAL, len(missing), len(unexpected)), flush=True)
lifter = Pi3Lifter(device=dev); tracker = CoTrackerTracker(device=dev)
ds = StreamingClipDataset(K=K, stride=stride, seed=777, boundary_ratio=4.0, load_actions=False)
it = iter(ds)
c = next(it)
while not (c.get("instruction") or "").strip():
    c = next(it)
# ---- prep ONE clip once ----
res = lifter.lift(c["frames"], conf_thr=0.1, edge_rtol=0.03)
pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
g0, uv = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0, return_uv=True)
N = len(g0)
img_hw = (imgs.shape[-2], imgs.shape[-1])     # lifted-image (H,W) that uv is in -> normalize for grid_sample
gen = torch.Generator(device=dev).manual_seed(0)
ctrl_idx = torch.randperm(N, device=dev, generator=gen)[:M]
control_uv = uv[ctrl_idx]                       # [M,2] frame-0 (x=col,y=row) pixel coords of the controls
tracks, vis = tracker.track(imgs, uv[ctrl_idx])
gt_pos = sample_pointmaps_at(pts, tracks)                       # [K+1,M,3]
gt_traj = gt_pos[1:K + 1]; vis_traj = vis[1:K + 1]
init = g0.means[ctrl_idx]
radius = (g0.means - g0.means.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius
clean = g0.means[ctrl_idx]
knn_idx = torch.cdist(clean, clean).topk(9, largest=False).indices[:, 1:]
gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
img0 = (imgs[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
vi = model.encoder.build_inputs(c["instruction"], img0)
vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
          else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}
print("clip '%s' | N=%d M=%d GT frac>0.02=%.3f top5%%=%.3f" % (
    c["instruction"][:40], N, M, (gt_disp > 0.02).float().mean().item(),
    gt_disp.topk(M // 20).values.mean().item()))
# Higher LR on the freshly-initialized spatial-grounding params so they bootstrap fast and
# win over the already-trained head/AdaLN (which otherwise re-collapses to uniform translation).
sg_names = ("vis_tok", "vis_film", "vis_norm", "vis_vhead")
sg_params = [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sg_names)]
base_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sg_names)]
if SPATIAL and sg_params:
    opt = torch.optim.AdamW([{"params": base_params, "lr": 3e-4},
                             {"params": sg_params, "lr": 1e-3}])
    print("opt: base lr=3e-4 (%d tensors) | spatial lr=1e-3 (%d tensors)" % (len(base_params), len(sg_params)), flush=True)
else:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
amp = torch.autocast("cuda", dtype=torch.bfloat16)
gt_full = torch.cat([init[None], gt_traj], 0)
for s in range(STEPS + 1):
    with amp:
        out = model(vi, g0, K, ctrl_idx=ctrl_idx,
                    control_uv=control_uv, control_uv_hw=img_hw)
    OBJF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # 0 = plain L1 (clean capacity test)
    pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                   relevance=rel, obj_focus=OBJF)
    rot_l = rotation_loss(out["om"].float(), gt_full.float(), knn_idx, torch.cat([vis[:1], vis_traj], 0))
    loss = pos_l + vel_l + 0.2 * rot_l
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(gn):
        opt.step()
    if s % 50 == 0:
        with torch.no_grad():
            pd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius
            corr = torch.corrcoef(torch.stack([gt_disp.float(), pd.float()]))[0, 1]
            topk = gt_disp.topk(M // 20).indices
            print("step %3d  pos%.4f  PRED p50=%.3f p90=%.3f max=%.3f  corr=%.3f  topmover_ratio=%.2f" % (
                s, pos_l.item(), pd.median().item(), pd.quantile(0.9).item(), pd.max().item(),
                corr.item(), (pd[topk].mean() / gt_disp[topk].mean().clamp_min(1e-6)).item()), flush=True)
