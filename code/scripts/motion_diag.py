"""Diagnose WHY the predicted rollout barely moves: compare GT control-point motion vs the
model's PREDICTED control-point motion, per control. Answers: does the GT trajectory have
localized motion (arm/object) that the model UNDER-predicts (loss dilution), or do the
randomly-sampled controls miss the motion entirely (sampling)?"""
import numpy as np, torch, sys
sys.path.insert(0, "code")
from igsw.data.streaming import StreamingClipDataset
from igsw.lifting import Pi3Lifter, points_to_gaussians
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel

dev = "cuda"; K = 16; stride = 8
_ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stream13_anchor/ckpt_0008000.pt"
_cond = sys.argv[2] if len(sys.argv) > 2 else "aggregator"
print("CKPT", _ckpt, "cond_mode", _cond)
ck = torch.load(_ckpt, map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"]); M = ck.get("M", 2048)
model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16), cond_mode=_cond).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
lifter = Pi3Lifter(device=dev); tracker = CoTrackerTracker(device=dev)
ds = StreamingClipDataset(K=K, stride=stride, seed=777, boundary_ratio=4.0, load_actions=False)
it = iter(ds)

def pct(x):
    x = x.float().cpu().numpy()
    return "p50=%.4f p90=%.4f p99=%.4f max=%.4f" % (np.percentile(x, 50), np.percentile(x, 90), np.percentile(x, 99), x.max())

n = 0
while n < 4:
    c = next(it)
    if not (c.get("instruction") or "").strip():
        continue
    res = lifter.lift(c["frames"], conf_thr=0.1, edge_rtol=0.03)
    pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
    g0, uv = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0, return_uv=True)
    N = len(g0)
    if N < M:
        continue
    gen = torch.Generator(device=dev).manual_seed(0)
    ctrl_idx = torch.randperm(N, device=dev, generator=gen)[:M]
    tracks, vis = tracker.track(imgs, uv[ctrl_idx])
    gt_pos = sample_pointmaps_at(pts, tracks)                          # [K+1,M,3]
    radius = (g0.means - g0.means.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius            # [M] GT displacement (frac radius)
    if getattr(cfg, "feature_dim", 0) > 0:   # feed GT-motion relevance as input (matches training)
        from igsw.gaussians import GaussianSet
        gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
        rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
        feat = g0.means.new_zeros(len(g0), cfg.feature_dim); feat[ctrl_idx, 0] = rel
        g0 = GaussianSet(g0.means, g0.quats, g0.scales, g0.opacities, g0.colors, feat)
    img0 = (imgs[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    vi = model.encoder.build_inputs(c["instruction"], img0)
    vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
              else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(vi, g0, K, ctrl_idx=ctrl_idx)
    pred_disp = (out["ctrl"][K - 1].float() - g0.means[ctrl_idx]).norm(dim=-1) / radius   # [M]
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pred_disp.float()]))[0, 1]
    # how much of the GT motion magnitude does the model reproduce, on the top-moving controls?
    topk = gt_disp.topk(max(1, M // 20)).indices                       # top 5% movers (the arm/object)
    print("clip%d  '%s'" % (n, (c["instruction"] or "")[:44]))
    print("  GT   disp:", pct(gt_disp), " frac>0.02=%.3f" % (gt_disp > 0.02).float().mean().item())
    print("  PRED disp:", pct(pred_disp), " frac>0.02=%.3f" % (pred_disp > 0.02).float().mean().item())
    print("  corr(GT,PRED)=%.3f | on GT-top5%% movers: GT mean=%.4f PRED mean=%.4f (ratio %.2f)" % (
        corr.item(), gt_disp[topk].mean().item(), pred_disp[topk].mean().item(),
        (pred_disp[topk].mean() / gt_disp[topk].mean().clamp_min(1e-6)).item()))
    n += 1
