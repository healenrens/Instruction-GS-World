"""Diagnostic: trace per-control variance through the dynamics with spatial grounding.
Where does the per-control visual signal die? Check: vis_tok output variance, pre-head
feature variance, and whether the head CAN map a per-control input to per-control output."""
import numpy as np, torch, sys
sys.path.insert(0, "code")
from igsw.data.streaming import StreamingClipDataset
from igsw.lifting import Pi3Lifter, points_to_gaussians
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at
from igsw.dynamics.model import DynamicsConfig, GaussianState
from igsw.model_full import InstructGSWorldModel

dev = "cuda"; K = 16; stride = 8
ck = torch.load("checkpoints/stream11c_infonce/ckpt_0006000.pt", map_location="cpu", weights_only=False)
cfg = DynamicsConfig(**ck["cfg"]); M = ck.get("M", 2048)
model = InstructGSWorldModel(cfg, n_control=M, n_query=ck.get("n_query", 16), cond_mode="aggregator",
                             spatial_ground=True).to(dev)
model.load_state_dict(ck["model"], strict=False)
model.eval()

# ---- head / ada non-zero? (does the resumed ckpt carry trained modulation -> film grad flows) ----
hw = model.dynamics.head.weight
print("HEAD weight: norm=%.4f  is_zero=%s" % (hw.norm().item(), bool(hw.abs().sum()==0)))
ada0 = model.dynamics.blocks[0].ada[-1].weight
print("ada[0] last weight: norm=%.4f  is_zero=%s" % (ada0.norm().item(), bool(ada0.abs().sum()==0)))

lifter = Pi3Lifter(device=dev); tracker = CoTrackerTracker(device=dev)
ds = StreamingClipDataset(K=K, stride=stride, seed=777, boundary_ratio=4.0, load_actions=False)
it = iter(ds); c = next(it)
while not (c.get("instruction") or "").strip():
    c = next(it)
res = lifter.lift(c["frames"], conf_thr=0.1, edge_rtol=0.03)
pts = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
g0, uv = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0, return_uv=True)
N = len(g0); img_hw = (imgs.shape[-2], imgs.shape[-1])
gen = torch.Generator(device=dev).manual_seed(0)
ctrl_idx = torch.randperm(N, device=dev, generator=gen)[:M]
control_uv = uv[ctrl_idx]
img0 = (imgs[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
vi = model.encoder.build_inputs(c["instruction"], img0)
vi = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
          else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in vi.items()}

with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
    # 1) per-control visual feature variance
    vt, vf, vh = model._control_visual(vi, control_uv, img_hw, dev, torch.bfloat16)
    print("\nvis_tok   out [M,d]:  per-control std (avg over d)=%.4f  (0 => no per-control signal)" %
          vt.float().std(0).mean().item())
    print("vis_film  out [M,2d]: per-control std=%.4f" % vf.float().std(0).mean().item())
    print("vis_vhead out [M,3]:  per-control std=%.4f  (zero at init -> warm-start identity)" %
          vh.float().std(0).mean().item())

    # 2) trace pre-head feature variance: monkeypatch final_norm to capture input
    captured = {}
    fn = model.dynamics.final_norm
    def hook(mod, inp, out): captured["x"] = inp[0].detach()
    h = fn.register_forward_hook(hook)
    out = model(vi, g0, K, ctrl_idx=ctrl_idx, control_uv=control_uv, control_uv_hw=img_hw)
    h.remove()
    x = captured["x"][0].float()    # [M,d] pre-head features at last rollout step
    print("\npre-head x [M,d]: per-control std=%.4f  (if ~0 -> self-attn homogenized the tokens)" % x.std(0).mean().item())
    v = out["v"][K-1].float()       # last-step control velocity [M,3]
    print("output v [M,3]: per-control std=%.4f  (the actual per-control motion spread)" % v.std(0).mean().item())

# 3) CAPABILITY of the head alone: feed RANDOM strongly-per-control features into final_norm+head
with torch.no_grad():
    xr = torch.randn(M, cfg.d_model, device=dev)
    vr = model.dynamics.head(model.dynamics.final_norm(xr))[..., :3]
    print("\nHEAD capability: random per-control input -> output per-control std=%.4f "
          "(if >0, the head CAN produce localized output; the bottleneck is upstream)" % vr.float().std(0).mean().item())
print("\n[diag done]")
