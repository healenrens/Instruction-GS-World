"""§54 VISUAL proof of language causality: same scene, render the MODEL's prediction under the TRUE
instruction vs a SWAP instruction. Cols: REAL | PRED(true) | PRED(swap). The named object should move
in PRED(true) and stay PUT in PRED(swap). Usage: _libero_langswap_video.py <ckpt> <clip> "<swap>" <out.mp4>"""
import sys; sys.path.insert(0, "code")
import torch, numpy as np, imageio.v2 as iio, cv2
from igsw.gaussians import GaussianSet, render_gaussianset
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import _to_dev
from scripts.eval_langswap import build_model, uniform_controls

ckpt, clip_p, swap, out_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = torch.load(clip_p, map_location="cuda", weights_only=False)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
mdl = build_model(ck)
g0 = GaussianSet(c["means"].cuda(), c["quats"].cuda(), c["scales"].cuda(), c["opacities"].cuda(), c["colors"].cuda(), None)
seg = c["seg_per_g"].cuda().long(); K = int(c["Kf"]); N = len(seg)
ci = uniform_controls(seg, N - int(c.get("n_fill", 0)), ck.get("M", 2048))
img0 = c["gt_rgb"][0].cpu().numpy()


def predict(instr):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(mdl.encoder.build_inputs(instr, img0), "cuda")
        out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=c["uv"].cuda()[ci],
                  control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg)
    return out["means"].float()   # [K,N,3] dense


pm_true = predict(c["instruction"])
pm_swap = predict(swap)
sc = c["scales"].cuda(); vm = c["viewmat"][None].cuda().float(); Ki = c["K_intr"][None].cuda().float()
W, H = int(c["W"]), int(c["H"])


def R(mn):
    col, _, _ = render_gaussianset(GaussianSet(mn, c["quats"].cuda(), sc, c["opacities"].cuda(), c["colors"].cuda(), None), vm, Ki, W, H)
    return (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def lbl(im, t):
    im = im.copy(); cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2); return im


frames = []
for t in range(K + 1):
    real = c["gt_rgb"][t].cpu().numpy()
    pt = R(pm_true[t - 1] if t > 0 else g0.means)
    ps = R(pm_swap[t - 1] if t > 0 else g0.means)
    frames.append(np.concatenate([lbl(real, f"REAL t{t}"), lbl(pt, "PRED: TRUE instr"), lbl(ps, "PRED: SWAP instr")], 1))
frames = [frames[0]] * 2 + frames + [frames[-1]] * 3
iio.mimwrite(out_p, frames, fps=2, quality=8)
print(f"TRUE={c['instruction']!r}\nSWAP={swap!r}\nsaved {out_p}")
