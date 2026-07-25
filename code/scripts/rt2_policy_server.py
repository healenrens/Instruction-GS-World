"""VLA policy HTTP server (runs in instruct_gs_world/.venv with the v2 model). The RoboTwin2 rollout
client POSTs one observation; the causal-v1 path reconstructs GPSTokens from the current head RGB with
single-frame VGGT on the same fixed grid used for training.

  POST /act  {rgb_b64, depth_b64, K:[9], qpos:[14], instruction:str, policy_seed:int}
        ->   {action: [[14] x chunk] absolute qpos targets, viz_png_b64: <3D-pred overlay>}

Launch (our venv, 1 GPU):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/rt2_policy_server.py \
    --ckpt checkpoints/vla_causal_v1_from20k_2n8g/vla_050000.pt \
    --port 9010 --wrist 1 --placement entropy
"""
import os, sys, io, json, base64, argparse

for _key, _value in {
    "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
}.items():
    os.environ.setdefault(_key, _value)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from train_vla import make_model, mv_in, placement_saliency
from igsw.causal_geometry import (CAUSAL_GEOMETRY_VERSION, causal_geometry_from_prediction,
                                  preprocessed_rgb)
from igsw.gpstoken_wm import place_tokens

DEV = "cuda"
POLICY_SEED_MODE = "request_v1"
G = SimpleNamespace(model=None, enc=None, args=None, meta=None, geometry_model=None,
                    geometry_preprocess=None)


def _b64_to_np(b64):
    return np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)


def _np_to_b64(a):
    buf = io.BytesIO(); np.save(buf, a); return base64.b64encode(buf.getvalue()).decode()


def _bilinear_sample(image, xs, ys):
    """Sample a 2D numpy image at floating-point pixel coordinates."""
    h, w = image.shape
    x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1); y0 = np.clip(y0, 0, h - 1)
    wx = xs - x0; wy = ys - y0
    return ((1 - wx) * (1 - wy) * image[y0, x0] + wx * (1 - wy) * image[y0, x1]
            + (1 - wx) * wy * image[y1, x0] + wx * wy * image[y1, x1])


def _geometry_samples(rgb, depth_mm, K, mode):
    """Return the head image, intrinsics, and depth samples used to lift token candidates."""
    rgb0 = np.ascontiguousarray(np.asarray(rgb).astype(np.uint8))
    if mode == "vggt_t1_grid48":
        frame = torch.from_numpy(rgb0).permute(2, 0, 1).float()[None]
        video = G.geometry_preprocess(frame)[None]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = G.geometry_model(video.cuda() / 255.0)
        geom = causal_geometry_from_prediction(pred["points_map"], pred["intrs"], 48)
        uv = geom["uv"].detach().cpu().numpy()
        depth = geom["means"][:, 2].detach().cpu().numpy()
        return (preprocessed_rgb(video).cpu().numpy(), geom["K_intr"].detach().cpu().numpy(),
                uv[:, 0], uv[:, 1], depth)

    depth_m = np.asarray(depth_mm, np.float32) / 1000.0
    K_out = np.asarray(K, np.float32).copy()

    if mode == "raw_random6000":
        ys, xs = np.nonzero(depth_m > 1e-3)
        if len(xs) > 6000:
            sel = np.random.RandomState(0).choice(len(xs), 6000, replace=False)
            xs, ys = xs[sel], ys[sel]
        return rgb0, K_out, xs.astype(np.float32), ys.astype(np.float32), depth_m[ys, xs]

    if mode != "train_grid48_simdepth":
        raise ValueError(f"unknown obs_preprocess mode: {mode!r}")

    # SpaTracker training clips resize the 320x240 head view to 518x392 and start from a 48x48
    # regular query grid. Keep that token layout while lifting causally from simulator depth.
    from PIL import Image
    import torch.nn.functional as F
    src_h, src_w = rgb0.shape[:2]
    out_h, out_w = 392, 518
    rgb0 = np.asarray(Image.fromarray(rgb0).resize((out_w, out_h)), dtype=np.uint8)
    depth_m = F.interpolate(torch.from_numpy(depth_m)[None, None], size=(out_h, out_w),
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    K_out[0, 0] *= out_w / src_w; K_out[0, 2] *= out_w / src_w
    K_out[1, 1] *= out_h / src_h; K_out[1, 2] *= out_h / src_h

    grid_size = 48
    margin = out_w // 64
    gy, gx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    ys = margin + gy.reshape(-1) / float(grid_size - 1) * (out_h - 2 * margin)
    xs = margin + gx.reshape(-1) / float(grid_size - 1) * (out_w - 2 * margin)
    d = _bilinear_sample(depth_m, xs, ys).astype(np.float32)
    valid = d > 1e-3
    return rgb0, K_out, xs[valid].astype(np.float32), ys[valid].astype(np.float32), d[valid]


def build_obs_batch(rgb, depth_mm, K, qpos, instruction, left_rgb=None, right_rgb=None):
    """Reconstruct the inference dict from one current observation. Causal-v1 uses current-RGB VGGT;
    legacy diagnostic modes use simulator depth. Entropy placement then selects the GPSTokens.
    When --wrist and left/right rgb are provided, [head,left,right] are fed to Qwen (head image #0 -> 3D
    unchanged; wrist enriches context), matching the --wrist training."""
    args, enc = G.args, G.enc
    rgb0, K_obs, xs, ys, d = _geometry_samples(rgb, depth_mm, K, args.obs_preprocess)
    H, W = rgb0.shape[:2]
    if len(xs) == 0:
        raise ValueError("no valid depth")
    fx, fy, cx, cy = (float(K_obs[0, 0]), float(K_obs[1, 1]),
                      float(K_obs[0, 2]), float(K_obs[1, 2]))
    X = (xs - cx) * d / fx; Y = (ys - cy) * d / fy; Z = d
    means = torch.tensor(np.stack([X, Y, Z], 1), dtype=torch.float32, device=DEV)   # [N,3] camera frame
    uv = torch.tensor(np.stack([xs, ys], 1), dtype=torch.float32, device=DEV)        # [N,2]
    n_keep = means.shape[0]
    imgs = rgb0
    if getattr(args, "wrist", 0):
        if left_rgb is None or right_rgb is None:
            raise ValueError("wrist checkpoint requires both left_rgb and right_rgb")
        imgs = [rgb0, np.ascontiguousarray(np.asarray(left_rgb).astype(np.uint8)),
                np.ascontiguousarray(np.asarray(right_rgb).astype(np.uint8))]   # head #0 -> 3D head-only
    vlm0 = mv_in(enc.build_inputs(instruction, imgs), DEV)
    sal = placement_saliency(args, enc, vlm0, uv, None, n_keep, H, W)     # entropy -> None (no GT)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, DEV, sal=sal, beta=args.beta)
    center = means[:n_keep].mean(0, keepdim=True)
    radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
    b = {
        "vlm0": vlm0, "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
        "tok_xyz0": means[idx], "center": center, "radius": radius,
        "K_intr": torch.tensor(K_obs, device=DEV),
        "viewmat": torch.eye(4, device=DEV), "H": H, "W": W, "rgb0_np": rgb0,
        "anchor": torch.tensor(np.asarray(qpos, np.float32), device=DEV),
    }
    return b


def viz_3d(rgb, b, xyz1_pred):
    """Overlay GPSToken centers (frame0) + arrows to the predicted future 2D position (project xyz1_pred
    via K, viewmat=I). Returns a PNG (uint8 RGB) as bytes."""
    import cv2
    K = b["K_intr"].cpu().numpy()
    cen = b["cen"].cpu().numpy()                                          # [M,2] frame0 uv
    p = xyz1_pred.cpu().numpy()                                           # [M,3] camera frame
    z = np.clip(p[:, 2], 1e-3, None)
    u1 = K[0, 0] * p[:, 0] / z + K[0, 2]; v1 = K[1, 1] * p[:, 1] / z + K[1, 2]
    img = np.ascontiguousarray(rgb[..., ::-1].copy())                    # RGB->BGR for cv2
    for i in range(cen.shape[0]):
        x0, y0 = int(cen[i, 0]), int(cen[i, 1]); x1, y1 = int(u1[i]), int(v1[i])
        mag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        col = (0, 0, 255) if mag > 1.5 else (0, 200, 0)                  # red = moving, green = static
        cv2.circle(img, (x0, y0), 1, col, -1)
        if mag > 1.5:
            cv2.arrowedLine(img, (x0, y0), (x1, y1), (0, 0, 255), 1, tipLength=0.3)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def act(rgb, depth_mm, K, qpos, instruction, policy_seed, left_rgb=None, right_rgb=None):
    policy_seed = int(policy_seed)
    if policy_seed < 0 or policy_seed >= 2 ** 63:
        raise ValueError(f"policy_seed must be in [0, 2**63), got {policy_seed}")
    b = build_obs_batch(rgb, depth_mm, K, qpos, instruction, left_rgb, right_rgb)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dq, xyz1 = G.model.rollout_predict(b)                            # dq [A,14] raw Δqpos, xyz1 [M,3]
    dq = dq.float().cpu().numpy()
    anchor = np.asarray(qpos, np.float32)
    qpos_abs = anchor[None] + np.cumsum(dq, axis=0)                      # [A,14] absolute
    qpos_abs[:, [6, 13]] = np.clip(qpos_abs[:, [6, 13]], 0.0, 1.0)       # grippers in [0,1]
    viz_png = viz_3d(b["rgb0_np"], b, xyz1)
    return qpos_abs.astype(np.float32), viz_png


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, status, payload):
        out = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, G.meta)

    def do_POST(self):
        if self.path != "/act":
            self._send_json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            rgb = _b64_to_np(req["rgb_b64"]); depth = _b64_to_np(req["depth_b64"])
            K = np.asarray(req["K"], np.float32).reshape(3, 3)
            left = _b64_to_np(req["left_b64"]) if req.get("left_b64") else None     # optional wrist views
            right = _b64_to_np(req["right_b64"]) if req.get("right_b64") else None
            policy_seed = int(req["policy_seed"])
            action, viz = act(rgb, depth, K, req["qpos"], req.get("instruction", ""),
                              policy_seed, left, right)
            self._send_json(200, {"action": action.tolist(),
                                  "viz_png_b64": base64.b64encode(viz).decode(),
                                  "policy_seed": policy_seed})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send_json(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=9010)
    ap.add_argument("--norm_stats", default=os.path.join(PROJECT_ROOT, "data/rt2_act/norm_stats.pt"))
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--placement", default="entropy")
    ap.add_argument("--obs_preprocess",
                    choices=["vggt_t1_grid48", "raw_random6000", "train_grid48_simdepth"],
                    default="vggt_t1_grid48", help="causal-v1 checkpoints require vggt_t1_grid48; legacy "
                    "modes remain only for old diagnostic checkpoints")
    ap.add_argument("--spatrack_root", default="/mnt/pfs/public/xuhaoming/SpaTrackerV2")
    ap.add_argument("--wrist", type=int, default=0, help="1 = feed left+right wrist views to Qwen (use ONLY "
                    "with a --wrist-trained ckpt; client must POST left_b64/right_b64). head stays image #0.")
    a = ap.parse_args()
    args = SimpleNamespace(geom_mode="xyz", fdim=128, feat_source="qwen", dino_imgsize=518, traj_pred=0, wrist=a.wrist,
                           img_loss=1, w_depth=0.5, cam_cond=0, L=a.L, beta=a.beta, init_from="",
                           norm_stats=a.norm_stats, action_dim=14, action_steps=50, d_act=704, n_heads_act=11,
                           n_state_tokens=1, mlp_ratio=4.0, w_flow=1.0, w_act=1.0, placement=a.placement,
                           obs_preprocess=a.obs_preprocess,
                           causal_geometry_version=(CAUSAL_GEOMETRY_VERSION
                                                    if a.obs_preprocess == "vggt_t1_grid48" else ""))
    ckpt = torch.load(a.ckpt, map_location=DEV, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    expected = {
        "wrist": a.wrist, "placement": a.placement, "L": a.L, "geom_mode": "xyz",
        "img_loss": 1, "action_steps": 50, "d_act": 704, "n_heads_act": 11,
        "n_state_tokens": 1,
    }
    if a.obs_preprocess == "vggt_t1_grid48":
        expected["causal_geometry_version"] = CAUSAL_GEOMETRY_VERSION
    mismatches = {k: {"checkpoint": ckpt_args.get(k), "runtime": v}
                  for k, v in expected.items() if ckpt_args.get(k) != v}
    ckpt_geometry = ckpt_args.get("causal_geometry_version", "")
    if ckpt_geometry != args.causal_geometry_version:
        mismatches["causal_geometry_version"] = {
            "checkpoint": ckpt_geometry, "runtime": args.causal_geometry_version}
    if mismatches:
        raise ValueError(f"checkpoint/runtime config mismatch: {mismatches}")

    probe = sorted(__import__("glob").glob(os.path.join(PROJECT_ROOT, "data/rt2_joint/*_train.pt")))
    Kf = int(torch.load(probe[0], map_location="cpu", weights_only=False)["Kf"]) if probe else 12
    model = make_model(args, DEV, Kf=Kf)
    sd = ckpt["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    non_encoder_miss = [m for m in miss if not m.startswith("encoder.")]
    if non_encoder_miss or unexp:
        raise ValueError(f"checkpoint state mismatch: missing={non_encoder_miss} unexpected={unexp}")
    print(f"[server] loaded {a.ckpt}: {len(sd)} tensors, non-encoder-missing="
          f"{len(non_encoder_miss)} unexpected={len(unexp)}", flush=True)
    model.eval()
    if a.obs_preprocess == "vggt_t1_grid48":
        sys.path.insert(0, a.spatrack_root)
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
        G.geometry_model = VGGT4Track.from_pretrained(
            "Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
        G.geometry_preprocess = preprocess_image
    G.model, G.enc, G.args, G.meta = model, model.encoder, args, {
        "status": "ok", "checkpoint": os.path.abspath(a.ckpt), "step": int(ckpt["step"]),
        "wrist": a.wrist, "placement": a.placement, "L": a.L, "action_steps": args.action_steps,
        "model_tensors": len(sd), "obs_preprocess": a.obs_preprocess,
        "causal_geometry_version": args.causal_geometry_version,
        "policy_seed_mode": POLICY_SEED_MODE,
    }
    print(f"[server] ready on :{a.port} {json.dumps(G.meta, sort_keys=True)}", flush=True)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
