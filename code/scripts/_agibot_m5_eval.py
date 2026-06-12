"""§79 R3 m5: ZERO-SHOT manipulator eval on real AgiBot — the model (trained on LIBERO sim) predicts
on the real supermarket clip; its gripper-region motion, converted to METERS via the m4 scale, is
judged against the REAL EEF proprioception. Honest distribution-shift baseline vs the static baseline.

Pipeline: frames -> Pi3 g0 -> openvocab seg -> v11_rigid forward (instruction) -> gripper-box controls'
predicted displacement -> x m4-scale -> compare with EEF arm1 displacement (magnitude + direction after
the m4 similarity alignment maps EEF into the Pi3 frame)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from igsw.data.lerobot_agibot import list_tasks, AgiBotLeRobotTask          # noqa: E402
from igsw.lifting.pi3_lifter import Pi3Lifter                               # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians                   # noqa: E402
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at     # noqa: E402
from openvocab_seg import load_models, _gd_detect, segment_frame_amg        # noqa: E402
from scripts.eval_langswap import build_model, uniform_controls             # noqa: E402
from scripts.eval_sim_generalization import _to_dev                         # noqa: E402
from scripts._agibot_eefcal import umeyama                                  # noqa: E402
from igsw.gaussians import GaussianSet                                      # noqa: E402
from PIL import Image                                                       # noqa: E402

TASK, EP, WIN, K = "task_327", 0, 48, 12
CKPT = "checkpoints/libero_v11_rigid/ckpt_last.pt"
HEAD = "observation.images.head"


def main():
    dev = "cuda"
    root = next(r for r in list_tasks() if r.rstrip("/").endswith(TASK))
    t = AgiBotLeRobotTask(root)
    pq = t.read_parquet(EP, ["observation.states.end.position"])
    eef = pq["observation.states.end.position"].reshape(-1, 2, 3)
    Tt = eef.shape[0]
    segs = t.subtasks(EP)
    cand = [(float(np.linalg.norm(eef[min(int(s["end_frame"]), Tt - 1)] - eef[int(s["start_frame"])], axis=-1).max()),
             int(s["start_frame"]), s.get("action_text", "")) for s in segs
            if int(s.get("end_frame", 0)) - int(s.get("start_frame", 0)) >= WIN]
    _, a, instr = sorted(cand, reverse=True)[0]
    a = max(0, min(a, Tt - WIN - 1))
    widx = np.clip(np.unique(np.linspace(a, a + WIN, K + 1).round().astype(int)), 0, Tt - 1)
    Kf = len(widx) - 1
    frames = np.asarray(t.decode_frames(EP, HEAD, widx.tolist()))
    H0, W0 = frames.shape[1:3]
    print(f"[m5] window [{a},{a+WIN}] instr={instr[:60]!r}")

    lifter = Pi3Lifter(device=dev)
    res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.0)
    local = res["local_points"].numpy()
    poses = res["camera_poses"].numpy().astype(np.float64)
    imgs = res["images"]
    Hm, Wm = local.shape[1], local.shape[2]
    rel = np.stack([np.linalg.inv(poses[0]) @ poses[i] for i in range(Kf + 1)]).astype(np.float32)
    canon = (np.einsum("tij,thwj->thwi", rel[:, :3, :3], local) + rel[:, None, None, :3, 3]).astype(np.float32)
    z0 = local[0][..., 2].astype(np.float32)
    valid = np.isfinite(local[0]).all(-1) & (z0 > 1e-4) & (res["conf"].numpy()[0] > 0.1)
    g0, uv = points_to_gaussians(torch.from_numpy(local[0]).float()[None], imgs[0:1],
                                 torch.from_numpy(valid)[None], opacity_init=0.9,
                                 scale_factor=0.6, return_uv=True)
    g0 = g0.to(dev); uv = uv.to(dev)
    N = len(g0)

    # openvocab seg at frame-0 -> seg_per_g via uv (model px -> original px)
    idm = segment_frame_amg(frames[0], instr, device=dev)                  # [H0,W0] ids
    uv_o = (uv.cpu().numpy() * np.array([W0 / float(Wm), H0 / float(Hm)]))
    ui = np.clip(uv_o[:, 0].round().astype(int), 0, W0 - 1)
    vi = np.clip(uv_o[:, 1].round().astype(int), 0, H0 - 1)
    seg = torch.from_numpy(idm[vi, ui].astype(np.int64)).to(dev)
    print(f"[m5] N={N} | seg ids: {dict(zip(*[x.tolist() for x in torch.unique(seg, return_counts=True)]))}")

    # m4 scale: per-track consensus on the RIGHT-arm gripper box
    models = load_models(dev)
    dets = _gd_detect(models, Image.fromarray(frames[0]), ["robot gripper", "robotic arm"], 0.2, 0.2)
    boxes = [d["box"] for d in dets if "gripper" in d["label"]][:2]
    ct = CoTrackerTracker(device=dev)
    frT = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    best = None
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b]
        gy, gx = torch.meshgrid(torch.linspace(y0 + 5, y1 - 5, 5), torch.linspace(x0 + 5, x1 - 5, 5), indexing="ij")
        tracks, vis = ct.track(frT, torch.stack([gx.flatten(), gy.flatten()], -1).to(dev))
        tr_m = tracks.cpu() * torch.tensor([Wm / float(W0), Hm / float(H0)])
        p3 = sample_pointmaps_at(torch.from_numpy(canon), tr_m).numpy()
        ok = np.isfinite(p3).all(-1).all(0) & (vis.cpu().numpy().mean(0) > 0.5)
        zmed0 = float(np.nanmedian(canon[0][..., 2]))
        for arm in (0, 1):
            E = eef[widx, arm]
            disp = float(np.linalg.norm(E[-1] - E[0]))
            if disp < 0.03:
                continue
            for qi in np.where(ok)[0]:
                s, R, tt, resid = umeyama(p3[:, qi], E)
                if resid / disp < 0.15 and 0.3 < zmed0 * s < 5.0:
                    if best is None or resid / disp < best[0]:
                        best = (resid / disp, s, R, tt, arm, b)
    if best is None:
        print("[m5] FAIL: no sane scale fit"); return
    relr, s, R, tt, arm, gbox = best
    print(f"[m5] scale={s:.3f} (rel {relr*100:.0f}%) arm{arm} gripper-box={[int(v) for v in gbox]}")

    # model forward (zero-shot)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mdl = build_model(ck)
    ci = uniform_controls(seg, N, ck.get("M", 2048))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vinp = _to_dev(mdl.encoder.build_inputs(instr, frames[0]), "cuda")
        out = mdl(vinp, g0, K, ctrl_idx=ci, control_uv=uv[ci],
                  control_uv_hw=(Hm, Wm), seg_per_g=seg)
    epK = out["ctrl"][K - 1].float()
    init = g0.means[ci]
    seg_c = seg[ci]

    # gripper-region controls = seg==8 (robot) AND inside the gripper box (original px)
    uvo_c = uv_o[ci.cpu().numpy()]
    x0, y0, x1, y1 = [float(v) for v in gbox]
    inbox = (uvo_c[:, 0] >= x0) & (uvo_c[:, 0] <= x1) & (uvo_c[:, 1] >= y0) & (uvo_c[:, 1] <= y1)
    gm = (seg_c == 8).cpu().numpy() & inbox
    print(f"[m5] gripper-region controls: {int(gm.sum())}")
    if gm.sum() < 5:
        gm = inbox
    pred_d = (epK[gm] - init[gm]).mean(0).cpu().numpy()                    # Pi3 gauge
    pred_m = pred_d * s                                                     # meters (rotation-free magnitude)
    E = eef[widx, arm]
    gt_d = E[-1] - E[0]                                                     # meters, base frame
    gt_in_pi3 = (R.T @ gt_d) / s                                            # direction comparison in Pi3 frame
    cos = float(np.dot(pred_d, gt_in_pi3) / (np.linalg.norm(pred_d) * np.linalg.norm(gt_in_pi3) + 1e-9))
    print(f"\n[m5] ZERO-SHOT manipulator vs REAL EEF (meters):")
    print(f"  GT  |ΔEEF|   = {np.linalg.norm(gt_d)*100:.1f}cm")
    print(f"  PRED |Δgrip| = {np.linalg.norm(pred_m)*100:.1f}cm   (mag-ratio {np.linalg.norm(pred_m)/max(np.linalg.norm(gt_d),1e-9):.2f}x)")
    print(f"  direction cos (Pi3 frame) = {cos:+.2f}")
    pred_in_base = s * (R @ pred_d)                                         # Pi3 -> meters, base frame
    print(f"  EPE3D endpoint = {np.linalg.norm(pred_in_base - gt_d)*100:.1f}cm   "
          f"vs STATIC baseline {np.linalg.norm(gt_d)*100:.1f}cm")
    p_dyn = torch.sigmoid(out["p_dyn"][gm].float()).mean().item() if "p_dyn" in out else float("nan")
    print(f"  gate(gripper region) = {p_dyn:.2f}")


if __name__ == "__main__":
    main()
