"""§66 v10-rigid unit tests for entity_rigid_aggregate (no real data / no model needed).

Verifies:
  1. RIGID field in -> identity out (a genuine per-entity SE(3) field is reproduced exactly).
  2. SCATTER field -> extent-ratio collapses to ~1.0 while the centroid direction is preserved.
  3. DEGENERATE entity (<4 pts, and collinear) -> translation-only fallback, finite, no NaN.
  4. bf16 autocast around the call is safe (fp32 island inside) + gradients flow finite.
  5. excluded ids (background 0) pass through unchanged.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.dynamics.rigid_agg import entity_rigid_aggregate     # noqa: E402
from igsw.dynamics.manifold import axis_angle_to_quat, quat_to_rotmat  # noqa: E402


def _ext(X):
    return ((X - X.mean(0)) ** 2).sum(1).mean().clamp_min(0).sqrt().item()


def _rand_rot(dev, scale=0.2):
    aa = torch.randn(3, device=dev) * scale
    return quat_to_rotmat(axis_angle_to_quat(aa[None]))[0], aa  # [3,3], axis-angle [3]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ok = True

    # ---- build a 2-entity scene: id 1 (40 pts), id 2 (60 pts), + background id 0 (20 pts) ----
    x1 = torch.randn(40, 3, device=dev) * 0.1 + torch.tensor([0.3, 0.0, 1.0], device=dev)
    x2 = torch.randn(60, 3, device=dev) * 0.1 + torch.tensor([-0.3, 0.1, 1.1], device=dev)
    x0 = torch.randn(20, 3, device=dev) * 0.5                   # background
    x = torch.cat([x1, x2, x0], 0)                              # [120,3]
    seg = torch.cat([torch.ones(40), torch.full((60,), 2.0), torch.zeros(20)]).long().to(dev)

    # ===== TEST 1: a TRUE rigid field (CONSISTENT v + omega) is reproduced exactly =====
    # §74: rotation now comes from the supervised omega, so a rigid field must carry both the rigid
    # displacement (v) AND the matching per-control axis-angle (omega).
    (R1, aa1), t1 = _rand_rot(dev), torch.tensor([0.05, -0.02, 0.03], device=dev)
    (R2, aa2), t2 = _rand_rot(dev), torch.tensor([-0.04, 0.01, 0.02], device=dev)
    y1 = x1 @ R1.T + t1
    y2 = x2 @ R2.T + t2
    y0 = x0                                                     # bg: no motion
    v_true = (torch.cat([y1, y2, y0], 0) - x)[None]            # [1,120,3]
    om_in = torch.zeros_like(v_true)
    om_in[0, :40] = aa1                                         # entity 1's rotation
    om_in[0, 40:100] = aa2                                      # entity 2's rotation
    v_hat, om_hat = entity_rigid_aggregate(x, v_true.clone(), om_in.clone(), seg, w=None)
    err = (v_hat - v_true).abs().max().item()
    t1_ok = err < 1e-4
    print(f"[1] rigid-in(v+omega)->identity: max|v_hat - v_true| = {err:.2e}  {'PASS' if t1_ok else 'FAIL'}")
    ok &= t1_ok

    # ===== TEST 2: a SCATTER field collapses to rigid, centroid direction preserved =====
    trans = torch.tensor([0.10, 0.05, -0.03], device=dev)      # the intended object translation
    scatter = torch.randn(40, 3, device=dev) * 0.08            # per-point incoherent noise (the bug)
    v_scat = torch.cat([(trans + scatter),                     # id1: translation + scatter
                        torch.zeros(60, 3, device=dev),        # id2 static
                        torch.zeros(20, 3, device=dev)], 0)[None]
    ext_before = _ext(x1 + v_scat[0, :40])
    v_hat2, _ = entity_rigid_aggregate(x, v_scat.clone(), torch.zeros_like(v_scat), seg, w=None)
    y1_hat = x1 + v_hat2[0, :40]
    ext_after = _ext(y1_hat)
    ratio_before = ext_before / _ext(x1)
    ratio_after = ext_after / _ext(x1)
    dir_before = (v_scat[0, :40].mean(0))
    dir_after = (v_hat2[0, :40].mean(0))
    dcos = torch.cosine_similarity(dir_before[None], dir_after[None]).item()
    t2_ok = ratio_after < 1.05 and ratio_before > 1.1 and dcos > 0.98
    print(f"[2] scatter->rigid: extent-ratio {ratio_before:.2f}->{ratio_after:.2f}  "
          f"centroid dir-cos {dcos:.3f}  {'PASS' if t2_ok else 'FAIL'}")
    ok &= t2_ok

    # ===== TEST 3: degenerate entities (too few pts; collinear) -> translation fallback, finite =====
    xs = torch.tensor([[0., 0, 1], [0, 0, 1.1], [0, 0, 1.2]], device=dev)   # 3 collinear pts (<min_pts AND rank1)
    seg_s = torch.ones(3, device=dev).long()
    v_s = (torch.tensor([0.02, 0.01, 0.0], device=dev)[None].expand(3, 3) + 0.0)[None]
    v_hat3, om_hat3 = entity_rigid_aggregate(xs, v_s.clone(), torch.zeros_like(v_s), seg_s, w=None)
    t3_ok = torch.isfinite(v_hat3).all().item() and torch.isfinite(om_hat3).all().item() \
        and om_hat3.abs().max().item() < 1e-4      # rotation suppressed -> pure translation
    print(f"[3] degenerate->translation: finite={torch.isfinite(v_hat3).all().item()} "
          f"|om|max={om_hat3.abs().max().item():.2e}  {'PASS' if t3_ok else 'FAIL'}")
    ok &= t3_ok

    # ===== TEST 4: bf16 autocast safe + gradients finite =====
    t4_ok = True
    if dev == "cuda":
        v_req = v_scat.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_h, om_h = entity_rigid_aggregate(x, v_req, torch.zeros_like(v_req), seg, w=None)
            loss = (v_h ** 2).sum() + (om_h ** 2).sum()
        loss.backward()
        t4_ok = torch.isfinite(v_h).all().item() and v_req.grad is not None \
            and torch.isfinite(v_req.grad).all().item()
        print(f"[4] bf16+grad: out-finite={torch.isfinite(v_h).all().item()} "
              f"grad-finite={torch.isfinite(v_req.grad).all().item()}  {'PASS' if t4_ok else 'FAIL'}")
    else:
        print("[4] bf16+grad: SKIP (cpu)")
    ok &= t4_ok

    # ===== TEST 5: background id 0 passes through unchanged =====
    v_bg_in = v_true.clone()
    v_hat5, _ = entity_rigid_aggregate(x, v_bg_in, torch.zeros_like(v_bg_in), seg, w=None)
    t5_ok = (v_hat5[0, 100:] - v_bg_in[0, 100:]).abs().max().item() < 1e-6
    print(f"[5] bg(id0) passthrough: {'PASS' if t5_ok else 'FAIL'}")
    ok &= t5_ok

    print("\n==== ALL PASS ====" if ok else "\n==== FAILURES ABOVE ====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
