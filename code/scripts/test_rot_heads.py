"""§87 unit tests for the two rotation redesigns (no real data needed).

  1. rot6d: zero delta -> exactly I; random deltas -> orthonormal (R^T R = I, det=+1).
  2. entity_rot head: zero-init => v/omega outputs UNCHANGED (warm-start identity); grads finite
     through a position loss; erot_log records per-step (R_e, uniq).
  3. motion_bases head (pure): zero-init => v=0, omega=0 (identity motion); coefficients softmax
     (sum to 1); grads finite.
  4. entity_rot_position_loss: zero rotation on a rotating GT -> loss>0; PERFECT per-step rotation
     -> loss ~0 (the supervision is satisfiable).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from igsw.dynamics.rot6d import rot6d_delta_to_matrix, matrix_to_axis_angle  # noqa: E402
from igsw.dynamics.model import GaussianDynamics, DynamicsConfig, GaussianState  # noqa: E402
from igsw.training.losses import entity_rot_position_loss  # noqa: E402
from igsw.dynamics.manifold import axis_angle_to_quat, quat_to_rotmat  # noqa: E402


def small_cfg():
    return DynamicsConfig(d_model=64, n_heads=4, n_layers=2, lang_dim=32, use_checkpoint=False)


def mk_state(N, dev):
    return GaussianState(torch.randn(1, N, 3, device=dev) * 0.2,
                         torch.tensor([1.0, 0, 0, 0], device=dev).repeat(1, N, 1),
                         torch.full((1, N, 3), 0.01, device=dev),
                         torch.full((1, N), 0.9, device=dev),
                         torch.rand(1, N, 3, device=dev))


def run_heads(m, st, seg, dev):
    N = st.means.shape[1]
    ctx = torch.randn(1, m.cfg.n_layers, 7, m.cfg.d_model, device=dev)
    mask = torch.ones(1, 7, dtype=torch.bool, device=dev)
    cond = torch.randn(1, m.cfg.d_model, device=dev)
    step = torch.zeros(1, dtype=torch.long, device=dev)
    return m.predict_deltas(st, ctx, mask, cond, step, seg_local=seg)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ok = True

    # ===== 1. rot6d =====
    R0 = rot6d_delta_to_matrix(torch.zeros(5, 6, device=dev))
    e1 = (R0 - torch.eye(3, device=dev)).abs().max().item()
    d6 = torch.randn(64, 6, device=dev) * 0.3
    R = rot6d_delta_to_matrix(d6)
    ortho = (R.transpose(-1, -2) @ R - torch.eye(3, device=dev)).abs().max().item()
    det = torch.det(R).min().item()
    t1 = e1 < 1e-6 and ortho < 1e-4 and det > 0.99
    print(f"[1] rot6d: zero->I err={e1:.1e} ortho={ortho:.1e} det_min={det:.4f}  {'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # ===== 2. entity_rot head: identity at init + finite grads =====
    N = 96
    seg = torch.cat([torch.ones(40), torch.full((40,), 2.0), torch.zeros(16)]).long().to(dev)
    st = mk_state(N, dev)
    m_off = GaussianDynamics(small_cfg()).to(dev)
    m_rot = GaussianDynamics(small_cfg(), entity_rot=True).to(dev)
    m_rot.load_state_dict(m_off.state_dict(), strict=False)      # same trunk weights
    torch.manual_seed(1); o_off = run_heads(m_off, st, seg, dev)
    m_rot.erot_log = []
    torch.manual_seed(1); o_rot = run_heads(m_rot, st, seg, dev)
    dv = (o_off[0] - o_rot[0]).abs().max().item()
    dw = (o_off[1] - o_rot[1]).abs().max().item()
    loss = o_rot[0].square().sum() + o_rot[1].square().sum()
    loss.backward()
    g_ok = all(torch.isfinite(p.grad).all() for p in m_rot.parameters() if p.grad is not None)
    t2 = dv < 1e-5 and dw < 1e-5 and g_ok and len(m_rot.erot_log) == 1
    print(f"[2] entity_rot: warm-start dv={dv:.1e} dw={dw:.1e} grads_finite={g_ok} "
          f"log={len(m_rot.erot_log)}  {'PASS' if t2 else 'FAIL'}")
    ok &= t2

    # ===== 3. motion_bases (pure): identity at init, coef softmax, finite grads =====
    m_bas = GaussianDynamics(small_cfg(), motion_bases=4, bases_mode="pure").to(dev)
    st3 = mk_state(N, dev)
    o_bas = run_heads(m_bas, st3, seg, dev)
    v_mag = o_bas[0].abs().max().item()
    w_mag = o_bas[1].abs().max().item()
    csum = (m_bas.coef_last.sum(-1) - 1).abs().max().item()
    loss3 = (o_bas[0] - 0.01).square().sum()
    loss3.backward()
    g3 = all(torch.isfinite(p.grad).all() for p in m_bas.parameters() if p.grad is not None)
    t3 = v_mag < 1e-5 and w_mag < 1e-5 and csum < 1e-4 and g3
    print(f"[3] motion_bases: init |v|={v_mag:.1e} |om|={w_mag:.1e} coef_sum_err={csum:.1e} "
          f"grads_finite={g3}  {'PASS' if t3 else 'FAIL'}")
    ok &= t3

    # ===== 4. entity_rot_position_loss: satisfiable supervision =====
    K, Mc = 4, 60
    seg4 = torch.cat([torch.ones(40), torch.zeros(20)]).long().to(dev)
    X0 = torch.randn(Mc, 3, device=dev) * 0.1
    aa_step = torch.tensor([0.0, 0.0, 0.10], device=dev)         # 0.1 rad/step about z
    Rstep = quat_to_rotmat(axis_angle_to_quat(aa_step[None]))[0]
    traj = [X0]
    cur = X0.clone()
    c0 = X0[:40].mean(0)
    for t in range(K):
        nxt = cur.clone()
        nxt[:40] = (cur[:40] - cur[:40].mean(0)) @ Rstep.T + cur[:40].mean(0) + 0.01
        traj.append(nxt); cur = nxt
    traj = torch.stack(traj)                                      # [K+1,Mc,3]
    uniq = torch.unique(seg4)                                     # [0,1] sorted
    E = uniq.numel()
    perfect = [(torch.stack([torch.eye(3, device=dev) if int(e) == 0 else Rstep for e in uniq]), uniq)
               for _ in range(K)]
    zero = [(torch.eye(3, device=dev).expand(E, 3, 3).clone(), uniq) for _ in range(K)]
    l_perf = float(entity_rot_position_loss(perfect, traj, seg4))
    l_zero = float(entity_rot_position_loss(zero, traj, seg4))
    t4 = l_perf < 1e-4 and l_zero > 5 * max(l_perf, 1e-9)
    print(f"[4] erot loss: perfect={l_perf:.2e} zero-rot={l_zero:.2e}  {'PASS' if t4 else 'FAIL'}")
    ok &= t4

    print("\n==== ALL PASS ====" if ok else "\n==== FAILURES ====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
