"""Correctness test for the manifold math vs `roma` (ground truth).

Guards the SO(3) exp-map, Hamilton product, and quat->rotmat used by the dynamics
delta application. roma uses scalar-LAST quaternions [x,y,z,w]; ours are scalar-
FIRST [w,x,y,z], so we convert with index [1,2,3,0] for comparison.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.dynamics.manifold import axis_angle_to_quat, quat_mul, quat_to_rotmat  # noqa: E402


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def main():
    import roma
    torch.manual_seed(0)
    N = 10000
    dev = "cpu"

    # 1) exp map: axis_angle -> unit quat (include tiny + large angles)
    omega = torch.randn(N, 3)
    omega[:5] *= 1e-7   # near-zero stability
    omega[5:10] *= 10   # large
    q_mine = axis_angle_to_quat(omega)
    q_roma = roma.rotvec_to_unitquat(omega)             # xyzw
    # align hemisphere (q and -q are same rotation) before comparing
    qm = wxyz_to_xyzw(q_mine)
    sign = torch.sign((qm * q_roma).sum(-1, keepdim=True)); sign[sign == 0] = 1
    err_exp = (qm * sign - q_roma).abs().max().item()
    print(f"[exp map]    max|q_mine - q_roma| = {err_exp:.2e}")
    assert err_exp < 1e-5, "axis_angle_to_quat mismatch"

    # 2) Hamilton product
    a = roma.random_unitquat(N); b = roma.random_unitquat(N)   # xyzw
    a_w = a[..., [3, 0, 1, 2]]; b_w = b[..., [3, 0, 1, 2]]     # -> wxyz
    prod_mine = wxyz_to_xyzw(quat_mul(a_w, b_w))
    prod_roma = roma.quat_product(a, b)
    sign = torch.sign((prod_mine * prod_roma).sum(-1, keepdim=True)); sign[sign == 0] = 1
    err_mul = (prod_mine * sign - prod_roma).abs().max().item()
    print(f"[quat_mul]   max|p_mine - p_roma| = {err_mul:.2e}")
    assert err_mul < 1e-5, "quat_mul mismatch"

    # 3) quat -> rotmat
    R_mine = quat_to_rotmat(a_w)
    R_roma = roma.unitquat_to_rotmat(a)
    err_R = (R_mine - R_roma).abs().max().item()
    print(f"[quat2rot]   max|R_mine - R_roma| = {err_R:.2e}")
    assert err_R < 1e-5, "quat_to_rotmat mismatch"

    # 4) consistency: applying Exp(omega) as rotmat == rotvec_to_rotmat
    R_exp = quat_to_rotmat(axis_angle_to_quat(omega))
    R_ref = roma.rotvec_to_rotmat(omega)
    err_c = (R_exp - R_ref).abs().max().item()
    print(f"[exp->rot]   max|R_exp - R_ref|  = {err_c:.2e}")
    assert err_c < 1e-5, "exp->rotmat mismatch"

    print("[OK] manifold math matches roma to <1e-5 (including near-zero & large angles).")


if __name__ == "__main__":
    main()
