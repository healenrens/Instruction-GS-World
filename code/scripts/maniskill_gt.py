"""CLEAN ground-truth data pipeline from ManiSkill3 simulation for the
language-conditioned 3DGS dynamics model (agent.md section 38 data pivot).

Real-data GT (Pi3 monocular depth + CoTracker 2D tracking) was too occlusion-noisy to
learn from. Sim gives EXACT rendered depth + EXACT per-entity 3D pose (the simulator
knows every body's pose even when it is visually occluded), so we can build a perfectly
clean, occlusion-free per-Gaussian 3D trajectory.

Pipeline (this file):
  1. Generate ONE meaningful manipulation episode with a scripted motion-planning solver
     (objects actually get moved). Record per frame: RGB, depth, SEGMENTATION (per-pixel
     entity id), camera intrinsics + extrinsics (world->cam, OpenCV), and the world pose
     of EVERY entity (actors AND articulation links, keyed by segmentation id).
  2. Frame-0 3DGS from RGB-D (clean, NOT Pi3): backproject frame-0 depth with K + extrinsic
     -> world colored point cloud -> GaussianSet (reuse points_to_gaussians scale/opacity
     init). Keep each Gaussian's frame-0 pixel uv (for control_uv) and its entity id (seg).
  3. Analytic per-Gaussian 3D motion: for a Gaussian belonging to entity e, its position at
     frame t = T_{e,t} . T_{e,0}^{-1} . X_0, with T built from the entity raw_pose. Static
     entities -> identity. Exact, occlusion-free trajectory.
  4. CRITICAL VALIDATION: move the frame-0 Gaussians by the analytic transforms to frame t,
     render from the SAME camera with our gsplat renderer, compare to the ACTUAL ManiSkill
     RGB at frame t. They must MATCH (high PSNR). If not, the pose math / backprojection is
     wrong.
  5. Save the clip (g0 dense, per-pixel uv + entity id, K, camera viewmats, gt trajectory of
     a sampled control set, instruction) to disk for the overfit.

ManiSkill conventions (verified):
  - depth: int16 in MILLIMETERS -> /1000 = meters.
  - extrinsic_cv [3,4] = world->camera, OpenCV (x-right, y-down, z-forward). gsplat wants the
    same world->cam viewmat -> pad to 4x4 directly.
  - intrinsic_cv [3,3] pinhole.
  - raw_pose = [x,y,z, qw,qx,qy,qz] (wxyz quaternion = gsplat convention).
  - u.segmentation_id_map: per_scene_id (int) -> entity (Actor or articulation Link), both have
    .pose.raw_pose. goal_site is a translucent goal marker (not real geometry) -> excluded.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.gaussians.types import GaussianSet
from igsw.gaussians.render import render_gaussianset, psnr
from igsw.lifting.to_gaussians import points_to_gaussians, _per_view_neighbor_scale


# --------------------------------------------------------------------------- #
# pose / geometry helpers
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q [...,4] wxyz (unit) -> R [...,3,3]. (Matches gsplat / ManiSkill raw_pose order.)"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def pose_to_matrix(raw_pose: torch.Tensor) -> torch.Tensor:
    """raw_pose [...,7] = [x,y,z, qw,qx,qy,qz] -> homogeneous T [...,4,4] (world transform)."""
    t = raw_pose[..., :3]
    R = quat_wxyz_to_rotmat(raw_pose[..., 3:7])
    T = torch.zeros(*raw_pose.shape[:-1], 4, 4, dtype=raw_pose.dtype, device=raw_pose.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def extrinsic_to_viewmat(extrinsic_cv: torch.Tensor) -> torch.Tensor:
    """extrinsic_cv [3,4] (world->cam, OpenCV) -> viewmat [4,4] (world->cam) for gsplat."""
    V = torch.eye(4, dtype=extrinsic_cv.dtype, device=extrinsic_cv.device)
    V[:3, :4] = extrinsic_cv
    return V


def backproject_to_world(depth_m: torch.Tensor, K: torch.Tensor, viewmat_w2c: torch.Tensor):
    """depth_m [H,W] meters, K [3,3], viewmat_w2c [4,4] world->cam (OpenCV).
    Returns world_points [H,W,3] and a finite/positive-depth mask [H,W]."""
    H, W = depth_m.shape
    dev = depth_m.device
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth_m
    # OpenCV camera frame: x right, y down, z forward.
    x_c = (uu - cx) / fx * z
    y_c = (vv - cy) / fy * z
    pts_cam = torch.stack([x_c, y_c, z], dim=-1)                         # [H,W,3]
    cam2world = torch.linalg.inv(viewmat_w2c)                            # [4,4]
    R = cam2world[:3, :3]; t = cam2world[:3, 3]
    pts_world = pts_cam @ R.T + t                                        # [H,W,3]
    mask = torch.isfinite(z) & (z > 1e-4)
    return pts_world, mask


# --------------------------------------------------------------------------- #
# WHOLE-VIDEO temporal fusion -> ONE complete canonical Gaussian set
# --------------------------------------------------------------------------- #
def _voxel_first_indices(P: torch.Tensor, voxel: float) -> torch.Tensor:
    """Indices that keep ONE point per `voxel`-sized cell (the first in input order). Used to dedupe
    the multi-frame point pile; with the canonical frame placed FIRST, its points win each cell."""
    q = torch.floor(P / voxel).to(torch.int64)
    q = q - q.min(dim=0).values
    nx = int(q[:, 0].max().item()) + 1
    ny = int(q[:, 1].max().item()) + 1
    key = q[:, 0] + q[:, 1] * nx + q[:, 2] * (nx * ny)
    ks, order = torch.sort(key, stable=True)
    first = torch.ones_like(ks, dtype=torch.bool)
    first[1:] = ks[1:] != ks[:-1]
    return order[first]


def _project_to_canonical_px(P: torch.Tensor, K: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """Project canonical world points P[N,3] into the canonical camera -> pixel uv[N,2] (OpenCV)."""
    Ph = torch.cat([P, torch.ones(P.shape[0], 1, device=P.device)], dim=-1)
    cam = (Ph @ viewmat.T)[:, :3]
    z = cam[:, 2].clamp_min(1e-6)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return torch.stack([u, v], dim=-1)


def _fuse_canonical_gaussians(rec, start_idx: int, device: str, exclude_seg, depth_max: float,
                              fuse_stride: int = 3, voxel: float = 0.004, scale_factor: float = 0.6,
                              scale_pct=(0.01, 0.7), opacity_init: float = 0.9):
    """★ The unified whole-video reconstruction (replaces the single-frame G0). Back-project EVERY
    (strided) frame's depth, and for each entity REGISTER its points into the canonical (start_idx)
    pose via the KNOWN per-entity poses (`X_canon = T_{e,0} · T_{e,f}^{-1} · X_f`; static/no-pose ->
    identity since the camera is static), accumulate across the whole video, then voxel-dedupe into
    ONE complete, denoised canonical GaussianSet. Each kept point carries its source frame's grid-
    neighbour scale (so detail quality matches the single-frame path). Moving entities reveal new
    faces over time and the arm's occlusion-shadows on the static scene get filled -> a COMPLETE,
    low-uncertainty canonical geometry that the analytic trajectory then drives. The temporal-fusion
    methodology transfers to real monocular video (one camera over time), unlike multi-camera tricks.
    Returns (GaussianSet, uv[N,2], seg_per_g[N])."""
    n_sim = len(rec.frames)
    fcanon = rec.frames[start_idx]
    K = fcanon["K"].to(device).float()
    viewmat = extrinsic_to_viewmat(fcanon["extrinsic_cv"].to(device).float())
    poses0 = rec.entity_poses[start_idx]
    exclude_seg = set(int(s) for s in (exclude_seg or set()))
    # canonical frame FIRST (wins the per-voxel dedupe), then the rest of the video
    fuse_idx = [start_idx] + [f for f in range(0, n_sim, max(1, fuse_stride)) if f != start_idx]
    P_all, C_all, S_all, SC_all = [], [], [], []
    for f in fuse_idx:
        fr = rec.frames[f]
        depth = fr["depth"].to(device).float() / 1000.0
        Kf = fr["K"].to(device).float()
        vmf = extrinsic_to_viewmat(fr["extrinsic_cv"].to(device).float())
        pts_w, dmask = backproject_to_world(depth, Kf, vmf)            # [H,W,3] world at frame f
        seg = fr["seg"].to(device).long()
        rgb = fr["rgb"].to(device).float() / 255.0                    # [H,W,3]
        scl = _per_view_neighbor_scale(pts_w[None])[0]                # [H,W] grid spacing (rotation-invariant)
        keep = dmask & (seg > 0) & (depth < float(depth_max)) & torch.isfinite(scl)
        for sid in exclude_seg:
            keep &= (seg != sid)
        if not bool(keep.any()):
            continue
        Pk = pts_w[keep]; Ck = rgb[keep].clamp(0, 1); Sk = seg[keep]; SCk = scl[keep]
        posef = rec.entity_poses[f]
        Xc = Pk.clone()
        for sid in torch.unique(Sk).tolist():
            sid = int(sid)
            rp0 = poses0.get(sid); rpf = posef.get(sid)
            if rp0 is None or rpf is None:
                continue                                               # static / no pose -> identity
            T0 = pose_to_matrix(rp0.to(device).float())
            Tf = pose_to_matrix(rpf.to(device).float())
            T_rel = T0 @ torch.linalg.inv(Tf)                          # frame-f world -> canonical world
            m = (Sk == sid)
            Xh = torch.cat([Pk[m], torch.ones(int(m.sum()), 1, device=device)], dim=-1)
            Xc[m] = (Xh @ T_rel.T)[:, :3]
        P_all.append(Xc); C_all.append(Ck); S_all.append(Sk); SC_all.append(SCk)
    P = torch.cat(P_all); C = torch.cat(C_all); S = torch.cat(S_all); SC = torch.cat(SC_all)
    idx = _voxel_first_indices(P, voxel)                              # dedupe overlapping points
    P, C, S, SC = P[idx], C[idx], S[idx], SC[idx]
    if SC.numel() >= 16:                                              # same robust scale clamp as points_to_gaussians
        lo = torch.quantile(SC, scale_pct[0]); hi = torch.quantile(SC, scale_pct[1])
        SC = SC.clamp(min=max(1e-4, float(lo)), max=float(hi)) * scale_factor
    else:
        SC = SC.clamp_min(1e-4) * scale_factor
    N = P.shape[0]
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    g0 = GaussianSet(means=P.contiguous(), quats=quats,
                     scales=SC[:, None].repeat(1, 3).contiguous(),
                     opacities=torch.full((N,), float(opacity_init), device=device),
                     colors=C.contiguous()).validate()
    uv = _project_to_canonical_px(P, K, viewmat)
    return g0, uv, S


# --------------------------------------------------------------------------- #
# episode generation
# --------------------------------------------------------------------------- #
class _ObsRecorder:
    """Wraps a ManiSkill env so we capture the full obs dict on every step()/reset()."""
    def __init__(self, env):
        self.env = env
        self.frames = []          # list of obs dicts
        self.entity_poses = []     # list of {seg_id: raw_pose[7] tensor}

    def _entity_pose_snapshot(self):
        u = self.env.unwrapped
        snap = {}
        for sid, ent in u.segmentation_id_map.items():
            if hasattr(ent, "pose"):
                snap[int(sid)] = ent.pose.raw_pose[0].detach().cpu().clone()
        return snap

    def _record(self, obs):
        # detach + move the camera tensors to cpu so we do not hold sim memory across the rollout
        cam = obs["sensor_data"]["base_camera"]
        rec = {
            "rgb": cam["rgb"][0].detach().cpu().clone(),                 # [H,W,3] uint8
            "depth": cam["depth"][0, ..., 0].detach().cpu().clone(),    # [H,W] int16 mm
            "seg": cam["segmentation"][0, ..., 0].detach().cpu().clone(),# [H,W] int16
            "K": obs["sensor_param"]["base_camera"]["intrinsic_cv"][0].detach().cpu().clone(),
            "extrinsic_cv": obs["sensor_param"]["base_camera"]["extrinsic_cv"][0].detach().cpu().clone(),
        }
        self.frames.append(rec)
        self.entity_poses.append(self._entity_pose_snapshot())

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.frames = []; self.entity_poses = []
        self._record(obs)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self._record(out[0])
        return out

    # passthroughs so the motion planner sees a normal env
    def __getattr__(self, name):
        return getattr(self.env, name)


# Per-task language instruction (what the scripted policy actually does).
INSTRUCTIONS = {
    "PickCube-v1": "Pick up the red cube and move it to the goal position.",
    "StackCube-v1": "Pick up the red cube and stack it on top of the green cube.",
    "PushCube-v1": "Push the cube to the goal region.",
    "PullCube-v1": "Pull the cube to the goal region.",
}


def _ee_action(env, target_p, grip, gain=8.0, target_q_delta=None):
    """Build a pd_ee_delta_pose action [dx,dy,dz, drx,dry,drz, gripper] that drives the TCP
    toward target_p with a proportional gain. The delta is expressed in the EE frame for this
    control mode, but for small steps the world-delta works well and is robust."""
    u = env.unwrapped
    tcp = u.agent.tcp.pose.p[0].cpu().numpy()
    d = (np.asarray(target_p) - tcp) * gain
    act = np.zeros(env.action_space.shape[-1], dtype=np.float32)
    act[:3] = np.clip(d, -1.0, 1.0)
    if target_q_delta is not None:
        act[3:6] = np.clip(target_q_delta, -1.0, 1.0)
    act[-1] = grip
    return act


def _script_pick(env, rec, goal_offset=(0.0, 0.0, 0.0)):
    """Scripted grasp-and-CARRY for PickCube / StackCube (no mplib): approach above the cube,
    descend, grasp, lift, then carry the grasped cube through a MULTI-WAYPOINT path so the scene
    has ~6-7 s of CONTINUOUS 3D motion (=> a random 4 s sub-window always contains real change).
    The first carry target is the task goal, so 'pick & move to goal' still matches the motion;
    the extra waypoints keep the grasped cube travelling through the workspace (lateral + vertical)."""
    u = env.unwrapped
    OPEN, CLOSE = 1.0, -1.0
    target = u.cube if hasattr(u, "cube") else u.cubeA   # PickCube: cube; StackCube: cubeA

    def cube_p():
        return target.pose.p[0].cpu().numpy()

    # phase 1: hover above the cube (open gripper)
    for _ in range(14):
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.06]), OPEN))
    # phase 2: descend onto the cube
    for _ in range(10):
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.005]), OPEN, gain=6.0))
    # phase 3: close the gripper to grasp
    for _ in range(6):
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.005]), CLOSE, gain=4.0))
    # phase 4: lift straight up
    for _ in range(10):
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.12]), CLOSE, gain=6.0))
    # task goal = FIRST carry waypoint (so the instruction still describes the motion)
    if hasattr(u, "goal_site"):
        goal = u.goal_site.pose.p[0].cpu().numpy()
    elif hasattr(u, "cubeB"):
        goal = u.cubeB.pose.p[0].cpu().numpy() + np.array([0, 0, 0.04])
    else:
        goal = cube_p() + np.array([0.10, 0.10, 0.12])
    goal = goal + np.array(goal_offset)
    # phase 5+: multi-waypoint carry of the GRASPED cube -> ~5 s of continuous 3D motion
    base = cube_p()                                       # lifted cube position
    waypoints = [
        goal,
        base + np.array([ 0.09,  0.00, 0.02]),
        base + np.array([ 0.09,  0.10, 0.07]),
        base + np.array([-0.07,  0.10, 0.01]),
        base + np.array([-0.07, -0.09, 0.07]),
        base + np.array([ 0.07, -0.09, 0.01]),
        base + np.array([ 0.00,  0.00, 0.05]),
    ]
    for wp in waypoints:
        for _ in range(14):
            rec.step(_ee_action(env, wp, CLOSE, gain=6.0))


def _manip_object(u):
    """Return the manipulated object actor regardless of the env's attribute name.
    PickCube uses `cube`; PushCube/PullCube use `obj`; StackCube uses `cubeA`."""
    for nm in ("cube", "obj", "cubeA"):
        if hasattr(u, nm):
            return getattr(u, nm)
    raise AttributeError("no manipulated-object attribute (cube/obj/cubeA) on env")


def _script_push(env, rec):
    """Scripted MULTI-SEGMENT push for PushCube: push the cube through a sequence of targets
    (re-approaching behind it before each segment) so the scene has ~6 s of continuous motion
    (=> a random 4 s sub-window always contains real change). The first target is the task goal;
    the rest zigzag in a bounded area around it so the cube stays in the workspace."""
    u = env.unwrapped
    CLOSE = -1.0
    obj = _manip_object(u)
    cube0 = obj.pose.p[0].cpu().numpy()
    goal = u.goal_region.pose.p[0].cpu().numpy() if hasattr(u, "goal_region") else (
        u.goal_site.pose.p[0].cpu().numpy() if hasattr(u, "goal_site") else cube0 + np.array([0.10, 0, 0]))
    # sequence of push targets: start at the goal, then zigzag in a bounded region around it
    targets = [
        np.array([goal[0],        goal[1],        cube0[2]]),
        np.array([goal[0] + 0.07, goal[1] + 0.08, cube0[2]]),
        np.array([goal[0] - 0.07, goal[1] + 0.08, cube0[2]]),
        np.array([goal[0] - 0.07, goal[1] - 0.05, cube0[2]]),
    ]
    for tgt in targets:
        cube = obj.pose.p[0].cpu().numpy()
        direction = tgt[:2] - cube[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        behind = cube + np.array([-direction[0] * 0.04, -direction[1] * 0.04, 0.02])
        for _ in range(10):                              # re-approach behind the cube
            rec.step(_ee_action(env, behind, CLOSE))
        for _ in range(20):                              # push it to this segment's target
            cube = obj.pose.p[0].cpu().numpy()
            tp = np.array([tgt[0], tgt[1], cube[2]])
            rec.step(_ee_action(env, tp, CLOSE, gain=5.0))


def _quat_angle(qa, qb):
    """Angle (rad) of the relative rotation between two wxyz quaternions."""
    qa = np.asarray(qa) / (np.linalg.norm(qa) + 1e-9)
    qb = np.asarray(qb) / (np.linalg.norm(qb) + 1e-9)
    return float(2.0 * np.arccos(min(1.0, abs(float(np.dot(qa, qb))))))


def _script_rotate(env, rec, spin_steps: int = 64, drz: float = 0.9, rng=None):
    """Track B (rotation-rich data): grasp the cube CENTRALLY, lift, then SPIN the wrist in place.
    The rigidly-grasped cube rotates about ~its own centroid. When `rng` is given, the spin DIRECTION
    (cw/ccw) and MAGNITUDE are RANDOMIZED per episode => VARIED rotation (angle ~40-150deg, both
    signs) instead of a constant ~120deg. This kills the constant-rotation confound (the model must
    express the rotation through the per-token field, not memorize one fixed angle)."""
    if rng is not None:
        spin_steps = int(rng.integers(30, 73))
        drz = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.0))
    u = env.unwrapped
    OPEN, CLOSE = 1.0, -1.0
    target = u.cube if hasattr(u, "cube") else u.cubeA

    def cube_p():
        return target.pose.p[0].cpu().numpy()

    for _ in range(14):                                              # hover above the cube (open)
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.06]), OPEN))
    for _ in range(10):                                              # descend onto it
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.005]), OPEN, gain=6.0))
    for _ in range(6):                                               # grasp
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.005]), CLOSE, gain=4.0))
    for _ in range(8):                                               # minimal lift to just clear the table
        rec.step(_ee_action(env, cube_p() + np.array([0, 0, 0.05]), CLOSE, gain=6.0))
    hold = cube_p()                                                  # hold this position while spinning
    for _ in range(spin_steps):                                      # spin the wrist (yaw) -> cube rotates
        rec.step(_ee_action(env, hold, CLOSE, gain=5.0, target_q_delta=[0.0, 0.0, drz]))


def generate_episode(env_id: str, seed: int, cam_w: int, cam_h: int, max_retries: int = 8,
                     policy: str = "auto"):
    """Drive the manipulation with a deterministic SCRIPTED end-effector policy (pd_ee_delta_pose;
    avoids mplib which segfaults in this headless container) while capturing obs every step.
    Returns (recorder, instruction, success). Success here means 'the object moved a lot'."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 (registers envs)

    base = gym.make(
        env_id, obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", sim_backend="cpu", num_envs=1,
        sensor_configs=dict(width=cam_w, height=cam_h),
    )
    doc = (type(base.unwrapped).__doc__ or "").strip().split("\n")
    doc = " ".join(l.strip() for l in doc if l.strip() and not l.strip().startswith("**"))[:200]
    instruction = INSTRUCTIONS.get(env_id, doc or env_id)
    if policy == "rotate":
        instruction = "Pick up the red cube and rotate it in place."
    rec = _ObsRecorder(base)

    s = seed
    for _ in range(max_retries):
        rec.reset(seed=s)
        u = base.unwrapped
        try:
            obj = _manip_object(u)
            cube0 = obj.pose.p[0].cpu().numpy().copy()
            cube0_q = obj.pose.q[0].cpu().numpy().copy()
        except AttributeError:
            obj, cube0, cube0_q = None, None, None
        if policy == "rotate":
            _script_rotate(base, rec, rng=np.random.default_rng(s))
        elif env_id in ("PushCube-v1", "PullCube-v1"):
            _script_push(base, rec)
        else:
            _script_pick(base, rec)
        # 'meaningful motion' check: translation for carry/push; ROTATION angle for the rotate policy.
        moved, rot = 0.0, 0.0
        if obj is not None and cube0 is not None:
            moved = float(np.linalg.norm(obj.pose.p[0].cpu().numpy() - cube0))
            rot = _quat_angle(cube0_q, obj.pose.q[0].cpu().numpy())
        ok = (rot > 0.5 if policy == "rotate" else moved > 0.04) and len(rec.frames) > 4
        print(f"  episode seed={s}: cube moved {moved:.3f} m / rot {np.degrees(rot):.0f}deg over "
              f"{len(rec.frames)} frames ({'OK' if ok else 'retry'})", flush=True)
        if ok:
            return rec, instruction, True
        s += 1
    print(f"  WARNING: no clearly-moving episode in {max_retries} tries; using last.", flush=True)
    return rec, instruction, False


# --------------------------------------------------------------------------- #
# build clip: frame-0 gaussians + analytic GT
# --------------------------------------------------------------------------- #
EXCLUDE_NAMES = {"goal_site"}     # translucent goal marker = not real geometry


def build_clip(rec: _ObsRecorder, K_frames: int, device: str, exclude_seg: set[int] | None = None,
               depth_max: float = 2.0, start_frac: float = 0.0,
               window_steps: int | None = None, rng=None, fuse_stride: int = 0,
               fuse_voxel: float = 0.002):
    """Subsample to K_frames+1 frames over a WINDOW of the episode, backproject the FIRST sampled
    frame -> GaussianSet, attach per-Gaussian entity (seg) id, and compute the analytic per-Gaussian
    world trajectory over the window. The window is `window_steps` sim steps long (None = to the end of
    the episode = the legacy whole-episode clip). Its START is RANDOM in [0, n_sim-1-window] when an
    `rng` (np.random.Generator) is given, else placed deterministically by `start_frac` within the same
    valid range. This yields 'predict the next ~window-seconds of change from a RANDOM mid-episode state'
    clips (vs always starting at the task beginning)."""
    n_sim = len(rec.frames)
    win = (n_sim - 1) if window_steps is None else int(min(max(1, window_steps), n_sim - 1))
    max_start = max(0, (n_sim - 1) - win)
    if rng is not None:                                                  # RANDOM start within the episode
        start = int(rng.integers(0, max_start + 1))
    else:                                                                # deterministic start_frac within the valid range
        start = int(round(max(0.0, min(1.0, start_frac)) * max_start))
    end = start + win
    idx = np.linspace(start, end, K_frames + 1).round().astype(int)
    idx = np.unique(idx)
    if len(idx) < K_frames + 1:                                          # pad if the window is short
        idx = np.linspace(start, end, K_frames + 1).round().astype(int)
    frames = [rec.frames[i] for i in idx]
    poses = [rec.entity_poses[i] for i in idx]
    Kf = len(idx) - 1                                                    # actual K used

    f0 = frames[0]
    K = f0["K"].to(device).float()
    viewmat = extrinsic_to_viewmat(f0["extrinsic_cv"].to(device).float())   # [4,4] world->cam (shared static cam)
    depth0 = f0["depth"].to(device).float() / 1000.0                    # mm -> m
    rgb0 = (f0["rgb"].to(device).float() / 255.0).permute(2, 0, 1)      # [3,H,W]
    seg0 = f0["seg"].to(device).long()                                   # [H,W]
    H, W = depth0.shape

    # exclude unwanted seg ids (goal marker) and invalid depth from the gaussian set
    u = rec.env.unwrapped
    name_by_sid = {int(sid): getattr(ent, "name", str(sid)) for sid, ent in u.segmentation_id_map.items()}
    exclude_seg = set(exclude_seg or set())
    for sid, nm in name_by_sid.items():
        if nm in EXCLUDE_NAMES:
            exclude_seg.add(int(sid))

    if fuse_stride > 0:
        # ★ WHOLE-VIDEO unified reconstruction: ONE complete canonical Gaussian set fused over the
        # whole video (registers every frame's per-entity points into the canonical pose). Fills the
        # occluded/back geometry the single-frame shell leaves UNCERTAIN -> certain increments.
        g0, uv, seg_per_g = _fuse_canonical_gaussians(
            rec, int(idx[0]), device, exclude_seg, depth_max, fuse_stride=fuse_stride, voxel=fuse_voxel)
        g0 = g0.to(device)
    else:
        # legacy single-frame canonical G0 (one view of the start frame only)
        pts_world, dmask = backproject_to_world(depth0, K, viewmat)          # [H,W,3], [H,W]
        keep = dmask.clone()
        for sid in exclude_seg:
            keep &= (seg0 != sid)
        keep &= (seg0 > 0)                                                   # drop background/no-hit
        # workspace crop: drop FAR ground points (grazing ground stretches to >10m) that inflate the
        # scene radius + carry huge grazing-angle scales. Workspace is ~within depth_max m of the cam.
        keep &= (depth0 < float(depth_max))
        # tighter scale clamp (0.01..0.7 pct) + 0.6 factor (verified ~22 dB frame-0 reconstruction).
        g0, uv = points_to_gaussians(
            pts_world[None], rgb0[None], keep[None],
            opacity_init=0.9, scale_factor=0.6, scale_pct=(0.01, 0.7), return_uv=True,
        )
        seg_per_g = seg0[keep]                                               # [M]
        g0 = g0.to(device)

    # ---- analytic per-Gaussian trajectory ----
    # group gaussians by entity; T_rel(e,t) = T(e,t) @ inv(T(e,0)); X_t = (T_rel @ X0_homog)
    X0 = g0.means                                                        # [N,3]
    N = X0.shape[0]
    traj = torch.empty(Kf + 1, N, 3, device=device, dtype=torch.float32)
    traj[0] = X0
    uniq = torch.unique(seg_per_g).tolist()
    # cache frame-0 entity transforms
    for t in range(Kf + 1):
        pose_t = poses[t]
        if t == 0:
            traj[0] = X0
            continue
        Xt = X0.clone()
        for sid in uniq:
            sid = int(sid)
            m = (seg_per_g == sid)
            rp0 = poses[0].get(sid)
            rpt = pose_t.get(sid)
            if rp0 is None or rpt is None:
                continue                                                # static (no pose) -> identity
            T0 = pose_to_matrix(rp0.to(device).float())
            Tt = pose_to_matrix(rpt.to(device).float())
            T_rel = Tt @ torch.linalg.inv(T0)                           # world delta of this entity
            Xg = X0[m]
            Xg_h = torch.cat([Xg, torch.ones(Xg.shape[0], 1, device=device)], dim=-1)  # [m,4]
            Xt[m] = (Xg_h @ T_rel.T)[:, :3]
        traj[t] = Xt

    return {
        "g0": g0, "uv": uv, "seg_per_g": seg_per_g, "traj": traj,
        "K_intr": K, "viewmat": viewmat, "H": H, "W": W, "Kf": Kf,
        "frames": frames, "poses": poses, "img_hw": (H, W),
        "name_by_sid": name_by_sid, "uniq_seg": uniq,
        "start": int(start), "win": int(win), "n_sim": int(n_sim),
    }


def apply_traj_to_gaussians(g0: GaussianSet, X_t: torch.Tensor, seg_per_g, poses_t, poses_0,
                            device) -> GaussianSet:
    """Move a copy of g0 to frame t: set means = X_t and ROTATE quats by each entity's relative
    rotation (so the rendered appearance matches, since gaussians are anisotropic surfels)."""
    g = g0.clone()
    g.means = X_t
    quats = g0.quats.clone()
    for sid in torch.unique(seg_per_g).tolist():
        sid = int(sid)
        rp0 = poses_0.get(sid); rpt = poses_t.get(sid)
        if rp0 is None or rpt is None:
            continue
        R0 = quat_wxyz_to_rotmat(rp0.to(device).float()[3:7])
        Rt = quat_wxyz_to_rotmat(rpt.to(device).float()[3:7])
        R_rel = Rt @ R0.T                                               # [3,3]
        from igsw.dynamics.manifold import quat_to_rotmat  # noqa
        m = (seg_per_g == sid)
        # compose R_rel onto each gaussian's rotation: q_new s.t. R_new = R_rel @ R_old
        R_old = quat_wxyz_to_rotmat(quats[m])                           # [k,3,3]
        R_new = R_rel[None] @ R_old                                     # [k,3,3]
        quats[m] = rotmat_to_quat_wxyz(R_new)
    g.quats = quats
    return g


def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """R [...,3,3] -> q [...,4] wxyz. Stable branchless-ish via the standard trace method."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    # case trace > 0
    s = torch.sqrt((t + 1.0).clamp_min(1e-12)) * 2.0
    qw = 0.25 * s
    qx = (m[..., 2, 1] - m[..., 1, 2]) / s
    qy = (m[..., 0, 2] - m[..., 2, 0]) / s
    qz = (m[..., 1, 0] - m[..., 0, 1]) / s
    q[..., 0], q[..., 1], q[..., 2], q[..., 3] = qw, qx, qy, qz
    # for numerical robustness when trace <= 0, fall back per-element
    bad = t <= 0
    if bad.any():
        idx = bad.nonzero(as_tuple=True)
        for i in zip(*idx):
            Ri = m[i]
            qi = _rotmat_to_quat_single(Ri)
            q[i] = qi
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _rotmat_to_quat_single(R: torch.Tensor) -> torch.Tensor:
    d = R.device
    K_ = torch.tensor([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ], device=d, dtype=R.dtype) / 3.0
    w, V = torch.linalg.eigh(K_)
    qxyzw = V[:, torch.argmax(w)]
    return torch.stack([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])  # -> wxyz


# --------------------------------------------------------------------------- #
# validation: render analytic-moved gaussians vs real future RGB
# --------------------------------------------------------------------------- #
def _moving_seg_ids(clip: dict, device: str, thresh: float = 0.01):
    """Seg ids of entities whose mean per-gaussian displacement (frame0->frameK) exceeds `thresh` m."""
    seg = clip["seg_per_g"]; X0 = clip["traj"][0]; XK = clip["traj"][clip["Kf"]]
    out = []
    for sid in clip["uniq_seg"]:
        m = seg == int(sid)
        if m.any() and (XK[m] - X0[m]).norm(dim=-1).mean().item() > thresh:
            out.append(int(sid))
    return out


def validate(clip: dict, device: str, out_dir: str | None = None):
    """Move frame-0 gaussians by the ANALYTIC transforms to frame t, render from the SAME camera,
    compare to the ACTUAL ManiSkill RGB. Reports two PSNRs per frame:
      - full: rendered cloud composited over the static frame-0 background (overall geometry/motion).
      - dyn:  PSNR restricted to the DYNAMIC image region = union of the real-segmentation pixels of
              the moving entities at frame 0 (where the object WAS) and frame t (where it IS). This
              directly tests that the analytically-moved gaussians land where the real object moved
              to (the background composite cannot hide a motion error here).
    """
    g0 = clip["g0"]; seg_per_g = clip["seg_per_g"]; traj = clip["traj"]
    K = clip["K_intr"]; viewmat = clip["viewmat"]; H, W = clip["H"], clip["W"]
    poses = clip["poses"]; Kf = clip["Kf"]; frames = clip["frames"]
    move_ids = _moving_seg_ids(clip, device)
    seg0_img = frames[0]["seg"].to(device).long()                        # [H,W] real seg at frame 0
    dyn0 = torch.zeros(H, W, dtype=torch.bool, device=device)
    for sid in move_ids:
        dyn0 |= (seg0_img == sid)

    full, dyn = [], []
    for t in range(Kf + 1):
        g_t = apply_traj_to_gaussians(g0, traj[t], seg_per_g, poses[t], poses[0], device)
        colors, alphas, _ = render_gaussianset(g_t, viewmat[None], K[None], W, H)
        pred = colors[0].clamp(0, 1)                                     # [H,W,3] (over black)
        gt = frames[t]["rgb"].to(device).float() / 255.0                # [H,W,3]
        a = alphas[0].clamp(0, 1)                                        # [H,W,1]
        bg0 = frames[0]["rgb"].to(device).float() / 255.0
        pred_c = pred + (1.0 - a) * bg0                                  # composite over static bg
        full.append(psnr(pred_c, gt))
        # dynamic-region mask = (object pixels at t in REAL seg) ∪ (object pixels at 0)
        segt_img = frames[t]["seg"].to(device).long()
        dynt = dyn0.clone()
        for sid in move_ids:
            dynt |= (segt_img == sid)
        if dynt.any():
            mse = (((pred_c - gt) ** 2)[dynt]).mean().clamp_min(1e-12)
            dyn.append(float(10.0 * torch.log10(1.0 / mse)))
        else:
            dyn.append(float("nan"))
        if out_dir is not None:
            import imageio.v3 as iio
            os.makedirs(out_dir, exist_ok=True)
            comp = (torch.cat([gt, pred_c], dim=1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            iio.imwrite(os.path.join(out_dir, f"val_t{t:02d}_psnr{full[-1]:.1f}.png"), comp)
    return full, dyn, [clip["name_by_sid"].get(s, s) for s in move_ids]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PickCube-v1")
    ap.add_argument("--policy", default="auto", choices=["auto", "rotate"],
                    help="rotate = grasp-then-spin-in-place (Track B rotation-rich data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--cam", type=int, default=512, help="square camera resolution")
    ap.add_argument("--depth_max", type=float, default=2.0, help="drop backprojected points beyond this (m)")
    ap.add_argument("--start_frac", type=float, default=0.0, help="anchor the clip at a MID-EPISODE frame (predict continuation from mid-manipulation)")
    ap.add_argument("--out", default="data/maniskill/clip.pt")
    ap.add_argument("--valdir", default="outputs/maniskill_val")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"[maniskill_gt] env={args.env} seed={args.seed} K={args.K} cam={args.cam} dev={dev}", flush=True)

    rec, instruction, success = generate_episode(args.env, args.seed, args.cam, args.cam, policy=args.policy)
    print(f"[maniskill_gt] instruction: {instruction!r}", flush=True)
    clip = build_clip(rec, args.K, dev, depth_max=args.depth_max, start_frac=args.start_frac)
    print(f"[maniskill_gt] dense gaussians N={len(clip['g0'])}  Kf={clip['Kf']}  "
          f"entities={[clip['name_by_sid'].get(s, s) for s in clip['uniq_seg']]}", flush=True)

    # report per-entity motion magnitude (sanity: cube + arm move, table/ground do not).
    # Normalize displacements by a workspace radius computed from the MOVING entities' spread
    # (not the full ground), so the corr/topmover metrics in the overfit are meaningful.
    X0 = clip["traj"][0]; XK = clip["traj"][clip["Kf"]]
    disp = (XK - X0).norm(dim=-1)
    move_ids = _moving_seg_ids(clip, dev)
    movemask = torch.zeros(len(clip["seg_per_g"]), dtype=torch.bool, device=dev)
    for sid in move_ids:
        movemask |= (clip["seg_per_g"] == sid)
    ref = X0[movemask] if movemask.any() else X0
    radius = (ref - ref.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    print(f"[maniskill_gt] workspace radius(p90 over movers)={radius:.3f}  GT disp/radius: "
          f"p50={disp.median()/radius:.3f} p90={disp.quantile(0.9)/radius:.3f} max={disp.max()/radius:.3f}  "
          f"frac>0.02r={(disp/radius > 0.02).float().mean():.3f}", flush=True)
    for sid in clip["uniq_seg"]:
        m = clip["seg_per_g"] == int(sid)
        if m.any():
            d = (XK[m] - X0[m]).norm(dim=-1).mean() / radius
            tag = "  <-- MOVER" if int(sid) in move_ids else ""
            print(f"    entity {clip['name_by_sid'].get(int(sid), sid):24s} n={int(m.sum()):6d} "
                  f"mean_disp={d:.3f}r{tag}", flush=True)

    print("[maniskill_gt] === CRITICAL VALIDATION: analytic-moved gaussians vs real RGB ===", flush=True)
    full, dyn, mover_names = validate(clip, dev, out_dir=args.valdir)
    print(f"[maniskill_gt] moving entities: {mover_names}", flush=True)
    print(f"[maniskill_gt] full-frame PSNR/frame:    " + " ".join(f"{p:.1f}" for p in full), flush=True)
    print(f"[maniskill_gt] dynamic-region PSNR/frame: " + " ".join(f"{p:.1f}" for p in dyn), flush=True)
    print(f"[maniskill_gt] FULL: frame0={full[0]:.2f}  mean(t>=1)={np.mean(full[1:]):.2f}  min(t>=1)={np.min(full[1:]):.2f}", flush=True)
    dyn1 = [x for x in dyn[1:] if x == x]
    print(f"[maniskill_gt] DYN : frame0={dyn[0]:.2f}  mean(t>=1)={np.mean(dyn1):.2f}  min(t>=1)={np.min(dyn1):.2f}", flush=True)
    psnrs = full

    # save the clip (cpu tensors) for the overfit
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save = {
        "means": clip["g0"].means.cpu(), "quats": clip["g0"].quats.cpu(),
        "scales": clip["g0"].scales.cpu(), "opacities": clip["g0"].opacities.cpu(),
        "colors": clip["g0"].colors.cpu(),
        "uv": clip["uv"].cpu(), "seg_per_g": clip["seg_per_g"].cpu(),
        "traj": clip["traj"].cpu(), "K_intr": clip["K_intr"].cpu(),
        "viewmat": clip["viewmat"].cpu(), "H": clip["H"], "W": clip["W"], "Kf": clip["Kf"],
        "instruction": instruction, "env": args.env, "seed": args.seed,
        "val_psnr": psnrs,
        # store the real RGB frames so the overfit / further checks can reuse them
        "gt_rgb": torch.stack([f["rgb"] for f in clip["frames"]], 0),     # [Kf+1,H,W,3] uint8
    }
    torch.save(save, args.out)
    print(f"[maniskill_gt] saved clip -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
