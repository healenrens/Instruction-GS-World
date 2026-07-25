"""Multi-view scan ManiSkill env wrappers for Stage-B DynamicGS clips."""
from __future__ import annotations

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from scripts.maniskill_gt import (
    INSTRUCTIONS,
    _ObsRecorder,
    _manip_object,
    _script_pick,
    _script_push,
)


def _scan_configs(env):
    target = [0.0, 0.0, 0.05]
    eyes = [
        [0.30, 0.00, 0.60],
        [0.00, 0.36, 0.56],
        [0.00, -0.36, 0.56],
        [-0.34, 0.00, 0.58],
    ]
    cams = []
    for i, eye in enumerate(eyes):
        pose = sapien_utils.look_at(eye=eye, target=target)
        cams.append(CameraConfig(f"scan_{i}", pose, env.scan_w, env.scan_h, np.pi / 2, 0.01, 100))
    return cams


class PickCubeScanEnv(PickCubeEnv):
    scan_w = 256
    scan_h = 256

    @property
    def _default_sensor_configs(self):
        return _scan_configs(self)


class PushCubeScanEnv(PushCubeEnv):
    scan_w = 256
    scan_h = 256

    @property
    def _default_sensor_configs(self):
        return _scan_configs(self)


class StackCubeScanEnv(StackCubeEnv):
    scan_w = 256
    scan_h = 256

    @property
    def _default_sensor_configs(self):
        return _scan_configs(self)


class MultiViewRecorder(_ObsRecorder):
    def _record(self, obs):
        rec = {"cams": {}}
        for uid, cam in obs["sensor_data"].items():
            prm = obs["sensor_param"][uid]
            rec["cams"][uid] = {
                "rgb": cam["rgb"][0].detach().cpu().clone(),
                "depth": cam["depth"][0, ..., 0].detach().cpu().clone(),
                "seg": cam["segmentation"][0, ..., 0].detach().cpu().clone(),
                "K": prm["intrinsic_cv"][0].detach().cpu().clone(),
                "extrinsic_cv": prm["extrinsic_cv"][0].detach().cpu().clone(),
            }
        self.frames.append(rec)
        self.entity_poses.append(self._entity_pose_snapshot())


SCAN_ENVS = {
    "PickCube-v1": ("PickCubeScanStageB-v1", PickCubeScanEnv, _script_pick),
    "PushCube-v1": ("PushCubeScanStageB-v1", PushCubeScanEnv, _script_push),
    "StackCube-v1": ("StackCubeScanStageB-v1", StackCubeScanEnv, _script_pick),
}


def generate_scan_episode(env_id: str, seed: int, cam: int, max_retries: int = 8):
    if env_id not in SCAN_ENVS:
        raise ValueError(f"unsupported Stage-B scan env: {env_id}")
    scan_id, cls, policy = SCAN_ENVS[env_id]
    cls.scan_w = int(cam)
    cls.scan_h = int(cam)
    if scan_id not in gym.envs.registry:
        gym.register(scan_id, entry_point=cls)
    base = gym.make(
        scan_id, obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", sim_backend="cpu", num_envs=1,
    )
    instruction = INSTRUCTIONS[env_id]
    rec = MultiViewRecorder(base)

    s = int(seed)
    for _ in range(int(max_retries)):
        rec.reset(seed=s)
        obj = _manip_object(base.unwrapped)
        start = obj.pose.p[0].cpu().numpy().copy()
        policy(base, rec)
        moved = float(np.linalg.norm(obj.pose.p[0].cpu().numpy() - start))
        ok = moved > 0.04 and len(rec.frames) > 4
        print(f"[mv] env={env_id} seed={s} moved={moved:.3f} frames={len(rec.frames)} "
              f"{'OK' if ok else 'retry'}", flush=True)
        if ok:
            return rec, instruction
        s += 1
    raise RuntimeError(f"failed to generate moving scan episode for {env_id} seed={seed}")
