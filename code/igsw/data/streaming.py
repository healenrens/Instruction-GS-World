"""Streaming clip dataset — JIT decode, no pre-cached storage (scale-up).

An infinite IterableDataset: CPU DataLoader workers pick random (task, episode,
f0) across ALL tasks and seek-decode a short head-camera window on the fly. The
trainer then lifts (Pi3) + trains + discards — so we stream over the entire
multi-thousand-hour corpus with bounded memory and zero clip storage.

Each yielded item = {frames uint8[K+1,H,W,3], instruction, task, ep, f0}.
Workers/ranks use disjoint RNG seeds so each samples a different random stream.
"""

from __future__ import annotations

import os
import random

import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from .lerobot_agibot import AgiBotLeRobotTask, list_tasks, DEFAULT_BETA_ROOT


def build_clip_index(task_roots, K, stride, margin):
    """Segment-level index: [(task_idx, ep, seg_start, seg_end, action_text)].

    Uses AgiBot's fine-grained sub-task segments (episodes.jsonl::action_config) so each
    sampled clip lies WITHIN one coherent sub-action and carries that segment's action_text
    (fine-grained, clip-varying language). Falls back to the whole episode + task string when
    no action_config exists.
    """
    span = K * stride
    index = []
    for ti, tr in enumerate(task_roots):
        try:
            t = AgiBotLeRobotTask(tr)
        except Exception:
            continue
        for ep in t.episode_indices:
            L = t.episode_meta(ep).length
            segs = t.subtasks(ep)
            if segs:
                for s in segs:
                    a, b = int(s.get("start_frame", 0)), int(s.get("end_frame", 0))
                    b = min(b, L)
                    txt = (s.get("action_text") or "").strip()
                    if txt and (b - a) > span + 2 * margin:
                        index.append((ti, ep, a, b, txt))
            elif L > span + 2 * margin:                  # fallback: whole episode + task string
                index.append((ti, ep, margin, L - margin, t.language(ep)))
    return index


class StreamingClipDataset(IterableDataset):
    def __init__(self, task_roots=None, K=12, stride=3, margin=30,
                 cam="observation.images.head", seed=0, beta_root=DEFAULT_BETA_ROOT,
                 load_actions=True, boundary_frac=0.35, boundary_ratio=4.0, middle_weight=0.3):
        super().__init__()
        self.task_roots = task_roots or list_tasks(beta_root)
        self.K = K; self.stride = stride; self.margin = margin; self.cam = cam; self.seed = seed
        self.load_actions = load_actions
        # boundary-biased sampling: oversample the ONSET of each sub-task (where the static scene
        # is least informative about which action follows -> language is necessary), and down-weight
        # mid-action clips (auxiliary). boundary_ratio = sampling odds boundary:middle.
        self.boundary_frac = boundary_frac
        self.boundary_ratio = boundary_ratio
        self.middle_weight = middle_weight
        self.index = build_clip_index(self.task_roots, K, stride, margin)
        if not self.index:
            raise RuntimeError("empty clip index")
        self._tasks: dict[int, AgiBotLeRobotTask] = {}
        self._act_cache: dict = {}   # (ti,ep) -> (eef[T,2,3], grip[T,2]); per-worker

    @property
    def n_episodes(self):
        return len(self.index)

    def _task(self, ti):
        if ti not in self._tasks:
            self._tasks[ti] = AgiBotLeRobotTask(self.task_roots[ti])
        return self._tasks[ti]

    def _action_seq(self, ti, ep, f0):
        """Per-step end-effector ACTION sequence for the clip: [K, 8] =
        dual-arm Δposition(6) + dual-gripper Δ(2). The COMMANDED action is the
        control input that causes the future — non-redundant with the scene."""
        import numpy as np
        key = (ti, ep)
        if key not in self._act_cache:
            t = self._task(ti)
            cols = t.read_parquet(ep, columns=["actions.end.position", "actions.effector.position"])
            self._act_cache[key] = (np.asarray(cols["actions.end.position"], dtype=np.float32),
                                    np.asarray(cols["actions.effector.position"], dtype=np.float32))
            if len(self._act_cache) > 32:
                self._act_cache.pop(next(iter(self._act_cache)))
        eef, grip = self._act_cache[key]                       # [T,2,3], [T,2]
        idx = [f0 + j * self.stride for j in range(self.K + 1)]
        eef_c = eef[idx].reshape(self.K + 1, -1)               # [K+1,6]
        grip_c = grip[idx]                                     # [K+1,2]
        dpos = eef_c[1:] - eef_c[:-1]                          # [K,6]
        dgrip = grip_c[1:] - grip_c[:-1]                       # [K,2]
        return np.concatenate([dpos, dgrip], axis=-1).astype(np.float32)   # [K,8]

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rng = random.Random(self.seed + 100003 * rank + 9973 * wid)
        span = self.K * self.stride
        p_boundary = self.boundary_ratio / (self.boundary_ratio + 1.0)
        while True:
            ti, ep, seg_a, seg_b, instr = rng.choice(self.index)
            lo = seg_a + self.margin
            hi_full = seg_b - span - self.margin                    # latest f0 (clip fits in segment)
            # onset window = clip STARTS in [lo, lo + boundary_frac*seg_len]; middle = strictly after it
            hi_bnd = min(lo + int(self.boundary_frac * (seg_b - seg_a)), hi_full)
            want_boundary = (rng.random() < p_boundary)
            if want_boundary and hi_bnd >= lo:
                f0 = rng.randint(lo, hi_bnd); is_boundary = True
            elif hi_full > hi_bnd:
                f0 = rng.randint(hi_bnd + 1, hi_full); is_boundary = False   # strictly mid-action
            else:                                                            # segment too short for a middle
                f0 = rng.randint(lo, hi_full); is_boundary = True
            boundary_weight = 1.0 if is_boundary else self.middle_weight
            try:
                t = self._task(ti)
                frames = t.decode_window(ep, self.cam, f0, self.stride, self.K + 1)
            except Exception:
                continue
            if frames.shape[0] != self.K + 1:
                continue
            item = {"frames": frames, "instruction": instr, "boundary_weight": boundary_weight,
                    "is_boundary": is_boundary,
                    "task": os.path.basename(self.task_roots[ti]), "ep": ep, "f0": f0}
            if self.load_actions:
                try:
                    item["actions"] = self._action_seq(ti, ep, f0)   # [K,8]
                except Exception:
                    continue
            yield item
