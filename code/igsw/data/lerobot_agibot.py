"""Faithful reader for the AgiBot-World-beta LeRobot v2.1 datasets on the server.

Dataset layout (confirmed 2026-06-05):
    <root>/meta/info.json          # features, fps, paths, chunks_size
    <root>/meta/tasks.jsonl        # {"task_index": int, "task": str}
    <root>/meta/episodes.jsonl     # {"episode_index": int, "length": int, "tasks": [...]}
    <root>/data/chunk-{c:03d}/episode_{e:06d}.parquet   # per-frame proprio/actions
    <root>/videos/chunk-{c:03d}/<video_key>/episode_{e:06d}.mp4   # AV1, 30 fps

`<root>` for a task is e.g.
    /mnt/pfs/public/agibot-world-beta-lerobot/agibot-world-beta-lerobot/task_327/task_327

No camera calibration is present in this conversion (intrinsics/extrinsics were
dropped); poses must be estimated by the geometry model downstream.

This module performs NO simplification of the format: parquet columns are
reshaped to their declared feature shapes, video frames are decoded with PyAV
(libdav1d) which is the only reliably-AV1-capable path verified on the server.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np


# Canonical camera keys present in AgiBot-World-beta LeRobot.
RGB_CAMERAS = [
    "observation.images.head",                 # 480x640 pinhole-ish (best for lifting)
    "observation.images.head_center_fisheye",  # 768x960 fisheye
    "observation.images.head_left_fisheye",    # 768x960 fisheye
    "observation.images.head_right_fisheye",   # 768x960 fisheye
    "observation.images.hand_left",            # 480x640 wrist
    "observation.images.hand_right",           # 480x640 wrist
    "observation.images.back_left_fisheye",    # 768x960 fisheye
    "observation.images.back_right_fisheye",   # 768x960 fisheye
]


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int
    tasks: list[str]
    # fine-grained per-segment sub-task annotation from episodes.jsonl::action_config
    # each entry: {action_text, skill, start_frame, end_frame}
    action_config: list = None


class AgiBotLeRobotTask:
    """One AgiBot task directory in LeRobot v2.1 format."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        info_path = os.path.join(self.root, "meta", "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"info.json not found under {self.root}/meta")
        with open(info_path) as f:
            self.info: dict = json.load(f)

        self.fps: int = int(self.info["fps"])
        self.chunks_size: int = int(self.info["chunks_size"])
        self.features: dict = self.info["features"]
        self.data_path_tmpl: str = self.info["data_path"]
        self.video_path_tmpl: str = self.info["video_path"]

        # task_index -> language string
        self._tasks: dict[int, str] = {}
        for row in _load_jsonl(os.path.join(self.root, "meta", "tasks.jsonl")):
            self._tasks[int(row["task_index"])] = row["task"]

        # episode_index -> EpisodeMeta
        self._episodes: dict[int, EpisodeMeta] = {}
        for row in _load_jsonl(os.path.join(self.root, "meta", "episodes.jsonl")):
            ei = int(row["episode_index"])
            self._episodes[ei] = EpisodeMeta(
                episode_index=ei,
                length=int(row.get("length", 0)),
                tasks=list(row.get("tasks", [])),
                action_config=list(row.get("action_config", []) or []),
            )

    # ---- discovery -------------------------------------------------------
    @property
    def episode_indices(self) -> list[int]:
        return sorted(self._episodes.keys())

    def episode_meta(self, ep: int) -> EpisodeMeta:
        return self._episodes[ep]

    # ---- fine-grained sub-task semantics (action_config) ----------------
    def subtasks(self, ep: int) -> list:
        """Sub-task segments [{action_text, skill, start_frame, end_frame}, ...]."""
        em = self._episodes.get(ep)
        return em.action_config if (em and em.action_config) else []

    def subtask_at(self, ep: int, frame: int):
        """The sub-task segment containing `frame` (else nearest), or None."""
        segs = self.subtasks(ep)
        if not segs:
            return None
        for s in segs:
            if int(s.get("start_frame", 0)) <= frame < int(s.get("end_frame", 1 << 30)):
                return s
        return min(segs, key=lambda s: abs(int(s.get("start_frame", 0)) - frame))

    def subtask_text_at(self, ep: int, frame: int) -> str:
        """Fine-grained instruction for the segment at `frame`; falls back to task string."""
        s = self.subtask_at(ep, frame)
        if s and s.get("action_text"):
            return s["action_text"]
        return self.language(ep)

    def chunk_of(self, ep: int) -> int:
        return ep // self.chunks_size

    @property
    def video_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    # ---- path helpers ----------------------------------------------------
    def parquet_path(self, ep: int) -> str:
        return os.path.join(
            self.root,
            self.data_path_tmpl.format(episode_chunk=self.chunk_of(ep), episode_index=ep),
        )

    def video_path(self, ep: int, video_key: str) -> str:
        return os.path.join(
            self.root,
            self.video_path_tmpl.format(
                episode_chunk=self.chunk_of(ep), video_key=video_key, episode_index=ep
            ),
        )

    # ---- language --------------------------------------------------------
    def language(self, ep: int) -> str:
        """Primary natural-language instruction for an episode.

        Prefer the per-episode task list (meta/episodes.jsonl); fall back to the
        task_index column of the parquet -> tasks.jsonl mapping.
        """
        em = self._episodes.get(ep)
        if em and em.tasks:
            return em.tasks[0]
        # fall back to parquet task_index
        cols = self.read_parquet(ep, columns=["task_index"])
        ti = int(np.asarray(cols["task_index"]).reshape(-1)[0])
        return self._tasks.get(ti, "")

    # ---- proprio / actions parquet --------------------------------------
    def read_parquet(self, ep: int, columns: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """Read a per-frame parquet into a dict of np arrays reshaped to feature shape.

        Each non-scalar feature is reshaped to [T, *declared_shape].
        """
        import pyarrow.parquet as pq

        path = self.parquet_path(ep)
        table = pq.read_table(path, columns=list(columns) if columns else None)
        out: dict[str, np.ndarray] = {}
        for name in table.column_names:
            col = table.column(name)
            arr = col.to_numpy(zero_copy_only=False)
            # list-typed (and nested-list) columns come back as object arrays of
            # per-row arrays/lists; to_pylist + np.array recovers the full
            # multi-dim float tensor faithfully (e.g. dual-arm [T,2,4] quats).
            if arr.dtype == object:
                arr = np.array(col.to_pylist())
            else:
                arr = np.asarray(arr)
            feat = self.features.get(name, {})
            shape = feat.get("shape")
            if shape and list(shape) != [1]:
                try:
                    arr = arr.reshape((arr.shape[0], *shape))
                except ValueError:
                    pass  # leave as-is if it does not match (defensive)
            out[name] = arr
        return out

    # ---- video frames (PyAV / libdav1d) ----------------------------------
    def num_video_frames(self, ep: int, video_key: str) -> int:
        import av

        with av.open(self.video_path(ep, video_key)) as container:
            stream = container.streams.video[0]
            n = stream.frames
            if n and n > 0:
                return int(n)
            # some AV1 muxes report 0; count by decoding
            return sum(1 for _ in container.decode(stream))

    def decode_frames(
        self, ep: int, video_key: str, indices: Iterable[int]
    ) -> np.ndarray:
        """Decode the requested 0-based frame indices, returns uint8 [N,H,W,3] RGB.

        Decodes sequentially (AV1 random seek is unreliable across muxers); the
        returned order matches the *input* `indices` order. Robust and exact.
        """
        import av

        want = list(indices)
        want_sorted = sorted(set(want))
        target = set(want_sorted)
        max_idx = want_sorted[-1]
        collected: dict[int, np.ndarray] = {}

        with av.open(self.video_path(ep, video_key)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            i = 0
            for frame in container.decode(stream):
                if i in target:
                    collected[i] = frame.to_ndarray(format="rgb24")
                if i >= max_idx:
                    break
                i += 1

        missing = [j for j in want_sorted if j not in collected]
        if missing:
            raise IndexError(
                f"frames {missing[:8]}... not decoded (video has fewer frames) "
                f"for ep {ep} cam {video_key}"
            )
        return np.stack([collected[j] for j in want], axis=0)

    def decode_window(
        self, ep: int, video_key: str, start: int, stride: int, count: int
    ) -> np.ndarray:
        """Efficiently decode a strided window via keyframe SEEK (for streaming).

        Returns uint8 [count,H,W,3] for indices [start, start+stride, ...,
        start+stride*(count-1)]. Seeks to the keyframe at/just-before `start` then
        decodes forward, mapping each frame to its index via pts — so random clips
        deep in long videos don't decode from frame 0.
        """
        import av

        wanted = {start + i * stride for i in range(count)}
        last = start + stride * (count - 1)
        collected: dict[int, np.ndarray] = {}
        path = self.video_path(ep, video_key)
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = stream.time_base
            # seek to just before `start` (in stream time_base units), land on keyframe
            target_ts = int(start / self.fps / tb)
            try:
                container.seek(target_ts, backward=True, any_frame=False, stream=stream)
            except av.AVError:
                container.seek(0, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                idx = int(round(float(frame.pts * tb) * self.fps))
                if idx in wanted:
                    collected[idx] = frame.to_ndarray(format="rgb24")
                if idx >= last:
                    break
        missing = [i for i in sorted(wanted) if i not in collected]
        if missing:
            # fall back to exact sequential decode for the missing ones
            extra = self.decode_frames(ep, video_key, missing)
            for j, i in enumerate(missing):
                collected[i] = extra[j]
        return np.stack([collected[start + i * stride] for i in range(count)], axis=0)

    def iter_frames(
        self, ep: int, video_key: str, start: int = 0, stop: int | None = None, stride: int = 1
    ):
        """Generator yielding (index, uint8 HxWx3 RGB) for a contiguous strided range."""
        import av

        with av.open(self.video_path(ep, video_key)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            i = 0
            for frame in container.decode(stream):
                if stop is not None and i >= stop:
                    break
                if i >= start and (i - start) % stride == 0:
                    yield i, frame.to_ndarray(format="rgb24")
                i += 1


# Default beta root on the server (the nested layout).
DEFAULT_BETA_ROOT = (
    "/mnt/pfs/public/agibot-world-beta-lerobot/agibot-world-beta-lerobot"
)


def list_tasks(beta_root: str = DEFAULT_BETA_ROOT) -> list[str]:
    """Return the LeRobot roots for every task (the inner task_XXX/task_XXX dir)."""
    out = []
    for name in sorted(os.listdir(beta_root)):
        if not name.startswith("task_"):
            continue
        inner = os.path.join(beta_root, name, name)
        if os.path.isfile(os.path.join(inner, "meta", "info.json")):
            out.append(inner)
    return out
