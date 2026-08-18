"""Manifest contract for source-balanced RGB robot-video training."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


MULTISOURCE_VIDEO_CONTRACT = "multisource_robot_video_v1"
SUPPORTED_ADAPTERS = {"rgb_episode_cache", "video_file"}


@dataclass(frozen=True)
class VideoSource:
    name: str
    adapter: str
    weight: float


@dataclass(frozen=True)
class VideoEpisode:
    source_index: int
    episode_index: int
    split: str
    group: str
    path: str
    fps: float
    frame_count: int
    frame_offset: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_multisource_index(path: str) -> tuple[list[VideoSource], list[VideoEpisode], dict]:
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("contract") == MULTISOURCE_VIDEO_CONTRACT, "unknown video index contract")
    raw_sources = payload.get("sources")
    raw_episodes = payload.get("episodes")
    _require(isinstance(raw_sources, list) and raw_sources, "video index has no sources")
    _require(isinstance(raw_episodes, list) and raw_episodes, "video index has no episodes")

    sources = []
    names = set()
    for raw in raw_sources:
        name = str(raw.get("name", ""))
        adapter = str(raw.get("adapter", ""))
        weight = float(raw.get("weight", 0.0))
        _require(name and name not in names, f"duplicate or empty source name: {name}")
        _require(adapter in SUPPORTED_ADAPTERS, f"unsupported source adapter: {adapter}")
        _require(weight > 0.0, f"source weight must be positive: {name}")
        names.add(name)
        sources.append(VideoSource(name, adapter, weight))

    episodes = []
    identities = set()
    checked_paths = set()
    for raw in raw_episodes:
        source_index = int(raw.get("source_index", -1))
        episode_index = int(raw.get("episode_index", -1))
        split = str(raw.get("split", ""))
        group = str(raw.get("group", ""))
        episode_path = os.path.abspath(str(raw.get("path", "")))
        fps = float(raw.get("fps", 0.0))
        frame_count = int(raw.get("frame_count", 0))
        frame_offset = int(raw.get("frame_offset", 0))
        _require(0 <= source_index < len(sources), "episode source index is outside source table")
        identity = (source_index, episode_index)
        _require(identity not in identities, f"duplicate episode identity: {identity}")
        _require(split and group, f"episode {identity} has an empty split or group")
        if episode_path not in checked_paths:
            _require(os.path.isfile(episode_path), f"episode payload is missing: {episode_path}")
            checked_paths.add(episode_path)
        _require(fps > 0.0 and frame_count > 0 and frame_offset >= 0, f"invalid timing for {identity}")
        identities.add(identity)
        episodes.append(
            VideoEpisode(
                source_index, episode_index, split, group, episode_path,
                fps, frame_count, frame_offset,
            )
        )
    return sources, episodes, payload
