"""Read frozen teacher shards; RGB history is independent of future target selection."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .grounded_motion_sources_v68 import SOURCES, permitted_camera
from .grounded_motion_export_v68 import SELECTION_POLICY
from .tracker_visual_review_v67 import decode_case


class GroundedMotionDatasetV68(Dataset):
    def __init__(self, manifest, points=256, seed=17, horizons=(1.0, 3.0)):
        self.path = str(Path(manifest).resolve())
        self.manifest = json.loads(Path(manifest).read_text())
        self.root = Path(self.manifest["root"])
        self.entries = [e for e in self.manifest["entries"] if e["source"] in SOURCES and e["object_targets"] > 0
                        and permitted_camera(e["source"], e["camera"])]
        self.points, self.seed, self.horizons = points, seed, horizons

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        entry = self.entries[index]
        data = torch.load(self.root / entry["path"], map_location="cpu", weights_only=False)
        case, native, queries = data["case"], data["native"], data["queries"]
        fps, frames = case["record"]["fps"], len(native["tracks"])
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + index)
        history_count = int(rng.integers(1, 5))
        stride = max(1, round(.1 * fps))
        current = int(rng.integers(3 * stride, frames - round(max(self.horizons) * fps)))
        history = torch.tensor([current - j * stride for j in (3, 2, 1, 0)])
        future = torch.tensor([current + round(seconds * fps) for seconds in self.horizons])
        indices = torch.cat((history, future))
        rgb = decode_case(case, indices + case["first_frame"])
        history_valid = torch.arange(4) >= 4 - history_count
        rgb[:4][~history_valid] = 0
        visible = native["visibility"] & native["in_bounds"] & torch.isfinite(native["tracks"]).all(-1)
        available = visible[current] & (queries["frames"] <= case["first_frame"] + current)
        all_roles = data.get("selection_policy") == SELECTION_POLICY
        selection = data["motion_target_mask"] if all_roles else data["object_motion_target_mask"]
        if all_roles:
            available &= selection
        # These queries are loss quadrature points, never inputs to encode_history().
        ids = torch.where(available)[0].numpy()
        rng.shuffle(ids)
        selected = ids[:self.points]
        count = len(selected)
        shape = (len(indices), self.points)
        coordinates = torch.zeros((*shape, 2))
        coordinates[:, :count] = native["tracks"][indices[:, None], torch.as_tensor(selected)[None]]
        valid = torch.zeros(shape, dtype=torch.bool)
        valid[:, :count] = visible[indices[:, None], torch.as_tensor(selected)[None]]
        # Invalid teacher locations have no geometric target, including no NaN payload.
        coordinates = torch.where(valid[..., None], coordinates, torch.zeros_like(coordinates))
        motion = torch.zeros(self.points, dtype=torch.bool)
        motion[:count] = selection[selected]
        reliable = torch.zeros((2, self.points), dtype=torch.bool)
        reliable[:, :count] = data["target_valid"][future[:, None], torch.as_tensor(selected)[None]]
        # The given original query frame is not an independent tracking observation.
        reliable[:, :count] &= data["target_valid"][current, selected][None]
        reliable &= valid[3:4]
        h, w = rgb.shape[-2:]
        normalized = coordinates / torch.tensor([w - 1, h - 1]).clamp_min(1) * 2 - 1
        regions = torch.full((self.points,), -1, dtype=torch.long)
        roles = torch.full((self.points,), -1, dtype=torch.long)
        names = sorted({queries["metadata"][int(i)]["region_id"] for i in selected})
        for j, i in enumerate(selected):
            if all_roles:
                continue
            regions[j] = names.index(queries["metadata"][int(i)]["region_id"])
            role = queries["metadata"][int(i)]["role"]
            roles[j] = {"robot_context": 1, "scene_context": 2}.get(role, -1)
            if motion[j]:
                roles[j] = 0
        residual = data["motion_evidence"]["compensated_coordinates"]
        delta = torch.zeros((self.points, 2))
        delta[:count] = residual[future[0], selected] - residual[current, selected]
        noise = torch.full((self.points,), float("inf"))
        noise[:count] = data["motion_evidence"]["threshold_px"][selected]
        different = (delta[:, None] - delta[None]).norm(dim=-1) > noise[:, None] + noise[None]
        different &= reliable[0, :, None] & reliable[0, None, :]
        if not all_roles:
            different &= regions[:, None] != regions[None]
        point_ids = torch.full((self.points,), -1, dtype=torch.long)
        point_ids[:count] = torch.as_tensor(selected)
        return {"video_rgb": rgb, "history_valid": history_valid, "coordinates": normalized,
                "point_valid": valid, "target_valid": reliable, "motion_mask": motion, "region_ids": regions,
                "roles": roles, "different_motion_evidence": different,
                "frame_times": indices.float() / fps, "native_image_hw": torch.tensor([h, w]),
                "point_ids": point_ids, "frame_indices": indices + case["first_frame"],
                "source": entry["source"], "case_id": entry["case_id"], "sample_index": index,
                "data_epoch": epoch, "sampler_seed": self.seed,
                "selection_policy": data.get("selection_policy", "legacy_object_region_selection")}


def collate_grounded_motion_v68(samples):
    height = (max(s["video_rgb"].shape[-2] for s in samples) + 3) // 4 * 4
    width = (max(s["video_rgb"].shape[-1] for s in samples) + 3) // 4 * 4
    rgb = torch.zeros((len(samples), 6, 3, height, width), dtype=torch.uint8)
    pixels = torch.zeros((len(samples), 6, height, width), dtype=torch.bool)
    for i, sample in enumerate(samples):
        h, w = sample["video_rgb"].shape[-2:]
        rgb[i, :, :, :h, :w] = sample["video_rgb"]
        pixels[i, :, :h, :w] = True
        pixels[i, :4] &= sample["history_valid"][:, None, None]
    result = {"video_rgb": rgb, "video_pixel_valid": pixels}
    for key in samples[0]:
        if key != "video_rgb":
            result[key] = torch.stack([s[key] for s in samples]) if isinstance(samples[0][key], torch.Tensor) else [s[key] for s in samples]
    return result


class GroundedMotionSamplerV68(Sampler):
    def __init__(self, dataset, rank, world, seed, batch):
        self.dataset, self.rank, self.world, self.seed, self.batch = dataset, rank, world, seed, batch
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = [[i for i, e in enumerate(self.dataset.entries) if e["source"] == source] for source in SOURCES]
        groups = [g for g in groups if g]
        count = max(map(len, groups))
        mixed = []
        for group in groups:
            permutation = torch.randperm(len(group), generator=generator).tolist()
            mixed.append([group[permutation[i % len(group)]] for i in range(count)])
        indices = [row for part in zip(*mixed) for row in part]
        global_batch = self.batch * self.world
        padding = (-len(indices)) % global_batch
        indices += (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
        return iter((index, self.epoch) for index in indices[self.rank::self.world])

    def __len__(self):
        groups = [sum(e["source"] == s for e in self.dataset.entries) for s in SOURCES]
        size = max(groups) * sum(n > 0 for n in groups)
        global_batch = self.batch * self.world
        return ((size + global_batch - 1) // global_batch) * self.batch
