"""Read lossless simulated observations; evaluator truth stays outside model batches."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SequenceConfig, write_json


def requested_times(config):
    return np.concatenate((np.linspace(-config.history_seconds, 0., config.history_frames),
                           np.arange(1, config.future_frames+1)*config.future_seconds/config.future_frames))


def nearest_indices(recorded, requested):
    right = np.searchsorted(recorded, requested)
    right = np.minimum(right, len(recorded)-1)
    left = np.maximum(right-1, 0)
    return np.where(abs(recorded[left]-requested) <= abs(recorded[right]-requested), left, right)


def load_episode(path):
    path = Path(path)
    metadata = json.loads(path.read_text())
    with np.load(path.parent/metadata["rgb"], allow_pickle=False) as stored:
        rgb, times = stored["rgb"], stored["times_seconds"]
    return metadata, rgb, times


def load_truth(path, indices):
    path = Path(path)
    metadata = json.loads(path.read_text())
    with np.load(path.parent/metadata["truth"], allow_pickle=False) as stored:
        result = {key: stored[key][indices] for key in
                  ("segmentation", "poses_xyz_xyzw", "visible_pixels", "times_seconds", "physics_ticks")}
        result["body_ids"] = stored["body_ids"]
    return result


def prepare_simulation_manifest(collection, output, config=None):
    collection = Path(collection).resolve()
    source = json.loads(collection.read_text())
    config = config or SequenceConfig(history_seconds=source["history_seconds"],
                                      future_seconds=source["future_seconds"], history_frames=16, future_frames=50,
                                      sampling="fixed_anchor")
    entries = []
    for entry in source["episodes"]:
        episode = collection.parent/entry["episode"]
        meta = json.loads(episode.read_text())
        entries.append({**entry, "episode": str(episode), "source": "cliport_ordered",
                        "group": str(entry["scene_seed"]), "episode_index": entry["scene_seed"],
                        "language_windows": [{"window_id": entry["case_id"]+"_fixed_horizon",
                            "instruction": meta["instruction"], "language_provenance": {
                                "source": "simulator_program", "instruction_scope": "requested_event_order",
                                "program_realized": meta["status"] == "completed"}}]})
    manifest = {"contract": "object_event_sim_sequence_v1", "root": str(collection.parent),
                "source_collection": str(collection), "sequence": vars(config), "entries": entries,
                "time_unit": "seconds", "split_use": source["partition"],
                "sampling": "scene_uniform_train_then_uniform_available_instruction",
                "language_training": "only completed oracle programs; incomplete RGB retained for joint reconstruction",
                "truth_used_by_model_input": False, "plan": source.get("plan")}
    write_json(output, manifest)
    return manifest


class SimulationVideoDataset(Dataset):
    def __init__(self, manifest, partition="train", config=None, seed=17, text_cache="", sampling=None):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.config = config or SequenceConfig(**self.manifest["sequence"])
        self.seed = seed
        grouped = defaultdict(list)
        for entry in self.manifest["entries"]:
            if entry["partition"] == partition:
                if sampling == "language_windows" and entry["status"] != "completed":
                    continue
                key = entry["scene_seed"] if partition == "train" else entry["case_id"]
                grouped[key].append(entry)
        self.episodes = [grouped[key] for key in sorted(grouped)]
        self.text_store = None
        if text_cache:
            from .text import TextFeatureStore
            self.text_store = TextFeatureStore(text_cache)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        index, epoch, occurrence = item if isinstance(item, tuple) else (item, 0, item)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, occurrence, index]))
        entry = self.episodes[index][int(rng.integers(len(self.episodes[index])))]
        meta, rgb, times = load_episode(entry["episode"])
        desired = requested_times(self.config)
        indices = nearest_indices(times, desired)
        valid = np.isclose(times[indices], desired, atol=.5/meta["physics_hz"], rtol=0)
        rgb = torch.from_numpy(rgb[indices].copy()).permute(0, 3, 1, 2)
        window = entry["language_windows"][0]
        parent = Path(entry["episode"]).parent
        sample = {"rgb": rgb, "frame_valid": torch.from_numpy(valid),
                  "times": torch.tensor(desired, dtype=torch.float32), "time_unit": "seconds",
                  "time_source": "requested_fixed_grid_matched_to_physics_ticks",
                  "history_frames": self.config.history_frames, "frame_indices": torch.from_numpy(indices),
                  "native_hw": torch.tensor(rgb.shape[-2:]), "instruction": meta["instruction"],
                  "language_provenance": window["language_provenance"], "window_id": window["window_id"],
                  "case_id": entry["case_id"], "source": entry["source"], "group": entry["group"],
                  "episode_index": entry["episode_index"], "partition": entry["partition"],
                  "camera": "cliport_front", "media_path": str(parent/meta["rgb"]),
                  "adapter": "simulator_npz", "frame_offset": 0,
                  "fps": meta["physics_hz"]/meta["capture_every_physics_steps"],
                  "observation_record": {"adapter": "simulator_npz", "path": str(parent/meta["history_rgb"])},
                  "episode_path": entry["episode"], "sample_index": index, "epoch": epoch, "occurrence": occurrence}
        if self.text_store is not None:
            sample["text_tokens"] = self.text_store[meta["instruction"]]
        return sample
