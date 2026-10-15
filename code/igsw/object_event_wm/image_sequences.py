"""Author-published RGB sequences with frame-step, not physical-time, coordinates."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


AUTHOR_EXCLUDED_EPISODES = {"episode07564", "episode09031", "episode13755", "episode11237"}


@dataclass(frozen=True)
class ImageSequenceConfig:
    history_frames: int = 1
    future_frames: int = 9
    sampling: str = "fixed_start"


def prepare_textocvp_manifest(archive_path, output, history_frames=1, future_frames=9):
    from .data import write_json

    archive_path = Path(archive_path).resolve()
    episodes = defaultdict(list)
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            position = next((i for i, part in enumerate(parts)
                             if part.startswith("episode") and part[7:].isdigit()), None)
            if position is not None and not name.endswith("/"):
                episodes[str(PurePosixPath(*parts[:position+1]))].append(name)
        entries, excluded = [], []
        for path, names in sorted(episodes.items()):
            parts = PurePosixPath(path).parts
            split = next(part for part in parts if part in ("train", "val", "valid", "test", "eval"))
            partition = {"train": "train", "val": "development", "valid": "development",
                         "test": "test", "eval": "test"}[split]
            episode = parts[-1]
            frames = sorted((name for name in names if PurePosixPath(name).parent.name == "color"
                             and name.endswith("_color.png")),
                            key=lambda name: int(PurePosixPath(name).name.split("_")[0]))
            reason = ("author_loader_exclusion" if episode in AUTHOR_EXCLUDED_EPISODES else
                      "short_sequence" if len(frames) < history_frames+future_frames else None)
            if reason:
                excluded.append({"episode": path, "reason": reason, "frames": len(frames)})
                continue
            caption_path = str(PurePosixPath(path) / "task_description.txt")
            instruction = archive.read(caption_path).decode("utf-8").strip()
            case_id = f"textocvp_{split}_{episode}"
            provenance = {"source": "author_released_caption", "archive": str(archive_path),
                          "member": caption_path, "instruction_scope": "episode_goal"}
            entries.append({"case_id": case_id, "source": "textocvp_cliport", "group": instruction,
                            "episode_index": int(episode[7:]), "partition": partition,
                            "original_partition": split,
                            "record": {"adapter": "image_zip", "path": str(archive_path),
                                       "frame_files": frames},
                            "language_windows": [{"window_id": case_id+"_first_frame",
                                                  "instruction": instruction,
                                                  "language_provenance": provenance}]})
    payload = {"contract": "object_event_image_sequence_v1", "root": str(archive_path.parent),
               "source_archive": str(archive_path), "entries": entries,
               "sequence": vars(ImageSequenceConfig(history_frames, future_frames)),
               "time_unit": "sampled_frame", "physical_fps": None,
               "sampling": "first_contiguous_frames_in_author_episode",
               "author_loader": "https://github.com/angelvillar96/TextOCVP/blob/master/src/data/CLIPort.py",
               "split_use": {"train": "author_train", "development": "author_val", "test": "author_test"},
               "independent_object_truth_used": False,
               "report": {"episodes": len(entries), "partitions": dict(Counter(e["partition"] for e in entries)),
                          "instructions": len({e["group"] for e in entries}), "excluded": excluded}}
    write_json(output, payload)
    return payload


def read_image_sequence(record, indices, archive=None):
    def read(opened):
        frames = []
        for index in indices:
            with opened.open(record["frame_files"][int(index)]) as stream, Image.open(stream) as image:
                frames.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        return torch.stack(frames)

    if archive is not None:
        return read(archive)
    with ZipFile(record["path"]) as opened:
        return read(opened)


class ImageSequenceDataset(Dataset):
    def __init__(self, manifest, partition="train", config=None, seed=17, text_cache="", sampling=None):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.config = config or ImageSequenceConfig(**self.manifest["sequence"])
        self.seed = seed
        self.episodes = [[entry] for entry in self.manifest["entries"] if entry["partition"] == partition]
        self.archives = {}
        self.text_store = None
        if text_cache:
            from .text import TextFeatureStore
            self.text_store = TextFeatureStore(text_cache)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        index, epoch, occurrence = item if isinstance(item, tuple) else (item, 0, item)
        entry = self.episodes[index][0]
        record = entry["record"]
        if record["path"] not in self.archives:
            self.archives[record["path"]] = ZipFile(record["path"])
        th = self.config.history_frames
        indices = torch.arange(th+self.config.future_frames)
        rgb = read_image_sequence(record, indices, self.archives[record["path"]])
        window = entry["language_windows"][0]
        sample = {"rgb": rgb, "frame_valid": torch.ones(len(indices), dtype=torch.bool),
                  "times": (indices-th+1).float(), "time_unit": "sampled_frame",
                  "time_source": "author_sequence_order_physical_fps_unspecified",
                  "history_frames": th, "frame_indices": indices,
                  "native_hw": torch.tensor(rgb.shape[-2:]),
                  "instruction": window["instruction"], "language_provenance": window["language_provenance"],
                  "window_id": window["window_id"], "case_id": entry["case_id"], "source": entry["source"],
                  "group": entry["group"], "episode_index": entry["episode_index"],
                  "partition": entry["partition"], "camera": "author_provided_view",
                  "media_path": record["path"], "adapter": record["adapter"], "frame_offset": 0,
                  "fps": None, "observation_record": record,
                  "sample_index": index, "epoch": epoch, "occurrence": occurrence}
        if self.text_store is not None:
            sample["text_tokens"] = self.text_store[window["instruction"]]
        return sample
