"""History-only native RGB and small, precomputed fixed-teacher effect labels."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .object_video_manifest_v69 import load_object_video_manifest_v69, resolve_object_video_case_v69
from .object_video_rgb_v69 import read_object_video_frames_v69


class LanguageEffectDatasetV70(Dataset):
    """Entries keep episode sampler keys; tuple visits do not resample fixed windows.

    Labels contain mean/logvar and query={xy, frame_index, features, valid},
    with teacher metadata. The parent's flat query_* export is also accepted.
    Coordinates and feature ordering are consumed unchanged from the teacher.
    load_labels=False still decodes only history; offline full-RGB export is
    independent and must explicitly read all 41 entry.frame_indices.
    """

    def __init__(self, manifest, partition="train", load_labels=True):
        self.path = str(Path(manifest).resolve())
        self.manifest = json.loads(Path(self.path).read_text(encoding="utf-8"))
        stage2_path = (Path(self.path).parent / self.manifest["stage2_manifest"]).resolve()
        root = load_object_video_manifest_v69(stage2_path)["root"]
        self.entries = [{**entry, "case": resolve_object_video_case_v69(entry["case"], root),
                         "label_path": str((Path(self.path).parent / entry["label_path"]).resolve())}
                        for entry in self.manifest["entries"] if entry["partition"] == partition]
        self.load_labels = load_labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        index, epoch, occurrence = item if isinstance(item, tuple) else (item, 0, item)
        entry = self.entries[index]
        indices = torch.tensor(entry["frame_indices"], dtype=torch.long)
        history_indices = indices[:16]
        rgb, timestamps, time_source = read_object_video_frames_v69(entry["case"], history_indices)
        frame_valid = torch.ones(16, dtype=torch.bool)
        frame_valid[:-1] = history_indices[:-1] != history_indices[1:]
        sample = {"rgb": rgb, "frame_valid": frame_valid, "times": (timestamps - timestamps[-1]).float(),
                  "native_hw": torch.tensor(rgb.shape[-2:]), "history_frames": 16,
                  "instruction": entry["instruction"], "window_id": entry["window_id"],
                  "case_id": entry["case_id"], "partition": entry["partition"],
                  "stage2_partition": entry["stage2_partition"],
                  "source": entry["source"], "group": entry["group"], "episode_index": entry["episode_index"],
                  "frame_indices": indices, "case": entry["case"], "time_source": time_source,
                  "sample_index": index, "epoch": epoch, "occurrence": occurrence,
                  "language_provenance": entry["language_provenance"]}
        if self.load_labels:
            label = torch.load(entry["label_path"], map_location="cpu", weights_only=False)
            query = label["query"] if "query" in label else {
                "xy": label["query_xy"], "frame_index": label["query_frame_index"],
                "features": label["query_features"], "valid": label["query_valid"]}
            sample.update(target_mean=label["mean"], target_logvar=label["logvar"],
                          query_xy=query["xy"], query_frame_index=query["frame_index"],
                          query_features=query["features"], query_valid=query["valid"],
                          teacher_metadata=label.get("teacher_metadata", label.get("teacher")))
        return sample


def collate_language_effect_v70(samples):
    """V69 top-left RGB padding to multiples of 16, with native coordinate scale."""
    height = (max(sample["rgb"].shape[-2] for sample in samples) + 15) // 16 * 16
    width = (max(sample["rgb"].shape[-1] for sample in samples) + 15) // 16 * 16
    rgb = torch.zeros((len(samples), 16, 3, height, width), dtype=torch.uint8)
    pixel_valid = torch.zeros((len(samples), 16, height, width), dtype=torch.bool)
    for index, sample in enumerate(samples):
        h, w = sample["rgb"].shape[-2:]
        rgb[index, :, :, :h, :w] = sample["rgb"]
        pixel_valid[index, :, :h, :w] = sample["frame_valid"][:, None, None]
    batch = {"rgb": rgb, "pixel_valid": pixel_valid, "history_frames": 16}
    for key in samples[0]:
        if key in ("rgb", "history_frames"):
            continue
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return batch
