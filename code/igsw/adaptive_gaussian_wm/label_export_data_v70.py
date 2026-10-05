"""CPU decode workers and native-resolution batches for offline effect labels."""

import torch
from torch.utils.data import Dataset

from .object_video_rgb_v69 import read_object_video_frames_v69
from .object_video_sequence_dataset_v69 import collate_object_video_v69, distinct_sequence_frames
from .video_file_decoder import VideoDecodeError


def full_window(entry):
    indices = torch.tensor(entry["frame_indices"], dtype=torch.long)
    rgb, times, time_source = read_object_video_frames_v69(entry["case"], indices)
    if isinstance(rgb, VideoDecodeError):
        return {"decode_error": str(rgb), "window_id": entry["window_id"]}
    th = entry["history_frames"]
    return {"rgb": rgb, "times": (times-times[th-1]).float(),
            "frame_valid": distinct_sequence_frames(indices, th), "history_frames": th,
            "native_hw": torch.tensor(rgb.shape[-2:]), "time_source": time_source}


class LabelWindowDatasetV70(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return index, full_window(self.entries[index])


def collate_label_windows_v70(items):
    groups, errors = {}, []
    for index, sample in items:
        if "decode_error" in sample:
            errors.append(sample)
            continue
        key = (*sample["rgb"].shape, sample["history_frames"])
        groups.setdefault(key, []).append((index, sample))
    batches = []
    for group in groups.values():
        indices, samples = zip(*group)
        batches.append((indices, collate_object_video_v69(samples)))
    return {"groups": batches, "errors": errors}
