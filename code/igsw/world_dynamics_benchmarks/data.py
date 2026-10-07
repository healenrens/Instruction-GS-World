"""Decode only the declared observation; labels never enter an encoder call."""

import io

import av
import h5py
import numpy as np
import torch
from PIL import Image


def unique_indices(length, budget):
    count = min(length, budget)
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def read_observation(row, frame_budget):
    if row["format"] == "physion_hdf5":
        indices, times = row["frame_indices"], row["times"]
        with h5py.File(row["path"]) as sample:
            keys = sorted(sample["frames"])
            images = [np.array(Image.open(io.BytesIO(
                sample[f"frames/{keys[index]}/images/_img"][()].tobytes())).convert("RGB"))
                for index in indices]
    else:
        with av.open(row["path"]) as container:
            frames = list(container.decode(video=0))
            # All methods use the same even number of unique frames for the tubelet-2 baseline.
            count = min(len(frames), frame_budget)
            count -= count % 2
            indices = unique_indices(len(frames), count)
            images = [frames[index].to_ndarray(format="rgb24") for index in indices]
            times = [float(frames[index].pts * frames[index].time_base) for index in indices]
    pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()[None]
    h, w = pixels.shape[-2:]
    return {"rgb": pixels, "times": torch.tensor(times, dtype=torch.float32)[None] - times[0],
            "pixel_valid": torch.ones((1, len(images), h, w), dtype=torch.bool),
            "native_hw": torch.tensor([[h, w]]), "frame_indices": indices}


def move_observation(batch, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()}
