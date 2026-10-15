"""The deployment path: observed RGB, instruction features, and requested times."""

import json
from pathlib import Path

import torch

from igsw.adaptive_gaussian_wm.object_video_rgb_v69 import read_object_video_frames_v69
from .codec import letterbox_video
from .state import coordinates


def read_history_request(path, device):
    request = json.loads(Path(path).read_text())
    indices = torch.tensor(request["frame_indices"], dtype=torch.long)
    rgb, times, _ = read_object_video_frames_v69({"record": request["record"]}, indices)
    observed = {"rgb": rgb[None].to(device), "native_hw": torch.tensor([rgb.shape[-2:]], device=device),
                "frame_valid": torch.ones((1, len(rgb)), dtype=torch.bool, device=device),
                "times": (times-times[-1]).float()[None].to(device)}
    return observed


def make_history_request(sample):
    th = sample["history_frames"]
    return {"record": {"path": sample["media_path"], "fps": sample["fps"],
                       "frame_offset": sample["frame_offset"], "adapter": sample["adapter"]},
            "frame_indices": sample["frame_indices"][:th].tolist(), "case_id": sample["case_id"],
            "input": "observed_frames_only"}


def history_batch(batch):
    count = batch["history_frames"]
    return {"rgb": batch["rgb"][:, :count], "native_hw": batch["native_hw"],
            "frame_valid": batch["frame_valid"][:, :count], "times": batch["times"][:, :count]}


def observe_history(model, codec, observed):
    images, pixel_valid = letterbox_video(observed, codec.image_size)
    features = codec.encode(images)
    states = model.observe(features, codec.patch_valid(pixel_valid), observed["times"])
    return states


def generate_from_history(model, codec, prior, states, requested_times, text_tokens=None,
                          text_valid=None, generator=None, deterministic=False):
    result = prior.generate(states, requested_times, model.dynamics, text_tokens, text_valid,
                            generator=generator, deterministic=deterministic)
    features, masks = model.render_features(result["states"], coordinates(codec.grid_size, requested_times.device))
    return {**result, "rgb": codec.decode(features), "masks": masks}
