"""Runtime helpers for Stage-B multiview DynamicGS clips."""
from __future__ import annotations

import torch


def is_stage_b_clip(clip: dict) -> bool:
    return clip.get("contract_version") == "dynamic_gs_stage_b_v1"


def move_vlm_inputs(inputs: dict, dev, dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
        else:
            out[k] = v
    return out


def image_uint8_numpy(image):
    if torch.is_tensor(image):
        return image.detach().to(torch.uint8).cpu().numpy()
    return image


def build_single_view_inputs(encoder, instruction: str, image, dev, dtype):
    img = None if image is None else image_uint8_numpy(image)
    return move_vlm_inputs(encoder.build_inputs(instruction, img), dev, dtype)


def build_multiview_inputs(encoder, instruction: str, scan_rgb, policy_view: int, dev, dtype):
    views = [
        build_single_view_inputs(encoder, instruction, scan_rgb[v], dev, dtype)
        for v in range(int(scan_rgb.shape[0]))
    ]
    return views[int(policy_view)], views
