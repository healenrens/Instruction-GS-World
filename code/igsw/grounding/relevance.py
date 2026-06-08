"""Helpers to turn 2D role masks into per-Gaussian task relevance, and to derive
role phrases from an instruction (v7-min). Background phrases are fixed (static
structure); foreground = manipulator + instruction object-nouns.
"""

from __future__ import annotations

import re
import torch

# confident STATIC structure -> frozen by L_bg_static
BACKGROUND_PHRASES = ["shelf", "rack", "wall", "floor", "table", "cabinet", "counter", "ground", "background"]
# the manipulator always belongs to the moving foreground
MANIPULATOR_PHRASES = ["robot gripper", "robot arm", "robot hand"]

_STOP = set("the a an in on of to into from and or for with at is are be robot please pick place "
            "put move take get item items object objects supermarket environment positioned front "
            "scene this that it its their there here you your we our".split())


def instruction_object_phrases(instruction: str, max_n: int = 4) -> list[str]:
    """Crude noun-ish extraction: content words from the instruction (before any '|')."""
    head = instruction.split("|")[0].lower()
    words = re.findall(r"[a-z]{3,}", head)
    out = []
    for w in words:
        if w not in _STOP and w not in out:
            out.append(w)
    return out[:max_n]


def sample_mask_at_uv(mask: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """mask [H,W] (float), uv [M,2] (x=col,y=row, pixel coords) -> [M] nearest-sample."""
    H, W = mask.shape
    u = uv[:, 0].round().long().clamp(0, W - 1)
    v = uv[:, 1].round().long().clamp(0, H - 1)
    return mask[v, u]


def role_phrases_for_clip(instruction: str) -> tuple[list[str], dict[str, str]]:
    """Return (background_phrases, foreground_role_phrases). Foreground = manipulator +
    instruction object-nouns; each mapped to a role name."""
    fg = {f"manip{i}": p for i, p in enumerate(MANIPULATOR_PHRASES)}
    for i, p in enumerate(instruction_object_phrases(instruction)):
        fg[f"obj{i}"] = p
    return BACKGROUND_PHRASES, fg
