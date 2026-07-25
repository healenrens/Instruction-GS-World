"""Loss-weight contract shared by training modes and the joint objective."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveGaussianLossWeights:
    future: float = 1.0
    history: float = 0.5
    flow: float = 0.1
    feature: float = 0.1
    allocator: float = 0.1
    slot: float = 0.01
    action: float = 0.1
    action_specificity: float = 0.0
    geometry: float = 0.0
    rgb: float = 1.0
