"""CPU end-to-end tensor and backward test for v62 E0 and E1."""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_object_observation_v62 import (  # noqa: E402
    ContinuousObjectObservationV62,
)
from igsw.adaptive_gaussian_wm.teacher_object_autoencoder_v62 import (  # noqa: E402
    TeacherObjectAutoencoderV62,
)
from igsw.adaptive_gaussian_wm.teacher_transition_oracle_v62 import (  # noqa: E402
    TeacherTransitionOracleV62,
)
from igsw.adaptive_gaussian_wm.v62_config import ObjectTransitionConfigV62  # noqa: E402


def synthetic_observation(config):
    batch, frames, points = 4, 3, 32
    coordinates = torch.rand(batch, frames, points, 2) * 2.0 - 1.0
    coordinates[:, 1:] += torch.tensor((0.08, -0.04))
    dino = F.normalize(torch.randn(batch, frames, points, config.semantic_dim), dim=-1)
    siglip = F.normalize(dino + 0.15 * torch.randn_like(dino), dim=-1)
    center = coordinates[:, :, :1]
    support = ((coordinates - center).norm(dim=-1) < 0.65).float()
    visibility = torch.ones_like(support)
    visibility[:, 1, 4:8] = 0.0
    lifecycle = torch.zeros(batch, frames, 3)
    lifecycle[..., 0] = 1.0
    membership = support[:, 0]
    return ContinuousObjectObservationV62(
        coordinates=coordinates,
        dino=dino,
        siglip=siglip,
        support=support,
        visibility=visibility,
        lifecycle=lifecycle,
        membership=membership,
        object_valid=torch.ones(batch, dtype=torch.bool),
        seed_track=torch.zeros(batch, dtype=torch.long),
    )


def finite_gradients(model):
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return len(gradients), all(bool(torch.isfinite(value).all()) for value in gradients)


def main():
    torch.manual_seed(17)
    config = ObjectTransitionConfigV62()
    observation = synthetic_observation(config)
    e0 = TeacherObjectAutoencoderV62(config)
    e0_output = e0(observation)
    e0_output["loss"].backward()
    e0_count, e0_finite = finite_gradients(e0)
    if not e0_finite:
        raise RuntimeError("v62 E0 CPU test produced non-finite gradients")

    e1 = TeacherTransitionOracleV62(config)
    e1.load_codec_state(e0.state_dict())
    frame_times = torch.tensor((0.0, 0.1, 0.2))[None].expand(4, -1)
    source_index = torch.zeros(4, dtype=torch.long)
    e1_output = e1(observation, frame_times, source_index)
    e1_output["loss"].backward()
    e1_count, e1_finite = finite_gradients(e1)
    if not e1_finite:
        raise RuntimeError("v62 E1 CPU test produced non-finite gradients")
    correct_zero_difference = (
        (e1_output["correct"].carriers - e1_output["zero"].carriers).abs().max()
    )
    if not bool(correct_zero_difference > 0.0):
        raise RuntimeError("v62 E1 correct effect does not alter predicted state")
    report = {
        "status": "passed",
        "e0_loss": float(e0_output["loss"].detach()),
        "e0_gradient_tensors": e0_count,
        "e1_loss": float(e1_output["loss"].detach()),
        "e1_gradient_tensors": e1_count,
        "effect_shape": list(e1_output["effect"].value.shape),
        "correct_zero_max_difference": float(correct_zero_difference.detach()),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
