#!/usr/bin/env python3
"""Optional server-side end-to-end check, not a training admission gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from igsw.adaptive_gaussian_wm.grounded_motion_dataset_v68 import GroundedMotionDatasetV68, collate_grounded_motion_v68
from igsw.adaptive_gaussian_wm.grounded_object_transport_v68 import GroundedObjectTransportV68
from igsw.adaptive_gaussian_wm.grounded_appearance_teacher_v68 import GroundedAppearanceTeacherV68
from igsw.adaptive_gaussian_wm.v67_config import ContinuousPredictiveObjectFieldConfigV67


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    dataset = GroundedMotionDatasetV68(args.manifest, points=128)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_grounded_motion_v68([dataset[0]]).items()}
    teacher = GroundedAppearanceTeacherV68(ContinuousPredictiveObjectFieldConfigV67(), device, "bf16", args.dino_checkpoint, args.siglip_checkpoint, 16, 16)
    appearance = teacher(batch)
    report = {}
    for stage in ("state", "dynamics"):
        model = GroundedObjectTransportV68(stage=stage).to(device).train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, appearance if stage == "state" else None)
        output["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        report[f"{stage}_loss"] = float(output["loss"].detach())
        report[f"{stage}_gradient_norm"] = float(norm)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            first = model.encode_history(batch)
            swapped = {**batch, "video_rgb": batch["video_rgb"].clone()}
            swapped["video_rgb"][:, 4:] = 255 - swapped["video_rgb"][:, 4:]
            second = model.encode_history(swapped)
            difference = float((first["slots"] - second["slots"]).abs().max())
            assert difference == 0.0
            report[f"{stage}_future_swap_max_difference"] = difference
            zero = torch.zeros((1, model.config.objects+2, model.config.effect_dim), device=device)
            coordinates = batch["coordinates"][:, 3]
            owner = model.encoder.assignment(first, coordinates)
            predicted, _, _ = model.predict_transport(first, zero, torch.ones(1, device=device), coordinates, owner)
            zero_error = float((predicted - coordinates).abs().max())
            assert zero_error == 0.0
            report[f"{stage}_zero_effect_max_difference"] = zero_error
        del model, output
    print(json.dumps({"status": "completed", "contract_only_not_object_quality": True, **report}), flush=True)


if __name__ == "__main__":
    main()
