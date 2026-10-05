#!/usr/bin/env python3
"""CPU export integration with real video workers and a small explicit teacher fixture."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import export_language_effect_labels_v70 as exporter
from igsw.adaptive_gaussian_wm.frozen_object_teacher_v70 import FixedEffectTeacherV70
from igsw.adaptive_gaussian_wm.label_export_data_v70 import full_window
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import collate_object_video_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.query_object_video_encoder_v69 import QueryObjectVideoEncoderV69
from igsw.adaptive_gaussian_wm.object_sequence_dynamics_v69 import ObjectSequencePosteriorV69
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69


class FixtureBackbone(nn.Module):
    def forward_features(self, images):
        tokens = F.avg_pool2d(images, 16, 16).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens.repeat(1, 1, 342)[..., :1024]}


def fixture_teacher(*args):
    torch.manual_seed(17)
    config = ObjectVideoConfigV69(width=32, heads=4, object_queries=4, local_carriers=2,
                                  observation_layers=1, memory_layers=1, posterior_layers=1,
                                  history_frames=2, future_frames=3)
    teacher = FixedEffectTeacherV70.__new__(FixedEffectTeacherV70)
    nn.Module.__init__(teacher)
    perception = PretrainedVisualEncoderV69.__new__(PretrainedVisualEncoderV69)
    nn.Module.__init__(perception)
    perception.kind, perception.frame_batch, perception.dtype = "dinov3_vitl16", 7, torch.float32
    perception.batch_across_samples = False
    perception.backbone = FixtureBackbone()
    perception.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
    perception.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
    teacher.config, teacher.perception = config, perception
    teacher.encoder, teacher.target_encoder = QueryObjectVideoEncoderV69(config), QueryObjectVideoEncoderV69(config)
    teacher.posterior = ObjectSequencePosteriorV69(config)
    teacher.teacher = {"fixture": True}
    return teacher.requires_grad_(False).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    entries = []
    for index, (h, w) in enumerate([(48, 64)] * 3 + [(64, 80)] * 2):
        video = out / f"video_{index}.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.height, stream.width, stream.pix_fmt = h, w, "yuv420p"
            for _ in range(5):
                frame = av.VideoFrame.from_ndarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        entries.append({"window_id": str(index), "case": {"record": {"adapter": "video",
                        "path": str(video), "frame_offset": 0, "fps": 30}},
                        "frame_indices": list(range(5)), "history_frames": 2,
                        "label_path": str(out / "labels" / f"{index}.pt"), "language_provenance": {}})
    missing = {**entries[0], "window_id": "missing", "label_path": str(out / "missing.pt"),
               "case": {"record": {**entries[0]["case"]["record"], "path": str(out / "absent.mp4")}}}
    entries.insert(1, missing)
    manifest = out / "language_manifest.json"
    manifest.write_text(json.dumps({"teacher_checkpoint": "fixture", "entries": entries}))
    context = SimpleNamespace(device="cpu", distributed=False, rank=0, world_size=1, is_main=True)
    command = ["export", "--manifest", str(manifest), "--batch", "4", "--workers", "2", "--prefetch", "1"]
    with patch.object(exporter, "FixedEffectTeacherV70", fixture_teacher), \
         patch.object(exporter, "init_torchrun", return_value=context), patch.object(sys, "argv", command):
        exporter.main()
        before = [(Path(e["label_path"]).stat().st_mtime_ns) for e in entries if e is not missing]
        exporter.main()
        after = [(Path(e["label_path"]).stat().st_mtime_ns) for e in entries if e is not missing]
    serial = fixture_teacher()
    max_difference = 0.
    for entry in entries:
        if entry is missing:
            continue
        expected = serial(collate_object_video_v69([full_window(entry)]))
        actual = torch.load(entry["label_path"], weights_only=False)
        for name, value in expected.items():
            max_difference = max(max_difference, float((value[0].float()-actual[name].float()).abs().max()))
            torch.testing.assert_close(value[0], actual[name], atol=1e-5, rtol=1e-4)
        assert actual["mean"].untyped_storage().nbytes() == actual["mean"].numel() * actual["mean"].element_size()
    result = json.loads((out / "labeled_manifest.json").read_text())
    assert len(result["entries"]) == 5 and result["unlabeled_windows"] == 1
    assert before == after
    print(json.dumps({"test": "cpu_video_worker_export_fixture", "labels": 5,
                      "decode_failures": 1, "workers": 2, "serial_batch_max_difference": max_difference,
                      "existing_labels_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
