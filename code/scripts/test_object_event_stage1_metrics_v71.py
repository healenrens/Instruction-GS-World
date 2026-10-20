#!/usr/bin/env python3
"""One complete metric/report exercise using moving, static and occluded actors."""

import json
import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.independent_metrics import score_rgb_outcomes
from igsw.object_event_wm.stage1_metrics import actor_change_records, summarize_stage1


def run_pipeline(rgb, labels):
    import av
    import torch
    from argparse import Namespace
    from copy import deepcopy
    from torch.nn import functional as F
    from igsw.object_event_wm.config import ModelConfig
    from igsw.object_event_wm.data import SequenceConfig, write_json
    from igsw.object_event_wm.model import ObjectEventWorldModel
    from igsw.object_event_wm.simulation_data import SimulationVideoDataset, prepare_simulation_manifest
    from igsw.object_event_wm.stage1_evaluation import evaluate_stage1
    from igsw.object_event_wm.reconstruction_probe import prepare_frames, train_probe

    class FixtureCodec(torch.nn.Module):
        image_size, grid_size = 32, 4

        def encode(self, images):
            pooled = F.adaptive_avg_pool2d(images.flatten(0, 1), (4, 4)).flatten(2).transpose(1, 2)
            return pooled.reshape(*images.shape[:2], 16, 3)

        def decode(self, tokens):
            pictures = tokens.reshape(-1, 16, 3).transpose(1, 2).reshape(-1, 3, 4, 4)
            return F.interpolate(pictures, (32, 32)).reshape(*tokens.shape[:-2], 3, 32, 32)

        def patch_valid(self, pixels):
            return F.adaptive_avg_pool2d(pixels.flatten(0, 1).float(), (4, 4)).flatten(1).reshape(
                *pixels.shape[:2], 16) > 0

    torch.set_num_threads(1)
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory(prefix="v71_held_pipeline_") as folder:
        root = Path(folder)
        times = np.arange(len(rgb), dtype=np.float64)
        np.savez_compressed(root/"rgb.npz", rgb=rgb, times_seconds=times)
        poses = np.zeros((len(rgb), 2, 7), dtype=np.float32)
        poses[:, 0, 0] = times*.01
        np.savez_compressed(root/"truth.npz", segmentation=labels, poses_xyz_xyzw=poses,
            visible_pixels=np.stack([(labels == body).sum((1, 2)) for body in (1, 2)], 1),
            times_seconds=times, physics_ticks=(times*240).astype(int), body_ids=np.array([1, 2]))
        write_json(root/"episode.json", {"rgb": "rgb.npz", "history_rgb": "rgb.npz", "truth": "truth.npz",
            "physics_hz": 240, "capture_every_physics_steps": 240, "instruction": "move the red block",
            "status": "completed", "seed": 17, "combination": "fixture", "robot_and_tool_body_ids": [9],
            "entities": [{"body_id": 1, "kind": "block"}, {"body_id": 2, "kind": "block"}],
            "program": [{"actor_body_id": 1}, {"actor_body_id": 2}]})
        write_json(root/"collection.json", {"episodes": [{"case_id": "fixture", "scene_seed": 17,
            "episode": "episode.json", "partition": "development", "status": "completed", "combination": "fixture"}],
            "history_seconds": 0, "future_seconds": 4, "partition": "fixture"})
        prepare_simulation_manifest(root/"collection.json", root/"manifest.json",
            SequenceConfig(history_seconds=0, future_seconds=4, history_frames=1, future_frames=4))
        dataset = SimulationVideoDataset(root/"manifest.json", partition="development")
        model = ObjectEventWorldModel(ModelConfig(feature_dim=3, width=16, heads=2, object_slots=2,
            local_tokens=2, observation_iterations=1, dynamics_layers=1, effect_dim=4, view_size=32))
        info = {"checkpoint_step": 0, "checkpoint": "fixture", "model_config": vars(model.config)}
        out = root/"evaluation"
        result = evaluate_stage1(model, FixtureCodec(), dataset, out, info, items=1, video_cases=1, precision="bf16")
        assert result["cases"] == result["scenes"] == 1 and len(result["summary"]["by_case"]) == 8
        metrics_path = out/"cases"/"fixture"/"metrics.json"
        before = metrics_path.stat().st_mtime_ns
        again = evaluate_stage1(model, FixtureCodec(), dataset, out, info, items=1,
                                video_cases=1, precision="bf16", resume=True)
        assert result["summary"] == again["summary"] and metrics_path.stat().st_mtime_ns == before
        with av.open(out/"cases"/"fixture"/"comparison.mp4") as video:
            decoded = list(video.decode(video=0))
        assert len(decoded) == len(rgb) and decoded[0].width == 768
        assert (out/"index.html").is_file() and (out/"cases"/"fixture"/"terminal.jpg").is_file()
        bank = prepare_frames(dataset, FixtureCodec(), 32, 1, [0, 1, 2, 4], torch.device("cpu"), "bf16")
        assert len(bank["images"]) == 4
        original = deepcopy(model.state_dict())
        options = dict(checkpoint="fixture", initialization="checkpoint", source_revision="fixture",
            batch=2, precision="bf16", lr=.001, weight_decay=.01, log_every=1, save_every=2, steps=4)
        uninterrupted = root/"probe_uninterrupted"
        uninterrupted.mkdir()
        report = train_probe(model, FixtureCodec(), bank, Namespace(out=str(uninterrupted), **options))
        complete = deepcopy(model.state_dict())
        assert any(not torch.equal(complete[k], original[k]) for k in original if k.startswith("encoder."))
        assert any(not torch.equal(complete[k], original[k]) for k in original if k.startswith("renderer."))
        assert all(torch.equal(complete[k], original[k]) for k in original
                   if k.startswith(("posterior.", "dynamics.")))
        assert all(np.isfinite(value) for value in report["after"].values())
        assert (uninterrupted/"frames"/"frame_03.png").is_file()
        model.load_state_dict(original)
        interrupted = root/"probe_resumed"
        interrupted.mkdir()
        train_probe(model, FixtureCodec(), bank, Namespace(out=str(interrupted), **{**options, "steps": 2}))
        checkpoint = torch.load(interrupted/"latest.pt", map_location="cpu", weights_only=False)
        train_probe(model, FixtureCodec(), bank, Namespace(out=str(interrupted), **options), resume=checkpoint)
        difference = max(float((model.state_dict()[k]-complete[k]).abs().max()) for k in complete)
        assert difference == 0
        return {"conditions": 8, "video_frames": len(decoded), "resume_reused_case": True,
                "reconstruction_probe_frames": 4, "reconstruction_probe_steps": 4,
                "only_encoder_renderer_updated": True, "probe_resume_max_difference": difference,
                "scope": "CPU BF16 real small ObjectEventWorldModel; fixture visual codec, not DINO/RAE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", action="store_true", help="Also run model, evaluator, video and resume on CPU.")
    args = parser.parse_args()
    frames, height, width = 5, 24, 32
    labels = np.zeros((frames, height, width), dtype=np.int64)
    rgb = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for frame in range(frames):
        left = 2+frame*3
        labels[frame, 5:10, left:left+4] = 1
        rgb[frame, 5:10, left:left+4, 0] = 255
        labels[frame, 15:19, 4:8] = 2
        rgb[frame, 15:19, 4:8, 1] = 255
        labels[frame, :3] = 9
        rgb[frame, :3, :, 2] = frame*40
    times = np.arange(1, frames, dtype=np.float32)
    valid = np.ones(frames-1, dtype=bool)
    truth, anchor = rgb[1:], rgb[0]
    copy = np.broadcast_to(anchor, truth.shape)
    lag = rgb[:-1]
    variants = {"posterior": truth, "zero_effect": copy, "source_state_copy": copy,
                "rgb_copy": copy, "reversed_effect_order": lag}
    conditions = {}
    for name, prediction in variants.items():
        conditions[name] = {
            "rgb": score_rgb_outcomes(prediction, truth, labels[0], labels[1:], [1, 2], [9], times),
            "change": actor_change_records(prediction, anchor, truth, anchor, labels[1:], labels[0],
                                           [1, 2], [9], times, valid),
            "feature": []}
    exact = conditions["posterior"]["change"]["actors"]
    static = conditions["rgb_copy"]["change"]["actors"]
    assert exact[0]["anchor_change_l1"] == exact[0]["step_change_l1"] == 0
    assert static[0]["anchor_change_l1"] > 0 and abs(static[0]["change_normalized_error"]-1) < 1e-7
    assert static[1]["anchor_change_l1"] == 0 and static[1]["change_normalized_error"] is None
    robot_corrupt = truth.copy()
    robot_corrupt[:, :3] = 255
    robot_score = actor_change_records(robot_corrupt, anchor, truth, anchor, labels[1:], labels[0],
                                       [1, 2], [9], times, valid)
    assert robot_score["actors"][0]["anchor_change_l1"] == 0
    missing_end = valid.copy()
    missing_end[-1] = False
    missing = actor_change_records(truth, anchor, truth, anchor, labels[1:], labels[0],
                                    [1, 2], [9], times, missing_end)
    assert missing["actors"][0]["terminal_rgb_l1"] is None
    gap = valid.copy()
    gap[1] = False
    gapped = actor_change_records(truth, anchor, truth, anchor, labels[1:], labels[0],
                                   [1, 2], [9], times, gap)
    assert gapped["frames"][1]["query_frame_index"] == 2
    assert gapped["frames"][1]["actors"][0]["step_change_l1"] is None
    hidden = labels[1:].copy()
    hidden[-1, hidden[-1] == 1] = 9
    occluded = actor_change_records(truth, anchor, truth, anchor, hidden, labels[0],
                                     [1, 2], [9], times, valid)
    assert occluded["actors"][0]["terminal_rgb_l1"] is None
    cases = [{"case_id": f"paired_{index}", "scene_seed": 17, "conditions": conditions} for index in range(2)]
    report = summarize_stage1(cases)
    paired = report["paired_posterior_gain"]["rgb_copy"]["actor_region_rgb_l1"]
    assert paired["case_gain"]["count"] == 2 and paired["scene_gain"]["count"] == 1
    assert paired["case_wins"] == 2 and paired["scene_wins"] == 1
    restored = json.loads(json.dumps(report, allow_nan=False))
    assert restored == report
    result = {"status": "passed", "moving_copy_change_ratio": static[0]["change_normalized_error"],
        "oracle_actor_error": conditions["posterior"]["rgb"]["actor_region_rgb_l1"],
        "copy_actor_error": conditions["rgb_copy"]["rgb"]["actor_region_rgb_l1"],
        "scene_count": paired["scene_gain"]["count"],
        "missing_and_occluded_terminal_not_substituted": True,
        "scope": "CPU metric/report pipeline; no pretrained-model or GPU execution"}
    if args.pipeline:
        result["pipeline"] = run_pipeline(rgb, labels)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
