"""One complete CPU/CUDA fixture integration, not a pretrained benchmark result."""

import argparse
import io
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import av
import h5py
import numpy as np
import torch
from PIL import Image

from igsw.adaptive_gaussian_wm.object_sequence_dynamics_v69 import ObjectSequencePosteriorV69
from igsw.adaptive_gaussian_wm.query_object_video_encoder_v69 import QueryObjectVideoEncoderV69
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69
from igsw.world_dynamics_benchmarks.compare import compare_models
from igsw.world_dynamics_benchmarks.data import move_observation, read_observation
from igsw.world_dynamics_benchmarks.features import FrozenRepresentation, export_features
from igsw.world_dynamics_benchmarks.io import experiment_root, read_json, write_json
from igsw.world_dynamics_benchmarks.prepare import build_manifest, inspect_schema
from igsw.world_dynamics_benchmarks.probe import evaluate, run_probe, train_attention


HUB_FIXTURE = '''import torch
from torch import nn
from torch.nn import functional as F

class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 1024, 16, 16)
    def forward_features(self, x):
        return {"x_norm_patchtokens": self.patch(x).flatten(2).transpose(1, 2)}

class TinyVjepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv3d(3, 1024, (2, 16, 16), (2, 16, 16))
    def forward(self, x):
        return self.patch(x).flatten(2).transpose(1, 2)

def dinov3_vitl16(pretrained=False):
    return TinyDino()

def vjepa2_vit_large(pretrained=False):
    return TinyVjepa(), nn.Identity()

def vjepa2_preprocessor(crop_size=32):
    def transform(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, (crop_size, crop_size), mode="bilinear", align_corners=False)
        return [x.permute(1, 0, 2, 3)]
    return transform
'''


def images(label, length, width=32):
    output = []
    for frame in range(length):
        image = np.zeros((32, width, 3), dtype=np.uint8)
        image[:, :, 0] = 25 + label * 120
        offset = frame % (width - 6)
        image[8:16, offset:offset + 6, 1] = 210
        output.append(image)
    return output


def make_hdf5(path, label, stimulus):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.create_dataset("static/stimulus_name", data=stimulus.encode())
        file.create_dataset("static/object_ids", data=[3, 2, 1])
        file.create_dataset("static/object_segmentation_colors", data=np.array(
            [[166, 180, 55], [218, 76, 142], [241, 241, 236]], dtype=np.uint8))
        file.create_dataset("static/target_id", data=2)
        file.create_dataset("static/zone_id", data=1)
        for frame, pixels in enumerate(images(label, 40)):
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            file.create_dataset(f"frames/{frame:04d}/images/_img", data=np.frombuffer(buffer.getvalue(), dtype=np.uint8))
            segmentation = np.zeros_like(pixels)
            offset = frame % (pixels.shape[1] - 6)
            segmentation[8:16, offset:offset + 6] = [218, 76, 142]
            segmentation[20:28, 24:30] = [241, 241, 236]
            segmentation[:4, :4] = [166, 180, 55]
            buffer = io.BytesIO()
            Image.fromarray(segmentation).save(buffer, format="PNG")
            file.create_dataset(f"frames/{frame:04d}/images/_id", data=np.frombuffer(buffer.getvalue(), dtype=np.uint8))
            file.create_dataset(f"frames/{frame:04d}/labels/target_contacting_zone", data=label and frame == 39)


def make_video(path, label, width):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = width, 32, "yuv420p"
        for pixels in images(label, 10, width):
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def setup(root, device):
    repository = root / "fixture_backbones"
    repository.mkdir()
    (repository / "hubconf.py").write_text(HUB_FIXTURE)
    dino = torch.hub.load(str(repository), "dinov3_vitl16", source="local", pretrained=False)
    torch.save(dino.state_dict(), root / "dino.pt")
    vjepa, _ = torch.hub.load(str(repository), "vjepa2_vit_large", source="local", pretrained=False)
    torch.save({"target_encoder": vjepa.state_dict()}, root / "vjepa.pt")
    state_config = ObjectVideoConfigV69(width=16, heads=4, object_queries=2, local_carriers=2,
        observation_layers=1, memory_layers=1, posterior_layers=1, effect_tokens=2, effect_dim=4)
    encoder = QueryObjectVideoEncoderV69(state_config)
    posterior = ObjectSequencePosteriorV69(state_config)
    weights = {**{"encoder." + key: value for key, value in encoder.state_dict().items()},
               **{"target_encoder." + key: value for key, value in encoder.state_dict().items()},
               **{"posterior." + key: value for key, value in posterior.state_dict().items()}}
    torch.save({"model": weights, "perception": dino.state_dict(), "config": asdict(state_config),
                "step": 8750}, root / "teacher.pt")
    return {"seed": 17, "attempt": "complete", "device": device, "scope": "generated_fixture",
        "num_classes": 2, "output_root": str(root / "outputs"),
        "export": {"frames": 8, "token_budget": 64, "frame_batch": 4, "observed_segments": 2},
        "models": {"dino_repository": str(repository), "dino_weights": str(root / "dino.pt"),
                   "vjepa_repository": str(repository), "vjepa_weights": str(root / "vjepa.pt"),
                   "vjepa_factory": "vjepa2_vit_large", "vjepa_checkpoint_key": "target_encoder",
                   "vjepa_crop": 32, "teacher_checkpoint": str(root / "teacher.pt")},
        "probe": {"width": 16, "heads": 4, "batch_size": 2, "epochs": 2, "lr": .001,
                  "weight_decay": .01, "save_every": 2, "linear_tokens": 4, "linear_cv": 2,
                  "linear_logspace": [-1, 1, 3], "linear_max_iter": 100}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="igsw-benchmark-integration-"))
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    base = setup(root, args.device)
    configs = []
    physion = {**base, "benchmark": "physion", "protocol": "fixture_custom_hdf5_pair_cued_prefix",
        "manifest": str(root / "physion.json"), "data": {"root": str(root / "physion"), "scenarios": ["Collide"],
            "input_frame_indices": [0, 5, 10, 15, 20, 25, 30, 36], "fps": 30, "dev_fraction": .33,
            "prefix_source": "verified fixture boundary", "pair_cue_alpha": .65}}
    for split, count in (("readout_training", 12), ("testing", 4)):
        for index in range(count):
            # Both directories and splits reuse basenames and static stimulus names, as real bundles do.
            make_hdf5(root / "physion" / split / "Collide" / f"experiment_{index // 6}" / f"{index % 6:04d}.hdf5",
                      index % 2, f"train_readout_{index % 2:04d}")
    configs.append(physion)
    videos = root / "ssv2"
    videos.mkdir()
    write_json(videos / "labels.json", {"moving something left": "0", "moving something right": "1"})
    for split, count in (("train", 12), ("validation", 4)):
        annotations = []
        for index in range(count):
            identity = f"{split}_{index}"
            make_video(videos / f"{identity}.mp4", index % 2, 32 if index % 2 == 0 else 48)
            annotations.append({"id": identity, "template": "moving [something] " + ("left" if index % 2 == 0 else "right")})
        write_json(videos / f"{split}.json", annotations)
    configs.append({**base, "benchmark": "ssv2", "protocol": "fixture_official_annotation_interface",
        "manifest": str(root / "ssv2.json"), "data": {"root": str(videos), "videos": str(videos),
        "labels": str(videos / "labels.json"), "train": str(videos / "train.json"),
        "validation": str(videos / "validation.json"), "extension": ".mp4", "dev_fraction": .33,
        "pilot_per_class": {"train": 4, "dev": 2, "test": 2}}})
    reports = {}
    for config in configs:
        manifest = build_manifest(config)
        assert manifest["split_counts"] == {"train": 8, "dev": 4, "test": 4}
        assert len({row["id"] for row in manifest["rows"]}) == len(manifest["rows"])
        if config["benchmark"] == "ssv2":
            assert manifest["class_counts"] == {"train": {"0": 4, "1": 4}, "dev": {"0": 2, "1": 2},
                                                 "test": {"0": 2, "1": 2}}
        if config["benchmark"] == "physion":
            assert len({row["stimulus_name"] for row in manifest["rows"]}) == 2
            assert all(row["time_basis"]["measured_clock"] is False for row in manifest["rows"])
            assert all(row["times"][-1] == 1.2 and row["time_basis"]["paper_observed_prefix_seconds"] == 1.5
                       for row in manifest["rows"])
            row = manifest["rows"][0]
            observed = read_observation(row, 8)["rgb"][0].permute(0, 2, 3, 1).numpy()
            with h5py.File(row["path"]) as file:
                for index, pixels in zip(row["frame_indices"], observed):
                    raw = np.array(Image.open(io.BytesIO(file[f"frames/{index:04d}/images/_img"][()].tobytes())))
                    segmentation = np.array(Image.open(io.BytesIO(file[f"frames/{index:04d}/images/_id"][()].tobytes())))
                    masks = [(segmentation == color).all(-1) for color in ([218, 76, 142], [241, 241, 236])]
                    assert np.array_equal(pixels[~(masks[0] | masks[1])], raw[~(masks[0] | masks[1])])
                    for mask, tint in zip(masks, ([255, 0, 0], [255, 255, 0])):
                        assert np.array_equal(pixels[mask], np.rint(.35 * raw[mask] + .65 * np.array(tint)).astype(np.uint8))
        for model in ("dino", "vjepa2", "state", "state_z"):
            export_features(config, model)
            # Run the normal exporter twice: completed cache entries are retained.
            export_features(config, model)
            cached_manifest = read_json(experiment_root(config, model) / "features.json")
            feature = torch.load(cached_manifest["rows"][0]["feature_path"], weights_only=False)
            public_times = torch.tensor(feature["times"])
            if model in ("state", "state_z"):
                state_times = torch.tensor(feature["state_clock_times"])
                assert state_times[0, -1] == 0 and (state_times <= 0).all()
                assert torch.allclose(state_times.diff(dim=1), public_times.diff(dim=1), atol=1e-6)
                assert feature["provenance"]["state_time_anchor"] == "last_observed_frame_zero"
                if model == "state_z":
                    for segment in feature["effect_source_references"]:
                        assert segment["source_time"] == [0.0]
                        assert max(segment["query_frame_indices"][0]) <= segment["source_frame"]
                        expected = public_times[:, segment["source_frame"] + 1:segment["target_end_frame"] + 1] - public_times[:, segment["source_frame"]:segment["source_frame"] + 1]
                        assert torch.equal(torch.tensor(segment["target_times"]), expected)
            else:
                assert "state_clock_times" not in feature and public_times[0, 0] == 0
            run_probe(config, model, "attention", "probe", False)
            for split in ("dev", "test"):
                report = evaluate(config, model, "attention", read_json(experiment_root(config, model) / "features.json"), split)
                assert report["overall"]["count"] == 4 and report["split"] == split
            output = experiment_root(config, model) / "attention"
            cases = [{json.loads(line)["id"] for line in (output / f"cases_{split}.jsonl").read_text().splitlines()}
                     for split in ("dev", "test")]
            assert cases[0].isdisjoint(cases[1])
        for split in ("dev", "test"):
            compare_models(config, split)
        reports[config["benchmark"]] = read_json(experiment_root(config, "state_z") / "attention/report_dev.json")
    run_probe(physion, "state_z", "linear", "probe", False)
    for split in ("dev", "test"):
        run_probe(physion, "state_z", "linear", "evaluate", False, split)
    manifest = read_json(experiment_root(physion, "state_z") / "features.json")
    resume_config = {**physion, "attempt": "resumed"}
    train_attention(resume_config, "state_z", manifest, stop_step=2)
    train_attention(resume_config, "state_z", manifest, resume=True)
    uninterrupted = torch.load(experiment_root(physion, "state_z") / "attention/latest.pt", weights_only=False)
    resumed = torch.load(experiment_root(resume_config, "state_z") / "attention/latest.pt", weights_only=False)
    maximum = max(float((value - resumed["model"][name]).abs().max()) for name, value in uninterrupted["model"].items())
    assert maximum == 0
    traces = [Path(path).read_text() for path in (experiment_root(physion, "state_z") / "attention/trace.jsonl",
                                                experiment_root(resume_config, "state_z") / "attention/trace.jsonl")]
    assert traces[0] == traces[1]
    row = read_json(physion["manifest"])["rows"][0]
    runtime = FrozenRepresentation(physion, "state_z")
    seen_clock, posterior_clocks = [], []
    runtime.encoder.register_forward_pre_hook(lambda module, args: seen_clock.append(args[0].times.detach().cpu()))
    runtime.posterior.register_forward_pre_hook(lambda module, args: posterior_clocks.append(
        (args[0].time.detach().cpu(), torch.stack([state.time for state in args[1]], 1).detach().cpu())))
    original_observation = read_observation(row, 8)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        original = runtime(move_observation(original_observation, args.device))
    assert torch.equal(seen_clock[0], original_observation["times"] - original_observation["times"][:, -1:])
    assert seen_clock[0][0, -1] == 0
    for (source_time, target_times), segment in zip(posterior_clocks, runtime.effect_source_references):
        assert torch.equal(source_time, torch.zeros_like(source_time)) and (target_times > 0).all()
        expected = original_observation["times"][:, segment["source_frame"] + 1:segment["target_end_frame"] + 1] - original_observation["times"][:, segment["source_frame"]:segment["source_frame"] + 1]
        assert torch.equal(target_times, expected)
    device_observation = move_observation(original_observation, args.device)
    first, stop = 2, 4
    movie_times = device_observation["times"]
    altered_rgb = device_observation["rgb"].clone()
    altered_rgb[:, first + 1:stop + 1] = 255 - altered_rgb[:, first + 1:stop + 1]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        perception = runtime.perception(device_observation["rgb"], device_observation["pixel_valid"],
            movie_times - movie_times[:, -1:], device_observation["native_hw"])
        changed_perception = runtime.perception(altered_rgb, device_observation["pixel_valid"],
            movie_times - movie_times[:, -1:], device_observation["native_hw"])
        source_queries, source, targets, effect = runtime.observed_effect(perception, movie_times, first, stop)
        changed_queries, changed_source, changed_targets, changed_effect = runtime.observed_effect(changed_perception, movie_times, first, stop)
    assert torch.equal(source.tokens, changed_source.tokens) and torch.equal(source.centers, changed_source.centers)
    assert torch.equal(source_queries.features, changed_queries.features)
    assert not torch.equal(targets[-1].tokens, changed_targets[-1].tokens)
    assert not torch.equal(effect["value"], changed_effect["value"])
    assert torch.equal(perception.times, movie_times - movie_times[:, -1:])
    with h5py.File(row["path"], "r+") as file:
        for frame in range(37, 40):
            pixels = images(1, 1)[0]
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            key = f"frames/{frame:04d}/images/_img"
            del file[key]
            file.create_dataset(key, data=np.frombuffer(buffer.getvalue(), dtype=np.uint8))
            buffer = io.BytesIO()
            Image.fromarray(np.full_like(pixels, [241, 241, 236])).save(buffer, format="PNG")
            key = f"frames/{frame:04d}/images/_id"
            del file[key]
            file.create_dataset(key, data=np.frombuffer(buffer.getvalue(), dtype=np.uint8))
    changed_observation = read_observation(row, 8)
    assert torch.equal(original_observation["rgb"], changed_observation["rgb"])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        changed = runtime(move_observation(changed_observation, args.device))
    assert torch.equal(original["tokens"], changed["tokens"])
    assert not any(parameter.requires_grad for parameter in runtime.parameters())
    write_json(root / "integration_result.json", {"status": "passed", "device": args.device,
        "fixture_not_pretrained_benchmark": True, "resume_max_difference": maximum,
        "trace_exact": True, "prefix_future_swap_exact": True, "pair_cue_pixels_exact": True,
        "future_mask_swap_observation_exact": True, "benchmarks": list(reports),
        "dev_test_artifacts_separate": True,
        "class_balanced_sampling": True,
        "state_last_observed_clock_zero": True,
        "state_clock_intervals_preserved": True,
        "native_posterior_source_zero_targets_positive": True,
        "segment_target_swap_source_exact": True,
        "segment_target_swap_effect_changed": True,
        "hdf5_schema": inspect_schema(row["path"]), "results": str(root / "outputs")})
    print(json.dumps({"status": "passed", "integration_result": str(root / "integration_result.json")}), flush=True)


if __name__ == "__main__":
    main()
