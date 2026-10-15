"""Separate image-codec, state, and posterior-rollout reconstruction evidence."""

from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import time

import av
import numpy as np
import torch

from igsw.training.losses import ssim
from .codec import letterbox_video
from .data import collate_object_event_video, write_json
from .state import coordinates
from .training import device_batch


def image_records(prediction, target, mask, current, times, perceptual=None):
    rows = []
    for frame in range(len(target)):
        active = mask[frame, 0]
        locations = torch.where(active)
        if not len(locations[0]):
            continue
        y0, y1 = int(locations[0].min()), int(locations[0].max())+1
        x0, x1 = int(locations[1].min()), int(locations[1].max())+1
        predicted = prediction[frame:frame+1, :, y0:y1, x0:x1].float().clamp(0, 1)
        truth = target[frame:frame+1, :, y0:y1, x0:x1].float().clamp(0, 1)
        initial = current[None, :, y0:y1, x0:x1]
        error = (predicted-truth).abs()
        mse = (predicted-truth).square().mean()
        # Pixel change includes illumination and camera effects, not only physical motion.
        change = (truth-initial).abs().mean(1, keepdim=True)
        weighted_error = (error*change).sum()/(change.sum()*3).clamp_min(1e-8)
        row = {"frame": frame, "seconds": float(times[frame]), "rgb_l1": float(error.mean()),
               "psnr": float(-10*torch.log10(mse.clamp_min(1e-12))), "ssim": float(ssim(predicted, truth)),
               "appearance_change_weighted_l1": float(weighted_error) if change.sum() > 0 else None,
               "rgb_change_amplitude": float(change.mean())}
        if perceptual is not None:
            row["lpips"] = float(perceptual(predicted*2-1, truth*2-1).mean())
        if frame:
            previous_pred = prediction[frame-1:frame, :, y0:y1, x0:x1].float().clamp(0, 1)
            previous_true = target[frame-1:frame, :, y0:y1, x0:x1].float().clamp(0, 1)
            if mask[frame-1].any():
                row["temporal_rgb_delta_l1"] = float(((predicted-previous_pred)-(truth-previous_true)).abs().mean())
        rows.append(row)
    return rows


def write_video(path, frames, times):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = Fraction((len(frames)-1)/float(times[-1]-times[0])).limit_denominator(1000)
    images = (frames.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255).round().astype(np.uint8)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.height, stream.width = images.shape[1:3]
        stream.pix_fmt = "yuv420p"
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def balanced_indices(dataset, items_per_source):
    by_source = defaultdict(list)
    for index, episodes in enumerate(dataset.episodes):
        by_source[episodes[0]["source"]].append(index)
    return [index for source in sorted(by_source) for index in by_source[source][:items_per_source]]


@torch.no_grad()
def evaluate(model, codec, dataset, out, items_per_source=8, precision="bf16", videos_per_source=1,
             perceptual=None):
    model.eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    counts, records, excluded = defaultdict(int), [], []
    with (out / "cases.jsonl").open("w") as stream:
        for index in balanced_indices(dataset, items_per_source):
            sample = dataset[index]
            if "decode_error" in sample:
                excluded.append(sample)
                continue
            batch = device_batch(collate_object_event_video([sample]), device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                result = model(batch, codec, deterministic=True, render_all=True)
                effects = torch.stack([row["value"] for row in result["effects"]], 1)
                times = batch["times"][:, batch["history_frames"]:]
                xy = coordinates(codec.grid_size, device)
                zero_states = model.rollout(result["source"], torch.zeros_like(effects), times)
                reverse_states = model.rollout(result["source"], effects.flip(1), times)
                zero = codec.decode(model.render_features(zero_states, xy)[0])
                reverse = codec.decode(model.render_features(reverse_states, xy)[0])
                source_image = codec.decode(model.render_features([result["source"]], xy)[0])
                reference = codec.decode(result["target_features"])
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            images, _ = letterbox_video(batch, codec.image_size)
            th = batch["history_frames"]
            current = images[0, th-1]
            target = result["target_rgb"][0]
            outputs = {"codec_reference": reference[0], "observed_state": result["observed_rgb"][0],
                       "posterior": result["posterior_rgb"][0], "zero_effect": zero[0],
                       "reversed_effect_order": reverse[0], "source_state_copy": source_image[0].expand_as(target),
                       "rgb_copy": current[None].expand_as(target)}
            case = {"case_id": sample["case_id"], "source": sample["source"],
                    "episode_index": sample["episode_index"], "partition": sample["partition"],
                    "frame_indices": sample["frame_indices"].tolist(), "times": sample["times"].tolist(),
                    "instruction": sample["instruction"], "history_frames": th,
                    "image_size": codec.image_size, "elapsed_seconds_all_conditions": seconds,
                    "effect_update_fraction": float(result["parts"]["effect_update_fraction"]),
                    "conditions": {}}
            for condition, prediction in outputs.items():
                frame_rows = image_records(prediction, target, result["pixel_valid"][0],
                                           current, times[0], perceptual)
                case["conditions"][condition] = frame_rows
                if counts[sample["source"]] < videos_per_source:
                    full = torch.cat((images[0, :th], prediction), 0)
                    write_video(out / "videos" / sample["case_id"] / f"{condition}.mp4", full, batch["times"][0])
            if counts[sample["source"]] < videos_per_source:
                write_video(out / "videos" / sample["case_id"] / "target.mp4", images[0], batch["times"][0])
            counts[sample["source"]] += 1
            records.append(case)
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps({"event": "evaluated", "case_id": sample["case_id"], "cases": len(records)}), flush=True)
    scores = defaultdict(lambda: defaultdict(list))
    for case in records:
        for condition, frames in case["conditions"].items():
            for key in ("rgb_l1", "psnr", "ssim", "lpips", "appearance_change_weighted_l1", "temporal_rgb_delta_l1"):
                values = [row[key] for row in frames if row.get(key) is not None]
                if values:
                    scores[condition][key].append(float(np.mean(values)))
    report = {"task": "posterior_conditioned_reconstruction", "cases": len(records),
              "sources": dict(counts), "excluded": excluded,
              "conditional_prior_evaluated": False, "independent_object_semantics_evaluated": False,
              "conditions": {condition: {key: {"cases": len(values), "mean": float(np.mean(values)),
                                                      "median": float(np.median(values))}
                                          for key, values in metrics.items()}
                             for condition, metrics in scores.items()}}
    write_json(out / "report.json", report)
    return report
