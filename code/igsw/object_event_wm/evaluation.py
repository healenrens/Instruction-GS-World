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
from .prediction import history_batch, observe_history, generate_from_history, make_history_request


def image_records(prediction, target, mask, current, times, perceptual=None, time_unit="seconds"):
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
        time_key = "seconds" if time_unit == "seconds" else "sampled_frame_offset"
        row = {"frame": frame, time_key: float(times[frame]), "rgb_l1": float(error.mean()),
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


def write_video(path, frames, times, time_unit="seconds"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frame-only benchmarks use 5 FPS for viewing; no physical FPS is inferred.
    if time_unit == "sampled_frame":
        times = times/5
    fps = Fraction((len(frames)-1)/float(times[-1]-times[0])).limit_denominator(1000)
    timestamps = (times-times[0]).detach().double().cpu().tolist()
    time_base = Fraction(1, 1000000)
    images = (frames.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255).round().astype(np.uint8)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.height, stream.width = images.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        for image, timestamp in zip(images, timestamps):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = round(timestamp / time_base)
            frame.time_base = time_base
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def balanced_indices(dataset, items_per_source):
    by_source = defaultdict(list)
    for index, episodes in enumerate(dataset.episodes):
        by_source[episodes[0]["source"]].append(index)
    rng = np.random.default_rng(dataset.seed)
    selected = []
    for source in sorted(by_source):
        indices = by_source[source]
        selected.extend(rng.choice(indices, size=min(items_per_source, len(indices)), replace=False).tolist())
    return selected


def summarize_cases(records):
    scores = defaultdict(lambda: defaultdict(list))
    for case in records:
        for condition, frames in case["conditions"].items():
            for key in ("rgb_l1", "psnr", "ssim", "lpips", "appearance_change_weighted_l1", "temporal_rgb_delta_l1"):
                values = [row[key] for row in frames if row.get(key) is not None]
                if values:
                    scores[condition][key].append(float(np.mean(values)))
    return {condition: {key: {"cases": len(values), "mean": float(np.mean(values)),
                              "median": float(np.median(values))} for key, values in metrics.items()}
            for condition, metrics in scores.items()}


@torch.no_grad()
def evaluate(model, codec, dataset, out, items_per_source=8, precision="bf16", videos_per_source=1,
             perceptual=None, prior=None, text_store=None, samples=1):
    model.eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    counts, records, excluded = defaultdict(int), [], []
    instruction_groups = defaultdict(set)
    if text_store is not None:
        for episode in dataset.episodes:
            for entry in episode:
                for window in entry["language_windows"]:
                    instruction_groups[entry["source"]].add((entry["group"], window["instruction"]))
    with (out / "cases.jsonl").open("w") as stream:
        for index in balanced_indices(dataset, items_per_source):
            sample = dataset[index]
            if "decode_error" in sample:
                excluded.append(sample)
                continue
            batch = device_batch(collate_object_event_video([sample]), device)
            time_unit = sample.get("time_unit", "seconds")
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                result = model(batch, codec, deterministic=True, render_all=True)
                effects = torch.stack([row["value"] for row in result["effects"]], 1)
                times = batch["times"][:, batch["history_frames"]:]
                xy = coordinates(codec.grid_size, device)
                zero_states = model.rollout(result["source"], torch.zeros_like(effects), times)
                reverse_states = model.rollout(result["source"], effects.flip(1), times)
                zero = model.decode_features(codec, model.render_features(zero_states, xy)[0])
                reverse = model.decode_features(codec, model.render_features(reverse_states, xy)[0])
                source_image = model.decode_features(codec, model.render_features([result["source"]], xy)[0])
                reference = model.decode_features(codec, result["target_features"])
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            images, _ = letterbox_video(batch, model.view_size(codec))
            th = batch["history_frames"]
            current = images[0, th-1]
            target = result["target_rgb"][0]
            outputs = {"codec_reference": reference[0], "observed_state": result["observed_rgb"][0],
                       "posterior": result["posterior_rgb"][0], "zero_effect": zero[0],
                       "reversed_effect_order": reverse[0], "source_state_copy": source_image[0].expand_as(target),
                       "rgb_copy": current[None].expand_as(target)}
            prediction_times, history_seconds, shuffled_instruction, shuffled_group = {}, None, None, None
            if prior is not None:
                torch.cuda.synchronize()
                history_started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    history = observe_history(model, codec, history_batch(batch))
                torch.cuda.synchronize()
                history_seconds = time.perf_counter()-history_started
                correct_text, correct_valid = (text_store.batch([sample["instruction"]], device)
                                               if text_store is not None else (None, None))
                conditions = [(f"correct_language_sample_{number}", correct_text, correct_valid, number)
                              for number in range(samples)] if text_store is not None else [
                                  (f"history_only_sample_{number}", None, None, number) for number in range(samples)]
                if text_store is not None:
                    conditions.append(("no_language_sample_0", None, None, 0))
                    alternatives = sorted((group, text) for group, text in instruction_groups[sample["source"]]
                                          if group != sample["group"] and text != sample["instruction"])
                    if alternatives:
                        chosen = int(np.random.default_rng(dataset.seed+index).integers(len(alternatives)))
                        shuffled_group, shuffled_instruction = alternatives[chosen]
                        shuffled_text, shuffled_valid = text_store.batch([shuffled_instruction], device)
                        conditions.append(("shuffled_language_sample_0", shuffled_text, shuffled_valid, 0))
                for name, text, valid, number in conditions:
                    torch.cuda.synchronize()
                    prediction_started = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                        forecast = generate_from_history(model, codec, prior, history, times, text, valid,
                            generator=torch.Generator(device=device).manual_seed(dataset.seed+index+number*len(dataset)))
                    torch.cuda.synchronize()
                    prediction_times[name] = time.perf_counter()-prediction_started
                    outputs[name] = forecast["rgb"][0]
                    del forecast
            case = {"case_id": sample["case_id"], "source": sample["source"],
                    "group": sample["group"], "camera": sample["camera"],
                    "native_hw": sample["native_hw"].tolist(), "language_provenance": sample["language_provenance"],
                    "episode_index": sample["episode_index"], "partition": sample["partition"],
                    "frame_indices": sample["frame_indices"].tolist(), "times": sample["times"].tolist(),
                    "time_unit": time_unit,
                    "preview_fps_not_physical": 5 if time_unit == "sampled_frame" else None,
                    "instruction": sample["instruction"], "history_frames": th,
                    "shuffled_instruction": shuffled_instruction,
                    "shuffled_group": shuffled_group,
                    "language_intervention": "different_group_within_source_not_verified_counterfactual" if shuffled_instruction else None,
                    "image_size": model.view_size(codec), "codec_decoder_size": codec.image_size,
                    "posterior_control_seconds": seconds,
                    "elapsed_seconds_all_conditions": seconds+(history_seconds or 0.)+sum(prediction_times.values()),
                    "effect_update_fraction": float(result["parts"]["effect_update_fraction"]),
                    "prior_history_encode_seconds": history_seconds,
                    "prior_dynamics_render_seconds": prediction_times,
                    "prior_single_sample_seconds_with_cached_text": {
                        key: history_seconds+value for key, value in prediction_times.items()},
                    "conditions": {}}
            for condition, prediction in outputs.items():
                frame_rows = image_records(prediction, target, result["pixel_valid"][0],
                                           current, times[0], perceptual, time_unit)
                case["conditions"][condition] = frame_rows
                if counts[sample["source"]] < videos_per_source:
                    full = torch.cat((images[0, :th], prediction), 0)
                    write_video(out / "videos" / sample["case_id"] / f"{condition}.mp4", full, batch["times"][0], time_unit)
            if counts[sample["source"]] < videos_per_source:
                write_video(out / "videos" / sample["case_id"] / "target.mp4", images[0], batch["times"][0], time_unit)
                write_json(out / "videos" / sample["case_id"] / "history_request.json", make_history_request(sample))
            sample_names = [key for key in outputs if key.startswith(("correct_language_sample_", "history_only_sample_"))]
            if sample_names:
                per_sample = {key: np.mean([row["rgb_l1"] for row in case["conditions"][key]]) for key in sample_names}
                case["generation"] = {"samples": len(sample_names), "expected_sequence_rgb_l1": float(np.mean(list(per_sample.values()))),
                                      "oracle_selected_sample_by_sequence_rgb_l1": min(per_sample, key=per_sample.get),
                                      "oracle_best_sequence_rgb_l1": float(min(per_sample.values()))}
            counts[sample["source"]] += 1
            records.append(case)
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps({"event": "evaluated", "case_id": sample["case_id"], "cases": len(records)}), flush=True)
    report = {"task": "paired_conditional_generation" if prior is not None else "posterior_conditioned_reconstruction", "cases": len(records),
              "dataset_contract": dataset.manifest["contract"],
              "time_unit": dataset.manifest.get("time_unit", "seconds"),
              "sampling": "fixed_seed_episode_uniform_within_source", "sampling_seed": dataset.seed,
              "sources": dict(counts), "excluded": excluded,
              "conditional_prior_evaluated": prior is not None, "independent_object_semantics_evaluated": False,
              "sampling_count": samples if prior is not None else 0,
              "text_encoding_latency_included": False,
              "conditions": summarize_cases(records),
              "by_source": {source: summarize_cases([row for row in records if row["source"] == source])
                            for source in counts}}
    generated = [row["generation"] for row in records if "generation" in row]
    if generated:
        report["generation"] = {key: {"cases": len(generated), "mean": float(np.mean([row[key] for row in generated])),
                                      "median": float(np.median([row[key] for row in generated]))}
                                for key in ("expected_sequence_rgb_l1", "oracle_best_sequence_rgb_l1")}
        report["oracle_selection"] = "One entire sampled sequence selected by its mean RGB L1; not deployment accuracy."
    write_json(out / "report.json", report)
    return report
