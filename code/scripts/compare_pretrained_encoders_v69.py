#!/usr/bin/env python3
"""Frozen encoder localization, temporal correspondence, and train/held transition probes."""

import argparse
import gc
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69, sample_perception_v69
from igsw.adaptive_gaussian_wm.object_sequence_evaluation_v69 import distribution_v69
from igsw.adaptive_gaussian_wm.object_sequence_annotations_v69 import independent_measurements_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


@torch.no_grad()
def measure(runtime, sample, config, device, independent=False):
    batch = move_batch_v69(collate_object_video_v69([sample]), device)
    labels = None
    if independent:
        batch, _, labels = independent_measurements_v69(batch, sample["annotation"])
    torch.cuda.synchronize()
    begin = time.monotonic()
    field = runtime(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
    torch.cuda.synchronize()
    elapsed = time.monotonic()-begin
    teacher = batch["teacher"]
    times, points = teacher["xy"].shape[1:3]
    features, feature_valid = sample_perception_v69(field, teacher["xy"], torch.arange(times, device=device)[None])
    ids = torch.arange(points, device=device)
    reference = features[0, teacher["reference_index"][0], ids]
    native_scale = (batch["native_hw"][0, [1, 0]].float()-1)*.5
    rows, xs, ys = [], [], []
    for frame in range(config.history_frames, times):
        valid = teacher["valid"][0, frame] & teacher["point_present"][0] & feature_valid[0, frame]
        measured = torch.where(valid)[0]
        token_ids = torch.where(field.valid[0, frame])[0]
        if not len(measured) or not len(token_ids):
            continue
        tokens = F.normalize(field.features[0, frame, token_ids].float(), dim=-1)
        similarity = reference[measured] @ tokens.T
        best = token_ids[similarity.argmax(-1)]
        predicted = field.coordinates[0, best]
        actual = teacher["xy"][0, frame, measured]
        errors = ((predicted-actual)*native_scale).norm(dim=-1)
        nearest = torch.cdist(actual*native_scale, field.coordinates[0, token_ids]*native_scale).min(-1).values
        displacement = ((actual-teacher["reference_xy"][0, measured])*native_scale).norm(dim=-1)
        ids_list = measured.cpu().tolist()
        rows.append({"case": sample["case_id"], "source": sample["source"], "seconds": float(batch["times"][0, frame]),
                     "encoder": runtime.kind, "measurement_source": "independent" if independent else "tracker",
                     "metric": "observed_frame_feature_correspondence_not_future_prediction", "epe_px": errors.cpu().tolist(),
                     "measurement_ids": teacher["point_ids"][0, measured].cpu().tolist(),
                     "displacement_px": displacement.cpu().tolist(), "native_hw": batch["native_hw"][0].cpu().tolist(),
                     "object_extent_px": [labels["object_extent_px"][i] for i in ids_list] if labels else [None]*len(ids_list),
                     "spatial_quantization_floor_px": distribution_v69(nearest), **distribution_v69(errors)})
        take = measured[:16]
        delta_feature = features[0, frame, take]-reference[take]
        delta_xy = teacher["xy"][0, frame, take]-teacher["reference_xy"][0, take]
        xs.append(delta_feature.cpu())
        ys.append(delta_xy.cpu())
    x = torch.cat(xs) if xs else torch.empty((0, config.perception_dim))
    y = torch.cat(ys) if ys else torch.empty((0, 2))
    return rows, x, y, {"case": sample["case_id"], "seconds": elapsed, "tokens_per_frame": field.features.shape[2],
                       "peak_memory_gb": torch.cuda.max_memory_allocated()/1024**3}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dinov3_repository", required=True)
    p.add_argument("--dinov3_weights", required=True)
    p.add_argument("--vjepa_repository", required=True)
    p.add_argument("--vjepa_weights", required=True)
    p.add_argument("--items", type=int, default=400)
    p.add_argument("--probe_train_items", type=int, default=64)
    p.add_argument("--frame_batch", type=int, default=2)
    p.add_argument("--annotations", default="")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--wandb_project", default="instruct-gs-world")
    p.add_argument("--wandb_entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    p.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    p.add_argument("--wandb_name", default="object_video_v69_frozen_encoder_comparison")
    args = p.parse_args()
    config, device = ObjectVideoConfigV69(), torch.device("cuda:0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = ObjectVideoSequenceDatasetV69(args.manifest, config, args.seed, "train")
    held = ObjectVideoSequenceDatasetV69(args.manifest, config, args.seed, "held", args.annotations)
    train_order, held_order = list(range(len(train))), list(range(len(held)))
    rng = random.Random(args.seed)
    rng.shuffle(train_order)
    rng.shuffle(held_order)
    train_order, held_order = train_order[:args.probe_train_items], held_order[:args.items]
    all_rows, profiles, probes, failures = [], [], {}, []
    for name, repository, weights in (("dinov3_vitl16", args.dinov3_repository, args.dinov3_weights),
                                       ("vjepa2_1_vitl16", args.vjepa_repository, args.vjepa_weights)):
        runtime = PretrainedVisualEncoderV69(name, repository, weights, args.frame_batch, history_seconds=config.history_seconds).to(device)
        torch.cuda.reset_peak_memory_stats()
        train_x, train_y = [], []
        for index in train_order:
            sample = train[index]
            if "decode_error" in sample:
                failures.append(sample)
                continue
            _, x, y, profile = measure(runtime, sample, config, device)
            train_x.append(x)
            train_y.append(y)
            profiles.append({"encoder": name, "partition": "train", **profile})
        x, y = torch.cat(train_x).double(), torch.cat(train_y).double()
        mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
        design = torch.cat(((x-mean)/std, torch.ones((len(x), 1), dtype=x.dtype)), -1)
        # A fixed ridge probe, fitted on train episodes only; this is inverse-transition readability.
        gram = design.T @ design / max(1, len(x))
        coefficient = torch.linalg.solve(gram + .001*torch.eye(gram.shape[0], dtype=gram.dtype), design.T @ y/max(1, len(x)))
        held_errors, zero_errors, mean_errors, probe_cases = [], [], [], []
        train_mean = y.mean(0)
        for number, index in enumerate(held_order):
            sample = held[index]
            if "decode_error" in sample:
                failures.append(sample)
                continue
            rows, hx, hy, profile = measure(runtime, sample, config, device)
            all_rows.extend(rows)
            profiles.append({"encoder": name, "partition": "held", **profile})
            hd = torch.cat(((hx.double()-mean)/std, torch.ones((len(hx), 1), dtype=torch.float64)), -1)
            errors = (hd@coefficient-hy.double()).norm(dim=-1)
            zero = hy.double().norm(dim=-1)
            average = (hy.double()-train_mean).norm(dim=-1)
            held_errors.extend(errors.tolist())
            zero_errors.extend(zero.tolist())
            mean_errors.extend(average.tolist())
            probe_cases.append({"case": sample["case_id"], "source": sample["source"],
                                "probe": distribution_v69(errors), "zero_delta": distribution_v69(zero),
                                "train_mean_delta": distribution_v69(average), "per_point_probe_error": errors.tolist()})
            annotation = sample["annotation"]
            if (annotation and annotation["queries"] and annotation["tracks"] and annotation["uses_training_tracker"] is False
                    and annotation["provenance"] in ("human_annotation", "simulator_ground_truth")):
                independent_rows, _, _, independent_profile = measure(runtime, sample, config, device, independent=True)
                all_rows.extend(independent_rows)
                profiles.append({"encoder": name, "partition": "independent", **independent_profile})
            print(f"[encoder-compare-v69] encoder={name} held={number+1}/{len(held_order)} case={sample['case_id']}", flush=True)
        probes[name] = {"metric": "two_observed_frames_inverse_transition_probe_normalized_xy_error",
                        "future_image_used": True, "not_a_forecasting_metric": True,
                        "train_vectors": len(x), "held": distribution_v69(torch.tensor(held_errors)),
                        "target_positions_used_to_sample_observed_features": True,
                        "zero_delta": distribution_v69(torch.tensor(zero_errors)),
                        "train_mean_delta": distribution_v69(torch.tensor(mean_errors)), "cases": probe_cases,
                        "encoder_parameters": runtime.provenance["parameters"]}
        torch.save({"mean": mean, "std": std, "coefficient": coefficient, "train_cases": train_order}, out / f"{name}_probe.pt")
        del runtime, x, y, design, gram, coefficient
        gc.collect()
        torch.cuda.empty_cache()
    report = {"status": "completed_frozen_encoder_measurements", "rows": all_rows, "profiles": profiles, "probes": probes,
              "failures": failures, "config": config.to_dict(), "automatic_encoder_selection": False,
              "comparison": "same native resolution, sampled times, points and nearest-feature readout; no fine-tuning"}
    write_json(out / "report.json", report)
    if args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="object-video-sequence-v69", job_type="encoder-comparison", mode=args.wandb_mode, config=vars(args))
        for start in range(0, len(all_rows), 5000):
            table = wandb.Table(columns=["encoder", "case", "source", "seconds", "measurement_source", "count", "mean", "p50", "p90", "p95"])
            for row in all_rows[start:start+5000]:
                table.add_data(*[row[name] for name in table.columns])
            run.log({f"encoder_comparison/cases_{start//5000:04d}": table})
        run.summary.update({"probes": probes, "measured_rows": len(all_rows), "automatic_encoder_selection": False})
        artifact = wandb.Artifact(args.wandb_name, type="frozen-encoder-comparison")
        artifact.add_dir(str(out))
        run.log_artifact(artifact)
        run.finish()
    print(f"[encoder-compare-v69] report={out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
