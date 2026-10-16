#!/usr/bin/env python3
"""Read upstream LPWM outputs with the shared per-frame RGB measurements."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_baselines.lpwm import LPWMConfig, LPWMVideoModel, local_reconstruction_loss
from igsw.object_event_wm.codec import letterbox_video
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import balanced_indices, image_records, summarize_cases, write_video
from igsw.object_event_wm.prediction import history_batch, make_history_request
from igsw.object_event_wm.training import device_batch, read_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--items-per-source", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    saved = read_checkpoint(args.checkpoint)
    assets = saved["baseline_assets"]
    model = LPWMVideoModel(LPWMConfig(**saved["model_config"]), assets["repository"]).to(device).eval()
    model.load_state_dict(saved["model"])
    criterion = local_reconstruction_loss(assets, model.config.native["normalize_rgb"]).to(device).eval()
    dataset = build_video_dataset(args.manifest, partition=args.partition, text_cache=saved["args"]["text_cache"])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with (output/"cases.jsonl").open("w") as stream:
        for index in balanced_indices(dataset, args.items_per_source):
            sample = dataset[index]
            batch = device_batch(collate_object_event_video([sample]), device)
            images, mask = letterbox_video(batch, model.config.native["image_size"])
            th, tf = batch["history_frames"], images.shape[1]-batch["history_frames"]
            language = batch["text_tokens"].float()
            empty_language = dataset.text_store.batch([""], device)[0]
            predictions = {"posterior": model.reconstruct_posterior(batch, language)[0],
                           "current_image_copy": images[0, th-1:th].expand(tf, -1, -1, -1)}
            timings = {}
            for count in range(args.samples):
                for name, text in (("correct_language", language), ("no_language", empty_language)):
                    key = f"{name}_sample_{count}"
                    print(json.dumps({"event": "predicting_condition", "case_id": sample["case_id"],
                                      "condition": key}), flush=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    predictions[key] = model.predict_history(history_batch(batch), tf, text,
                                                             seed=dataset.seed+index*1000+count)[0]
                    torch.cuda.synchronize()
                    timings[key] = time.perf_counter()-start
            case = {"case_id": sample["case_id"], "source": sample["source"], "group": sample["group"],
                    "instruction": sample["instruction"], "partition": args.partition,
                    "model": "upstream_lpwm", "native_hw": sample["native_hw"].tolist(),
                    "metric_image_size": model.config.native["image_size"], "image_policy": model.config.image_policy,
                    "time_unit": sample.get("time_unit", "seconds"), "conditions": {},
                    "generation_seconds": timings, "timing_includes_text_encoding": False,
                    "posterior_uses_future": True, "prior_uses_future": False,
                    "no_language_outcomes_are_ambiguous": True,
                    "no_language_condition": "T5-encoded empty instruction; language tokens retained",
                    "sample_reporting": "sample_0 primary; remaining samples reported individually, not best-of-N"}
            folder = output/"videos"/sample["case_id"]
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder/"history_request.json", make_history_request(sample))
            for name, prediction in predictions.items():
                case["conditions"][name] = image_records(prediction, images[0, th:], mask[0, th:], images[0, th-1],
                    batch["times"][0, th:], criterion.perceptual_loss, time_unit=case["time_unit"])
                write_video(folder/f"{name}.mp4", torch.cat((images[0, :th], prediction)),
                            batch["times"][0], time_unit=case["time_unit"])
            records.append(case)
            stream.write(json.dumps(case, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps({"case_id": case["case_id"], "cases": len(records)}), flush=True)
    write_json(output/"report.json", {"checkpoint": args.checkpoint, "manifest": args.manifest,
        "partition": args.partition, "cases": len(records), "metrics": summarize_cases(records),
        "inventory": model.inventory(), "native_config_changes": saved["native_config_changes"],
        "metric_image_size": model.config.native["image_size"], "time_unit": dataset.manifest.get("time_unit", "seconds"),
        "no_language_condition": "T5-encoded empty instruction; language tokens retained",
        "measurement_scope": "per-frame RGB; not independent object identity or event-order success"})


if __name__ == "__main__":
    main()
