#!/usr/bin/env python3
"""Export native-view LPWM predictions for the shared independent CLIPort scorer."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_baselines.lpwm import LPWMConfig, LPWMVideoModel
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import write_video
from igsw.object_event_wm.native_view import native_rgb, paired_indices
from igsw.object_event_wm.prediction import history_batch, make_history_request
from igsw.object_event_wm.text import TextFeatureStore
from igsw.object_event_wm.training import device_batch, read_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    saved = read_checkpoint(args.checkpoint)
    model = LPWMVideoModel(LPWMConfig(**saved["model_config"]), saved["baseline_assets"]["repository"]).to(device).eval()
    model.load_state_dict(saved["model"])
    texts = TextFeatureStore(saved["args"]["text_cache"])
    data = build_video_dataset(args.manifest, partition=args.partition, seed=args.seed)
    groups = defaultdict(list)
    for entry in data.manifest["entries"]:
        if entry["partition"] == args.partition:
            groups[entry["scene_seed"]].append(entry)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exported = []
    with (out/"cases.jsonl").open("w") as stream:
        for index in paired_indices(data, args.items):
            sample = data[index]
            meta = json.loads(Path(sample["episode_path"]).read_text())
            alternative = next((entry for entry in groups[meta["seed"]] if entry["case_id"] != sample["case_id"]
                and entry["status"] == meta["status"] == "completed"), None)
            batch = device_batch(collate_object_event_video([sample]), device)
            th, tf = sample["history_frames"], data.config.future_frames
            language, _ = texts.batch([sample["instruction"]], device)
            folder = out/sample["case_id"]
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder/"history_request.json", make_history_request(sample))
            case = {"case_id": sample["case_id"], "scene_seed": meta["seed"], "instruction": sample["instruction"],
                    "combination": meta["combination"], "oracle_status": meta["status"], "model": "native_lpwm",
                    "model_image_size": model.native.image_size, "native_hw": sample["native_hw"].tolist(),
                    "conditions": {}, "timing_includes_text_encoding": False}

            def save(mode, tensor, sample_index=None, seed=None, seconds=None):
                name = mode if sample_index in (None, 0) else f"{mode}_sample_{sample_index}"
                rgb = native_rgb(tensor[0], sample["native_hw"].tolist())
                path = folder/f"{name}.npz"
                np.savez_compressed(path, rgb=rgb, times_seconds=sample["times"][th:].numpy())
                history = sample["rgb"][:th].float()/255.
                write_video(folder/f"{name}.mp4", torch.cat((history, torch.from_numpy(rgb).permute(0, 3, 1, 2))), sample["times"])
                case["conditions"][name] = {"mode": mode, "sample_index": sample_index, "seed": seed,
                    "uses_future_rgb": mode == "posterior", "file": str(path.relative_to(out)), "seconds": seconds}

            save("posterior", model.reconstruct_posterior(batch, language))
            if meta["status"] == "completed":
                instructions = {"correct_language": sample["instruction"], "no_language": ""}
                if alternative is not None:
                    other = json.loads(Path(alternative["episode"]).read_text())
                    instructions["opposite_order_language"] = other["instruction"]
                for count in range(args.samples):
                    seed = int(np.random.SeedSequence([args.seed, meta["seed"], count]).generate_state(1)[0])
                    for mode, instruction in instructions.items():
                        context = texts.batch([instruction], device)[0]
                        print(json.dumps({"event": "predicting_condition", "case_id": sample["case_id"],
                                          "condition": mode, "sample": count}), flush=True)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        prediction = model.predict_history(history_batch(batch), tf, context, seed=seed)
                        torch.cuda.synchronize()
                        save(mode, prediction, count, seed, time.perf_counter()-started)
            case["conditions"]["current_image_copy"] = {"mode": "current_image_copy", "uses_future_rgb": False}
            stream.write(json.dumps(case, allow_nan=False)+"\n")
            stream.flush()
            exported.append(case["case_id"])
            print(json.dumps({"case_id": case["case_id"], "exported": len(exported)}), flush=True)
    write_json(out/"report.json", {"contract": "native_view_cliport_predictions_v1", "manifest": args.manifest,
        "checkpoint": args.checkpoint, "case_ids": exported, "partition": args.partition, "samples": args.samples,
        "inventory": model.inventory(), "upstream_revision": model.config.upstream_revision,
        "native_config_changes": saved["native_config_changes"], "image_policy": model.config.image_policy,
        "prior_uses_future_rgb": False, "posterior_uses_future_rgb": True, "entity_labels_exported": False,
        "no_language_condition": "T5-encoded empty instruction; language tokens retained"})


if __name__ == "__main__":
    main()
