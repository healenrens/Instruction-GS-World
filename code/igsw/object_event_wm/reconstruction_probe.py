"""Fixed-frame reconstruction diagnosis without posterior or dynamics objectives."""

from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .codec import letterbox_video
from .data import collate_object_event_video, write_json
from .independent_metrics import score_rgb_outcomes
from .model import weighted_mean
from .native_view import native_rgb
from .simulation_data import load_truth
from .state import ObjectState, coordinates
from .training import device_batch


class PerFrameStates(torch.nn.Module):
    def __init__(self, tokens, centers):
        super().__init__()
        self.tokens = torch.nn.Parameter(tokens.detach().float().clone())
        self.center_logits = torch.nn.Parameter(centers.detach().float().atanh())

    def forward(self, indices):
        return ObjectState(self.tokens[indices], self.center_logits[indices].tanh(),
                           self.tokens.new_zeros(len(indices)))


@torch.no_grad()
def initialize_frame_states(model, codec, bank, batch_size, precision):
    device = next(model.parameters()).device
    tokens, centers = [], []
    model.eval()
    for first in range(0, len(bank["features"]), batch_size):
        features = bank["features"][first:first+batch_size].to(device)
        valid = bank["patch_valid"][first:first+batch_size].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            state = model.observe(features[:, None], valid[:, None], features.new_zeros((len(features), 1)))[0]
        tokens.append(state.tokens)
        centers.append(state.centers)
    return PerFrameStates(torch.cat(tokens), torch.cat(centers))


def set_probe_trainable(model, frame_states=None):
    model.requires_grad_(False)
    model.renderer.requires_grad_(True)
    if frame_states is None:
        model.encoder.requires_grad_(True)
        return list(model.encoder.parameters()) + list(model.renderer.parameters())
    return list(frame_states.parameters()) + list(model.renderer.parameters())


def probe_state_dict(model):
    return {"encoder": model.encoder.state_dict(), "renderer": model.renderer.state_dict()}


def load_probe_state(model, state):
    model.encoder.load_state_dict(state["encoder"])
    model.renderer.load_state_dict(state["renderer"])


@torch.no_grad()
def prepare_frames(dataset, codec, view_size, scenes, seconds, device, precision):
    indices = np.random.default_rng(dataset.seed).permutation(len(dataset))[:scenes]
    images, pixels, features, patch_valid, metadata, truth = [], [], [], [], [], []
    skipped = []
    for index in indices:
        sample = dataset[int(index)]
        selected = list(dict.fromkeys(int((sample["times"]-second).abs().argmin()) for second in seconds))
        valid = [frame for frame in selected if bool(sample["frame_valid"][frame])]
        skipped.extend({"case_id": sample["case_id"], "sequence_frame": frame, "reason": "unrecorded_time"}
                       for frame in selected if frame not in valid)
        native = sample["rgb"][valid].permute(0, 2, 3, 1).numpy()
        anchor = sample["history_frames"]-1
        labels = load_truth(sample["episode_path"], sample["frame_indices"][[anchor]+valid].numpy())["segmentation"]
        episode = json.loads(Path(sample["episode_path"]).read_text())
        subset = {**sample, "rgb": sample["rgb"][valid], "frame_valid": sample["frame_valid"][valid],
                  "times": sample["times"][valid], "frame_indices": sample["frame_indices"][valid]}
        batch = device_batch(collate_object_event_video([subset]), device)
        rgb, masks = letterbox_video(batch, view_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            encoded = codec.encode(rgb)
        images.append(rgb[0].cpu())
        pixels.append(masks[0].cpu())
        features.append(encoded[0].float().cpu())
        patch_valid.append(codec.patch_valid(masks)[0].cpu())
        for position, frame in enumerate(valid):
            metadata.append({"case_id": sample["case_id"], "scene_seed": episode["seed"],
                "partition": sample["partition"], "sequence_frame": frame,
                "recorded_frame": int(sample["frame_indices"][frame]), "seconds": float(sample["times"][frame]),
                "episode": sample["episode_path"], "native_hw": sample["native_hw"].tolist(),
                "instruction": sample["instruction"]})
            truth.append({"rgb": native[position], "anchor_labels": labels[0], "labels": labels[position+1],
                "actors": sorted({event["actor_body_id"] for event in episode["program"]}),
                "robots": episode["robot_and_tool_body_ids"]})
    return {"images": torch.cat(images), "pixel_valid": torch.cat(pixels),
            "features": torch.cat(features), "patch_valid": torch.cat(patch_valid),
            "metadata": metadata, "truth": truth, "skipped": skipped}


def single_frame_forward(model, codec, features, valid, frame_states=None, indices=None):
    # Each image is an independent observation. No frame ID or prior image enters this path.
    times = features.new_zeros((len(features), 1))
    states = (model.observe(features[:, None], valid[:, None], times) if frame_states is None
              else [frame_states(indices)])
    rendered, _ = model.render_features(states, coordinates(codec.grid_size, features.device))
    rgb = model.decode_features(codec, rendered)
    return rendered[:, 0], rgb[:, 0]


@torch.no_grad()
def evaluate_frames(model, codec, bank, batch_size, precision, frame_states=None):
    model.eval()
    device = next(model.parameters()).device
    records, predicted = [], []
    for start in range(0, len(bank["images"]), batch_size):
        end = start+batch_size
        target = bank["features"][start:end].to(device)
        valid = bank["patch_valid"][start:end].to(device)
        images = bank["images"][start:end].to(device)
        pixels = bank["pixel_valid"][start:end].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            indices = torch.arange(start, start+len(target), device=device)
            rendered, rgb = single_frame_forward(model, codec, target, valid, frame_states, indices)
        for offset in range(len(rgb)):
            index = start+offset
            external = bank["truth"][index]
            native = native_rgb(rgb[offset:offset+1], bank["metadata"][index]["native_hw"])
            actor = score_rgb_outcomes(native, external["rgb"][None], external["anchor_labels"],
                external["labels"][None], external["actors"], external["robots"],
                [bank["metadata"][index]["seconds"]])
            records.append({"index": index,
                "feature_mse": float(weighted_mean((rendered[offset]-target[offset]).square(), valid[offset, :, None])),
                "rgb_l1_unclamped": float(weighted_mean((rgb[offset]-images[offset]).abs(), pixels[offset])),
                "full_native_rgb_l1": actor["full_rgb_l1"], "actor_rgb_l1": actor["actor_region_rgb_l1"],
                "actors": actor["frames"][0]["actors"]})
        predicted.append(rgb.float().clamp(0, 1).cpu())
    return torch.cat(predicted), records


def mean_scores(records):
    keys = ("feature_mse", "rgb_l1_unclamped", "full_native_rgb_l1", "actor_rgb_l1")
    return {key: float(np.mean([row[key] for row in records if row[key] is not None]))
            for key in keys if any(row[key] is not None for row in records)}


@torch.no_grad()
def codec_reference(model, codec, bank, batch_size, precision):
    device = next(model.parameters()).device
    images = []
    for features in bank["features"].split(batch_size):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            rgb = model.decode_features(codec, features[:, None].to(device))[:, 0]
        images.append(rgb.float().clamp(0, 1).cpu())
    return torch.cat(images)


def reference_scores(bank, reference):
    rows = []
    for index, image in enumerate(reference):
        external = bank["truth"][index]
        native = native_rgb(image[None], bank["metadata"][index]["native_hw"])
        result = score_rgb_outcomes(native, external["rgb"][None], external["anchor_labels"],
            external["labels"][None], external["actors"], external["robots"], [bank["metadata"][index]["seconds"]])
        rows.append({"index": index, "full_native_rgb_l1": result["full_rgb_l1"],
                     "actor_rgb_l1": result["actor_region_rgb_l1"], "actors": result["frames"][0]["actors"]})
    return rows


@torch.no_grad()
def load_encoder_comparison(model, codec, bank, path, batch_size, precision):
    original = {name: {key: value.detach().clone() for key, value in state.items()}
                for name, state in probe_state_dict(model).items()}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_probe_state(model, checkpoint["model"])
    images, records = evaluate_frames(model, codec, bank, batch_size, precision)
    load_probe_state(model, original)
    return {"rgb": images, "records": records, "checkpoint": str(path), "step": checkpoint["step"]}


def write_probe_review(out, bank, reference, before, after, report, comparison=None):
    out = Path(out)
    folder = out/"frames"
    folder.mkdir(parents=True, exist_ok=True)
    sections = []
    panels = [("target", bank["images"]), ("codec_reference", reference), ("before", before)]
    if comparison is not None:
        panels.append(("encoder_after", comparison["rgb"]))
    panels.append(("free_state_after" if report["state_source"] == "free" else "after", after))
    for index, meta in enumerate(bank["metadata"]):
        canvas = Image.new("RGB", (256*len(panels), 280), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (name, tensor) in enumerate(panels):
            image = Image.fromarray((tensor[index].permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype(np.uint8))
            canvas.paste(image.resize((256, 256)), (column*256, 24))
            draw.text((column*256+4, 5), name, fill="black")
        name = f"frame_{index:02d}.png"
        canvas.save(folder/name)
        sections.append(f'<section><h2>{escape(meta["case_id"])} / {meta["seconds"]:.1f}s</h2>'
                        f'<img src="frames/{name}" alt="Fixed-frame reconstruction comparisons"></section>')
    write_json(out/"report.json", report)
    (out/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>V71 Reconstruction Probe</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;letter-spacing:0}'
        'section{padding:16px 0;border-top:1px solid #ccc}img{max-width:100%}h2{font-size:18px}</style>'
        f'<h1>Fixed-frame reconstruction / step {report["step"]}</h1>'
        '<p>In-sample reconstruction, not held generalization or object dynamics. '
        'No posterior, effects or dynamics in the optimization path.</p>'
        '<p><a href="report.json">Compact report</a></p>'+''.join(sections)+'</html>')


def train_probe(model, codec, bank, args, tracker=None, resume=None, comparison=None):
    out = Path(args.out)
    state_source = getattr(args, "state_source", "encoder")
    if resume is not None:
        load_probe_state(model, resume["model"])
    frame_states = (initialize_frame_states(model, codec, bank, args.batch, args.precision)
                    if state_source == "free" else None)
    parameters = set_probe_trainable(model, frame_states)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    if resume is not None:
        if frame_states is not None:
            frame_states.load_state_dict(resume["free_states"])
        optimizer.load_state_dict(resume["optimizer"])
        start = resume["step"]
        initial = torch.load(out/"initial.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    else:
        before, initial_records = evaluate_frames(model, codec, bank, args.batch, args.precision, frame_states)
        reference = codec_reference(model, codec, bank, args.batch, args.precision)
        initial = {"before": before, "reference": reference, "records": initial_records,
                   "codec_records": reference_scores(bank, reference), "comparison": comparison}
        torch.save(initial, out/"initial.pt")
    device = next(model.parameters()).device
    count = len(bank["images"])
    config = asdict(model.config)

    def export(step):
        after, rows = evaluate_frames(model, codec, bank, args.batch, args.precision, frame_states)
        report = {"contract": "v71_fixed_frame_observed_reconstruction_v1", "step": step,
            "base_checkpoint": args.checkpoint, "initialization": args.initialization,
            "source_revision": args.source_revision, "model_config": config,
            "state_source": state_source,
            "trainable_modules": ["per_frame_states", "renderer"] if frame_states is not None else ["encoder", "renderer"],
            "state_shape_per_frame": [model.config.slots, model.config.local_tokens, model.config.width],
            "frames": count, "scenes": len({row["scene_seed"] for row in bank["metadata"]}),
            "partition": bank["metadata"][0]["partition"],
            "input": ("per-image optimized tokens/centers initialized from the original encoder"
                      if frame_states is not None else "one RGB frame, state reset for every image; zero model time"),
            "objective": "original feature MSE + original RGB L1; no KL, change, posterior or dynamics losses",
            "feature_weight": model.config.feature_weight, "rgb_weight": model.config.rgb_weight,
            "effective_batch": count, "micro_batch": args.batch,
            "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay, "clip_norm": 1.0},
            "codec": "frozen; target DINO features cached only for this small fixed set",
            "gt_masks_role": "scoring only; never input or training loss",
            "scope": "in-sample optimization diagnosis; no generalization or physical-object validity claim",
            "before": mean_scores(initial["records"]), "after": mean_scores(rows),
            "codec_reference": {key: float(np.mean([r[key] for r in initial["codec_records"] if r[key] is not None]))
                                for key in ("full_native_rgb_l1", "actor_rgb_l1")},
            "per_frame": [{**meta, "before": initial["records"][i], "after": rows[i],
                           "codec_reference": initial["codec_records"][i]} for i, meta in enumerate(bank["metadata"])]}
        reference_fit = initial.get("comparison")
        if reference_fit is not None:
            report["encoder_comparison"] = {key: reference_fit[key] for key in ("checkpoint", "step")}
            report["encoder_comparison"]["scores"] = mean_scores(reference_fit["records"])
            for index, row in enumerate(report["per_frame"]):
                row["encoder_after"] = reference_fit["records"][index]
        write_probe_review(out, bank, initial["reference"], initial["before"], after, report, reference_fit)
        evaluation = {"reconstruction/"+key: value for key, value in report["after"].items()}
        evaluation.update({"reference/initial_"+key: value for key, value in report["before"].items()})
        evaluation.update({"reference/codec_"+key: value for key, value in report["codec_reference"].items()})
        if reference_fit is not None:
            evaluation.update({"reference/encoder_"+key: value
                               for key, value in report["encoder_comparison"]["scores"].items()})
        for index, row in enumerate(rows):
            for actor in row["actors"]:
                if actor["rgb_l1"] is not None:
                    evaluation[f"frames/frame_{index:02d}/actor_{actor['body_id']}_rgb_l1"] = actor["rgb_l1"]
        print(json.dumps({"event": "reconstruction_probe_evaluation", "step": step, **evaluation}), flush=True)
        if tracker is not None:
            import swanlab
            swanlab.log(evaluation, step=step)
        return report

    def save(step):
        temporary = out/"latest.partial"
        torch.save({"format": "v71_observed_reconstruction_probe", "step": step,
            "model": probe_state_dict(model), "model_config": config, "optimizer": optimizer.state_dict(),
            "state_source": state_source,
            "free_states": frame_states.state_dict() if frame_states is not None else None,
            "args": vars(args), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "swanlab_run_id": tracker.id if tracker is not None else None}, temporary)
        temporary.replace(out/"latest.pt")

    report = export(start)
    with (out/"train.jsonl").open("a" if resume is not None else "x") as log:
        for step in range(start+1, args.steps+1):
            model.train()
            codec.eval()
            optimizer.zero_grad(set_to_none=True)
            feature_sum, rgb_sum = 0., 0.
            began = time.perf_counter()
            # All fixed frames contribute to every update, independent of micro-batch size.
            feature_denominator = bank["patch_valid"].sum().to(device)*bank["features"].shape[-1]
            rgb_denominator = bank["pixel_valid"].sum().to(device)*3
            for first in range(0, count, args.batch):
                last = first+args.batch
                target = bank["features"][first:last].to(device)
                valid = bank["patch_valid"][first:last].to(device)
                images = bank["images"][first:last].to(device)
                pixels = bank["pixel_valid"][first:last].to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    indices = torch.arange(first, first+len(target), device=device)
                    rendered, rgb = single_frame_forward(model, codec, target, valid, frame_states, indices)
                    feature_loss = ((rendered.float()-target).square()*valid[..., None]).sum()/feature_denominator
                    rgb_loss = ((rgb.float()-images).abs()*pixels).sum()/rgb_denominator
                    loss = model.config.feature_weight*feature_loss + model.config.rgb_weight*rgb_loss
                loss.backward()
                feature_sum += float(feature_loss.detach())
                rgb_sum += float(rgb_loss.detach())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {"step": step, "loss": model.config.feature_weight*feature_sum+model.config.rgb_weight*rgb_sum,
                    "feature_mse": feature_sum, "rgb_l1": rgb_sum, "grad_norm": grad_norm,
                    "seconds": time.perf_counter()-began, "effective_batch": count}
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                print(json.dumps({"event": "reconstruction_probe", **row}), flush=True)
                if tracker is not None:
                    import swanlab
                    swanlab.log(row, step=step)
            if step % args.save_every == 0 or step == args.steps:
                save(step)
                report = export(step)
    return report
