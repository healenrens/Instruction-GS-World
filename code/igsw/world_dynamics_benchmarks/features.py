"""Frozen token export. No Dynamics, tracker, labels, or training losses are called."""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from igsw.adaptive_gaussian_wm.frozen_object_teacher_v70 import module_state
from igsw.adaptive_gaussian_wm.object_sequence_dynamics_v69 import ObjectSequencePosteriorV69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.query_object_video_encoder_v69 import QueryObjectVideoEncoderV69, history_queries_v69
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69

from .data import move_observation, read_observation, unique_indices
from .io import experiment_root, read_json, save_tensor, write_json


def packet(values, coordinates, times, valid, kind=0, end_times=None):
    end_times = times if end_times is None else end_times
    metadata = torch.cat((coordinates, times[:, None], end_times[:, None],
                          torch.full_like(times[:, None], kind)), -1)
    # Zero padding equalizes the readout's parameter count, without a trainable representation adapter.
    return {"tokens": F.pad(values.float(), (0, 1024 - values.shape[-1])),
            "metadata": metadata.float(), "valid": valid.bool()}


def select_tokens(part, budget, group_size=1):
    groups = len(part["tokens"]) // group_size
    chosen = unique_indices(groups, max(1, budget // group_size))
    selected = torch.tensor([group * group_size + token for group in chosen for token in range(group_size)],
                            device=part["tokens"].device)
    return {name: value[selected] for name, value in part.items()}


class FrozenRepresentation(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config, self.model = config, model
        settings = config["models"]
        dtype = torch.bfloat16 if config["device"].startswith("cuda") else torch.float32
        self.provenance = {"model": model, "frozen": True}
        if model == "vjepa2":
            # V-JEPA2 clip features, not V69's per-frame V-JEPA2.1 adapter.
            self.encoder, predictor = torch.hub.load(settings["vjepa_repository"], settings["vjepa_factory"],
                                                      source="local", pretrained=False)
            del predictor
            saved = torch.load(settings["vjepa_weights"], map_location="cpu", weights_only=False)
            weights = {key.removeprefix("module.").removeprefix("backbone."): value
                       for key, value in saved[settings["vjepa_checkpoint_key"]].items()}
            # The official RoPE hub factory omits the checkpoint's unused absolute positional table.
            if "pos_embed" not in self.encoder.state_dict():
                weights.pop("pos_embed", None)
            self.encoder.load_state_dict(weights)
            self.transform = torch.hub.load(settings["vjepa_repository"], "vjepa2_preprocessor",
                                           source="local", crop_size=settings["vjepa_crop"])
            self.provenance.update({"version": "V-JEPA2", "factory": settings["vjepa_factory"],
                                   "repository": settings["vjepa_repository"], "weights": settings["vjepa_weights"],
                                   "input": "clip-level official single-center-crop transform", "crop": settings["vjepa_crop"]})
        elif model == "dino":
            self.perception = PretrainedVisualEncoderV69("dinov3_vitl16", settings["dino_repository"],
                                                          settings["dino_weights"],
                                                          config["export"]["frame_batch"], dtype=dtype)
            self.provenance.update(self.perception.provenance)
        else:
            saved = torch.load(settings["teacher_checkpoint"], map_location="cpu", weights_only=False)
            self.state_config = ObjectVideoConfigV69(**saved["config"])
            # State experiments retain the teacher's DINO implementation and native coordinates.
            self.perception = PretrainedVisualEncoderV69(self.state_config.encoder,
                settings["dino_repository"], settings["dino_weights"], config["export"]["frame_batch"],
                dtype=dtype, history_seconds=self.state_config.history_seconds, saved_backbone=saved["perception"])
            self.encoder = QueryObjectVideoEncoderV69(self.state_config)
            self.encoder.load_state_dict(module_state(saved, "encoder"))
            if model == "state_z":
                self.target_encoder = QueryObjectVideoEncoderV69(self.state_config)
                self.target_encoder.load_state_dict(module_state(saved, "target_encoder"))
                self.posterior = ObjectSequencePosteriorV69(self.state_config)
                self.posterior.load_state_dict(module_state(saved, "posterior"))
            self.provenance.update({"teacher": settings["teacher_checkpoint"], "step": saved["step"],
                                   "teacher_config": saved["config"], "posterior_scope": "observed intervals only"})
        self.requires_grad_(False).to(config["device"]).eval()

    def trim(self, part, budget, group_size=1):
        selected = select_tokens(part, budget, group_size)
        self.sampling.append({"original_tokens": len(part["tokens"]), "retained_tokens": len(selected["tokens"]),
                              "discarded_fraction": 1 - len(selected["tokens"]) / len(part["tokens"]),
                              "atomic_group_tokens": group_size})
        return selected

    @torch.no_grad()
    def forward(self, batch):
        self.sampling = []
        budget = self.config["export"]["token_budget"]
        times = batch["times"][0]
        if self.model == "vjepa2":
            frames = batch["rgb"][0].permute(0, 2, 3, 1).cpu().numpy()
            clip = self.transform(frames)[0][None].to(self.config["device"])
            encoded = self.encoder(clip)[0]
            gh = gw = self.config["models"]["vjepa_crop"] // 16
            tt = len(times) // 2
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, gh, device=times.device),
                                    torch.linspace(-1, 1, gw, device=times.device), indexing="ij")
            xy = torch.stack((xx, yy), -1).reshape(-1, 2).repeat(tt, 1)
            first, end = times[::2].repeat_interleave(gh * gw), times[1::2].repeat_interleave(gh * gw)
            part = packet(encoded, xy, first, torch.ones(len(encoded), device=times.device, dtype=torch.bool),
                          end_times=end)
            return self.trim(part, budget)
        perception = self.perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
        if self.model == "dino":
            values = perception.features[0].flatten(0, 1)
            xy = perception.coordinates[0].repeat(len(times), 1)
            part = packet(values, xy, times.repeat_interleave(perception.features.shape[2]),
                          perception.valid[0].flatten())
            return self.trim(part, budget)
        segments = min(self.config["export"]["observed_segments"], len(times) - 1)
        boundaries = np.linspace(0, len(times) - 1, segments + 1).round().astype(int).tolist()
        # Query initialization is identical for State and State+z, and sees only the first observed interval.
        queries = history_queries_v69(perception.prefix(boundaries[1] + 1), self.state_config.object_queries)
        states = self.encoder(perception, queries)
        values = torch.stack([state.tokens[0] for state in states]).flatten(0, 2)
        xy = torch.stack([state.centers[0] for state in states]).flatten(0, 2)
        count = self.state_config.object_queries * self.state_config.tokens_per_object
        valid = queries.valid[0].repeat_interleave(self.state_config.tokens_per_object).repeat(len(times))
        part = packet(values, xy, times.repeat_interleave(count), valid)
        # State content has the same budget in both internal ablations; z uses separate token positions.
        part = self.trim(part, budget * 3 // 4, count)
        if self.model == "state":
            return part
        effects = []
        for first, stop in zip(boundaries[:-1], boundaries[1:]):
            observed, targets = states[first].detach(), []
            for frame in range(first + 1, stop + 1):
                observed = self.target_encoder.observe(observed, perception.features[:, frame],
                    perception.coordinates, perception.valid[:, frame], perception.times[:, frame])
                targets.append(observed)
            effect = self.posterior(states[first], targets, deterministic=True)
            values = effect["value"][0].flatten(0, 1)
            xy = states[first].centers[0, :, 0].repeat_interleave(self.state_config.effect_tokens, 0)
            size = len(values)
            effects.append(packet(values, xy, times[first].expand(size),
                queries.valid[0].repeat_interleave(self.state_config.effect_tokens), kind=1,
                end_times=times[stop].expand(size)))
        z = self.trim({name: torch.cat([effect[name] for effect in effects]) for name in part}, budget // 4,
                      self.state_config.object_queries * self.state_config.effect_tokens)
        return {name: torch.cat((part[name], z[name])) for name in part}


def export_features(config, model):
    root = experiment_root(config, model)
    directory = root / "features"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_json(config["manifest"])
    runtime = FrozenRepresentation(config, model)
    rows = []
    for index, row in enumerate(manifest["rows"]):
        path = directory / (row["id"] + ".pt")
        if not path.exists():
            start = time.perf_counter()
            observation = read_observation(row, config["export"]["frames"])
            decoded = time.perf_counter()
            batch = move_observation(observation, config["device"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["device"].startswith("cuda")):
                feature = runtime(batch)
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            feature = {name: value.cpu() for name, value in feature.items()}
            # Keep native feature width on disk; common-width padding belongs to the readout batch.
            stored_width = runtime.state_config.width if model in ("state", "state_z") else 1024
            feature["tokens"] = feature["tokens"][:, :stored_width].half()
            save_tensor(path, {**feature,
                               "id": row["id"], "frame_indices": observation["frame_indices"],
                               "times": observation["times"].tolist(), "native_hw": observation["native_hw"].tolist(),
                               "time_basis": row["time_basis"],
                               "input_cue": observation["input_cue"],
                               "decode_seconds": decoded - start, "encoder_seconds": time.perf_counter() - decoded,
                               "token_sampling": runtime.sampling,
                               "provenance": runtime.provenance})
        rows.append({**row, "feature_path": str(path)})
        print(f"export model={model} {index + 1}/{len(manifest['rows'])} {row['id']}", flush=True)
    write_json(root / "features.json", {"config": config, "representation": runtime.provenance,
               "rows": rows, "num_classes": manifest["num_classes"], "scope": manifest["scope"],
               "protocol": manifest["protocol"], "benchmark": manifest["benchmark"]})
    return root / "features.json"
