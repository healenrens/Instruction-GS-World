#!/usr/bin/env python3
"""CPU wiring integration: tiny explicit fixture, not a Qwen/GPU capacity test."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import Dataset

from igsw.adaptive_gaussian_wm.language_effect_model_v70 import LanguageEffectModelV70
from igsw.adaptive_gaussian_wm.v70_training import add_training_arguments_v70, train_v70


class FixtureConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(8, 8).requires_grad_(False)
        self.text = nn.Linear(8, 2560)

    @property
    def text_blocks(self):
        return self.text

    @property
    def fsdp_ignored_modules(self):
        return (self.visual,)

    def prepare_inputs(self, batch):
        return {"values": batch["vlm_values"]}

    def forward(self, inputs):
        values = self.text(self.visual(inputs["values"]))
        return {"last_hidden_state": values, "attention_mask": torch.ones(values.shape[:2], dtype=torch.bool)}

    def parameter_inventory(self):
        return {}


class FixtureHistory(nn.Module):
    teacher = {"path": "explicit_cpu_fixture", "global_step": 0}

    def forward(self, batch):
        return {name: batch[name] for name in ("tokens", "centers", "times", "query_valid")}


class FixtureDataset(Dataset):
    entries = [{"source": "fixture", "group": "different_episode", "episode_index": i} for i in range(5)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        index, epoch, occurrence = key
        rng = torch.Generator().manual_seed(index*31 + epoch*13 + occurrence)
        return {"vlm_values": torch.randn(5, 8, generator=rng),
                "tokens": torch.randn(3, 3, 9, 512, generator=rng),
                "centers": torch.randn(3, 3, 9, 2, generator=rng),
                "times": torch.tensor([-3., -1., 0.]), "query_valid": torch.tensor([True, index % 2 == 0, True]),
                "target_mean": torch.randn(3, 4, 64, generator=rng)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    out = Path(options.out)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    initial = LanguageEffectModelV70("fixture", conditioner=FixtureConditioner(),
                                    expert_kwargs={"num_blocks": 2, "width": 32, "heads": 4, "ffn_dim": 64})
    base = add_training_arguments_v70(argparse.ArgumentParser()).parse_args([])
    base.out, base.steps, base.batch, base.accum = str(out/"uninterrupted"), 3, 2, 2
    base.workers, base.swanlab_mode, base.trace = 0, "disabled", True
    base.checkpoint_every, base.snapshot_every, base.retain = 1, 0, 2
    first_events, resumed_events = [], []
    full_model = deepcopy(initial)
    frozen = deepcopy(full_model.conditioner.visual.state_dict())
    train_v70(base, model=full_model, dataset=FixtureDataset(), history_encoder=FixtureHistory(),
              device="cpu", fsdp=False, trace_callback=first_events.append)
    stopped = deepcopy(base)
    stopped.out, stopped.stop_after = str(out/"resumed"), 1
    train_v70(stopped, model=deepcopy(initial), dataset=FixtureDataset(), history_encoder=FixtureHistory(),
              device="cpu", fsdp=False, trace_callback=resumed_events.append)
    resumed = deepcopy(stopped)
    resumed.resume, resumed.stop_after = str(out/"resumed/latest.json"), 0
    final_model = deepcopy(initial)
    train_v70(resumed, model=final_model, dataset=FixtureDataset(), history_encoder=FixtureHistory(),
              device="cpu", fsdp=False, trace_callback=resumed_events.append)
    difference = max(float((a.detach()-b.detach()).abs().max()) for a, b in zip(full_model.parameters(), final_model.parameters()))
    assert difference == 0, difference
    assert first_events == resumed_events
    assert all(torch.equal(value, full_model.conditioner.visual.state_dict()[name]) for name, value in frozen.items())
    assert not torch.equal(initial.conditioner.text.weight, full_model.conditioner.text.weight)
    assert not torch.equal(initial.expert.effect_output.weight, full_model.expert.effect_output.weight)
    # A topology-change warm start loads only weights, not optimizer/cursor/RNG.
    warm = deepcopy(base)
    warm.out, warm.init_from = str(out/"warm_start"), str(out/"uninterrupted/latest.json")
    warm.steps, warm.batch, warm.accum = 1, 1, 1
    warm.lr_text, warm.lr_expert = 0., 0.
    warm_model, warm_events = deepcopy(initial), []
    train_v70(warm, model=warm_model, dataset=FixtureDataset(), history_encoder=FixtureHistory(),
              device="cpu", fsdp=False, trace_callback=warm_events.append)
    warm_difference = max(float((a.detach()-b.detach()).abs().max())
                          for a, b in zip(full_model.parameters(), warm_model.parameters()))
    assert warm_difference == 0, warm_difference
    assert warm_events[0]["step"] == 1 and warm_events[0]["cursor"] == 0
    warm_metadata = json.loads((out/"warm_start/step_0000001/metadata.json").read_text())
    assert warm_metadata["step"] == 1 and warm_metadata["args"]["accum"] == 1
    # Query permutations must permute effects, not change their meaning by slot index.
    batch = torch.utils.data.default_collate([FixtureDataset()[(0, 0, 0)]])
    history = FixtureHistory()(batch)
    full_model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        condition = full_model.encode_condition(full_model.prepare_inputs(batch), history)
        noise = torch.randn_like(batch["target_mean"])
        predicted = full_model.sample(condition, steps=2, noise=noise)
        permutation = torch.tensor([2, 0, 1])
        reordered = {**history, "tokens": history["tokens"][:, :, permutation],
                     "centers": history["centers"][:, :, permutation], "query_valid": history["query_valid"][:, permutation]}
        other = full_model.sample(full_model.encode_condition(full_model.prepare_inputs(batch), reordered),
                                  steps=2, noise=noise[:, permutation])
    permutation_difference = float((predicted[:, permutation]-other).abs().max())
    assert torch.allclose(predicted[:, permutation], other, atol=.03, rtol=.03), permutation_difference
    report = {"test": "cpu_tiny_wiring_not_full_model", "resume_parameter_max_difference": difference,
              "warm_start_parameter_max_difference": warm_difference,
              "warm_start_new_step_and_cursor": True,
              "resume_trace_equal": True, "frozen_vision_unchanged": True, "text_and_expert_updated": True,
              "bf16_query_permutation_max_difference": permutation_difference}
    out.mkdir(parents=True, exist_ok=True)
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
