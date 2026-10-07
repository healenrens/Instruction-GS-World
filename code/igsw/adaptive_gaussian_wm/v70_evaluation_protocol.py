"""Fixed cases, paired randomness and decoded trajectory reporting for V70."""

from collections import defaultdict
import random

import torch

from .language_effect_evaluation_v70 import summarize_records


def evaluation_plan(entries, indices, seed):
    """Language donors are distinct episodes, preferably in the same source."""
    rng = random.Random(seed)
    ordered = sorted(entries, key=lambda entry: entry["window_id"])
    plan = []
    for index in indices:
        entry = entries[index]
        episode = (entry["source"], entry["group"], entry["episode_index"])
        candidates = [other for other in ordered
                      if (other["source"], other["group"], other["episode_index"]) != episode
                      and other["instruction"].strip() != entry["instruction"].strip()]
        same_source = [other for other in candidates if other["source"] == entry["source"]]
        donor = rng.choice(same_source or candidates) if candidates else None
        plan.append({"window_id": entry["window_id"], "source": entry["source"],
                     "group": entry["group"], "episode_index": entry["episode_index"],
                     "frame_indices": entry["frame_indices"], "instruction": entry["instruction"],
                     "shuffled_instruction": donor["instruction"] if donor else None,
                     "shuffled_donor_window": donor["window_id"] if donor else None})
    return plan


def paired_noise(target, seed, case_index, sample):
    generator = torch.Generator(device=target.device).manual_seed(seed + case_index * 1000003 + sample)
    return torch.randn(target.shape, dtype=torch.float32, device=target.device, generator=generator)


def effect_diagnostics(model, condition, target, noise, prediction):
    """Auxiliary teacher-forced flow loss, separate from pure-noise generation."""
    valid = condition["query_valid"][:, :, None, None]
    target = target.float().masked_fill(~valid, 0)
    noise = noise.masked_fill(~valid, 0)
    count = valid.sum().clamp_min(1) * target.shape[-2] * target.shape[-1]

    def mse(first, second):
        return float((first.float() - second.float()).masked_fill(~valid, 0).square().sum() / count)

    result = {"generated_mean_mse": mse(prediction, target),
              "generated_tanh_mse": mse(prediction.tanh(), target.tanh())}
    if model.mode == "flow":
        for value in (.1, .5, .9):
            tau = target.new_full((len(target),), value)
            u = (1 - value) * noise + value * target
            result[f"teacher_forced_flow_mse_tau_{value}"] = mse(model.expert(u, tau, condition), target - noise)
    return result


def evaluation_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
        grouped[f'{row["condition"]}/source/{row["source"]}'].append(row)
        if row["motion_selected"]:
            grouped[f'{row["condition"]}/motion_selected'].append(row)
        if row["reference_at_current"]:
            grouped[f'{row["condition"]}/reference_at_current'].append(row)
    summaries = {}
    for name, subset in grouped.items():
        summary = summarize_records(subset)
        clips = defaultdict(list)
        for row in subset:
            clips[row["window_id"]].append(row["ade_px"])
        summary["episode_balanced_ade_px"] = sum(sum(v)/len(v) for v in clips.values())/len(clips)
        summaries[name] = summary
    for condition in ("correct_language", "no_language", "shuffled_language"):
        samples = [row for row in rows if row["condition"].startswith(condition + "/sample_")]
        if not samples:
            continue
        summaries[condition + "/expected_over_samples"] = summarize_records(samples)
        by_case = defaultdict(lambda: defaultdict(list))
        for row in samples:
            by_case[row["window_id"]][row["condition"]].append(row)
        oracle = []
        for choices in by_case.values():
            oracle.extend(min(choices.values(), key=lambda group: sum(r["ade_px"] for r in group)/len(group)))
        summaries[condition + "/oracle_best_sample_per_clip"] = summarize_records(oracle)
    return summaries
