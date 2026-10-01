"""Observed-video State reconstruction and paired changes of the compact state."""

from collections import defaultdict
from dataclasses import replace
import random

import torch

from .episode_uniform_sampler_v69 import episode_key
from .object_sequence_evaluation_v69 import distribution_v69
from .pretrained_visual_encoder_v69 import sample_perception_v69


CONDITIONS = ("observed_state", "frozen_state", "shuffled_state", "frozen_tokens", "frozen_centers", "reference_copy")
METRICS = ("position_error_px", "change_error_px", "target_displacement_px", "predicted_displacement_px",
           "normalized_change_error", "direction_cosine_error")


def select_held_episodes_v69(dataset, per_source, seed):
    """One deterministically chosen window per episode; sources have equal caps."""
    sources = defaultdict(lambda: defaultdict(list))
    for index, entry in enumerate(dataset.entries):
        sources[entry["source"]][episode_key(entry)].append(index)
    rng = random.Random(seed)
    selected, coverage = [], []
    for source, episodes in sorted(sources.items()):
        keys = sorted(episodes)
        rng.shuffle(keys)
        chosen = keys[:per_source] if per_source else keys
        selected.extend(rng.choice(episodes[key]) for key in chosen)
        coverage.append({"source": source, "available_held_episodes": len(keys), "selected_episodes": len(chosen)})
    rng.shuffle(selected)
    return selected, coverage


def load_state_modules_v69(model, weights):
    """State-only evaluation strictly loads every used module, including V69 v1."""
    loaded = []
    for name in ("encoder", "target_encoder", "readout"):
        prefix = name + "."
        module_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        getattr(model, name).load_state_dict(module_weights, strict=True)
        loaded.append(name)
    return loaded


@torch.no_grad()
def observe_state_v69(model, perception, batch, supplied_queries=None):
    """Reuse training's encoding and history reference; future measurements only score it."""
    queries, history = model.encode_history(perception, supplied_queries)
    state, continuation = history[-1], []
    for frame in range(model.config.history_frames, perception.features.shape[1]):
        state = model.encoder.observe(state, perception.features[:, frame], perception.coordinates,
                                      perception.valid[:, frame], perception.times[:, frame])
        continuation.append(state)
    teacher = batch["teacher"]
    frames = torch.arange(teacher["xy"].shape[1], device=teacher["xy"].device)[None].expand(len(teacher["xy"]), -1)
    measured, measured_valid = sample_perception_v69(perception, teacher["xy"], frames)
    rows = torch.arange(len(measured), device=measured.device)[:, None]
    points = torch.arange(measured.shape[2], device=measured.device)[None]
    reference_feature = measured[rows, teacher["reference_index"], points]
    ownership, offset, local = model.reference_readout(history, teacher["reference_xy"], teacher["reference_index"], reference_feature)
    states = history + continuation
    decoded = model.render_sequence(states, teacher["reference_xy"], ownership, offset, local)
    return {"queries": queries, "history_states": history, "source": history[-1], "observed_states": states,
            "observed_positions": decoded["positions"], "reference_ownership": ownership, "reference_offset": offset,
            "reference_local_xy": local, "measurement_features": measured, "measurement_feature_valid": measured_valid}


def intervene_states_v69(states, history_frames, condition, seed):
    """Only replace continuation states. The calibration history stays identical."""
    source = states[history_frames - 1]
    continuation = states[history_frames:]
    if condition == "frozen_state":
        changed = [replace(state, tokens=source.tokens, centers=source.centers) for state in continuation]
    elif condition == "frozen_tokens":
        changed = [replace(state, tokens=source.tokens) for state in continuation]
    elif condition == "frozen_centers":
        changed = [replace(state, centers=source.centers) for state in continuation]
    elif condition == "shuffled_state":
        # A random cycle has no fixed frames when the continuation has more than one frame.
        order = list(range(len(continuation)))
        random.Random(seed).shuffle(order)
        permutation = list(range(len(order)))
        for first, second in zip(order, order[1:] + order[:1]):
            permutation[first] = second
        changed = [replace(continuation[permutation[index]], time=state.time) for index, state in enumerate(continuation)]
    else:
        changed = continuation
    return list(states[:history_frames]) + changed


@torch.no_grad()
def reconstruct_interventions_v69(model, output, batch, seed):
    teacher = batch["teacher"]
    predictions = {"observed_state": output["observed_positions"]}
    for condition in CONDITIONS[1:-1]:
        states = intervene_states_v69(output["observed_states"], model.config.history_frames, condition, seed)
        predictions[condition] = model.render_sequence(states, teacher["reference_xy"], output["reference_ownership"],
                                                       output["reference_offset"], output["reference_local_xy"])["positions"]
    predictions["reference_copy"] = teacher["reference_xy"][:, None].expand_as(predictions["observed_state"])
    return predictions


def state_change_values_v69(predictions, batch, history_frames, motion_threshold):
    teacher = batch["teacher"]
    scale = (batch["native_hw"][:, [1, 0]].float() - 1)[:, None, None] * .5
    target = teacher["xy"][:, history_frames:].float()
    target_change = (target - teacher["xy"][:, history_frames-1:history_frames].float()) * scale
    displacement = target_change.norm(dim=-1)
    position_valid = teacher["valid"][:, history_frames:] & teacher["point_present"][:, None]
    change_valid = position_valid & teacher["valid"][:, history_frames-1:history_frames]
    values = {}
    for condition, prediction in predictions.items():
        prediction = prediction.float()
        change = (prediction[:, history_frames:] - prediction[:, history_frames-1:history_frames]) * scale
        error = (change - target_change).norm(dim=-1)
        cosine = (change * target_change).sum(-1) / (change.norm(dim=-1) * displacement).clamp_min(1e-8)
        values[condition] = {"position_error_px": ((prediction[:, history_frames:] - target) * scale).norm(dim=-1),
                             "change_error_px": error, "target_displacement_px": displacement,
                             "predicted_displacement_px": change.norm(dim=-1),
                             "normalized_change_error": error / displacement.clamp_min(1e-8),
                             "direction_cosine_error": 1 - cosine.clamp(-1, 1)}
    return values, position_valid, change_valid, change_valid & (displacement >= motion_threshold)


def summarize_state_case_v69(predictions, batch, config, motion_threshold=5.0):
    """Point/frame slices use teacher evidence, identical for every intervention."""
    values, valid, change_valid, moving = state_change_values_v69(predictions, batch, config.history_frames, motion_threshold)
    displacement = values["observed_state"]["target_displacement_px"]
    motion_slices = {"all": torch.ones_like(valid), "under_1px": change_valid & (displacement < 1),
                     "1_to_5px": change_valid & (displacement >= 1) & (displacement < 5),
                     "5_to_20px": change_valid & (displacement >= 5) & (displacement < 20),
                     "20_to_50px": change_valid & (displacement >= 20) & (displacement < 50),
                     "at_least_50px": change_valid & (displacement >= 50), "motion_active": moving}
    seconds = batch["times"][:, config.history_frames:, None].expand_as(valid)
    horizon_slices = {f"{second}_to_{second+1}s": (seconds > second) & (seconds <= second+1)
                      for second in range(int(config.future_seconds + .999999))}
    pools = {"all_points": torch.ones_like(valid), "transport_pool": batch["teacher"]["transport_weight"][:, None].bool().expand_as(valid)}
    rows, paired = [], []
    for item, case in enumerate(batch["case_id"]):
        for pool, pool_mask in pools.items():
            specifications = [("motion", name, mask) for name, mask in motion_slices.items()]
            specifications += [("horizon", name, mask) for name, mask in horizon_slices.items()]
            for scope, name, selection in specifications:
                base_mask = selection[item] & pool_mask[item]
                for condition, metrics in values.items():
                    row = {"case": case, "source": batch["source"][item], "condition": condition,
                           "pool": pool, "scope": scope, "slice": name}
                    for metric, metric_values in metrics.items():
                        mask = valid[item] if metric == "position_error_px" else change_valid[item]
                        if metric in ("normalized_change_error", "direction_cosine_error"):
                            mask = moving[item]
                        stats = distribution_v69(metric_values[item][base_mask & mask])
                        row.update({metric + "_" + key: value for key, value in stats.items()})
                    rows.append(row)
            for condition in CONDITIONS[1:]:
                row = {"case": case, "source": batch["source"][item], "condition": condition, "pool": pool}
                for metric in ("position_error_px", "change_error_px"):
                    penalty = values[condition][metric][item] - values["observed_state"][metric][item]
                    stats = distribution_v69(penalty[moving[item] & pool_mask[item]])
                    row.update({metric + "_penalty_" + key: value for key, value in stats.items()})
                paired.append(row)
    return rows, paired


def aggregate_case_rows_v69(rows):
    """Report distributions of per-clip errors. Long clips do not get extra votes."""
    groups = defaultdict(list)
    for row in rows:
        for source in (row["source"], "__all_sources__"):
            groups[(source, row["condition"], row["pool"], row["scope"], row["slice"])].append(row)
    summaries = []
    for (source, condition, pool, scope, name), members in sorted(groups.items()):
        summary = {"source": source, "condition": condition, "pool": pool, "scope": scope, "slice": name,
                   "aggregation": "distribution_of_clip_statistics", "clips": len(members)}
        for metric in METRICS:
            for statistic in ("mean", "p50", "p90"):
                key = metric + "_" + statistic
                values = torch.tensor([row[key] for row in members if row[key] is not None], dtype=torch.float32)
                summary[key] = distribution_v69(values)
        summaries.append(summary)
    return summaries


def aggregate_paired_rows_v69(rows):
    groups = defaultdict(list)
    for row in rows:
        for source in (row["source"], "__all_sources__"):
            groups[(source, row["condition"], row["pool"])].append(row)
    result = []
    for (source, condition, pool), members in sorted(groups.items()):
        row = {"source": source, "condition": condition, "pool": pool}
        for metric in ("position_error_px", "change_error_px"):
            key = metric + "_penalty_mean"
            values = torch.tensor([member[key] for member in members if member[key] is not None])
            row[key] = distribution_v69(values)
            row[metric + "_observed_state_better_fraction"] = float((values > 0).float().mean()) if len(values) else None
        result.append(row)
    return result
