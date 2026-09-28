"""Absolute trajectory evidence and independent query/deletion measurements."""

import torch
from torch.nn import functional as F

from .object_sequence_readout_v69 import binding_probability_v69
from .object_sequence_objective_v69 import trajectory_error_v69


def distribution_v69(values):
    values = values.detach().float().cpu()
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "p50": float(values.quantile(.5)) if len(values) else None,
            "p90": float(values.quantile(.9)) if len(values) else None,
            "p95": float(values.quantile(.95)) if len(values) else None}


def known_entity_v69(name):
    return name is not None and name not in ("unknown", "unbound", "background", "scene")


def aggregate_query_objects_v69(ownership, query_objects):
    """Sum duplicate queries for each independently labeled entity; do not renormalize null mass."""
    names = sorted({name for name in query_objects if known_entity_v69(name)})
    membership = ownership.new_tensor([[float(label == name) for name in names] for label in query_objects])
    membership = membership.reshape(len(query_objects), len(names))
    with torch.autocast(ownership.device.type, enabled=False):
        grouped = ownership[..., :len(query_objects)].float() @ membership.float()
    return grouped, names


def sequence_metrics_v69(output, batch, config, stage):
    th = config.history_frames
    teacher = batch["teacher"]
    frames = range(len(batch["times"][0])) if stage == "state" else range(th, len(batch["times"][0]))
    predictions = {"observation_reconstruction": output["observed_positions"]} if stage == "state" else {
        "direct": output["direct_positions"], "rollout": output["rollout_positions"],
        "shuffled_effect": output["shuffled_positions"], "zero_effect": output["zero_positions"]}
    frames = list(frames)
    target = teacher["xy"][:, frames]
    predictions["last_observed_copy"] = teacher["reference_xy"][:, None].expand_as(target)
    rows = []
    for condition, prediction in predictions.items():
        _, epe = trajectory_error_v69(prediction, target, batch["native_hw"])
        for item in range(len(epe)):
            for local, frame in enumerate(frames):
                valid = teacher["valid"][item, frame] & teacher["point_present"][item]
                for subset, mask in (("all_measured", valid), ("primary_transport_pool", valid & teacher["transport_weight"][item].bool())):
                    points = teacher["point_ids"][item, mask].detach().cpu().tolist()
                    values = epe[item, local, mask]
                    rows.append({"case": batch["case_id"][item], "source": batch["source"][item], "condition": condition,
                                 "subset": subset, "seconds": float(batch["times"][item, frame]),
                                 "selection_status": "independent_truth" if batch.get("independent_truth") else batch["transport_selection_status"][item],
                                 "point_ids": points, "epe_px": values.detach().cpu().tolist(), **distribution_v69(values)})
    return rows


@torch.no_grad()
def independent_binding_v69(model, output, batch, labels):
    q_objects, p_objects = labels["query_objects"], labels["point_objects"]
    same = torch.tensor([[q == p and known_entity_v69(p) for q in q_objects] for p in p_objects], device=batch["rgb"].device, dtype=torch.bool)
    same &= output["queries"].valid[0, None]
    ownership = output["reference_ownership"][0, :, :len(q_objects)]
    mass = (ownership*same).sum(-1)
    eligible = same.any(-1) & batch["teacher"]["point_present"][0]
    metrics = {"independent_query_mass": distribution_v69(mass[eligible]),
               "independent_unrelated_mass": distribution_v69((ownership*(~same)).sum(-1)[eligible])}
    present = batch["teacher"]["point_present"][0]
    object_probability, object_names = aggregate_query_objects_v69(output["reference_ownership"][0], q_objects)
    truth = torch.tensor([[p == q for q in object_names] for p in p_objects], device=ownership.device).float()
    truth = truth.reshape(len(p_objects), len(object_names))
    full_ownership = output["reference_ownership"][0]
    allowed = torch.cat((output["queries"].valid[0], output["queries"].valid.new_ones(1))).float()
    uniform = allowed[None].expand_as(full_ownership)/allowed.sum().clamp_min(1)
    split_queries = torch.zeros_like(full_ownership)
    for point, label in enumerate(p_objects):
        matching = [i for i, q in enumerate(q_objects) if q == label and bool(output["queries"].valid[0, i]) and known_entity_v69(label)]
        split_queries[point, matching[point % len(matching)] if matching else -1] = 1
    controls = {"student_entity_mass": object_probability,
                "human_entity_reference": truth,
                "same_entity_split_queries": aggregate_query_objects_v69(split_queries, q_objects)[0],
                "uniform_queries": aggregate_query_objects_v69(uniform, q_objects)[0],
                "merge_all_entities": ownership.new_ones((len(p_objects), 1)),
                "all_unbound": torch.zeros_like(object_probability)}
    pair_same = torch.tensor([[a == b for b in p_objects] for a in p_objects], device=ownership.device)
    pair_valid = eligible[:, None] & eligible[None] & torch.triu(torch.ones_like(pair_same), diagonal=1)
    pair_rows = []
    for condition, probability in controls.items():
        with torch.autocast(probability.device.type, enabled=False):
            agreement = probability.float() @ probability.float().T
        for relation, mask in (("same_entity", pair_valid & pair_same), ("different_entity", pair_valid & ~pair_same)):
            target = 1.0 if relation == "same_entity" else 0.0
            pair_rows.append({"condition": condition, "relation": relation,
                              "brier_error": distribution_v69((agreement[mask]-target).square())})
    metrics["independent_grouping_controls"] = pair_rows
    metrics["independent_unbound_mass"] = distribution_v69(full_ownership[present, -1])
    metrics["grouping_eligible_points"] = int(eligible.sum())
    metrics["grouping_labeled_objects"] = object_names
    metrics["grouping_scope"] = "sum query probabilities by independent entity labels; null/unknown/scene excluded, no renormalization; identical query index is not entity identity"
    source = output["source"]
    frame = model.config.history_frames-1
    reference = batch["teacher"]["xy"][:, frame]
    target = output["measurement_features"][:, frame]
    current_valid = batch["teacher"]["valid"][0, frame] & output["measurement_feature_valid"][0, frame]
    current_owner = binding_probability_v69(model.readout.binding(source, reference, target), source.query_valid)
    base = model.readout(source, reference, current_owner)
    full_error = 1-F.cosine_similarity(base["appearance"].float(), target.float(), dim=-1)
    deletions = []
    classified = torch.tensor([p is not None and p not in ("unknown", "unbound") for p in p_objects], device=reference.device)
    for object_id in object_names:
        keep = torch.tensor([[q != object_id for q in q_objects]], device=reference.device)
        changed = model.readout(source, reference, current_owner, object_keep=keep)
        deleted_error = 1-F.cosine_similarity(changed["appearance"].float(), target.float(), dim=-1)
        difference = (deleted_error-full_error)[0]
        inside = torch.tensor([p == object_id for p in p_objects], device=reference.device) & current_valid
        outside = ~inside & current_valid & classified
        deletions.append({"object": object_id, "inside": distribution_v69(difference[inside]), "outside": distribution_v69(difference[outside])})
    metrics["independent_deletion"] = deletions
    metrics["deletion_measurement"] = "increase in frozen-feature reconstruction error at human-labeled current points; not a physical counterfactual"
    return metrics


@torch.no_grad()
def history_forecast_ablation_v69(model, perception, output, batch, perception_runtime):
    from .pretrained_visual_encoder_v69 import PerceptionSequenceV69
    th = model.config.history_frames
    queries = output["queries"]
    current = (queries.frame_index == th-1).all(0) & queries.valid.all(0)
    if not bool(current.any()):
        return {"status": "no_shared_current_queries", "rows": []}
    selected = {"xy": queries.xy[:, current], "frame_index": queries.frame_index[:, current], "valid": queries.valid[:, current]}
    effect = output["effect"]["value"][:, current]
    past = perception.prefix(th)
    reverse = torch.cat((torch.arange(th-2, -1, -1, device=past.features.device), torch.tensor([th-1], device=past.features.device)))
    last_valid = past.valid.clone()
    last_valid[:, :-1] = False
    variants = {"ordered": past,
                "reversed": PerceptionSequenceV69(past.features[:, reverse], past.coordinates, past.valid[:, reverse], past.times, past.native_hw, past.grid_hw),
                "last_frame": PerceptionSequenceV69(past.features, past.coordinates, last_valid, past.times, past.native_hw, past.grid_hw)}
    if perception_runtime.kind != "dinov3_vitl16":
        # A cached last video token already contains history; regenerate ablations from their actual RGB inputs.
        pixels = batch["pixel_valid"][:, :th].clone()
        pixels[:, :-1] = False
        variants["last_frame"] = perception_runtime(batch["rgb"][:, :th], pixels, past.times, past.native_hw)
        variants["reversed"] = perception_runtime(batch["rgb"][:, :th][:, reverse], batch["pixel_valid"][:, :th][:, reverse], past.times, past.native_hw)
    teacher = batch["teacher"]
    coordinates = teacher["xy"][:, th-1]
    valid = teacher["valid"][:, th:] & teacher["valid"][:, th-1:th]
    rows = []
    for condition, fields in variants.items():
        from .pretrained_visual_encoder_v69 import sample_perception_v69
        frame_index = torch.full((len(coordinates), 1), th-1, device=coordinates.device, dtype=torch.long)
        features, _ = sample_perception_v69(fields, coordinates[:, None], frame_index)
        features = features[:, 0]
        forecast, _ = model.forecast(fields, selected, effect, perception.times[:, th:], coordinates, features)
        _, epe = trajectory_error_v69(forecast["positions"], teacher["xy"][:, th:], batch["native_hw"])
        for item in range(len(epe)):
            for frame in range(epe.shape[1]):
                values = epe[item, frame, valid[item, frame]]
                rows.append({"case": batch["case_id"][item], "source": batch["source"][item], "condition": condition,
                             "seconds": float(perception.times[item, th+frame]), "epe_px": values.cpu().tolist(), **distribution_v69(values)})
    return {"status": "measured", "scope": "same current query coordinates and fixed effect; each variant re-encodes query/point features from its own history", "rows": rows}
