"""Object-labeled common-location readouts and effect interventions on cached states."""

from itertools import combinations

import torch

from .object_sequence_evaluation_v69 import distribution_v69, known_entity_v69
from .object_sequence_objective_v69 import trajectory_error_v69


def measurement_locations_v69(batch, frames, mask, item=0):
    frame, point = mask.nonzero(as_tuple=True)
    return {"point_ids": batch["teacher"]["point_ids"][item, point].cpu().tolist(),
            "times_seconds": batch["times"][item, frames][frame].cpu().tolist(),
            "frame_indices": batch["frame_indices"][item, frames][frame].cpu().tolist()}


def query_component_positions_v69(model, output, batch, point_indices=None):
    """Standalone query fields at the same measured positions; no new encoding or posterior call."""
    teacher = batch["teacher"]
    ids = torch.arange(teacher["xy"].shape[2], device=teacher["xy"].device) if point_indices is None else point_indices
    reference = teacher["reference_xy"][:, ids]
    indices = teacher["reference_index"][:, ids]
    owners = output["reference_ownership"][:, ids]
    local = output["reference_local_xy"][:, ids]
    offset = reference.new_zeros((*reference.shape[:2], owners.shape[-1]-1, 2))
    for frame, state in enumerate(output["history_states"]):
        field = model.readout(state, reference, owners, return_components=True)["component_position"]
        offset = torch.where((indices == frame)[..., None, None], field, offset)
    states = output["observed_states"] if model.stage == "state" else output["rollout_states"]
    fields = [model.readout(state, reference, owners, local_coordinates=local, return_components=True)["component_position"] for state in states]
    return reference[:, None, :, None] + torch.stack(fields, 1) - offset[:, None]


@torch.no_grad()
def effect_rate_and_sampling_v69(model, output, batch, samples=4, labels=None):
    if model.stage != "dynamics":
        return {"status": "not_applicable_state_stage_posterior_frozen"}
    effect, valid = output["effect"], output["queries"].valid
    rate = effect["kl"].float().sum((-1, -2))
    draws, component_draws, per_sample = [], [], []
    th = model.config.history_frames
    target, measured = batch["teacher"]["xy"][:, th:], batch["teacher"]["valid"][:, th:]
    measured = measured & batch["teacher"]["point_present"][:, None]
    for sample in range(samples):
        value = (effect["mean"] + torch.randn_like(effect["mean"]) * (.5*effect["logvar"]).exp()).tanh()
        states = model.dynamics(output["source"], value, batch["times"][:, th:], rollout=True)
        result = model.render_sequence(states, batch["teacher"]["reference_xy"], output["reference_ownership"],
                                       output["reference_offset"], output["reference_local_xy"])
        draws.append(result["positions"])
        if labels is not None:
            component_draws.append(query_component_positions_v69(model, {**output, "rollout_states": states}, batch))
        _, error = trajectory_error_v69(result["positions"], target, batch["native_hw"])
        for item in range(len(valid)):
            values = error[item][measured[item]]
            per_sample.append({"case": batch["case_id"][item], "sample": sample,
                               "epe_px": values.cpu().tolist(), **measurement_locations_v69(batch, slice(th, None), measured[item], item),
                               **distribution_v69(values)})
    rates, spread_rows = [], []
    for item in range(len(valid)):
        ids = torch.where(valid[item])[0].tolist()
        rates.append({"case": batch["case_id"][item], "query_ids": ids,
                      "kl_nats_by_query": rate[item, ids].cpu().tolist(), "clip_total_kl_nats": float(rate[item, ids].sum()),
                      "valid_queries": len(ids), "dimensions_per_query": effect["kl"].shape[-1]*effect["kl"].shape[-2]})
        if draws:
            scale = (batch["native_hw"][item, [1, 0]].float()-1)*.5
            pixels = torch.stack([draw[item] for draw in draws])*scale
            variance = pixels.float().var(0, unbiased=False).sum(-1).sqrt()
            values = variance[measured[item]]
            spread_rows.append({"case": batch["case_id"][item], "point_rms_spread_px": values.cpu().tolist(),
                                **measurement_locations_v69(batch, slice(th, None), measured[item], item), **distribution_v69(values)})
    query_samples, entity_rates = [], []
    if labels is not None:
        for name in sorted({q for q in labels["query_objects"] if known_entity_v69(q)}):
            ids = [i for i, q in enumerate(labels["query_objects"]) if q == name and bool(valid[0, i])]
            entity_rates.append({"object": name, "query_ids": ids, "query_count": len(ids), "total_kl_nats": float(rate[0, ids].sum())})
    if component_draws:
        scale = (batch["native_hw"][0, [1, 0]].float()-1)*.5
        values = torch.stack(component_draws)[:, 0]
        deviation = ((values-target[0, :, :, None])*scale).norm(dim=-1)
        spread = (values*scale).float().var(0, unbiased=False).sum(-1).sqrt()
        for query, name in enumerate(labels["query_objects"]):
            if not known_entity_v69(name) or not bool(valid[0, query]):
                continue
            points = torch.tensor([p == name for p in labels["point_objects"]], device=target.device)
            mask = measured[0] & points[None]
            query_samples.append({"query": query, "object": name,
                                  "sample_epe_px": [distribution_v69(deviation[s, :, :, query][mask]) for s in range(samples)],
                                  "point_rms_spread_px": spread[:, :, query][mask].cpu().tolist(),
                                  **measurement_locations_v69(batch, slice(th, None), mask),
                                  "spread_distribution": distribution_v69(spread[:, :, query][mask])})
    return {"status": "measured", "samples": samples, "rates": rates, "sample_reconstruction": per_sample,
            "independent_query_sampling": query_samples,
            "independent_entity_rates": entity_rates,
            "sampling_spread": spread_rows, "scope": "posterior reconstruction and sample spread, not calibrated predictive uncertainty or object bitrate"}


@torch.no_grad()
def independent_association_v69(model, output, batch, labels, effect_samples=4):
    teacher = batch["teacher"]
    frames = slice(None) if model.stage == "state" else slice(model.config.history_frames, None)
    target = teacher["xy"][:, frames]
    valid = teacher["valid"][:, frames] & teacher["point_present"][:, None]
    frame_ids = torch.arange(teacher["xy"].shape[1], device=target.device)[frames]
    # The reference readout is calibrated to this supplied position, so it is not reconstruction evidence.
    valid &= frame_ids[None, :, None] != teacher["reference_index"][:, None]
    scale = (batch["native_hw"][0, [1, 0]].float()-1)*.5
    components = query_component_positions_v69(model, output, batch)[0]
    error = ((components-target[0, :, :, None])*scale).norm(dim=-1)
    q_objects, p_objects = labels["query_objects"], labels["point_objects"]
    query_rows, consistency, interventions = [], [], []
    for query, name in enumerate(q_objects):
        if not known_entity_v69(name) or not bool(output["queries"].valid[0, query]):
            continue
        regions = sorted({region for point, region in zip(p_objects, labels["point_regions"]) if point == name})
        for region in [None, *regions]:
            points = torch.tensor([point == name and (region is None or part == region)
                                   for point, part in zip(p_objects, labels["point_regions"])], device=error.device)
            mask = valid[0] & points[None]
            values = error[:, :, query][mask]
            displacement = ((target[0]-teacher["reference_xy"][0])*scale).norm(dim=-1)[mask]
            row = {"query": query, "object": name, "query_region": labels["query_regions"][query], "measured_region": region,
                   "epe_px": values.cpu().tolist(), "actual_displacement_px": displacement.cpu().tolist(),
                   "prediction_xy_px": ((components[:, :, query]+1)*scale)[mask].cpu().tolist(),
                   "target_xy_px": ((target[0]+1)*scale)[mask].cpu().tolist(),
                   **measurement_locations_v69(batch, frames, mask), **distribution_v69(values)}
            if model.stage == "dynamics":
                row["kl_nats"] = float(output["effect"]["kl"][0, query].float().sum())
            query_rows.append(row)
    for first, second in combinations(range(len(q_objects)), 2):
        name = q_objects[first]
        if name != q_objects[second] or not known_entity_v69(name) or not bool(output["queries"].valid[0, [first, second]].all()):
            continue
        points = torch.tensor([p == name for p in p_objects], device=error.device)
        mask = valid[0] & points[None]
        values = ((components[:, :, first]-components[:, :, second])*scale).norm(dim=-1)[mask]
        consistency.append({"query_a": first, "query_b": second, "object": name, "difference_px": values.cpu().tolist(),
                            **measurement_locations_v69(batch, frames, mask), **distribution_v69(values)})
    measurement_sets = []
    for name in sorted(set(labels["measurement_sets"])):
        ids = torch.tensor([i for i, group in enumerate(labels["measurement_sets"]) if group == name], device=error.device)
        subset = query_component_positions_v69(model, output, batch, ids)[0]
        delta = ((subset-components[:, ids])*scale).norm(dim=-1)
        selected = valid[0][:, ids][:, :, None] & output["queries"].valid[0, None, None]
        values = delta[selected.expand_as(delta)]
        measurement_sets.append({"measurement_set": name, "points": ids.cpu().tolist(), "same_cached_state_and_effect": True,
                                 "subset_prediction_difference_px": distribution_v69(values)})
    if model.stage == "dynamics":
        baseline = output["rollout_positions"]
        replacements = []
        representatives = {}
        for changed, name in enumerate(q_objects):
            if not known_entity_v69(name) or not bool(output["queries"].valid[0, changed]):
                continue
            representatives.setdefault(name, changed)
            donor = next((i for i, other in enumerate(q_objects) if known_entity_v69(other) and other != name
                          and bool(output["queries"].valid[0, i])), None)
            if donor is None:
                continue
            effect = output["effect"]["value"].clone()
            effect[:, changed] = output["effect"]["value"][:, donor]
            replacements.append(("replace_one_query", changed, donor, effect))
        for first, second in combinations(representatives.values(), 2):
            effect = output["effect"]["value"].clone()
            effect[:, first] = output["effect"]["value"][:, second]
            effect[:, second] = output["effect"]["value"][:, first]
            replacements.append(("swap_two_query_effects", first, second, effect))
        for intervention, changed, donor, effect in replacements:
            states = model.dynamics(output["source"], effect, batch["times"][:, model.config.history_frames:], rollout=True)
            result = model.render_sequence(states, teacher["reference_xy"], output["reference_ownership"],
                                           output["reference_offset"], output["reference_local_xy"])
            perturbation = ((result["positions"]-baseline)*scale).norm(dim=-1)[0]
            reconstruction = ((result["positions"]-target)*scale).norm(dim=-1)[0]
            for observed in sorted({p for p in p_objects if known_entity_v69(p)}):
                points = torch.tensor([p == observed for p in p_objects], device=error.device)
                mask = valid[0] & points[None]
                interventions.append({"intervention": intervention, "recipient_query": changed, "donor_query": donor,
                                      "recipient_object": q_objects[changed], "donor_object": q_objects[donor],
                                      "observed_object": observed, "response_px": distribution_v69(perturbation[mask]),
                                      "reconstruction_px": distribution_v69(reconstruction[mask]),
                                      **measurement_locations_v69(batch, frames, mask),
                                      "response_values_px": perturbation[mask].cpu().tolist()})
    sampling = effect_rate_and_sampling_v69(model, output, batch, effect_samples, labels)
    status = "measured_independent_annotations" if any(row["count"] for row in query_rows) else "not_measured_no_valid_entity_targets"
    return {"status": status, "case_tags": labels["case_tags"], "query_reconstruction": query_rows,
            "same_entity_common_location_consistency": consistency, "measurement_set_tests": measurement_sets,
            "query_effect_interventions": interventions, "sampling": sampling,
            "reference_calibration_frames_excluded": True,
            "scope": "standalone query fields at common externally labeled positions; no latent-equality, rigid-translation, zero-means-static or no-interaction requirement"}
