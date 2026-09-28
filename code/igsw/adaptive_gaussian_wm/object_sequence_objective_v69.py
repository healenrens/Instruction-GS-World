"""Observable sequence objectives with teacher validity distinct from model visibility."""

import torch
from torch.nn import functional as F
from .object_sequence_readout_v69 import binding_probability_v69


def masked_mean_v69(value, weight):
    weight = weight.float()
    numerator = (value.float()*weight).flatten(1).sum(1)
    denominator = weight.expand_as(value).flatten(1).sum(1).clamp_min(1)
    return (numerator/denominator).mean()


def trajectory_error_v69(prediction, target, native_hw):
    scale = (native_hw[:, [1, 0]].float()-1)[:, None, None] * .5
    pixel_difference = (prediction.float() - target.float()) * scale
    epe = pixel_difference.square().sum(-1).sqrt()
    # Optimize native-pixel error, not a resolution-shrunk surrogate. Epsilon only smooths the origin.
    distance = (pixel_difference.square().sum(-1)+1e-6).sqrt()-1e-3
    return distance, epe


def object_sequence_loss_v69(output, batch, config, stage):
    teacher = batch["teacher"]
    th = config.history_frames
    all_valid = teacher["valid"] & output["measurement_feature_valid"] & teacher["point_present"][:, None]
    parts = {}
    if stage == "state":
        binding_target = output["binding_target"].detach()
        owners = binding_probability_v69(output["binding_logits"], output["queries"].valid)
        binding_distance = (binding_target * (binding_target.clamp_min(1e-7).log()-owners.clamp_min(1e-7).log())).sum(-1)
        feature = 1-F.cosine_similarity(output["observed_appearance"].float(), output["measurement_features"].float(), dim=-1)
        parts["appearance_aux"] = masked_mean_v69(feature, all_valid)
        parts["appearance_binding_weak"] = masked_mean_v69(binding_distance, all_valid * output["binding_evidence"])
        first, second = owners[:, :-1], owners[:, 1:]
        middle = .5 * (first + second)
        js = .5 * ((first * (first.clamp_min(1e-7).log()-middle.clamp_min(1e-7).log())).sum(-1)
                   + (second * (second.clamp_min(1e-7).log()-middle.clamp_min(1e-7).log())).sum(-1))
        parts["track_correspondence"] = masked_mean_v69(js, all_valid[:, :-1] & all_valid[:, 1:])
        reconstruction, epe = trajectory_error_v69(output["observed_positions"], teacher["xy"], batch["native_hw"])
        trajectory_valid = teacher["valid"] & teacher["point_present"][:, None]
        parts["observed_transport_reconstruction"] = masked_mean_v69(reconstruction, trajectory_valid * teacher["transport_weight"][:, None])
        observation = output["observed_visibility"]
        observation_target = teacher["observation"]
        known = (observation_target >= 0) & teacher["point_present"][:, None]
        observation_error = F.binary_cross_entropy_with_logits(observation.float(), observation_target.clamp_min(0).float(), reduction="none")
        parts["tracker_observation_aux"] = masked_mean_v69(observation_error, known)
        loss = (config.feature_weight * parts["appearance_aux"] + config.binding_weight * parts["appearance_binding_weak"]
                + config.correspondence_weight * parts["track_correspondence"]
                + config.trajectory_weight * parts["observed_transport_reconstruction"]
                + config.observation_weight * parts["tracker_observation_aux"])
    else:
        target = teacher["xy"][:, th:]
        valid = teacher["valid"][:, th:] & teacher["point_present"][:, None]
        weight = valid * teacher["transport_weight"][:, None]
        direct_error, _ = trajectory_error_v69(output["direct_positions"], target, batch["native_hw"])
        roll_error, epe = trajectory_error_v69(output["rollout_positions"], target, batch["native_hw"])
        shuffled_error, _ = trajectory_error_v69(output["shuffled_positions"], target, batch["native_hw"])
        zero_error, _ = trajectory_error_v69(output["zero_positions"], target, batch["native_hw"])
        parts["direct_transport"] = masked_mean_v69(direct_error, weight)
        parts["rollout_transport"] = masked_mean_v69(roll_error, weight)
        parts["shuffled_transport"] = masked_mean_v69(shuffled_error.detach(), weight)
        parts["zero_effect_transport"] = masked_mean_v69(zero_error.detach(), weight)
        prediction = output["rollout_positions"]
        # Neighbor-relative displacement cancels shared translation; not a claim of 3D camera compensation.
        reference = teacher["reference_xy"]
        neighbors = torch.cdist(reference.float(), reference.float()).masked_fill(
            torch.eye(reference.shape[1], device=reference.device, dtype=torch.bool)[None] | ~teacher["point_present"][:, None], float("inf"))
        nearest = neighbors.argmin(-1)
        gather = nearest[:, None, :, None].expand(-1, prediction.shape[1], -1, 2)
        pair_pred = prediction - prediction.gather(2, gather)
        pair_target = target - target.gather(2, gather)
        pair_valid = valid & valid.gather(2, nearest[:, None].expand(-1, valid.shape[1], -1)) & (neighbors.min(-1).values.isfinite()[:, None])
        relative_error, _ = trajectory_error_v69(pair_pred, pair_target, batch["native_hw"])
        parts["relative_motion"] = masked_mean_v69(relative_error, pair_valid * teacher["transport_weight"][:, None])
        path, _ = trajectory_error_v69(prediction, output["direct_positions"].detach(), batch["native_hw"])
        parts["path_consistency"] = masked_mean_v69(path, weight)
        predicted = torch.stack([s.tokens for s in output["rollout_states"]], 1)
        latent_target = torch.stack([s.tokens for s in output["target_states"]], 1).detach()
        latent_error = (F.layer_norm(predicted.float(), (predicted.shape[-1],))-F.layer_norm(latent_target.float(), (latent_target.shape[-1],))).square().mean((-1, -2))
        latent_valid = output["queries"].valid[:, None].expand_as(latent_error) & batch["frame_valid"][:, th:, None]
        parts["future_latent_aux"] = masked_mean_v69(latent_error, latent_valid)
        parts["effect_rate"] = masked_mean_v69(output["effect"]["kl"].mean((-1, -2)), output["queries"].valid)
        query_kl = output["effect"]["kl"].detach().float().sum((-1, -2))
        query_valid = output["queries"].valid.float()
        parts["effect_clip_kl_nats"] = (query_kl*query_valid).sum(-1).mean()
        parts["effect_query_kl_nats"] = ((query_kl*query_valid).sum(-1)/query_valid.sum(-1).clamp_min(1)).mean()
        parts["effect_valid_query_count"] = query_valid.sum(-1).mean()
        # Shuffling can exchange equivalent motions; it is an intervention, not a negative GT label.
        parts["shuffled_minus_direct_px"] = parts["shuffled_transport"]-parts["direct_transport"].detach()
        observation_target = teacher["observation"][:, th:]
        known = (observation_target >= 0) & teacher["point_present"][:, None]
        observation_error = F.binary_cross_entropy_with_logits(output["rollout_visibility"].float(), observation_target.clamp_min(0).float(), reduction="none")
        parts["tracker_observation_aux"] = masked_mean_v69(observation_error, known)
        loss = (parts["direct_transport"] + parts["rollout_transport"] + config.relative_motion_weight*parts["relative_motion"]
                + config.latent_weight*parts["future_latent_aux"] + config.path_weight*parts["path_consistency"]
                + config.effect_rate_weight*parts["effect_rate"]
                + config.observation_weight*parts["tracker_observation_aux"])
    return loss, parts, epe
