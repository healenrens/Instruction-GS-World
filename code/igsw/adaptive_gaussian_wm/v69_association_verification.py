"""Object-association interface checks run inside the full single-GPU integration test."""

from dataclasses import replace

import torch

from .query_object_video_encoder_v69 import ObjectVideoStateV69
from .object_sequence_evaluation_v69 import aggregate_query_objects_v69


def reordered_state_v69(state, order):
    return ObjectVideoStateV69(state.tokens[:, order], state.centers[:, order], state.time, state.query_valid[:, order])


@torch.no_grad()
def verify_association_interfaces_v69(model, output, batch):
    device = output["source"].tokens.device
    with torch.autocast(device.type, enabled=False):
        observed = output["source"]
        source = ObjectVideoStateV69(observed.tokens[:1, :3].float().clone(), observed.centers[:1, :3].float().clone(),
                                     observed.time[:1].float(), torch.ones((1, 3), device=device, dtype=torch.bool))
        # Equal anchors must not turn query-bound effects back into an unordered set.
        source.tokens[:, :, 0] = source.tokens[:, :1, 0].clone()
        effect = torch.linspace(-.6, .6, 3*model.config.effect_tokens*model.config.effect_dim, device=device)
        effect = effect.reshape(1, 3, model.config.effect_tokens, model.config.effect_dim)
        exchanged = effect[:, [1, 0, 2]]
        local = model.dynamics.condition_local(source.tokens, effect, 0)
        local_swapped = model.dynamics.condition_local(source.tokens, exchanged, 0)
        local_changed = float((local[:, :2]-local_swapped[:, :2]).abs().max())
        local_unchanged = float((local[:, 2:]-local_swapped[:, 2:]).abs().max())
        assert local_changed > 1e-7 and local_unchanged == 0
        time = source.time + .4
        prediction = model.dynamics.advance(source, effect, time)
        swapped = model.dynamics.advance(source, exchanged, time)
        swap_difference = float((prediction.tokens-swapped.tokens).abs().max())
        assert swap_difference > 1e-7
        order = torch.tensor([2, 0, 1], device=device)
        permuted = model.dynamics.advance(reordered_state_v69(source, order), effect[:, order], time)
        assert torch.allclose(prediction.tokens[:, order], permuted.tokens, atol=2e-5, rtol=2e-5)
        assert torch.allclose(prediction.centers[:, order], permuted.centers, atol=2e-5, rtol=2e-5)
        permutation_difference = float((prediction.tokens[:, order]-permuted.tokens).abs().max())

        target = ObjectVideoStateV69(source.tokens.clone(), source.centers.clone(), time, source.query_valid)
        moved_centers = source.centers.clone()
        moved_centers[:, 0] += torch.tensor([.12, -.08], device=device)
        moved = ObjectVideoStateV69(source.tokens.clone(), moved_centers, time, source.query_valid)
        same = model.posterior(source, [target], deterministic=True)["mean"]
        changed = model.posterior(source, [moved], deterministic=True)["mean"]
        geometry_difference = float((same-changed).abs().max())
        if model.config.posterior_geometry:
            assert geometry_difference > 1e-7
        offset = torch.tensor([.17, -.13], device=device)
        shifted_source = ObjectVideoStateV69(source.tokens, source.centers+offset, source.time, source.query_valid)
        shifted_target = ObjectVideoStateV69(moved.tokens, moved.centers+offset, moved.time, moved.query_valid)
        shifted = model.posterior(shifted_source, [shifted_target], deterministic=True)["mean"]
        assert torch.allclose(changed, shifted, atol=2e-5, rtol=2e-5)
        original = model.posterior.config
        model.posterior.config = replace(original, posterior_geometry=False)
        feature_only_same = model.posterior(source, [target], deterministic=True)["mean"]
        feature_only_moved = model.posterior(source, [moved], deterministic=True)["mean"]
        model.posterior.config = original
        assert torch.equal(feature_only_same, feature_only_moved)

        owners = torch.tensor([[.5, .5, 0.], [.25, .75, 0.]], device=device)
        grouped, names = aggregate_query_objects_v69(owners, ["A", "A"])
        null = torch.tensor([[0., 0., 1.], [0., 0., 1.]], device=device)
        null_grouped, _ = aggregate_query_objects_v69(null, ["A", "A"])
        assert names == ["A"] and torch.equal(grouped, torch.ones_like(grouped))
        assert float((grouped @ grouped.T)[0, 1]) == 1.0
        assert float((null_grouped @ null_grouped.T)[0, 1]) == 0.0

        reference = batch["teacher"]["reference_xy"].float()
        ownership = output["reference_ownership"].float()
        coordinates = output["reference_local_xy"].float()
        current = ObjectVideoStateV69(observed.tokens.float(), observed.centers.float(), observed.time.float(), observed.query_valid)
        full = model.readout(current, reference, ownership, local_coordinates=coordinates)["position"]
        ids = torch.arange(0, reference.shape[1], 2, device=device)
        subset = model.readout(current, reference[:, ids], ownership[:, ids], local_coordinates=coordinates[:, ids])["position"]
        assert torch.allclose(full[:, ids], subset, atol=2e-5, rtol=2e-5)
    return {"status": "passed_interface_checks_not_object_semantics",
            "local_effect_swap_changed_max": local_changed, "local_other_query_change_max": local_unchanged,
            "identical_anchor_effect_swap_max": swap_difference, "joint_tuple_permutation_max": permutation_difference,
            "posterior_geometry_only_mean_difference": geometry_difference,
            "posterior_common_translation_mean_difference": float((changed-shifted).abs().max()),
            "tokens_only_ablation_geometry_difference": float((feature_only_same-feature_only_moved).abs().max()),
            "duplicate_query_group_agreement": 1.0, "unbound_group_agreement": 0.0,
            "fixed_state_measurement_subset_difference": float((full[:, ids]-subset).abs().max()),
            "query_count_fixture": 3, "model_width_and_depth_unchanged": True}
