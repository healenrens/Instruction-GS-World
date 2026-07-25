"""Integrated synthetic validation of object, density, and prior mechanisms."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json, os, sys, time
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
    adaptive_world_model_loss,
    make_oracle_mode_actions,
    make_synthetic_batch,
)
from igsw.adaptive_gaussian_wm.architecture_metrics import (  # noqa: E402
    mode_metrics,
    normalized_slot_error,
    swapped_density_reconstruction,
    temporal_identity_error,
)
from igsw.adaptive_gaussian_wm.action_cycle import (  # noqa: E402
    cross_context_action_cycle_loss,
)
from igsw.adaptive_gaussian_wm.action_regularization import (  # noqa: E402
    sparse_action_regularization,
)
from igsw.adaptive_gaussian_wm.metrics import pearson_correlation, slot_clustering_scores  # noqa: E402
from igsw.adaptive_gaussian_wm.prior_training import train_prior  # noqa: E402
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.training import (  # noqa: E402
    representation_pretrain_loss,
)
VARIANTS = {
    "full": {},
    "no_object": {"aggregation_mode": "global"},
    "independent_slots": {"aggregation_mode": "independent"},
    "fixed_density": {"density_mode": "fixed"},
    "independent_prior": {"joint_flow": False},
    "all_degraded": {
        "aggregation_mode": "global",
        "density_mode": "fixed",
        "joint_flow": False,
    },
}
def model_config(variant: str, feature_dim: int) -> AdaptiveGaussianWMConfig:
    return replace(AdaptiveGaussianWMConfig.tiny(feature_dim), **VARIANTS[variant])
def make_batch(
    config: AdaptiveGaussianWMConfig,
    batch_size: int,
    history_frames: int,
    future_steps: int,
    grid_size: int,
    device: torch.device,
    semantic_branch_strength: float = 0.0,
    balanced_ambiguity: bool = False,
) -> dict[str, torch.Tensor]:
    return make_synthetic_batch(
        config.feature_dim,
        batch_size,
        history_frames,
        future_steps,
        grid_size,
        device,
        paired_futures=True,
        irregular_gaps=True,
        mode_count=3,
        max_objects=config.object_slots,
        ambiguous_fraction=0.5,
        balanced_ambiguity=balanced_ambiguity,
        semantic_branch_strength=semantic_branch_strength,
    )
def train_model(
    config: AdaptiveGaussianWMConfig,
    steps: int, pretrain_steps: int, batch_size: int,
    grid_size: int, future_steps: int,
    device: torch.device,
    semantic_branch_strength: float,
    oracle_actions: bool, freeze_representation: bool,
    training_seed: int, single_history_probability: float,
    action_cycle_weight: float, action_sparsity_weight: float,
) -> tuple[AdaptiveGaussianObjectWorldModel, list[dict[str, float]]]:
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    torch.manual_seed(training_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=0.5,
        allocator=0.2,
        slot=0.2,
        action=0.0 if oracle_actions else 0.8,
    )
    trace = []
    model.train()
    for _ in range(pretrain_steps):
        history_frames = 1 if float(torch.rand(())) < single_history_probability else 3
        batch = make_batch(
            config,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            semantic_branch_strength,
        )
        loss = representation_pretrain_loss(model, batch)[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_target()
    if freeze_representation:
        model.update_target(momentum=0.0)
        for module in (model.allocator, model.object_aggregator):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    for step in range(1, steps + 1):
        history_frames = 1 if float(torch.rand(())) < single_history_probability else 3
        batch = make_batch(
            config,
            batch_size,
            history_frames,
            future_steps,
            grid_size,
            device,
            semantic_branch_strength,
        )
        actions_override = make_oracle_mode_actions(
            batch, config.action_tokens, config.action_dim
        ) if oracle_actions else None
        output = model(batch, actions_override=actions_override)
        loss, parts = adaptive_world_model_loss(model, batch, output, weights)
        action_cycle = (
            cross_context_action_cycle_loss(model, batch, output)
            if action_cycle_weight > 0.0
            else loss.new_zeros(())
        )
        action_sparse = (
            sparse_action_regularization(
                output["posterior_actions"],
                action_sparsity_weight,
            )[0]
            if action_sparsity_weight > 0.0
            else loss.new_zeros(())
        )
        loss = (
            loss
            + action_cycle_weight * action_cycle
            + action_sparse
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_target()
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "total": float(loss.detach()),
                    "future": float(parts["future"].detach()),
                    "feature": float(parts["feature"].detach()),
                    "flow": float(parts["flow"].detach()),
                    "action": float(parts["action"].detach()),
                    "action_cycle": float(action_cycle.detach()),
                    "action_sparse": float(action_sparse.detach()),
                    "allocator": float(parts["allocator"].detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return model, trace
@torch.no_grad()
def evaluate_model(
    model: AdaptiveGaussianObjectWorldModel,
    eval_groups: int, grid_size: int,
    future_steps: int, prior_samples: int,
    device: torch.device,
    semantic_branch_strength: float,
    oracle_actions: bool, evaluation_seed: int,
) -> dict:
    model.eval()
    torch.manual_seed(evaluation_seed)
    batch = make_batch(
        model.config,
        eval_groups * 3,
        3,
        future_steps,
        grid_size,
        device,
        semantic_branch_strength,
        balanced_ambiguity=True,
    )
    history_mask = torch.zeros(
        eval_groups * 3,
        3,
        model.config.object_slots,
        device=device,
        dtype=torch.bool,
    )
    actions_override = make_oracle_mode_actions(
        batch, model.config.action_tokens, model.config.action_dim
    ) if oracle_actions else None
    output = model(
        batch,
        history_mask=history_mask,
        actions_override=actions_override,
    )
    feature_error = (
        output["rendered_future_features"] - batch["future_features"]
    ).square().mean(dim=(1, 2, 3))
    copy_error = (
        batch["history_features"][:, -1, None] - batch["future_features"]
    ).square().mean(dim=(1, 2, 3))
    latent_error = normalized_slot_error(
        output["predicted_future_slots"],
        output["target_future_slots"],
    )
    copy_latent = output["online_history_slots"][:, -1, None]
    copy_latent_error = normalized_slot_error(
        copy_latent,
        output["target_future_slots"],
    )
    object_feature_error = normalized_slot_error(
        output["predicted_future_object_features"], output["target_future_object_features"])
    copy_object_feature_error = normalized_slot_error(
        output["online_history_object_features"][:, -1, None],
        output["target_future_object_features"])
    center_error = (
        output["predicted_future_centers"]
        - output["target_future_centers"]
    ).square().mean(dim=(1, 2, 3))
    copy_center = output["target_history_centers"][:, -1, None]
    copy_center_error = (
        copy_center - output["target_future_centers"]
    ).square().mean(dim=(1, 2, 3))
    effective = (
        output["history_token_states"][-1]
        .activation.sum(dim=1)
        .squeeze(-1)
    )
    density_error, swapped_density_error = swapped_density_reconstruction(
        output,
        batch,
    )
    low = batch["complexity"] <= torch.quantile(batch["complexity"], 1.0 / 3.0)
    high = batch["complexity"] >= torch.quantile(batch["complexity"], 2.0 / 3.0)
    object_scores = slot_clustering_scores(
        output,
        batch["history_labels"][:, -1],
    )
    return {
        "feature_mse": float(feature_error.mean()),
        "current_copy_feature_mse": float(copy_error.mean()),
        "latent_mse": float(latent_error.mean()),
        "current_copy_latent_mse": float(copy_latent_error.mean()),
        "object_feature_mse": float(object_feature_error.mean()),
        "current_copy_object_feature_mse": float(copy_object_feature_error.mean()),
        "center_mse": float(center_error.mean()),
        "current_copy_center_mse": float(copy_center_error.mean()),
        "object_scores": object_scores,
        "temporal_identity_shuffle_latent_mse": temporal_identity_error(
            model,
            output,
            batch,
        ),
        "effective_token_mean": float(effective.mean()),
        "effective_token_std": float(effective.std()),
        "effective_token_complexity_pearson": pearson_correlation(
            effective,
            batch["complexity"],
        ),
        "low_complexity_feature_mse": float(feature_error[low].mean()),
        "high_complexity_feature_mse": float(feature_error[high].mean()),
        "low_complexity_density_reconstruction_mse": float(
            density_error[low].mean()
        ),
        "high_complexity_density_reconstruction_mse": float(
            density_error[high].mean()
        ),
        "density_reconstruction_mse": float(density_error.mean()),
        "swapped_density_reconstruction_mse": float(
            swapped_density_error.mean()
        ),
        "mode": mode_metrics(
            model,
            output,
            batch,
            prior_samples,
        ),
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--pretrain_steps", type=int, default=300)
    parser.add_argument("--prior_steps", type=int, default=1000)
    parser.add_argument("--prior_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--eval_groups", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--future_steps", type=int, default=2)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--prior_samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_in")
    parser.add_argument("--calibrate_loaded_prior", action="store_true")
    parser.add_argument("--max_micro_tokens", type=int)
    parser.add_argument("--semantic_branch_strength", type=float, default=0.0)
    parser.add_argument("--slot_feature_fusion", action="store_true")
    parser.add_argument("--model_dim", type=int)
    parser.add_argument("--dynamics_layers", type=int)
    parser.add_argument("--oracle_mode_actions", action="store_true")
    parser.add_argument("--freeze_representation", action="store_true")
    parser.add_argument("--decoupled_jepa_slots", action="store_true")
    parser.add_argument("--action_query_modulation", action="store_true")
    parser.add_argument("--action_film_modulation", action="store_true")
    parser.add_argument("--kinematic_action_modulation", action="store_true")
    parser.add_argument("--learned_velocity_baseline", action="store_true")
    parser.add_argument("--spatial_slot_attention", action="store_true")
    parser.add_argument("--token_spatial_precision_floor", type=float, default=0.0)
    parser.add_argument("--single_history_probability", type=float, default=0.25)
    parser.add_argument("--evaluation_seed", type=int, default=9107)
    parser.add_argument("--action_dim", type=int)
    parser.add_argument("--action_cycle_weight", type=float, default=0.0)
    parser.add_argument("--action_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--disable_posterior_normalization", action="store_true")
    parser.add_argument("--canonical_center_action", action="store_true")
    parser.add_argument("--canonical_semantic_action", action="store_true")
    args = parser.parse_args()
    if args.batch_size % 3:
        raise ValueError("batch_size must be divisible by three")
    if args.semantic_branch_strength < 0.0:
        raise ValueError("semantic_branch_strength must be non-negative")
    if args.action_cycle_weight < 0.0:
        raise ValueError("action_cycle_weight must be non-negative")
    if args.action_sparsity_weight < 0.0:
        raise ValueError("action_sparsity_weight must be non-negative")
    if not 0.0 <= args.single_history_probability <= 1.0:
        raise ValueError("single_history_probability must be in [0, 1]")
    if min(args.steps, args.pretrain_steps, args.prior_steps, args.eval_groups) <= 0:
        raise ValueError("steps and eval_groups must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    started = time.time()
    if args.checkpoint_in:
        state = torch.load(
            args.checkpoint_in,
            map_location="cpu",
            weights_only=False,
        )
        if state["variant"] != args.variant:
            raise ValueError("checkpoint variant does not match --variant")
        config = AdaptiveGaussianWMConfig(**state["config"])
        model = AdaptiveGaussianObjectWorldModel(config).to(device)
        model.load_state_dict(state["model"], strict=True)
        trace = []
        prior_trace = (
            train_prior(
                model,
                args.prior_steps,
                args.batch_size,
                args.grid_size,
                args.future_steps,
                device,
                args.prior_lr,
                args.semantic_branch_strength,
                args.single_history_probability,
            )
            if args.calibrate_loaded_prior
            else []
        )
    else:
        config = model_config(args.variant, args.feature_dim)
        if args.max_micro_tokens:
            config = replace(config, max_micro_tokens=args.max_micro_tokens)
        if args.slot_feature_fusion:
            config = replace(config, slot_feature_fusion=True)
        if args.model_dim:
            config = replace(config, model_dim=args.model_dim)
        if args.dynamics_layers:
            config = replace(config, dynamics_layers=args.dynamics_layers)
        if args.action_dim:
            config = replace(config, action_dim=args.action_dim)
        if args.disable_posterior_normalization:
            config = replace(config, normalize_posterior=False)
        if args.canonical_center_action:
            config = replace(config, canonical_center_action=True)
        if args.canonical_semantic_action:
            config = replace(config, canonical_semantic_action=True)
        config = replace(config, decoupled_jepa_slots=args.decoupled_jepa_slots, action_query_modulation=args.action_query_modulation, action_film_modulation=args.action_film_modulation, kinematic_action_modulation=args.kinematic_action_modulation, learned_velocity_baseline=args.learned_velocity_baseline, spatial_slot_attention=args.spatial_slot_attention, token_spatial_precision_floor=args.token_spatial_precision_floor)
        model, trace = train_model(
            config,
            args.steps,
            args.pretrain_steps,
            args.batch_size,
            args.grid_size,
            args.future_steps,
            device,
            args.semantic_branch_strength,
            args.oracle_mode_actions,
            args.freeze_representation,
            args.seed + 1000,
            args.single_history_probability,
            args.action_cycle_weight,
            args.action_sparsity_weight,
        )
        prior_trace = train_prior(
            model,
            args.prior_steps,
            args.batch_size,
            args.grid_size,
            args.future_steps,
            device,
            args.prior_lr,
            args.semantic_branch_strength,
            args.single_history_probability,
        )
    metrics = evaluate_model(
        model,
        args.eval_groups,
        args.grid_size,
        args.future_steps,
        args.prior_samples,
        device,
        args.semantic_branch_strength,
        args.oracle_mode_actions,
        args.evaluation_seed,
    )
    if args.checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save(
            {
                "variant": args.variant,
                "seed": args.seed,
                "config": config.to_dict(),
                "model": model.state_dict(),
            },
            args.checkpoint,
        )
    report = {
        "status": "ok",
        "variant": args.variant,
        "seed": args.seed,
        "training_seed": args.seed + 1000,
        "evaluation_seed": args.evaluation_seed,
        "single_history_probability": args.single_history_probability,
        "action_cycle_weight": args.action_cycle_weight,
        "action_sparsity_weight": args.action_sparsity_weight,
        "posterior_normalization_disabled": (
            args.disable_posterior_normalization
        ),
        "canonical_center_action": args.canonical_center_action,
        "canonical_semantic_action": args.canonical_semantic_action,
        "config": config.to_dict(),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "steps": args.steps,
        "pretrain_steps": args.pretrain_steps,
        "prior_steps": args.prior_steps,
        "prior_lr": args.prior_lr,
        "semantic_branch_strength": args.semantic_branch_strength,
        "oracle_mode_actions": args.oracle_mode_actions,
        "oracle_dynamics_isolated": args.oracle_mode_actions,
        "prior_deferred_during_dynamics": True,
        "freeze_representation": args.freeze_representation,
        "elapsed_seconds": time.time() - started,
        "training_trace": trace,
        "prior_training_trace": prior_trace,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
if __name__ == "__main__":
    main()
