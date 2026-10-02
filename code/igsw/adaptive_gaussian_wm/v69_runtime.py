"""Shared execution configuration, metric records, and foreground DDP utilities."""

import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from .v69_config import ObjectVideoConfigV69


def add_v69_arguments(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", choices=("dinov3_vitl16", "vjepa2_1_vitl16"), default="dinov3_vitl16")
    parser.add_argument("--encoder_repository", required=True)
    parser.add_argument("--encoder_weights", required=True)
    parser.add_argument("--encoder_frame_batch", type=int, default=2)
    parser.add_argument("--config", default="")
    parser.add_argument("--stage", choices=("state", "dynamics"), default="state")
    parser.add_argument("--state_checkpoint", default="")
    parser.add_argument("--stage2_preset", choices=("legacy", "large"), default="legacy",
                        help="Fresh Stage2 capacity; large keeps State512 and uses Dynamics1024x12/Posterior1024x4, 16 heads.")
    parser.add_argument("--dynamics_checkpoint_blocks", action="store_true",
                        help="Recompute training Dynamics local/global blocks during backward, preserving RNG.")
    parser.add_argument("--posterior_geometry", choices=("inherit", "on", "off"), default="inherit")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--stop_after", type=int, default=0, help="Optional execution stop; scheduler still uses --steps.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic kernels for controlled resume comparisons.")
    parser.add_argument("--resume_trace", action="store_true", help="Save per-microbatch data, RNG and posterior evidence for the integration test.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    parser.add_argument("--wandb_name", default="object_video_sequence_v69")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def config_from_args(args, state_config=None):
    values = state_config.to_dict() if state_config is not None else {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text()))
    encoder = state_config.encoder if state_config is not None else args.encoder
    config = ObjectVideoConfigV69(**{**values, "encoder": encoder})
    if args.stage == "dynamics":
        config = replace(config, architecture="pretrained_query_object_video_sequence_v2")
        if args.stage2_preset == "large":
            config = replace(config, architecture="pretrained_query_object_video_sequence_v3_stage2_large",
                             dynamics_width=1024, dynamics_heads=16, dynamics_layers=12,
                             posterior_width=1024, posterior_heads=16, posterior_layers=4, posterior_geometry=True)
        if args.dynamics_checkpoint_blocks:
            config = replace(config, dynamics_checkpoint_blocks=True)
    return replace(config, posterior_geometry=args.posterior_geometry == "on") if args.posterior_geometry != "inherit" else config


@torch.no_grad()
def gradient_metrics_v69(model):
    """Preclip accumulated/DDP-synchronized gradients, including every trainable block."""
    modules = dict(model.named_children())
    for root in ("posterior", "dynamics"):
        for name, module in getattr(model, root).named_children():
            if isinstance(module, torch.nn.ModuleList):
                modules.update({f"{root}/{name}/{index}": block for index, block in enumerate(module)})
            else:
                modules[f"{root}/{name}"] = module
    records = {}
    for name, module in modules.items():
        gradients = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
        if gradients:
            records[name] = torch.stack((
                torch.stack([torch.linalg.vector_norm(g) for g in gradients]).norm(),
                torch.stack([g.abs().max() for g in gradients]).max(),
                torch.stack([g.isfinite().all() for g in gradients]).all().float()))
    values = torch.stack(list(records.values())).cpu().tolist()
    return {f"gradient_preclip/{name}/{statistic}": value
            for name, row in zip(records, values)
            for statistic, value in zip(("l2", "max_abs", "finite"), row)}


def batch_is_readable(batch, context, device):
    readable = torch.tensor(int("decode_errors" not in batch), device=device)
    if context.distributed:
        dist.all_reduce(readable, op=dist.ReduceOp.MIN)
    if "decode_errors" in batch:
        print(json.dumps({"event": "decode_skip", "rank": context.rank, "errors": batch["decode_errors"]}), flush=True)
    return bool(readable)


def case_metrics_v69(output, batch, config, stage):
    th = config.history_frames
    valid = batch["teacher"]["valid"] if stage == "state" else batch["teacher"]["valid"][:, th:]
    times = batch["times"] if stage == "state" else batch["times"][:, th:]
    rows = []
    for item in range(len(valid)):
        for frame in range(valid.shape[1]):
            all_values = output["epe_px"][item, frame][valid[item, frame]].detach().float().cpu()
            primary = valid[item, frame] & batch["teacher"]["transport_weight"][item].bool()
            primary_values = output["epe_px"][item, frame][primary].detach().float().cpu()
            row = {"case": batch["case_id"][item], "source": batch["source"][item],
                   "seconds": float(times[item, frame]), "valid_points": len(all_values),
                   "transport_points": len(primary_values), "epe_px": all_values.tolist(), "transport_epe_px": primary_values.tolist(),
                   "selection_status": batch["transport_selection_status"][item]}
            for name, values in (("all", all_values), ("transport", primary_values)):
                row[f"{name}_p50"] = float(values.quantile(.5)) if len(values) else None
                row[f"{name}_p90"] = float(values.quantile(.9)) if len(values) else None
            rows.append(row)
    return rows


def append_case_records(path, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
