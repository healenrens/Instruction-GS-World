"""Runtime reproducibility settings and inspectable resume comparisons, without hashes."""

import os
from pathlib import Path

import torch


def configure_reproducibility_v69(enabled):
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "task_attention": "explicit_qkv" if enabled else "automatic_sdpa"}


def step_inputs_v69(batch, device):
    teacher = batch["teacher"]
    return {"case_id": batch["case_id"], "sample_index": batch["sample_index"],
            "epoch": batch["epoch"], "occurrence": batch["occurrence"],
            "frame_indices": batch["frame_indices"].detach().cpu(),
            "times": batch["times"].detach().cpu(), "native_hw": batch["native_hw"].detach().cpu(),
            "point_ids": teacher["point_ids"].detach().cpu(),
            "teacher_xy": teacher["xy"].detach().cpu(), "teacher_valid": teacher["valid"].detach().cpu(),
            "rng_cpu_before_forward": torch.get_rng_state(),
            "rng_cuda_before_forward": torch.cuda.get_rng_state(device)}


def save_step_trace_v69(directory, step, microstep, inputs, output, model):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    trace = {**inputs, "source_tokens": output["source"].tokens.detach().cpu(),
             "source_centers": output["source"].centers.detach().cpu(),
             "loss": output["loss"].detach().cpu()}
    if "effect" in output:
        trace.update({"effect_"+key: output["effect"][key].detach().cpu()
                      for key in ("noise", "mean", "logvar", "value")})
        trace["posterior_query_gradient"] = model.posterior.queries.grad.detach().cpu()
    torch.save(trace, path / f"step_{step:07d}_micro_{microstep:04d}.pt")


def compare_values_v69(first, second, name="", atol=1e-6, rtol=1e-4):
    """Return mismatched leaves. Integer state/RNG comparisons are exact."""
    differences = []
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if first.shape != second.shape or first.dtype != second.dtype:
            return [{"field": name, "first_shape": list(first.shape), "second_shape": list(second.shape),
                     "first_dtype": str(first.dtype), "second_dtype": str(second.dtype)}]
        exact = not first.is_floating_point() or name.endswith("/effect_noise")
        equal = torch.equal(first, second) if exact else torch.allclose(first, second, atol=atol, rtol=rtol)
        if not equal:
            difference = (first.double()-second.double()).abs()
            differences.append({"field": name, "max_abs": float(difference.max()),
                                "l2": float(difference.square().sum().sqrt()),
                                "elements": first.numel()})
    elif isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            differences.append({"field": name, "missing_first": sorted(map(str, second.keys()-first.keys())),
                                "missing_second": sorted(map(str, first.keys()-second.keys()))})
        for key in first:
            if key in second:
                differences.extend(compare_values_v69(first[key], second[key], f"{name}/{key}", atol, rtol))
    elif isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            differences.append({"field": name, "first_length": len(first), "second_length": len(second)})
        for i, (a, b) in enumerate(zip(first, second)):
            differences.extend(compare_values_v69(a, b, f"{name}/{i}", atol, rtol))
    elif first != second:
        differences.append({"field": name, "first": first, "second": second})
    return differences


def compare_resume_v69(first, second, first_out, second_out):
    fields = ("model", "optimizer", "scheduler", "rng", "step", "epoch", "cursor", "grad_accum", "world_size", "config")
    mismatches = compare_values_v69({key: first[key] for key in fields}, {key: second[key] for key in fields})
    max_model_difference = max(float((value.float()-second["model"][name].float()).abs().max())
                               for name, value in first["model"].items())
    first_files = {p.name: p for p in (Path(first_out)/"resume_trace_rank0000").glob("*.pt")}
    second_files = {p.name: p for p in (Path(second_out)/"resume_trace_rank0000").glob("*.pt")}
    trace_mismatches = []
    for name in sorted(first_files.keys() & second_files.keys()):
        a = torch.load(first_files[name], map_location="cpu", weights_only=False)
        b = torch.load(second_files[name], map_location="cpu", weights_only=False)
        differences = compare_values_v69(a, b)
        if differences:
            trace_mismatches.append({"trace": name, "differences": differences})
    trace_inventory_equal = bool(first_files) and first_files.keys() == second_files.keys()
    return {"passed": not mismatches and not trace_mismatches and trace_inventory_equal,
            "atol": 1e-6, "rtol": 1e-4, "model_max_abs_difference": max_model_difference,
            "checkpoint_differences": mismatches, "trace_inventory_equal": trace_inventory_equal,
            "trace_count": len(first_files), "first_trace_mismatch": trace_mismatches[0] if trace_mismatches else None,
            "trace_mismatches": trace_mismatches}
