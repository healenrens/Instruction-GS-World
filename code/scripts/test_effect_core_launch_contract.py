"""CPU contract test for fail-closed two-node launch agreement."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prepare_effect_core_launch_contract.py",
)


def command(
    root: str,
    data: str,
    out: str,
    checkpoint: str,
    evidence: str,
    output: str,
    batch: int,
    grad_accum: int,
) -> list[str]:
    training = [
        "code/scripts/train_adaptive_gaussian_wm.py",
        "--data", data,
        "--data_format", "sequence",
        "--history_frames", "4",
        "--future_frames", "4",
        "--sequence_anchors", "3,5,8",
        "--out", out,
        "--profile", "full",
        "--representation_steps", "0",
        "--joint_steps", "100000",
        "--batch", str(batch),
        "--grad_accum", str(grad_accum),
        "--workers", "2",
        "--lr", "5e-5",
        "--lr_floor", "5e-6",
        "--warmup_steps", "5000",
        "--warmup_fraction", "0.0",
        "--weight_decay", "1e-4",
        "--save_every", "1000",
        "--log_every", "20",
        "--seed", "17",
        "--amp", "bf16",
        "--language_condition", "off",
        "--rgb_supervision", "on",
        "--rgb_short_side", "256",
        "--rgb_pad_multiple", "16",
        "--rgb_render_chunk", "8192",
        "--rgb_loss_weight", "0.5",
        "--rgb_ssim_weight", "0.2",
        "--rgb_change_loss_weight", "1.0",
        "--rgb_change_threshold", "0.04",
        "--language_effect_weight", "0.0",
        "--zero_action_margin_weight", "5.0",
        "--posterior_dynamics_gate",
        "--posterior_update_scope", "full",
        "--action_anchor", "object_slot",
        "--canonical_center_gate", "1.0",
        "--canonical_activity_gate",
        "--canonical_activity_power", "1.0",
        "--action_residual_dim", "8",
        "--action_residual_gate", "1.0",
        "--action_residual_dropout", "0.0",
        "--semantic_action_basis", "rgb",
        "--init_from", checkpoint,
    ]
    return [
        sys.executable,
        SCRIPT,
        "--root", root,
        "--out", out,
        "--data", data,
        "--attempt_key", "warm_start_joint_0012000",
        "--mode", "warm_start",
        "--checkpoint", checkpoint,
        "--nnodes", "2",
        "--nproc_per_node", "8",
        "--master_addr", "10.0.0.1",
        "--master_port", "29500",
        "--batch_per_gpu", str(batch),
        "--grad_accum", str(grad_accum),
        "--evidence", f"source={evidence}",
        "--output", output,
        "--training_args",
        *training,
    ]


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        data = os.path.join(root, "data")
        out = os.path.join(root, "out")
        os.makedirs(data)
        os.makedirs(out)
        checkpoint = os.path.join(root, "joint_0012000.pt")
        evidence = os.path.join(root, "gate.json")
        Path(checkpoint).write_bytes(b"checkpoint")
        Path(evidence).write_bytes(b"evidence")
        node0 = os.path.join(out, "node0.json")
        node1 = os.path.join(out, "node1.json")
        first = command(root, data, out, checkpoint, evidence, node0, 2, 8)
        subprocess.run(first, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(first, check=True, stdout=subprocess.DEVNULL)
        altered = first.copy()
        altered[altered.index("--lr") + 1] = "1e-4"
        profile_failure = subprocess.run(
            altered,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if profile_failure.returncode == 0:
            raise AssertionError("altered training profile passed")
        missing_flag = first.copy()
        missing_flag.remove("--canonical_activity_gate")
        flag_failure = subprocess.run(
            missing_flag,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if flag_failure.returncode == 0:
            raise AssertionError("missing training flag passed")
        node1_command = command(root, data, out, checkpoint, evidence, node1, 2, 8)
        node1_command[node1_command.index("--master_port") + 1] = "29501"
        subprocess.run(node1_command, check=True, stdout=subprocess.DEVNULL)
        if Path(node0).read_bytes() == Path(node1).read_bytes():
            raise AssertionError("different node contracts compare equal")
        mismatch = command(root, data, out, checkpoint, evidence, node0, 2, 8)
        mismatch[mismatch.index("--master_port") + 1] = "29501"
        failed = subprocess.run(
            mismatch,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if failed.returncode == 0:
            raise AssertionError("differing contract overwrote existing evidence")
    print({
        "status": "ok",
        "identical_reuse": True,
        "profile_mismatch_fails": True,
        "missing_flag_fails": True,
        "node_mismatch_fails": True,
    })


if __name__ == "__main__":
    main()
