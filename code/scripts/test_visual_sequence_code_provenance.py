"""Remote CPU contract test for code provenance exceptions."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


VERIFIER = Path(__file__).with_name("verify_visual_sequence_code_provenance.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, records: list[tuple[str, Path]]) -> None:
    path.write_text("".join(f"{sha256(source)}  {relative}\n" for relative, source in records))


def command(root: Path, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        str(VERIFIER),
        "--root",
        str(root),
        "--training_manifest",
        str(paths["training"]),
        "--runtime_manifest",
        str(paths["runtime"]),
        "--evaluation_manifest",
        str(paths["evaluation"]),
        "--exception",
        str(paths["exception"]),
    ]


def must_fail(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        raise AssertionError("invalid provenance fixture passed")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        code = root / "code"
        code.mkdir()
        train_file = code / "train.py"
        eval_file = code / "eval.py"
        train_file.write_text("launch training\n")
        eval_file.write_text("launch evaluation\n")
        launch_eval_hash = sha256(eval_file)
        paths = {
            name: root / f"{name}.sha256"
            for name in ("training", "runtime", "evaluation")
        }
        paths["exception"] = root / "exception.json"
        write_manifest(
            paths["training"],
            [("code/train.py", train_file), ("code/eval.py", eval_file)],
        )
        write_manifest(paths["runtime"], [("code/train.py", train_file)])
        eval_file.write_text("current evaluation\n")
        write_manifest(paths["evaluation"], [("code/eval.py", eval_file)])
        exception = {
            "schema_version": 1,
            "scope": "post_launch_evaluation_only_change",
            "root": str(root),
            "training_manifest": str(paths["training"]),
            "runtime_manifest": str(paths["runtime"]),
            "evaluation_manifest": str(paths["evaluation"]),
            "exclusions": [
                {
                    "path": "code/eval.py",
                    "role": "evaluation_only",
                    "launch_sha256": launch_eval_hash,
                    "current_sha256": sha256(eval_file),
                    "reason": "Synthetic evaluation-only change.",
                }
            ],
        }
        paths["exception"].write_text(json.dumps(exception, sort_keys=True) + "\n")
        cmd = command(root, paths)
        valid = subprocess.run(cmd, check=True, capture_output=True, text=True)
        valid_report = json.loads(valid.stdout)
        if valid_report["status"] != "pass" or valid_report["runtime_records"] != 1:
            raise AssertionError("valid provenance fixture did not pass")

        train_file.write_text("drifted training\n")
        must_fail(cmd)
        train_file.write_text("launch training\n")

        evaluation_text = paths["evaluation"].read_text()
        paths["evaluation"].write_text(
            f"{sha256(train_file)}  code/train.py\n"
        )
        must_fail(cmd)
        paths["evaluation"].write_text(evaluation_text)

        runtime_text = paths["runtime"].read_text()
        paths["runtime"].write_text(
            runtime_text + f"{sha256(eval_file)}  code/eval.py\n"
        )
        must_fail(cmd)
        paths["runtime"].write_text(runtime_text)

        eval_file.write_text("unfrozen evaluation drift\n")
        must_fail(cmd)

    report = {
        "status": "ok",
        "valid_exception_passes": True,
        "training_drift_fails_closed": True,
        "missing_dual_manifest_coverage_fails_closed": True,
        "runtime_manifest_scope_fails_closed": True,
        "evaluation_drift_fails_closed": True,
    }
    output = Path(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
