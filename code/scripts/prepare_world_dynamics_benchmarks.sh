#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-${VENV_ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PY}" -m igsw.world_dynamics_benchmarks --config "${CONFIG}" "$@"
