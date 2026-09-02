#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
REPORT="${REPORT:-${RUNTIME_ROOT}/outputs/v66_gates/${SOURCE_REVISION}_structural.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "$(dirname "${REPORT}")"

echo "[object-motion-field-v66] structural world=4 report=${REPORT}"
cd "${ROOT}"
exec "${VENV_ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node 4 \
  "${ROOT}/code/scripts/verify_object_motion_field_g0_v66.py" \
  --source_revision "${SOURCE_REVISION}" \
  --output "${REPORT}"
