#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
read -r DEPLOYED_REVISION < "${ROOT}/SOURCE_REVISION"
SOURCE_REVISION="${SOURCE_REVISION:-${DEPLOYED_REVISION}}"
RUN_ID="${RUN_ID:-object_motion_field_g0_v66_seed17_${SOURCE_REVISION:0:7}_256ps}"
REPORT="${REPORT:-${RUNTIME_ROOT}/outputs/object_motion_field_g0_v66/${RUN_ID}/object_motion_field_g0_audit.json}"

export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

exec "${VENV_ROOT}/.venv/bin/python" \
  "${ROOT}/code/scripts/upload_object_motion_field_g0_v66_report.py" \
  --report "${REPORT}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_name "${WANDB_NAME:-${RUN_ID}_report_upload}" \
  --wandb_group "${WANDB_GROUP:-object-motion-field-g0-v66}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
