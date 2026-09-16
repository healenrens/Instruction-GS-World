#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
RUN_NAME="${RUN_NAME:-continuous_predictive_object_field_v67_contract_audit_upload}"
REPORT="${REPORT:-${RUNTIME_ROOT}/outputs/v67_contract_audits/${RUN_NAME}.json}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RUNTIME_ROOT}/outputs/v67_contract_audits/${RUN_NAME}_evidence}"

export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
cd "${ROOT}"

exec "${VENV_ROOT}/.venv/bin/python" \
  "${ROOT}/code/scripts/upload_continuous_predictive_contracts_v67.py" \
  --report "${REPORT}" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-instruct-gs-world}" \
  --wandb_entity "${WANDB_ENTITY:-healenrenss-university-of-chinese-acadmic-and-science}" \
  --wandb_name "${WANDB_NAME:-${RUN_NAME}}" \
  --wandb_group "${WANDB_GROUP:-continuous-predictive-object-field-v67-contract-audit}" \
  --wandb_dir "${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"
