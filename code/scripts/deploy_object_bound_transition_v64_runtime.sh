#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
RELEASE_ROOT="${RELEASE_ROOT:-${RUNTIME_ROOT}/runtime/object_bound_transition_v64/releases/${SOURCE_REVISION}}"

mkdir -p "${RELEASE_ROOT}/code"
cp -a "${ROOT}/code/igsw" "${RELEASE_ROOT}/code/"
cp -a "${ROOT}/code/scripts" "${RELEASE_ROOT}/code/"
printf '%s\n' "${SOURCE_REVISION}" > "${RELEASE_ROOT}/SOURCE_REVISION"
printf '%s\n' "${ROOT}" > "${RELEASE_ROOT}/DEPLOYED_FROM"

echo "[v64-deploy] release_root=${RELEASE_ROOT}"
echo "[v64-deploy] source_revision=${SOURCE_REVISION}"
