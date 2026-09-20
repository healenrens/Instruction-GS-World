#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
BASE="${RUNTIME_ROOT}/runtime/grounded_motion_v68"
RELEASE="${BASE}/releases/${SOURCE_REVISION}"
mkdir -p "${RELEASE}/code"
cp -a "${ROOT}/code/igsw" "${RELEASE}/code/"
cp -a "${ROOT}/code/scripts" "${RELEASE}/code/"
printf '%s\n' "${SOURCE_REVISION}" > "${RELEASE}/SOURCE_REVISION"
printf '%s\n' "${SOURCE_REVISION}" > "${BASE}/DEPLOYED_REVISION"
echo "[motion-data-v68-deploy] release=${RELEASE}"
