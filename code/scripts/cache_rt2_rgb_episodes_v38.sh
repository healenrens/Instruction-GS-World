#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pfs/public/fanyupeng/dataset/robotwin2_lerobot}"
SOURCE_VARIANTS="${SOURCE_VARIANTS:-demo_clean,demo_randomized}"
EXPECTED_SOURCE_FPS="${EXPECTED_SOURCE_FPS:-30}"
OUT="${OUT:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/logs/rt2_visual_episodes_rgb_native_30hz_v4}"
CACHE_MODE="${CACHE_MODE:-auto}"
CACHE_JOBS="${CACHE_JOBS:-8}"
FRAME_BATCH="${FRAME_BATCH:-32}"
JPEG_WORKERS="${JPEG_WORKERS:-2}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
WINDOW_LENGTHS="${WINDOW_LENGTHS:-45,90,135,180}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
OVERWRITE="${OVERWRITE:-0}"
PY="${VENV_ROOT}/.venv/bin/python"

for path in "${ROOT}" "${RUNTIME_ROOT}" "${SOURCE_ROOT}"; do
    if [[ "${path}" != /* || ! -d "${path}" ]]; then
        echo "[v38-rgb] missing absolute directory: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${PY}" ]]; then
    echo "[v38-rgb] missing Python environment: ${PY}" >&2
    exit 2
fi
if [[ "${OUT}" != /* || "${LOG_ROOT}" != /* ]]; then
    echo "[v38-rgb] output paths must be absolute" >&2
    exit 2
fi
if [[ ! "${CACHE_JOBS}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${FRAME_BATCH}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${JPEG_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v38-rgb] cache worker counts must be positive" >&2
    exit 2
fi
if [[ "${CACHE_MODE}" != "auto" && "${CACHE_MODE}" != "fresh" \
    && "${CACHE_MODE}" != "resume" ]]; then
    echo "[v38-rgb] CACHE_MODE must be auto, fresh, or resume" >&2
    exit 2
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "[v38-rgb] OVERWRITE must be 0 or 1" >&2
    exit 2
fi
if ! git -C "${ROOT}" diff --quiet \
    || ! git -C "${ROOT}" diff --cached --quiet; then
    echo "[v38-rgb] tracked repository files are modified" >&2
    git -C "${ROOT}" status --short --untracked-files=no >&2
    exit 2
fi
"${PY}" -c 'import av, pyarrow, torch, torchvision'

mkdir -p "${OUT}" "${LOG_ROOT}"
LOCK_DIR="${OUT}/.v38_rgb_cache.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    owner="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ "${owner}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${owner}" 2>/dev/null; then
        echo "[v38-rgb] another cache process is alive: pid=${owner}" >&2
        exit 2
    fi
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
fi
printf '%s\n' "$$" >"${LOCK_DIR}/pid"
children=()
cleanup() {
    for pid in "${children[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    rm -rf "${LOCK_DIR}"
}
trap cleanup EXIT INT TERM

MANIFEST="${OUT}/episode_manifest.json"
CHECKSUM="${OUT}/episode_manifest.verified.sha256"
SOURCE_INDEX="${OUT}/episode_source_index.json"
PENDING="${OUT}/episode_manifest.pending.json"
VERIFY_REPORT="${LOG_ROOT}/verification.json"
if [[ -f "${MANIFEST}" && -f "${CHECKSUM}" ]] \
    && (cd "${OUT}" && sha256sum -c --status episode_manifest.verified.sha256); then
    if [[ "${CACHE_MODE}" == "fresh" ]]; then
        echo "[v38-rgb] fresh mode refuses a complete cache: ${OUT}" >&2
        exit 2
    fi
    "${PY}" "${ROOT}/code/scripts/verify_rt2_rgb_episode_cache_v38.py" \
        --data "${OUT}" --output "${VERIFY_REPORT}"
    echo "[v38-rgb] cache already complete: ${OUT}"
    exit 0
fi

if [[ "${CACHE_MODE}" == "auto" ]]; then
    if [[ -f "${SOURCE_INDEX}" && -f "${PENDING}" ]]; then
        CACHE_MODE=resume
    elif [[ ! -e "${SOURCE_INDEX}" && ! -e "${PENDING}" ]] \
        && ! compgen -G "${OUT}/*.pt" >/dev/null; then
        CACHE_MODE=fresh
    else
        echo "[v38-rgb] partial state lacks a resumable manifest pair" >&2
        exit 2
    fi
fi
if [[ "${CACHE_MODE}" == "resume" ]] \
    && [[ ! -f "${SOURCE_INDEX}" || ! -f "${PENDING}" ]]; then
    echo "[v38-rgb] resume requires source index and pending manifest" >&2
    exit 2
fi

cd "${ROOT}"
echo "[v38-rgb] commit=$(git rev-parse HEAD) mode=${CACHE_MODE} jobs=${CACHE_JOBS}"
"${PY}" code/scripts/verify_robotwin_lerobot_source_v37.py \
    --source_root "${SOURCE_ROOT}" \
    --source_variants "${SOURCE_VARIANTS}" \
    --expected_source_fps "${EXPECTED_SOURCE_FPS}" \
    --output "${LOG_ROOT}/source_preflight.json" \
    2>&1 | tee "${LOG_ROOT}/source_preflight.log"

common=(
    --source_root "${SOURCE_ROOT}"
    --source_variants "${SOURCE_VARIANTS}"
    --expected_source_fps "${EXPECTED_SOURCE_FPS}"
    --out "${OUT}"
    --window_lengths "${WINDOW_LENGTHS}"
    --sample_stride "${SAMPLE_STRIDE}"
    --jpeg_quality "${JPEG_QUALITY}"
    --frame_batch "${FRAME_BATCH}"
    --jpeg_workers "${JPEG_WORKERS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    common+=(--overwrite)
fi
"${PY}" code/scripts/cache_rt2_rgb_episodes_v38.py \
    "${common[@]}" --prepare --prepare_mode "${CACHE_MODE}" \
    2>&1 | tee "${LOG_ROOT}/prepare.log"

logs=()
for ((shard = 0; shard < CACHE_JOBS; shard++)); do
    log="${LOG_ROOT}/shard_${shard}.log"
    OMP_NUM_THREADS=2 "${PY}" code/scripts/cache_rt2_rgb_episodes_v38.py \
        "${common[@]}" --shard "${shard}" --nshard "${CACHE_JOBS}" \
        >"${log}" 2>&1 &
    children+=("$!")
    logs+=("${log}")
    echo "[v38-rgb] shard=${shard}/${CACHE_JOBS} pid=$! log=${log}"
done
failed=0
for index in "${!children[@]}"; do
    if ! wait "${children[${index}]}"; then
        failed=1
        echo "[v38-rgb] shard failed: ${logs[${index}]}" >&2
        tail -n 120 "${logs[${index}]}" >&2
    fi
done
children=()
if [[ "${failed}" -ne 0 ]]; then
    echo "[v38-rgb] one or more RGB cache shards failed" >&2
    exit 1
fi

"${PY}" code/scripts/cache_rt2_rgb_episodes_v38.py \
    "${common[@]}" --finalize 2>&1 | tee "${LOG_ROOT}/finalize.log"
(
    cd "${OUT}"
    sha256sum episode_manifest.json >episode_manifest.verified.sha256
)
"${PY}" code/scripts/verify_rt2_rgb_episode_cache_v38.py \
    --data "${OUT}" --output "${VERIFY_REPORT}" \
    2>&1 | tee "${LOG_ROOT}/verify.log"
echo "[v38-rgb] completed data=${OUT} verification=${VERIFY_REPORT}"
