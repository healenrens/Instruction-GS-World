#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/pfs/public/xuhaoming/instruct_gs_world}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}}"
DATA="${DATA:-${RUNTIME_ROOT}/data/rt2_visual_episodes_rgb_native_30hz_v4}"
MODEL_ROOT="${VIDEO_VAE_MODEL:-/mnt/pfs/public/xuhaoming/models/Wan2.2-TI2V-5B-Diffusers}"
PYTHON_ROOT="${VIDEO_VAE_PYTHONPATH:-${RUNTIME_ROOT}/v44_runtime/diffusers-0.35.2}"
REVISION="${VIDEO_VAE_REVISION:-b8fff7315c768468a5333511427288870b2e9635}"
CONTRACT="${VIDEO_VAE_CONTRACT:-${MODEL_ROOT}/vae_contract.json}"
MANIFEST="${VIDEO_VAE_MANIFEST:-${MODEL_ROOT}/vae_manifest.sha256}"
PY="${VENV_ROOT}/.venv/bin/python"
HF="${VENV_ROOT}/.venv/bin/hf"
PIP="${PIP:-pip}"
export PYTHONDONTWRITEBYTECODE=1

for path in "${ROOT}" "${RUNTIME_ROOT}" "${DATA}"; do
    [[ "${path}" == /* && -d "${path}" ]] || {
        echo "[wan-vae-v44] missing absolute directory: ${path}" >&2
        exit 2
    }
done
[[ -x "${PY}" && -x "${HF}" ]] || {
    echo "[wan-vae-v44] existing environment has no python/hf CLI" >&2
    exit 2
}
command -v "${PIP}" >/dev/null || {
    echo "[wan-vae-v44] no external pip is available for --target" >&2
    exit 2
}
[[ "${MODEL_ROOT}" == /* && "${PYTHON_ROOT}" == /* ]] || {
    echo "[wan-vae-v44] model and Python package roots must be absolute" >&2
    exit 2
}

mkdir -p "${MODEL_ROOT}" "${PYTHON_ROOT}"
if ! PYTHONPATH="${PYTHON_ROOT}" "${PY}" -c \
    'import diffusers; from diffusers import AutoencoderKLWan; assert diffusers.__version__ == "0.35.2"'; then
    "${PIP}" --python "${PY}" install --no-deps --target "${PYTHON_ROOT}" \
        'diffusers==0.35.2'
fi
PYTHONPATH="${PYTHON_ROOT}" "${PY}" -c \
    'import diffusers; from diffusers import AutoencoderKLWan; assert diffusers.__version__ == "0.35.2"'

"${HF}" download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --revision "${REVISION}" --include 'vae/*' --local-dir "${MODEL_ROOT}"
[[ -f "${MODEL_ROOT}/vae/config.json" \
    && -f "${MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors" ]] || {
    echo "[wan-vae-v44] official VAE files are incomplete" >&2
    exit 2
}
for forbidden in transformer text_encoder tokenizer image_encoder; do
    [[ ! -e "${MODEL_ROOT}/${forbidden}" ]] || {
        echo "[wan-vae-v44] forbidden full-model component exists: ${forbidden}" >&2
        exit 2
    }
done

PYTHONPATH="${PYTHON_ROOT}" "${PY}" \
    "${ROOT}/code/scripts/verify_wan_video_vae_v44.py" \
    --model "${MODEL_ROOT}" --data "${DATA}" --revision "${REVISION}" \
    --output "${CONTRACT}" --manifest "${MANIFEST}"
(cd "${MODEL_ROOT}" && sha256sum -c "$(basename "${MANIFEST}")")
echo "[wan-vae-v44] contract=${CONTRACT}"
echo "[wan-vae-v44] pythonpath=${PYTHON_ROOT}"
