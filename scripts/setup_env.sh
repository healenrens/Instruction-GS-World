#!/usr/bin/env bash
# Faithful base environment for Instruct-GS-World on the A100 server.
# uv-managed venv, torch matching system CUDA 12.6, core data/ML deps.
# VGGT / Pi3 / gsplat are added later once the design is locked.
set -euo pipefail

export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy

WS=/mnt/pfs/public/xuhaoming/instruct_gs_world
export UV_CACHE_DIR=$WS/.uv_cache
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache   # reuse existing HF cache
cd "$WS"
UV=/opt/conda/bin/uv

echo "[1/4] create venv (python 3.11)"
$UV venv --python 3.11 "$WS/.venv"
# shellcheck disable=SC1091
source "$WS/.venv/bin/activate"
python -V

echo "[2/4] install torch 2.8.0 + torchvision (cu126)"
$UV pip install --python "$WS/.venv/bin/python" \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo "[3/4] install core deps (default index)"
$UV pip install --python "$WS/.venv/bin/python" \
    numpy "pyarrow>=15" pandas pillow "opencv-python-headless" \
    einops tqdm safetensors "huggingface_hub" "transformers>=4.57" accelerate \
    roma plyfile imageio "imageio-ffmpeg" scipy matplotlib av tensorboard \
    "jaxtyping" rich

echo "[4/4] GPU sanity check"
python - <<'PY'
import torch, torchvision, platform
print("python   :", platform.python_version())
print("torch    :", torch.__version__, "cuda", torch.version.cuda)
print("torchvis :", torchvision.__version__)
print("cuda ok  :", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0     :", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item()
    print("matmul ok:", y is not None)
PY
echo "DONE base env at $WS/.venv"
