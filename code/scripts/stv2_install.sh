#!/bin/bash
# §80 StV2 install — robust against the flaky proxy: torch via PyPI CDN with wget -c (resumable),
# git deps already cloned locally, rest via requirements_nogit.txt with retries.
set -x
cd /mnt/pfs/public/xuhaoming/SpaTrackerV2
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 GIT_TERMINAL_PROMPT=0
L=/mnt/pfs/public/xuhaoming/instruct_gs_world/logs/stv2_install.log
PY=/mnt/pfs/public/xuhaoming/SpaTrackerV2/.venv_stv2/bin/python
PIP=/mnt/pfs/public/xuhaoming/SpaTrackerV2/.venv_stv2/bin/pip

echo "[stv2-install] SCRIPT start $(date)" >> "$L"
mkdir -p wheels && cd wheels

# wheel URLs pre-resolved interactively into wheels/{torch,tv}.url (in-script curl|py kept failing)
TORCH_URL=$(cat torch.url)
TV_URL=$(cat tv.url)
echo "[stv2-install] urls: $TORCH_URL $TV_URL" >> "$L"

for i in $(seq 1 60); do wget -c -q --timeout=90 "$TORCH_URL" && break; sleep 8; done
for i in $(seq 1 60); do wget -c -q --timeout=90 "$TV_URL" && break; sleep 8; done
echo "[stv2-install] wheels: $(ls -la *.whl 2>/dev/null | wc -l) files $(du -sh . | cut -f1)" >> "$L"

cd ..
"$PIP" install --no-index --find-links wheels wheels/torch-*.whl wheels/torchvision-*.whl >> "$L" 2>&1 \
  || "$PIP" install wheels/torch-*.whl wheels/torchvision-*.whl >> "$L" 2>&1
echo "[stv2-install] torch install exit=$?" >> "$L"

# rest of the deps (git deps pre-cloned; segment-anything/utils3d local)
grep -v "git+https" requirements.txt > requirements_nogit.txt
"$PIP" install --no-deps ./segment-anything ./utils3d >> "$L" 2>&1
for i in 1 2 3 4 5; do
  "$PIP" install --retries 15 --timeout 240 -r requirements_nogit.txt >> "$L" 2>&1 && break
  echo "[stv2-install] nogit attempt $i failed $(date)" >> "$L"
  sleep 20
done
echo "[stv2-install] REQS exit=$? $(date)" >> "$L"

# verify
"$PY" - >> "$L" 2>&1 << 'EOF'
import torch
print("[stv2-install] VERIFY torch", torch.__version__, "cuda", torch.cuda.is_available())
import segment_anything, utils3d, xformers
print("[stv2-install] VERIFY deps OK")
EOF
echo "[stv2-install] ALL-DONE $(date)" >> "$L"
