#!/bin/bash
# §61 B2 honesty test: regenerate the v2 dual-window clips with OPEN-VOCAB masks (--seg openvocab:
# GroundingDINO+SAM2, target point-prompted at the GT mover centroid = gen-time motion arbitration the
# plan allows). Identical episodes/windows to data/libero_pi3_v2 -> a clean A/B vs the GT-mask v9-lang.
# Usage: bash gen_libero_pi3_v2_ov.sh "<train epis>" "<heldtask epis>" "<heldseed epis>"
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
L=logs/gen_libero_pi3_v2_ov.log
mkdir -p data/libero_pi3_v2_ov logs
rm -f data/libero_pi3_v2_ov/*.pt
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GD_REPO=IDEA-Research/grounding-dino-base PYTHONDONTWRITEBYTECODE=1
find code/scripts/__pycache__ -name "pi3_video_gt*" -o -name "openvocab_seg*" 2>/dev/null | xargs rm -f 2>/dev/null
echo "[gen-v2-ov] start $(date)" > "$L"
i=0
gen_one() {  # epi split
  local epi=$1 split=$2 mode wm out gpu
  for mode in e c; do
    wm=center; [ "$mode" = e ] && wm=early
    out=data/libero_pi3_v2_ov/epi$(printf %06d "$epi")_${mode}_${split}.pt
    gpu=$((i % 4))
    CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python code/scripts/pi3_video_gt.py --epi "$epi" --K 12 \
      --out "$out" --split "$split" --window_mode "$wm" --seg openvocab \
      >> logs/vgen_pi3_v2_ov.log 2>&1 < /dev/null &
    i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
  done
}
for epi in $1; do gen_one "$epi" train; done
for epi in $2; do gen_one "$epi" heldtask; done
for epi in $3; do gen_one "$epi" heldseed; done
wait
NALL=$(ls data/libero_pi3_v2_ov/*.pt 2>/dev/null | wc -l)
NTRAIN=$(ls data/libero_pi3_v2_ov/*_train.pt 2>/dev/null | wc -l)
echo "[gen-v2-ov] done: $NALL clips ($NTRAIN train) $(date)" >> "$L"
# surface the per-clip open-vocab target-IoU (the honesty self-validation) into the summary log
grep -hE "\[ov\] open-vocab seg" logs/vgen_pi3_v2_ov.log >> "$L"
