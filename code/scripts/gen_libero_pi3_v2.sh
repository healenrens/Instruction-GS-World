#!/bin/bash
# §54 v8b data: Pi3 backend, TWO windows per episode — _e (EARLY: pre-contact, gripper far -> breaks the
# gripper-proximity shortcut) and _c (CENTER: current). Filename epi{N}_{e|c}_{split}.pt keeps the split
# suffix LAST so SimClipDataset.list_clips parses it. Usage: bash gen_libero_pi3_v2.sh "<train epis>" "<held epis>"
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
L=logs/gen_libero_pi3_v2.log
mkdir -p data/libero_pi3_v2 logs
rm -f data/libero_pi3_v2/*.pt
echo "[gen-v2] start $(date)" > "$L"
export https_proxy=http://10.66.65.186:18000 http_proxy=http://10.66.65.186:18000 HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache
i=0
gen_one() {  # epi split
  local epi=$1 split=$2 mode wm out gpu
  for mode in e c; do
    wm=center; [ "$mode" = e ] && wm=early
    out=data/libero_pi3_v2/epi$(printf %06d "$epi")_${mode}_${split}.pt
    gpu=$((i % 4))
    CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python code/scripts/pi3_video_gt.py --epi "$epi" --K 12 \
      --out "$out" --split "$split" --window_mode "$wm" >> logs/vgen_pi3_v2.log 2>&1 < /dev/null &
    i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
  done
}
for epi in $1; do gen_one "$epi" train; done
for epi in $2; do gen_one "$epi" heldtask; done    # held TASK (unseen target noun)
for epi in $3; do gen_one "$epi" heldseed; done    # held SEED (SEEN noun, unseen episode -> scene generalization)
wait
NALL=$(ls data/libero_pi3_v2/*.pt 2>/dev/null | wc -l)
NTRAIN=$(ls data/libero_pi3_v2/*_train.pt 2>/dev/null | wc -l)
echo "[gen-v2] done: $NALL clips ($NTRAIN train) $(date)" >> "$L"
grep -hE "\[dbg\] (manipulated|ent1:|hole)" logs/vgen_pi3_v2.log | tail -100 >> "$L"
