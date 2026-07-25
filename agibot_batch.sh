#!/bin/bash
set -u
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/mnt/pfs/public/xuhaoming/.cache
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
PY=.venv/bin/python
# args: task episode win_start win_frac
run(){ echo "===== $1 ep$2 (gpu$GPU) ====="; $PY code/scripts/agibot_spatrack_eval.py --task $1 --episode $2 --win_start $3 --win_frac $4 --kf 12 --grid 40 2>&1 | grep -E "\[agibot\]|SUMMARY|Error|Traceback|FileNotFound" ; }
for spec in "$@"; do
  IFS=, read t e ws wf <<< "$spec"
  run "$t" "$e" "$ws" "$wf"
done
echo "ALL_DONE_GPU$GPU"
