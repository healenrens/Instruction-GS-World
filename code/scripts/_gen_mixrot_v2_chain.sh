#!/bin/bash
# Chain the remaining mix_rot_v2 gen invocations after INV1 (PickCube@rotate x170) finishes.
# Each invocation is 4-way sharded across GPU0-3 and we wait for all shards before the next.
# Per the vetted scale-up spec (workflow whdi98cwb). All write to the SAME --out (resumable, --overwrite 0).
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
OUT=data/mix_rot_v2
COMMON="--K 16 --cam 512 --control_freq 20 --min_val_psnr 16 --min_movefrac 0.02 --out $OUT --overwrite 0"

run_sharded () {  # $1 = label, rest = gen args
  local label=$1; shift
  for sh in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$sh .venv/bin/python code/scripts/gen_sim_dataset.py "$@" --shard $sh --nshards 4 \
      > logs/gen_${label}_sh$sh.log 2>&1 &
  done
  wait
  echo "[$label] done: $(ls $OUT/*.pt 2>/dev/null | wc -l) total clips"
}

# wait for INV1 (already running) to finish
while pgrep -f "[g]en_sim_dataset" >/dev/null; do sleep 30; done
echo "INV1 done: $(ls $OUT/pickcuberot_*.pt 2>/dev/null | wc -l) rotate clips"

# INV2: translation backbone — PickCube + PushCube auto, varied mid-episode windows, heldseed 15%
run_sharded inv2 --tasks "PickCube-v1,PushCube-v1" --seeds 110 --seed_base 1000 --held_task none \
  --held_seed_frac 0.15 --window_sec 4 --random_start 1 $COMMON

# INV2b: StackCube auto, held ENTIRELY as cross-task translation generalization (bounded to 40 seeds)
run_sharded inv2b --tasks "StackCube-v1" --seeds 40 --seed_base 3000 --held_task StackCube-v1 \
  --window_sec 4 --random_start 1 $COMMON

# INV3: StackCube@rotate, held ENTIRELY as cross-task ROTATION generalization (the strongest test)
run_sharded inv3 --tasks "StackCube-v1@rotate" --seeds 45 --seed_base 7000 --held_task StackCube-v1 \
  --window_sec 0 --start_frac 0.72 --fuse_stride 3 $COMMON

echo "=== ALL GEN DONE ==="
ls $OUT/ | sed -E 's/_s[0-9].*//' | sort | uniq -c
echo "splits: train=$(ls $OUT/*_train.pt 2>/dev/null|wc -l) heldseed=$(ls $OUT/*heldseed*.pt 2>/dev/null|wc -l) heldtask=$(ls $OUT/*heldtask*.pt 2>/dev/null|wc -l)"
