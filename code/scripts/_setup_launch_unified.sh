#!/bin/bash
# Set up 3 train-data dirs + launch 3 unified training arms on the scaled mix_rot_v2 dataset.
#   nat (GPU0): natural mix (rotate ~44% of train) — realistic balanced model
#   os  (GPU1): rotate clips duplicated on disk (rotate ~61%) — does boosting rotation help w/o wrecking translation
#   rot (GPU2): rotate clips ONLY (100%) — rotation ceiling / dilution reference
# All eval on the SAME held splits living in data/mix_rot_v2. GPU3 reserved for eval/_gps_rotread.
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
P=$(pwd)

# real held clips (sim-to-real translation eval only; not *_train.pt so trainer ignores)
ln -sf $P/data/mix_v15/*heldreal*.pt  data/mix_rot_v2/ 2>/dev/null
ln -sf $P/data/mix_v15/*heldgoal*.pt  data/mix_rot_v2/ 2>/dev/null

# oversampled train dir: all train + a duplicate symlink of each rotate train clip
mkdir -p data/mix_rot_v2_os
ln -sf $P/data/mix_rot_v2/*_train.pt data/mix_rot_v2_os/ 2>/dev/null
for f in data/mix_rot_v2/pickcuberot_*_train.pt; do
  b=$(basename "$f" _train.pt); ln -sf $P/$f data/mix_rot_v2_os/${b}_dup_train.pt
done

# rotonly train dir: rotate train clips only
mkdir -p data/rot_only_v2
ln -sf $P/data/mix_rot_v2/pickcuberot_*_train.pt data/rot_only_v2/ 2>/dev/null

echo "nat train=$(ls data/mix_rot_v2/*_train.pt|wc -l)  os train=$(ls data/mix_rot_v2_os/*_train.pt|wc -l)  rotonly train=$(ls data/rot_only_v2/*_train.pt|wc -l)"

TA="--feat_source dino --dino_imgsize 518 --geom_mode xyz --L 1024 --steps 3000 --w_mag 0 --w_jepa 0.5 --w_sigreg 0.05 --w_ground 1.0 --lr 3e-4 --save_every 750 --log_every 20"
nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python code/scripts/train_gpstoken_wm.py --data data/mix_rot_v2    --out checkpoints/gpswm_uni_nat  $TA > logs/gpswm_uni_nat.log  2>&1 &
echo "launched uni_nat (GPU0) pid $!"
nohup env CUDA_VISIBLE_DEVICES=1 .venv/bin/python code/scripts/train_gpstoken_wm.py --data data/mix_rot_v2_os --out checkpoints/gpswm_uni_os   $TA > logs/gpswm_uni_os.log   2>&1 &
echo "launched uni_os  (GPU1) pid $!"
nohup env CUDA_VISIBLE_DEVICES=2 .venv/bin/python code/scripts/train_gpstoken_wm.py --data data/rot_only_v2   --out checkpoints/gpswm_rotonly2 $TA > logs/gpswm_rotonly2.log 2>&1 &
echo "launched rotonly (GPU2) pid $!"
