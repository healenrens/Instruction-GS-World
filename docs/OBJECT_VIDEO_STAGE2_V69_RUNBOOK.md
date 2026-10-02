# V69 Stage2 Large: Launch, Deploy and Resume

## Execution Contract

Stage1 is complete for this handoff. These commands do not retrain State.
The dedicated Stage2 wrappers leave the existing State launchers and defaults
unchanged. Stage2 uses `--stage dynamics --stage2_preset large
--dynamics_checkpoint_blocks`: frozen State,
EMA State and readout at width 512; internal width 1024, 16 heads, 12 Dynamics
layer groups, 4 Posterior layers and per-query continuous effect `4 x 64`.
Static parameter-shape accounting gives Posterior 51,103,872 and Dynamics
305,594,882 parameters, totaling 356,698,754 trainable parameters. The GPU
test records the actual `model_inventory.json` parameter counts.

The selected State is the exact snapshot assessed at step 6000 in W&B run
`gaiahhsw`, not the rolling training checkpoint:

```text
STATE_CHECKPOINT=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_state_change_held_aed14bb_20261001_235841/checkpoint_snapshot.pt
MANIFEST=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/dataset.json
ENCODER_REPOSITORY=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/encoder_source
ENCODER_WEIGHTS=/mnt/pfs/public/xuhaoming/instruct_gs_world/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

These baseline paths come from the user's W&B evaluation provenance; the
server filesystem was not inspected for this delivery. The evaluator saves
`checkpoint_snapshot.pt` before loading it. The training saver names numbered
checkpoints `step_{step:07d}.pt`, but no numbered step6000 file is assumed here.
Python owns State/EMA/readout migration and fresh Stage2 initialization; the
State optimizer is not resumed. New Stage2 training snapshots its own dataset
and encoder source into its separate OUT.

Runtime layout remains:

```text
/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/object_video_sequence_v69/DEPLOYED_REVISION
/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/object_video_sequence_v69/releases/<revision>/SOURCE_REVISION
/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/object_video_sequence_v69/releases/<revision>/configs/
```

Each block is independent of variables exported by another block. Test and
fresh training read the deployed pointer once and execute that release, not
the source checkout. Resume selects the checkpoint's saved release. Only the
sync block uses Git/network downloads; model execution uses local assets and
offline model-library flags. W&B online is allowed. Commands run in foreground,
and wrappers use `tee` with `pipefail`. The `bash` blocks keep `set -e` scoped
away from the interactive SSH shell; failures return normally to that shell.

## 1. Remote Pull and Deploy

Run after the integrated Python/config/shell change is published by the parent.
This is the only block that synchronizes remote source. It does not install
dependencies, download model weights, prepare a new dataset or run tests.

```bash
bash <<'BASH'
set -euo pipefail
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git fetch origin refs/heads/codex/object-video-sequence-v69
git switch --detach FETCH_HEAD
export SOURCE_REVISION="$(git rev-parse HEAD)"
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world \
  bash /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/deploy_object_video_stage2_v69_runtime.sh
BASH
```

The deploy wrapper copies `code/igsw`, `code/scripts`, the complete `configs/`
directory and the runbooks into the shared release, writes `SOURCE_REVISION`,
then publishes `DEPLOYED_REVISION`. It does not alter the old State wrappers.
No config file is required by default: the large preset defines the model.
An explicit `MODEL_CONFIG` override must name a local deployed file (the
wrappers pass it to `--config`); neither launcher fetches configurations.

## 2. Single-GPU Real-Checkpoint Full Stage2 Test

This calls the dedicated Stage2 test, not the old two-stage test which trains
State. It exercises the large Stage2 with the assessed State snapshot and real
clips, including interrupted/resumed and uninterrupted Stage2 paths. It is not
an eight-GPU resume comparison or a held scientific evaluation.

```bash
bash <<'BASH'
set -euo pipefail
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
REV="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
unset SOURCE_REVISION RESUME STAGE STOP_AFTER STATE_CHECKPOINT MODEL_CONFIG MANIFEST ENCODER ENCODER_REPOSITORY ENCODER_WEIGHTS
unset RUN_NAME OUT WANDB_RUN_ID WANDB_RESUME POSTERIOR_GEOMETRY
export WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${RUNTIME_ROOT}" TEST_GPU=0 \
  WANDB_MODE=online ENCODER_FRAME_BATCH=2 SEED=17 \
  bash "${ROOT}/code/scripts/test_object_video_stage2_v69.sh"
BASH
```

The wrapper creates a timestamped test OUT below
`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/` and prints its full path.
Its console log is `OUT/test.log`; results are `OUT/test_report.json` and
`OUT/dynamics_resume_comparison.json`, also uploaded to W&B. The default
test executes a two-update interrupted/resumed path and a two-update
uninterrupted path, with full 16+25 frames and batch 1. No result is claimed
until this command completes and the generated report is read.

## 3. Eight-GPU Foreground Training

Fresh Stage2 output, independent of the old State run and the single-GPU test:

```bash
bash <<'BASH'
set -euo pipefail
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
REV="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
unset CUDA_VISIBLE_DEVICES SOURCE_REVISION RESUME STAGE STOP_AFTER STATE_CHECKPOINT MODEL_CONFIG MANIFEST
unset ENCODER ENCODER_REPOSITORY ENCODER_WEIGHTS POSTERIOR_GEOMETRY DETERMINISTIC WANDB_RUN_ID WANDB_RESUME
export WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${RUNTIME_ROOT}" \
  RUN_NAME=object_video_v69_stage2_large_seed17_state6000 \
  OUT="${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_seed17_state6000" \
  NPROC_PER_NODE=8 BATCH_PER_GPU=2 TARGET_GLOBAL_BATCH=256 \
  ENCODER_FRAME_BATCH=2 WORKERS_PER_RANK=2 STEPS=30000 LR=0.0002 SEED=17 \
  SAVE_EVERY=2500 RECOVERY_EVERY=250 LOG_EVERY=20 WANDB_MODE=online \
  bash "${ROOT}/code/scripts/train_object_video_stage2_v69.sh"
BASH
```

Effective batch is 256 via 16 gradient-accumulation microsteps across eight
GPUs at batch 2 per GPU. The run does not silently reduce batch/world size or
download assets. The default OUT is reserved for this run; choose a new OUT and
RUN_NAME for a different fresh experiment, rather than reuse its dataset or
optimizer state. For interruption recovery use the next block.

## Independent Strict Resume

The path selects this Stage2 run's own `latest.pt`, not State's latest. This
block does not depend on `DEPLOYED_REVISION` or any previous shell variables.

```bash
bash <<'BASH'
set -euo pipefail
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export RESUME="${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_seed17_state6000/latest.pt"
REV="$("${VENV_ROOT}/.venv/bin/python" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",mmap=True,weights_only=False); print(c["args"]["source_revision"])' "${RESUME}")"
ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
unset CUDA_VISIBLE_DEVICES SOURCE_REVISION STOP_AFTER WANDB_RUN_ID WANDB_RESUME
export WANDB_MODE=online
bash "${ROOT}/code/scripts/train_object_video_stage2_v69.sh"
BASH
```

The wrapper restores world size, source revision, OUT and worker count from the
checkpoint. Python restores the saved config, manifest snapshot, encoder
source/backbone, batch/accumulation, optimizer, scheduler, sampler cursor,
per-rank RNG and W&B run ID. Newly exported hyperparameters do not replace
saved training arguments. The original release and run directory remain
required. This is continuation of Stage2, not a new State migration.

## Launcher CLI and Artifacts

Training delegates to `code/scripts/train_object_video_sequence_v69.py` with:

```text
--stage dynamics --stage2_preset large --state_checkpoint <assessed snapshot>
--dynamics_checkpoint_blocks --config ""
--manifest <State run>/dataset.json --out <Stage2 OUT>
--encoder dinov3_vitl16 --encoder_repository <State run>/encoder_source
--encoder_weights <local DINOv3 weights> --encoder_frame_batch 2
--resume "" --source_revision <release revision> --posterior_geometry inherit
--batch 2 --global_batch 256 --workers 2 --steps 30000 --stop_after 0
--lr 0.0002 --seed 17 --log_every 20 --save_every 2500 --recovery_every 250
--wandb_project instruct-gs-world
--wandb_entity healenrenss-university-of-chinese-acadmic-and-science
--wandb_name object_video_v69_stage2_large_seed17_state6000 --wandb_mode online
```

Optional `DETERMINISTIC=1` appends `--deterministic`. Test delegates to
`code/scripts/test_object_video_stage2_v69.py` with the same manifest,
checkpoint, large preset, config, encoder, seed, source revision and W&B
arguments. It adds `--stage dynamics` but does not take production optimizer
settings. The Python test supplies its own short full-model execution budget.

Production artifacts in the Stage2 OUT include `train.log`, `run.json`,
`model_inventory.json`, `dataset.json`, `encoder_source/`, `progress.json`,
`metrics.jsonl`, `cases_rankXXXX.jsonl`, `latest.pt` and `step_XXXXXXX.pt`.
The last two are Stage2 checkpoints and must not be used as State baselines.
Actual trainable parameter counts, migration evidence and complete test
reports are runtime outputs, not results of shell syntax validation.
