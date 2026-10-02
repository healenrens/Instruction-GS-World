# V69 Stage2 Large: Launch, Deploy and Resume

## Execution Contract

Stage1 is complete for this handoff. These commands do not retrain State.
Current V69 launchers use SwanLab; existing State model/training settings stay
unchanged. Only training scalar metrics are uploaded. Tests, evaluation,
case tables, images, videos and reports remain local. Fresh run names and outputs have a `_swanlab` suffix. Stage2 uses
`--stage dynamics --stage2_preset large
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
the source checkout. Resume also executes the current wrapper's release, not
the checkpoint's old W&B release. Git/network downloads occur only in the sync
and separate dependency-preparation blocks; model execution uses local assets
and offline model-library flags. SwanLab online logging is allowed. Commands run in foreground,
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

### Separate Dependency Preparation and Login

Prepare the SDK separately before model execution; no launcher installs it:

```bash
/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python -m pip install swanlab==0.10.1
```

Provide authentication through the process environment `SWANLAB_API_KEY`, or
perform an interactive login separately. No API key belongs in these commands,
source files or this document:

```bash
/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/swanlab login
```

This independent SDK-only offline test checks training scalar logging and
same-ID resume without importing or running a model. It does not upload
anything or test cloud authentication, and is not a launcher gate:

```bash
bash <<'BASH'
set -euo pipefail
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
REV="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
OUT="${RUNTIME_ROOT}/outputs/v69_swanlab_tracking_offline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"
"${RUNTIME_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/test_swanlab_tracking_v69.py" \
  --out "${OUT}" --swanlab_project instruct-gs-world --swanlab_workspace "" \
  --swanlab_name "$(basename "${OUT}")" --swanlab_mode offline \
  2>&1 | tee "${OUT}/test.log"
BASH
```

Read `OUT/test_report.json` and `OUT/tracking.json`. This test does not establish
cloud authentication or model correctness. Production training initializes
SwanLab normally with the configured account.

Shell logging defaults are `SWANLAB_PROJECT=instruct-gs-world`,
`SWANLAB_WORKSPACE=""` (personal workspace), `SWANLAB_MODE=online`, and
`--swanlab_name` from RUN_NAME or the OUT basename. Native modes are `online`,
`offline`, `local`, and `disabled`. Python keeps old `--wandb_*` CLI aliases
for queued commands; `WANDB_ENTITY` is never mapped to a SwanLab workspace.
Choose an explicit `SWANLAB_WORKSPACE` only for the intended workspace.
Non-training online requests do not start a cloud experiment; their computed
results are still written locally. Test/evaluation wrappers disable tracking.

This publishes a new release. It does not edit old cd865/ab12 releases or
change the code used by an already-running W&B job.

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
export SWANLAB_PROJECT=instruct-gs-world
export SWANLAB_WORKSPACE=""
RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${RUNTIME_ROOT}" TEST_GPU=0 \
  SWANLAB_MODE=disabled ENCODER_FRAME_BATCH=2 SEED=17 \
  bash "${ROOT}/code/scripts/test_object_video_stage2_v69.sh"
BASH
```

The wrapper creates a timestamped test OUT below
`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/` and prints its full path.
Its console log is `OUT/test.log`; results are `OUT/test_report.json` and
`OUT/dynamics_resume_comparison.json`. These results are local only and are not
uploaded to SwanLab. The default
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
export SWANLAB_PROJECT=instruct-gs-world
export SWANLAB_WORKSPACE=""
RUNTIME_ROOT="${RUNTIME_ROOT}" VENV_ROOT="${RUNTIME_ROOT}" \
  RUN_NAME=object_video_v69_stage2_large_seed17_state6000_swanlab \
  OUT="${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_seed17_state6000_swanlab" \
  NPROC_PER_NODE=8 BATCH_PER_GPU=4 TARGET_GLOBAL_BATCH=256 \
  ENCODER_FRAME_BATCH=2 WORKERS_PER_RANK=2 STEPS=30000 LR=0.0002 SEED=17 \
  SAVE_EVERY=2500 RECOVERY_EVERY=250 LOG_EVERY=20 SWANLAB_MODE=online \
  bash "${ROOT}/code/scripts/train_object_video_stage2_v69.sh"
BASH
```

Effective batch is 256 via 8 gradient-accumulation microsteps across eight
GPUs at batch 4 per GPU (`BATCH_PER_GPU` remains overridable). The run does not silently reduce batch/world size or
download assets. The default OUT is reserved for this run; choose a new OUT and
RUN_NAME for a different fresh experiment, rather than reuse its dataset or
optimizer state. For interruption recovery use the next block.

## Independent Strict Resume

The path selects this Stage2 run's own `latest.pt`, not State's latest. The
block selects the currently deployed SwanLab-capable release, independently
of previous shell variables. It does not switch back to a saved W&B release.

```bash
bash <<'BASH'
set -euo pipefail
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export RESUME="${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_seed17_state6000_swanlab/latest.pt"
REV="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
unset CUDA_VISIBLE_DEVICES SOURCE_REVISION STOP_AFTER WANDB_RUN_ID WANDB_RESUME
unset OUT RUN_NAME STATE_CHECKPOINT MODEL_CONFIG MANIFEST POSTERIOR_GEOMETRY
export SWANLAB_PROJECT=instruct-gs-world
export SWANLAB_WORKSPACE=""
export SWANLAB_MODE=online
bash "${ROOT}/code/scripts/train_object_video_stage2_v69.sh"
BASH
```

The wrapper restores world size, OUT and worker count from the checkpoint,
but retains its own release ROOT and passes that release revision to Python.
Python preserves the original checkpoint `source_revision` as training provenance
and records the current code separately as `execution_revision`. It restores
the saved config, manifest snapshot, encoder
source/backbone, batch/accumulation, optimizer, scheduler, sampler cursor,
per-rank RNG and complete model/optimizer state. Newly exported hyperparameters
do not replace saved training arguments: an old B2/accum16 checkpoint resumes
with B2/accum16, not the fresh-run B4/accum8 defaults. Logging can migrate from
an old W&B checkpoint to SwanLab; W&B IDs are not reused as SwanLab identities.
The saved run directory and assets remain required. This is continuation of
Stage2, not a new State migration.

For an existing W&B Stage2 run, use the same resume block but set RESUME to
`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_stage2_large_seed17_state6000/latest.pt`.
Resume retains that checkpoint's original OUT; it does not fork a new run into
the `_swanlab` directory. Stop the previous writer before resuming the same
checkpoint/OUT. A fresh `_swanlab` run is separate and does not compete for the
old run's files.

## Launcher CLI and Artifacts

Training delegates to `code/scripts/train_object_video_sequence_v69.py` with:

```text
--stage dynamics --stage2_preset large --state_checkpoint <assessed snapshot>
--dynamics_checkpoint_blocks --config ""
--manifest <State run>/dataset.json --out <Stage2 OUT>
--encoder dinov3_vitl16 --encoder_repository <State run>/encoder_source
--encoder_weights <local DINOv3 weights> --encoder_frame_batch 2
--resume "" --source_revision <release revision> --posterior_geometry inherit
--batch 4 --global_batch 256 --workers 2 --steps 30000 --stop_after 0
--lr 0.0002 --seed 17 --log_every 20 --save_every 2500 --recovery_every 250
--swanlab_project instruct-gs-world --swanlab_workspace ""
--swanlab_name object_video_v69_stage2_large_seed17_state6000_swanlab --swanlab_mode online
```

Optional `DETERMINISTIC=1` appends `--deterministic`. Test delegates to
`code/scripts/test_object_video_stage2_v69.py` with the same manifest,
checkpoint, large preset, config, encoder, seed and source revision, with
tracking disabled. It adds `--stage dynamics` but does not take production optimizer
settings. The Python test supplies its own short full-model execution budget.

Production artifacts in the Stage2 OUT include `train.log`, `run.json`,
`model_inventory.json`, `dataset.json`, `encoder_source/`, `progress.json`,
`metrics.jsonl`, `cases_rankXXXX.jsonl`, `latest.pt` and `step_XXXXXXX.pt`.
The last two are Stage2 checkpoints and must not be used as State baselines.
Actual trainable parameter counts, migration evidence and complete test
reports are runtime outputs, not results of shell syntax validation.

SwanLab uploads only scalar training metrics and training configuration.
Loss components, gradients, learning rate, memory and timing metrics keep their
original names and explicit steps. Case records stay in local JSONL files;
there is no case-table, image, video, report, dataset or checkpoint upload.
Training checkpoint/resume behavior and local evaluation outputs are unchanged.
