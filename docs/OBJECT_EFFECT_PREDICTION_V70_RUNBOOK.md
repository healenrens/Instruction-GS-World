# V70: Language-Conditioned Object Effect Prediction

## Decision

V70 predicts the fixed V69 Stage 2 Posterior's pre-tanh mean from observed video and a real instruction. It does not change the State representation or the effect dimensions. It does not predict robot actions.

- History: 16 frames over 3 seconds. Targets: the existing 25-frame, 5-second future.
- Effect: `[B,16,4,64]`. Four tokens jointly describe the complete future, not four time segments.
- Qwen3-VL-4B-Instruct: frozen vision and mergers; full language-Transformer tuning, including embeddings and final norm. No LoRA and no text-generation loss.
- Effect expert: 12 layers, width 1024, 16 heads, SwiGLU intermediate 4096; **308,660,288 parameters**, counted from the implementation.
- Every expert block reads query-local history, reads VLM context, then exchanges information across effect tokens. No learned query-index semantic embedding.
- Flow matching predicts `mean - noise` at `(1-tau)*noise + tau*mean`; inference uses ten Euler steps followed by the existing `tanh`.
- The original Dynamics and readout remain the only decoder in evaluation.

This decision replaces the earlier LoRA suggestion in ledger section 15.58. It does not establish that the Stage 2 representation has passed a new scientific evaluation.

## Data Comes First

The server's current Stage 2 `run.json` points to:

`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/dataset.json`

The read-only SWXC audit confirmed **17,816 clips and unique episodes**, with 16,034 train and 1,782 held. The earlier 20k name is a generation target, not the actual collection size.

| Source | Train | Held | Total |
|---|---:|---:|---:|
| Agibot | 3,600 | 400 | 4,000 |
| Bridge | 1,867 | 208 | 2,075 |
| HY | 3,600 | 400 | 4,000 |
| RoboMIND | 3,600 | 400 | 4,000 |
| RoboTwin | 3,367 | 374 | 3,741 |

The full metadata audit completed on 2026-10-06 through the authorized SWXC Codex chat and port-8600 server. It found the following **text-and-origin coverage**, not proof of semantic alignment to each five-second window:

| Source | Traced natural text | Train / Held | Excluded at this step |
|---|---:|---:|---:|
| Agibot | 4,000 | 3,600 / 400 | 0 |
| Bridge | 2,002 | 1,800 / 202 | 73 |
| HY | 3,393 | 3,053 / 340 | 607 |
| RoboMIND | 3,960 | 3,565 / 395 | 40 |
| RoboTwin | 3,741 | 3,367 / 374 | 0 |
| Total | 17,096 | 15,385 / 1,711 | 720 |

Of these, 9,962 texts were already embedded as `group` and exactly matched original episode metadata; 7,134 required original HY/RoboTwin metadata. Bridge excludes 67 `unknown task`, five reviewed garbled strings and one underspecified `take`; RoboMIND excludes 20 `xxxx` and 20 ambiguous `putegg`. HY excludes 591 video-offset mismatches and 16 episode-contract mismatches, concentrated in table_002 (512) and table_004 (95). No nearest-episode substitution is used.

HY needs additional care: 889 verified clips contain several frame-level task IDs, and episode-level text can describe a different action from the frame-level text. Text availability is therefore **not** the final usable-window count. HY uses the exact history-anchor frame's original task text and records its contiguous annotation interval; episode text is retained as evidence, not silently substituted. A subtask label is not a claim that the whole five-second future belongs to that subtask. Agibot episode goals stay episode-level in the first run; 3,986 clips also have overlapping timed action text, which is a separate observation.

The authoritative audit files are `outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.json` and `.jsonl` under the runtime root. There are 17,816 per-clip JSONL records. Natural-language-looking `group` values require provenance. Taskset identifiers and hashes are never instructions.

The original-video join uses source, raw path, frame offset, frame count and frame rate, not the reindexed episode number. Timed instructions and episode goals retain different provenance. Unsupported source schemas are reported, not guessed. No tracker or SAM is rerun.

Each usable clip yields at most four uniform legal windows. The 3s/5s sampling does not depend on future motion. Existing train episodes stay train; existing held episodes are split deterministically into diagnostic and final-test episodes. All windows of an episode remain together.

Offline labels store mean/logvar, history query coordinates/features/validity, exact frame indices, teacher snapshot metadata and language provenance. Dense patches and RGB are not copied. Training rereads only the history frames.

## 1. Sync and Dependencies

This block is for a preparation machine that may access GitHub and Hugging Face. None of the later runtime scripts fetch code or models.

```bash
(
  cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source &&
  git fetch origin refs/heads/codex/language-object-effect-v70 &&
  git switch --detach FETCH_HEAD &&
  export SOURCE_REVISION="$(git rev-parse HEAD)" &&
  export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world &&
  bash code/scripts/deploy_language_object_effect_v70_runtime.sh &&
  "${RUNTIME_ROOT}/.venv/bin/python" -m pip install -r code/requirements-v70.txt
)
```

Model download is a separate preparation action. It does not need to run again when the local model is already complete.

```bash
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${RUNTIME_ROOT}/.venv/bin/hf" download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "${RUNTIME_ROOT}/models/Qwen3-VL-4B-Instruct"
```

## 2. Read-Only Language Audit

Run this first and inspect the per-source counts and evidence. It does not load a neural model or use a GPU.

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=
"${RUNTIME_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/inspect_language_sources_v70.py" \
  --stage2_run "${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_b32_seed17_20261003_215438" \
  --verified_trace "${RUNTIME_ROOT}/outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.jsonl" \
  --output "${RUNTIME_ROOT}/outputs/v70_language_audits/current_collection.json"
```

Outputs: `current_collection.json`, a sibling CSV with per-clip evidence, and the CLI-reported source-schema/sidecar outputs. Do not treat an absent supported schema as evidence that the original source has no language.

## 3. Fix the Teacher and Prepare Windows

Teacher selection update (2026-10-06): the user requested step 8,500 instead of 7,500. The server has no `step_0008500.pt`; CPU reading of `latest.pt` returned step 8,750, matching `progress.json`. The 7,500-step path below is the earlier example, not the newly requested teacher. Do not run this preparation block until the user selects an available snapshot. No 8,500-step snapshot has been fabricated and no rolling checkpoint has been copied. Step 8,750 is the proposed alternative, pending the user's decision.

Set `STAGE2_CHECKPOINT` to a specific numbered Stage 2 checkpoint. Do not leave it pointing at a moving `latest.pt`. The snapshot deliberately omits optimizer state. Its output uses exclusive creation; choose a new `DATA_ROOT` for another teacher or another language policy.

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export DATA_ROOT="${RUNTIME_ROOT}/data/language_object_effect_v70"
export STAGE2_MANIFEST="${RUNTIME_ROOT}/outputs/object_video_v69_state_seed17_40e4586_20260929_003931/dataset.json"
export MODEL_PATH="${RUNTIME_ROOT}/models/Qwen3-VL-4B-Instruct"
# Earlier example only; see the pending teacher-selection update above.
export STAGE2_CHECKPOINT="${RUNTIME_ROOT}/outputs/object_video_v69_stage2_large_b32_seed17_20261003_215438/step_0007500.pt"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${RUNTIME_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/freeze_object_teacher_v70.py" \
  --checkpoint "${STAGE2_CHECKPOINT}" --output "${DATA_ROOT}/teacher.pt" &&
"${RUNTIME_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/prepare_language_manifest_v70.py" \
  --manifest "${STAGE2_MANIFEST}" --teacher_checkpoint "${DATA_ROOT}/teacher.pt" \
  --audit "${RUNTIME_ROOT}/outputs/v70_language_audits/current_collection.json" \
  --verified_trace "${RUNTIME_ROOT}/outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.jsonl" \
  --tokenizer_path "${MODEL_PATH}" --output "${DATA_ROOT}/language_manifest.json"
```

The real tokenizer limits instruction content to 512 tokens in preparation. Overlong instructions are reported and excluded, never truncated. Official chat/video timestamps add prompt tokens; their counts and any prompt-budget overflow are logged separately.

## 4. Export Labels

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_ROOT="${RUNTIME_ROOT}/data/language_object_effect_v70"
export CUDA_VISIBLE_DEVICES=0
export EXPORT_GPUS=1
export DINO_FRAME_BATCH=8
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" export
```

Successful files can be reused after an interrupted export. Decode failures are listed; `labeled_manifest.json` includes only windows with label files. The language report and final labeled counts are separate. This is label preparation, not a model-capability result.

## 5. Single-GPU Full Test

The user changed testing from four GPUs to **one GPU**. This test uses the complete 4B conditioner and 308.7M expert, real windows, one uninterrupted run and a matching interrupted/resumed run. It also measures future swapping in the offline teacher, records actual parameter updates and peak memory, and does not require every intermediate tensor to be BF16.

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_ROOT="${RUNTIME_ROOT}/data/language_object_effect_v70"
export MODEL_PATH="${RUNTIME_ROOT}/models/Qwen3-VL-4B-Instruct"
export CUDA_VISIBLE_DEVICES=0
export TEST_BATCH=1
export TEST_OUT="${RUNTIME_ROOT}/outputs/v70_tests/full_single_gpu_$(date +%Y%m%d_%H%M%S)"
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" test
```

Results: `${TEST_OUT}/resume_report.json`, `teacher_future_swap.json`, and each run's `module_update_report.json`, `run.json` and traces. Full optimizer checkpoints are large; this test retains one rolling checkpoint per test run, not a second copy of the teacher. Test batch 1 is not a claim that eight-GPU training batch 4 has been measured.

## 6. Eight-GPU Foreground Training

Submit this complete block to the eight-GPU job. It uses shared, already prepared files. It contains no Git, model download, `nohup`, or background process.

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_ROOT="${RUNTIME_ROOT}/data/language_object_effect_v70"
export MODEL_PATH="${RUNTIME_ROOT}/models/Qwen3-VL-4B-Instruct"
export MODE=flow
export RUN_NAME=language_object_effect_v70_flow_seed17_run1
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export BATCH_PER_GPU=4 GRAD_ACCUM=8 STEPS=10000 WORKERS_PER_RANK=2
export DINO_FRAME_BATCH=8
export SWANLAB_PROJ_NAME=instruct-gs-world
export SWANLAB_MODE=online
# SWANLAB_API_KEY must already be provided by the job environment.
unset RESUME SWANLAB_PROJECT SWANLAB_WORKSPACE SWANLAB_RUN_ID SWANLAB_RESUME
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" train
TRAIN_RC=$?
echo "TRAIN_RC=${TRAIN_RC}"
```

All eight visible GPUs must be provided by the scheduler. Do not inherit `CUDA_VISIBLE_DEVICES=0` from a single-GPU testing shell. Effective batch is `4 * 8 * 8 = 256`. The regression control uses the same complete block with `MODE=regression` and a distinct run/output name.

## 7. Explicit Resume

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export RUN_NAME=language_object_effect_v70_flow_seed17_run1
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export RESUME="${OUT}/latest.json"
export SWANLAB_MODE=online
unset SWANLAB_PROJECT SWANLAB_WORKSPACE SWANLAB_RUN_ID SWANLAB_RESUME
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" resume
TRAIN_RC=$?
echo "TRAIN_RC=${TRAIN_RC}"
```

The checkpoint restores numerical settings, model path, dataset, optimizer, scheduler, cursor, rank RNG and SwanLab run. Use the same eight-GPU topology. Changing topology or numerical settings is a separate experiment, not equivalent resume. New runs do not infer resume from old output contents.

## 8. Local Paired Evaluation

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/language_object_effect_v70/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/language_object_effect_v70/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_ROOT="${RUNTIME_ROOT}/data/language_object_effect_v70"
export RUN_NAME=language_object_effect_v70_flow_seed17_run1
export CHECKPOINT="${RUNTIME_ROOT}/outputs/${RUN_NAME}/latest.json"
export CUDA_VISIBLE_DEVICES=0
export PARTITION=diagnostic EVAL_CASES=128 EVAL_SAMPLES=4
export EVAL_OUT="${RUNTIME_ROOT}/outputs/v70_evaluation/${RUN_NAME}"
unset CONFLICTS
bash "${ROOT}/code/scripts/run_language_object_effect_v70.sh" evaluate
```

Read `${EVAL_OUT}/index.html`, `report.json`, and `trajectories.jsonl`. Videos: prediction red, tracker measurement green. All results remain on disk; SwanLab uploads training scalars only. These are tracker-derived measurements, not human-verified object identity ground truth.

Primary result is one fixed-seed sample. Expected error over four fixed samples and best-of-four oracle coverage are separate. Ratios are computed per trajectory only when its measured motion is at least `motion_floor_px` (default 1 native pixel, an explicit reporting floor, not a calibrated visibility threshold). An invisible final target has no endpoint error; no earlier frame substitutes for it.

Optional `CONFLICTS` is a local JSONL keyed by `window_id`, containing a different `instruction` and independently specified `targets`: `point_id`, `target_xy_px`, `tolerance_px`. Without those targets, conflict predictions are saved but have no success claim. Original-video error is not the score for a conflicting instruction. Run the deterministic regression experiment through this same evaluator before claiming a benefit from distribution modeling.

## Verification Status

Local CPU integration completed with the real training loop, AdamW, DCP, sampler and scheduler on a small injected fixture. Interrupted/resumed parameters matched uninterrupted parameters exactly (max difference 0), traces matched, frozen vision stayed unchanged and language/expert parameters updated. The official small Qwen processor/model interface was exercised on Transformers 4.57.1 and 5.18.0; this is not a full 4B model execution.

The server language audit above is complete. The committed CLI at `5dc062a3e917461a48317de9738ed4a6d6aef760` also ran through SWXC on the port-8600 server: exit 0, all 17,816 entries, 9,962 embedded, 7,134 joinable and 720 excluded, with zero teacher files loaded. `current_collection.json` and `.csv` already exist at the paths in section 2.

The full 4B single-GPU test and eight-GPU FSDP run have **not** been executed by local CPU tests. Single-GPU memory fit, production throughput and complete CUDA numerical replay remain to be measured by section 5. The numbered 7,500-step teacher is a fixed preparation choice, not a new claim that its scientific quality is sufficient. The new diagnostic/test split isolates V70 episodes, but earlier experiments' inspected held cases have not yet been cross-checked against the candidate test set; do not call it a historically untouched test set until that check is done.
