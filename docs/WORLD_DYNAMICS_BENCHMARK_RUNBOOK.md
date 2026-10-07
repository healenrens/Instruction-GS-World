# World Dynamics Representation Benchmarks

## Question and Boundaries

Does the frozen V69 State plus its **observed-transition effect** support physical outcome prediction and video-change recognition better than State alone and pretrained visual tokens, at a fixed shallow readout capacity?

This is an external task-utility experiment. It does not establish object identity, robot policy quality, or deployment language conditioning. No Stage 1/2/3 module is trained or changed. RoboTwin is not the primary benchmark. MOVi remains an optional diagnostic protocol, not a third training program.

Four representations: raw DINOv3 ViT-L/16 patch tokens; official V-JEPA2 ViT-L clip tokens; frozen V69 State; frozen V69 State plus `tanh(posterior mean)` over **already observed** intervals. V-JEPA2.1 is not used or relabeled. Posterior/Dynamics from an immutable Stage 2 snapshot are distinct from a language-conditioned predictor. Dynamics, tracker, readout and teacher losses are not loaded by this exporter.

## Official Sources and Protocol Departures

- [Physion v1 dataset and model protocol](https://github.com/cogtoolslab/physics-benchmarking-neurips2021): eight scenarios. V1.5 is separate and not mixed.
- [Physopt](https://github.com/neuroailab/physopt-physics-benchmarking) and [physics-models](https://github.com/neuroailab/physics-models): official readout is independent logistic regression per scenario. `pDEIT_MLP/physion.yaml` uses `STATE_LEN=7`, `SUBSAMPLE_FACTOR=6`; `pydata.py` with `random_seq=False` therefore exposes input HDF5 frame positions 0,6,...,36. `FROZEN.py` distinguishes `input_states` from later `observed_states`; the latter are NOT allowed in our OCP representation.
- The new matched protocol uses eight unique HDF5 positions `[0,5,10,15,20,25,30,36]`, to accommodate V-JEPA2's tubelet size 2, without extending the pDEIT-derived input boundary. No mapping from HDF5 positions to MP4 positions is inferred. [Physion paper section 2.2](https://arxiv.org/html/2106.08261v3) states that stimuli are 5-10 second movies rendered at **30 fps**; its testing protocol describes an **observed 1.5 second** prefix. These are separate facts: our custom frame36 endpoint is nominally `36/30=1.2` seconds, not a reproduction of the paper's 1.5 second observed protocol. The first real HDF5 has no timestamp/FPS field or attribute, so `times=frame_position/30` is inferred from the paper's movie rate, not measured from the file. Manifest/features/cases retain `time_basis.kind=nominal_frame_time`, `nominal_fps=30`, `measured_clock=false`, the paper URL, and the paper's observed-prefix duration separately. Generator default30 and visualization `BASE_FPS=30` are supporting conventions, not file clock measurements. `target_contacting_zone` over all HDF5 frames supplies the outcome label; only the custom prefix images reach models.
- Official readout training bundles are declared red/yellow by the Physion README. Coloring of downloaded testing HDF5 remains a review item: inspect an actual `_img` before claiming equivalence to `mp4s-redyellow`. Until then all scores are **custom HDF5 pilot**, not PhysionTest-Core reproduction. No unverified recoloring heuristic is applied. A Core MP4 does not replace HDF5 without an explicit verified time mapping and outcome join.
- The 8600 first real file `pilot_it2_collision_yeet_box_1_dis_1_occ_0034.hdf5` has `stimulus_name=train_readout_0034`, 152 frames, target id 2 red and zone id 1 yellow. This confirms coloring for that sample, not all scenarios/splits. `stimulus_name` is metadata, not a unique key. IDs now namespace scenario, original split and relative file path, retaining the original stimulus separately. Rebuild the manifest and use a new attempt after this change; do not reuse caches created under the old stimulus-based IDs/time scale.
- Testing Collide feedback shows that natural RGB does not always identify the queried pair: `pilot_it2_collision_assorted_targets_box_0001.hdf5` has152 frames, `object_ids=[1,2,3]`, target2, zone1, probe3, PNG-encoded `_img`/`_id`, and uint8 segmentation colors `[[241,241,236],[218,76,142],[166,180,55]]`. The parent obtained this remote-observed schema from SWXC actual stdout/receipts, not from user-supplied schema; this implementation agent did not inspect it locally. Artifacts are `logs/world_dynamics_benchmarks_20cd3946/physion_testing_collide_schema.json` and `physion_testing_collide_img0000.png` relative to the runtime root. [Official `_set_segmentation_colors`](https://github.com/neuroailab/tdw_physics/blob/master/tdw_physics/dataset.py#L730-L757) stores colors in `object_ids` order; [official first-frame cue](https://github.com/neuroailab/tdw_physics/blob/master/tdw_physics/dataset.py#L466-L490) indexes target/zone IDs then exact-matches `_id` RGB. Our shared reader applies target-red/zone-yellow tint with alpha0.65 to visible pixels in each **observed prefix frame**, leaving all other pixels unchanged. This is **custom HDF5 pair-cued input**, not official material rerendering. All train/dev/test representations receive the same processed RGB; masks, roles and cue metadata do not enter the encoder or probe as separate inputs. Outcome labels and future masks/images do not affect the cue. Regenerate the Physion manifest and use `attempt: cue01`; SSv2 input/config/cache/attempt remain unchanged.
- [SSv2 official downloads](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads): use authorized archives/URLs. Login/license acceptance remains a human task. Official train is split internally into train/dev by class; official **validation** is the held evaluation partition. Hidden official test is not evaluated or described as labeled. No horizontal flip changes directional classes. Video PTS is retained, short clips use fewer unique frames, and are never repeated to imitate an eight-second robot clip.
- [V-JEPA2 official hub](https://github.com/facebookresearch/vjepa2): `vjepa2_vit_large`, `target_encoder` checkpoint, single official 256 center crop. This is not its published SSv2 multi-view/multi-segment four-block probe protocol. DINO and State retain native geometry; all methods share the same decoded frame indices but their pretrained preprocessing differs and is reported.
- DINO token-budget selection is custom, not a claim about unrestricted raw-backbone performance. Default cap 4096; State retains complete query/carrier groups (normally all 16-frame states), z retains complete query/effect groups. Export artifacts record discarded fractions and actual counts. No arithmetic global pooling is used. Features are cached in FP16 at native width, then zero-padded to 1024 before a common trainable probe; this is capacity matching, not a claim that all representations have equal intrinsic dimension. Raw tokens can cost up to 8MiB per clip, so full SSv2 export can exceed 1TB **per representation**. First run is a bounded pilot; full export needs a separate storage decision.

## Assets and Preparation

The paths below are explicit requirements, not yet a verified remote inventory:

```bash
export ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
export VENV_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export PY="${VENV_ROOT}/.venv/bin/python"
export BENCH_ROOT="${VENV_ROOT}/data/world_dynamics_benchmarks"
export DINO_REPOSITORY=/PATH/TO/official_dinov3
export DINO_WEIGHTS=/PATH/TO/dinov3_vitl16.pth
export VJEPA_REPOSITORY=/PATH/TO/official_vjepa2
export VJEPA_WEIGHTS=/PATH/TO/vjepa2_vitl.pt
export TEACHER_CHECKPOINT=/PATH/TO/fixed_stage2_8750_teacher.pt
export CUDA_VISIBLE_DEVICES=0
```

Only preparation/synchronization may download. Install into the existing environment using uv:

```bash
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
uv pip install --python "${PY}" -r "${ROOT}/code/requirements-world-dynamics-benchmarks.txt"
```

Physion pilot preparation (not the ~1TB dynamics corpus):

```bash
export CONFIG="${ROOT}/code/configs/world_dynamics_benchmarks/physion.yaml"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" plan
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" download
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" inspect /PATH/TO/ONE.hdf5
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" manifest
```

`plan` reports filesystem free bytes and URLs. Default streaming download stops after 32 members per scenario/split; resumed downloads keep existing members. This first-archive pilot is not a random sample of the full corpus. Manifest/dev split is class-stratified; class counts are recorded. Some pilots may still lack outcome classes: expand source acquisition rather than treating their accuracy as representative. Full per-scenario archives are roughly 40GB; change `download_limit` to zero only after disk/data approval. Test-Core's 270MB archive is an optional authorized download-plan entry, not a default 380GB Test-Complete download.

For bounded tar-stream pilots, a destination already containing at least `max_files` complete `.hdf5` files is skipped **before opening the archive URL**. `.partial` files do not count; unlimited `max_files: 0` is not skipped. Remote feedback reported Physion terminal exit1 after384 complete files (six scenarios, both splits,32 each), because the README's old `Rollreadout_HDF5s.tar.gz` URL returns404. SWXC subsequently returned official bucket listing GET200 with actual key `Roll_readout_training_HDF5s.tar.gz`; the bucket key is the current download authority, not a guessed alias. The plan now uses the standard `<scenario>_<kind>_HDF5s.tar.gz` name for Roll as well. The parent read this remote-observed stdout/receipt; it is not user-supplied schema or a new local network verification. Completed rows need not be downloaded again, and HTTP failures remain visible. Support+Roll were continued through an explicit plan without deleting or rerunning the384 files or editing remote source; completion is recorded below.

SSv2 preparation:

```bash
export CONFIG="${ROOT}/code/configs/world_dynamics_benchmarks/ssv2.yaml"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" download --plan /PATH/TO/authorized_downloads.json
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" unpack --destination "${BENCH_ROOT}/ssv2" /PATH/TO/video.zip /PATH/TO/labels.zip
# For a single archive split into ordered parts, add --join-parts instead.
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" inspect "${BENCH_ROOT}/ssv2/labels/train.json"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" manifest
```

Download-plan format is a JSON list: `[{"url":"AUTHORIZED_URL","mode":"file","destination":"/ABS/PATH/archive.zip"}]`. CLI does not bypass authorization or fabricate links. The manifest accepts official `id`/`template` annotations and label mapping; schema inspection precedes adaptation if actual fields differ. The labels archive resolves to `${BENCH_ROOT}/ssv2/labels/{labels,train,validation}.json` in the current configuration. Default pilot is 256 per partition, selected round-robin from class-stratified random queues. Across174 classes this gives only about1-2 training examples per class: it is an engineering pilot and cannot support generalization or model-ranking conclusions. Expand the next pilot from development-set class counts, not test scores or test-selected research directions. Do not expand to full >1TB representation caches in this round. Full export, after a separate data/storage decision, uses `pilot_per_split: 0`, `scope: full`, and a new `attempt`. Reusing an attempt after changing input or checkpoint is not allowed by the experiment protocol.

Remote feedback reports DINO pilot export768/768 exit0 and dev epoch18 accuracy2/256 (0.0078125), near chance1/174 (0.00575). With only1-2 train examples/class this does not determine representation utility. The independent `ssv2_development.yaml` keeps the original preprocessing, assets and probe capacity, first assigns train-derived dev/official-validation held partitions, then samples per class: train16/dev8/test8. If all classes have sufficient videos, this yields2784/1392/1392 clips (5568 total); manifest class counts show actual availability. It writes `manifest_development.json` and `attempt: development01`, leaving the running `ssv2.yaml` pilot01 unchanged. Selection uses class counts and dev evidence, never held test scores. At the8MiB raw-token cap this is approximately43.5GiB per raw representation, or a conservative174GiB ceiling across four capped representations; compact caches are smaller. This is bounded development, not full >1TB export. Use the normal manifest/export/probe/evaluate commands with:

```bash
export CONFIG="${ROOT}/code/configs/world_dynamics_benchmarks/ssv2_development.yaml"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" manifest
```

## Offline Single-GPU Run

Set the complete asset block above and one `CONFIG`. Run in foreground; no Git/model download occurs:

```bash
unset http_proxy https_proxy
for MODEL in dino vjepa2 state state_z; do
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" export --model "${MODEL}"
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model "${MODEL}" --protocol attention
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model "${MODEL}" --protocol attention --split dev
done
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" compare --split dev
```

Physion linear reference, a separate protocol/output directory:

```bash
for MODEL in dino vjepa2 state state_z; do
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model "${MODEL}" --protocol linear
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model "${MODEL}" --protocol linear --split dev
done
```

Reference uses balanced training subsampling, StandardScaler, per-scenario logistic regression, train-only stratified CV for C, and reports class-balanced test accuracy. It flattens 16 selected token+position records, rather than each author's original unrestricted feature layout; therefore **official-style reference**, not paper reproduction. Small pilots require setting `linear_cv` to a fold count supported by each outcome count. No test result selects C, epoch, or config.

Matched probe is one 128-wide attention query plus a small MLP (no deep temporal network); all methods use identical trainable parameter count and train/dev schedule. Epoch selection uses internal dev only. Future targets, labels, task filenames and scenario labels never enter the representation or attention inputs. State+z is a test of observed change encoding, not future prediction supplied by a posterior.

`evaluate` and `compare` default to `--split dev`; these are development-selection results, not held generalization. `probe` only trains and selects best on dev, never evaluates test as a side effect. Existing SSv2 exports and probe checkpoints remain reusable: evaluate/compare dev without rerunning export or probe. After choices are fixed, held evaluation is explicit:

```bash
for MODEL in dino vjepa2 state state_z; do
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model "${MODEL}" --protocol attention --split test
done
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" compare --split test
```

For SSv2, local `test` means official public validation, not the hidden official test. Do not use it to select an architecture, epoch or pilot expansion.

## Resume and Results

Export repeats only missing feature files. Attention resume explicitly uses `--resume` and restores model, optimizer, CPU/CUDA RNG, epoch/batch cursor; existing trace is appended. Linear resume skips complete per-scenario fits. New probe runs create their trace exclusively, avoiding silent overwrite.

```bash
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" export --model state_z
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model state_z --protocol attention --resume
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model state_z --protocol attention --split dev
```

Outputs: `${BENCH_ROOT}/experiments/<benchmark>/<model>/seed17/<attempt>/` containing `features.json`, per-case `.pt`, and `attention/` or `linear/` probe checkpoints, `report_dev.json`/`report_test.json`, `cases_dev.jsonl`/`cases_test.jsonl`, `confusion_dev.csv`/`confusion_test.csv`. Comparisons use `comparison_dev.json`/`comparison_test.json` and split-specific paired cases; dev and test never overwrite each other. Cases include frame indices, time values with their provenance (paper-inferred for Physion, decoded PTS for SSv2), cue metadata, resolution, token retention, probabilities, decode/encoder latency, and readout latency for attention. Wilson CI describes example-level accuracy; per-class/per-scenario counts and balanced accuracy must accompany the aggregate. A missing class is not an evaluated class. No automatic SwanLab/W&B upload or raw data transfer.

Compare per-example paired correctness before aggregate claims. Physion demonstrates contact-outcome utility, SSv2 demonstrates video-change class utility, neither validates a query as a complete physical object. Future experiments require repeated seeds, full official partitions, adequate class coverage and independent object-level diagnostics. Pilot numbers do not choose a new world-model architecture.

## MOVi Optional Diagnostic

[Official Kubric MOVi](https://github.com/google-research/kubric/tree/main/challenges/movi) supplies synthetic object masks/trajectories. Preparation can use the same authorized archive downloader/unpacker; no MOVi classifier or training loader is included. A later diagnostic should evaluate object coverage, temporal correspondence and grouping against simulator truth, separately from Physion/SSv2 task utility. Do not silently replace external benchmarks with MOVi or mix synthetic mask labels into V69 pretraining.

## Execution Evidence

The updated CPU integration completed locally on 2026-10-08 in `/tmp/igsw-benchmark-integration-pair-cue-balanced-20261008`, with full log at `/tmp/igsw-benchmark-integration-pair-cue-balanced-20261008.log`: two generated dataset interfaces, duplicate Physion names across files/splits, paper-based time metadata, PNG segmentation pair-cue exact pixels, future image/mask isolation, per-class sampling, separate dev/test artifacts, all four feature exporters/probes, FP16/native-width cache, linear reference, paired report, and interrupted/resumed updates. Final parameter difference and trace difference were exactly zero. These fixture accuracy numbers have no research meaning.

The parent reports that SWXC reached **port8600**, completed CPU and CUDA fixture integrations with exit0, and obtained a first approximately25MB Collide HDF5. Its first-sample facts are recorded above. These are reported remote execution receipts, not a fresh local verification by this implementation agent. No scientific benchmark scores have been reported. Full scenario/split coverage, testing colors, pretrained encoder runs and measured throughput remain pending; HDF5 lacks a measured clock. Port8732 is not part of this first experiment. The single-GPU fixture entry remains:

Earlier remote feedback reported Physion PID173180 live at252/512 files, approximately28.53GB; the updated terminal384/404 receipt above supersedes that live status. CPU/CUDA fixture results are reported passed. SSv2 part1 is complete; part2 stopped around7.32GB with curl92 and SWXC resumed it and labels. Later feedback reported the768-clip DINO pilot export exit0 as recorded in the development section. The implementation agent has not read the concrete remote result paths. These receipts do not establish complete Physion acquisition or scientific benchmark results.

```bash
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" "${ROOT}/code/scripts/test_world_dynamics_benchmarks.py" --device cuda:0
```

This uses generated videos/HDF5 plus small fixture backbones and actual V69 State/Posterior. It tests interfaces/resume/leakage, not pretrained quality. It does not download anything or interfere with existing training.

### 2026-10-08 Remote-Observed Update

The State/State+z scores in this historical receipt used the first-frame-anchored clock. The subsequent adapter correction below invalidates those State comparisons as correctly adapted capability evidence; the DINO/V-JEPA2 paths are unaffected.

The parent read SWXC actual stdout/receipts for this update. It supersedes earlier partial-download/running receipts above, without changing their chronology. The implementation agent has not independently rerun those remote commands.

- Physion preparation completed:512 HDF5 files,60,397,021,088 bytes, eight scenarios x two splits x32. Support/Roll exit0. Manifest counts are train204/dev52/held256.
- Actual pair-cue check changed2559 pixels: target2034 plus zone525, outside-pair changes0. This confirms the pixel operation for the inspected case, not official-rerender equivalence or object validity.
- SSv2 pilot01 completed all four768-case exports and all four20-epoch/160-update probes with exit0. Dev evaluate/compare commands all exit0. Every probe has239406 trainable parameters and256 dev cases.

| Representation | Dev Accuracy | Macro Accuracy | Top5 | Best Epoch |
| --- | ---: | ---: | ---: | ---: |
| DINOv3 | 0.015625 | 0.011494 | 0.03125 | 0 |
| V-JEPA2 | 0.0625 | 0.063218 | 0.089844 | 19 |
| State | 0.04296875 | 0.04023 | 0.09375 | 18 |
| State+z | 0.04296875 | 0.04023 | 0.089844 | 15 |

Paired dev State+z minus State is0, bootstrap95% interval[-0.015625,0.015625], with2 improved and2 regressed cases. Minus DINO is0.02734375, interval[0,0.05859375]; minus V-JEPA2 is-0.01953125, interval[-0.05087890625,0.015625]. Source: `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/world_dynamics_benchmarks/experiments/ssv2/comparisons/seed17/pilot01/comparison_dev.json`.

These are development-selection scores from a pilot with only1-2 train examples/class. Neither the scores nor intervals establish held generalization or object validity, and test scores did not choose the research mainline. No observed gain from adding z to the complete State sequence does **not** show that z failed to learn change. Redundancy between two representations of the same observed clip is a structural hypothesis, not causal evidence.

The expanded manifest is2784 train/1392 dev/1392 held, exactly16/8/8 per class. The existing768 features per representation were hard-linked where teacher checkpoint/export configuration matched; new development probes do not inherit optimizer state. This remains a bounded cache experiment, not full >1TB export.

Execution source is `5859aac2df28335367ce4fc6e62588630a7e7f2b`. Server checkout: `/mnt/pfs/public/xuhaoming/instruct_gs_world/world_dynamics_benchmarks_5859aac2`. Log root: `/mnt/pfs/public/xuhaoming/instruct_gs_world/logs/world_dynamics_benchmarks_5859aac2/`, with `gpu_pipeline.state` and `gpu_stage_receipts.tsv`. At this receipt, controller PID971835 is live, state `running:ssv2:development01:dino:export`, latest2623/5568; Physion cue01 is queued after it. Expanded SSv2 and Physion research results remain in progress, so the research goal is not complete. This documentation update does not change that server source or execution.

### State Clock Adapter Correction

V69 `ObjectVideoSequenceDataset.__getitem__` subtracts the last history timestamp (`timestamps[th-1]`), so history ends at0 and earlier frames are negative. State `observe` embeds absolute time as well as elapsed time. The benchmark previously supplied first-frame-relative positive time, which violated this pretrained input contract. The correction is code-contract driven, not chosen from dev performance: only State/State+z perception receives `encoder_times = batch.times - batch.times[:, -1:]`; online State, target State and posterior source/target times share that clock. Adjacent intervals and posterior relative elapsed time are unchanged. Public movie PTS/metadata remain first-frame-relative; raw DINO/V-JEPA2 inputs and outputs do not change.

Code evidence: `code/igsw/adaptive_gaussian_wm/object_video_sequence_dataset_v69.py:149` subtracts the last history timestamp; `query_object_video_encoder_v69.py:103` embeds `(time,time-state.time)`; `pretrained_visual_encoder_v69.py:96` retains supplied times without normalization; `object_sequence_dynamics_v69.py:37` uses `target.time-source.time` in the posterior. The public packet still uses the original movie times; cache `state_clock_times` comes from the actual `perception.times` supplied to State.

State provenance now declares `state_time_anchor=last_observed_frame_zero`; every State cache records the actual negative-to-zero `state_clock_times`. This is audit metadata, not an extra probe input. Old pilot01 and development01 State/State+z caches and comparisons are historical incorrectly adapted results, not evidence of correctly adapted capability. Preserve the artifacts, but do not reuse their State caches or optimizer state.

The existing development config now uses `ss_development02_clock`, and Physion uses `physioncue02_clock`. Manifests, sample selection, readout settings, V69 training/model/checkpoint and public data clock stay unchanged. Initialize new probes, re-export State/State+z, and reuse only raw DINO/V-JEPA2 caches where sampling/teacher/export settings match; no new cache-management system is introduced. This correction does not establish benchmark success; correct-clock dev evaluation is the next result.

One local complete integration run passed with exit0 at `/tmp/igsw-benchmark-integration-state-clock-20261008/integration_result.json`, full log `/tmp/igsw-benchmark-integration-state-clock-20261008.log`: the encoder hook confirms the actual negative-to-zero input clock; adjacent intervals are preserved; resume parameter difference0.0, trace/future-swap exact. This is CPU fixture evidence, not new real-data benchmark scores.
