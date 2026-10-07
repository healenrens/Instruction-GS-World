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

SSv2 preparation:

```bash
export CONFIG="${ROOT}/code/configs/world_dynamics_benchmarks/ssv2.yaml"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" download --plan /PATH/TO/authorized_downloads.json
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" unpack --destination "${BENCH_ROOT}/ssv2" /PATH/TO/video.zip /PATH/TO/labels.zip
# For a single archive split into ordered parts, add --join-parts instead.
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" inspect "${BENCH_ROOT}/ssv2/something-something-v2-train.json"
bash "${ROOT}/code/scripts/prepare_world_dynamics_benchmarks.sh" manifest
```

Download-plan format is a JSON list: `[{"url":"AUTHORIZED_URL","mode":"file","destination":"/ABS/PATH/archive.zip"}]`. CLI does not bypass authorization or fabricate links. The manifest accepts official `id`/`template` annotations and label mapping; schema inspection precedes adaptation if actual fields differ. Default pilot is 256 per partition, selected round-robin from class-stratified random queues. Set `pilot_per_split: 0`, `scope: full`, and a new `attempt` for full data. Reusing an attempt after changing input or checkpoint is not allowed by the experiment protocol.

## Offline Single-GPU Run

Set the complete asset block above and one `CONFIG`. Run in foreground; no Git/model download occurs:

```bash
unset http_proxy https_proxy
for MODEL in dino vjepa2 state state_z; do
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" export --model "${MODEL}"
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model "${MODEL}" --protocol attention
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model "${MODEL}" --protocol attention
done
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" compare
```

Physion linear reference, a separate protocol/output directory:

```bash
for MODEL in dino vjepa2 state state_z; do
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model "${MODEL}" --protocol linear
  bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model "${MODEL}" --protocol linear
done
```

Reference uses balanced training subsampling, StandardScaler, per-scenario logistic regression, train-only stratified CV for C, and reports class-balanced test accuracy. It flattens 16 selected token+position records, rather than each author's original unrestricted feature layout; therefore **official-style reference**, not paper reproduction. Small pilots require setting `linear_cv` to a fold count supported by each outcome count. No test result selects C, epoch, or config.

Matched probe is one 128-wide attention query plus a small MLP (no deep temporal network); all methods use identical trainable parameter count and train/dev schedule. Epoch selection uses internal dev only. Future targets, labels, task filenames and scenario labels never enter the representation or attention inputs. State+z is a test of observed change encoding, not future prediction supplied by a posterior.

## Resume and Results

Export repeats only missing feature files. Attention resume explicitly uses `--resume` and restores model, optimizer, CPU/CUDA RNG, epoch/batch cursor; existing trace is appended. Linear resume skips complete per-scenario fits. New probe runs create their trace exclusively, avoiding silent overwrite.

```bash
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" export --model state_z
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" probe --model state_z --protocol attention --resume
bash "${ROOT}/code/scripts/run_world_dynamics_benchmarks.sh" evaluate --model state_z --protocol attention
```

Outputs: `${BENCH_ROOT}/experiments/<benchmark>/<model>/seed17/<attempt>/` containing `features.json`, per-case `.pt`, and `attention/` or `linear/` probe checkpoints, `report.json`, `cases.jsonl`, `confusion.csv`. Cases include frame indices, time values with their provenance (paper-inferred for Physion, decoded PTS for SSv2), resolution, token retention, probabilities, decode/encoder latency, and readout latency for attention. Wilson CI describes example-level accuracy; per-class/per-scenario counts and balanced accuracy must accompany the aggregate. A missing class is not an evaluated class. No automatic SwanLab/W&B upload or raw data transfer.

Compare per-example paired correctness before aggregate claims. Physion demonstrates contact-outcome utility, SSv2 demonstrates video-change class utility, neither validates a query as a complete physical object. Future experiments require repeated seeds, full official partitions, adequate class coverage and independent object-level diagnostics. Pilot numbers do not choose a new world-model architecture.

## MOVi Optional Diagnostic

[Official Kubric MOVi](https://github.com/google-research/kubric/tree/main/challenges/movi) supplies synthetic object masks/trajectories. Preparation can use the same authorized archive downloader/unpacker; no MOVi classifier or training loader is included. A later diagnostic should evaluate object coverage, temporal correspondence and grouping against simulator truth, separately from Physion/SSv2 task utility. Do not silently replace external benchmarks with MOVi or mix synthetic mask labels into V69 pretraining.

## Execution Evidence

The updated CPU integration completed locally on 2026-10-08 in `/tmp/igsw-benchmark-integration-paper-time-20261008`, with full log at `/tmp/igsw-benchmark-integration-paper-time-20261008.log`: two generated dataset interfaces, duplicate Physion names across files/splits, paper-based time metadata, all four feature exporters/probes, FP16/native-width cache, linear reference, paired report, interrupted/resumed updates, and HDF5 future swap. Final parameter difference and trace difference were exactly zero. These fixture accuracy numbers have no research meaning.

The parent reports that SWXC reached **port8600**, completed CPU and CUDA fixture integrations with exit0, and obtained a first approximately25MB Collide HDF5. Its first-sample facts are recorded above. These are reported remote execution receipts, not a fresh local verification by this implementation agent. No scientific benchmark scores have been reported. Full scenario/split coverage, testing colors, pretrained encoder runs and measured throughput remain pending; HDF5 lacks a measured clock. Port8732 is not part of this first experiment. The single-GPU fixture entry remains:

Latest remote feedback: Physion PID173180 is live, with252/512 files (approximately28.53GB), reaching Drape testing and using the correct HTTP/HTTPS proxy. CPU/CUDA fixture results are reported passed. SSv2 part1 is complete; part2 stopped around7.32GB with curl92 and SWXC is resuming it and acquiring labels. The implementation agent has not read the concrete remote result paths. These partial-download receipts do not establish complete acquisition or benchmark results.

```bash
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" "${ROOT}/code/scripts/test_world_dynamics_benchmarks.py" --device cuda:0
```

This uses generated videos/HDF5 plus small fixture backbones and actual V69 State/Posterior. It tests interfaces/resume/leakage, not pretrained quality. It does not download anything or interfere with existing training.
