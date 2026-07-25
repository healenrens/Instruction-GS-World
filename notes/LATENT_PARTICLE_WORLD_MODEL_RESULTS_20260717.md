# Latent Particle World Model 初步验证结果

日期：2026-07-17
主设计文档：`/Users/hela/Instruct-GS-World/notes/LATENT_PARTICLE_WORLD_MODEL_20260717.md`

## 1. 验证设计

新增隔离代码：

- `/Users/hela/Instruct-GS-World/code/igsw/latent_particle_wm/`
- `/Users/hela/Instruct-GS-World/code/scripts/build_latent_particle_probe.py`
- `/Users/hela/Instruct-GS-World/code/scripts/train_latent_particle_probe.py`
- `/Users/hela/Instruct-GS-World/code/scripts/analyze_latent_particle_codes.py`
- `/Users/hela/Instruct-GS-World/code/scripts/evaluate_latent_prior_coverage.py`
- `/Users/hela/Instruct-GS-World/code/scripts/audit_latent_particle_probe_cache.py`
- `/Users/hela/Instruct-GS-World/code/scripts/test_latent_particle_wm.py`

远端环境：

- Python：`/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python3`
- PyTorch：2.8.0+cu126
- 当前容器实际可见：2 张 A100 80GB
- 未重装环境
- 未使用 RoboTwin/XR-2 rollout 环境

probe 数据：

- 800 train clips。
- 200 held-seed clips。
- 200 held-task clips。
- 50 个任务。
- horizon：1、3、6、9、12。
- 每个当前帧固定 256 个 particle。
- dynamics 使用全部 256 个。
- current-only GPSToken active target 为 96，dedup 后约占 35%。
- 为隔离 motion 结构，probe 只从存在 tracker target 的 17,610 个窗口中分层采样；无 tracker 窗口未进入本轮 motion 对照。

严格不变量：

- 同一当前状态替换未来 target 后，所有模型 prior 参数最大变化均为 `0.0`。
- posterior 会随未来改变。
- future-conditioned input field 列表为空。

## 2. 数据质量

正式 audit：

- 本地：`/Users/hela/Instruct-GS-World/outputs/latent_particle_probe_v2/cache_audit.json`
- 远端：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/cache_audit.json`

训练集：

- geometry-valid：68.53%。
- geometry-valid 中 motion-valid：95.43%。
- tracker jump rejection：1.95%。
- mover 占 motion-valid：7.16%。
- active particle 占全部 proposal：35.03%。
- active mover recall：72.17%。
- active mover precision：15.69%。

held-seed active mover recall：73.76%。
held-task active mover recall：67.97%。

结论：

- current-only entropy GPSToken 不是“完全没工作”。
- 它用约 35% 粒子覆盖约 68% 到 74% movers。
- 长 horizon recall 会下降，因此 active set 不能代替完整 proposal set。
- 正确设计是 `M` 全部进入 dynamics，`L` 只用于 rendering/高分辨率 loss。

## 3. 模型结果

统一 mover 阈值为归一化 image flow `> 0.01`。

### 3.1 Held-seed

| 模型 | posterior EPE px | prior center EPE px | best-of-32 px | dCos | xyz EPE cm |
|---|---:|---:|---:|---:|---:|
| zero motion | 40.97 | 40.97 | - | - | 5.59 |
| deterministic | 35.14 | 35.14 | - | 0.403 | 4.94 |
| independent local two-stage | **33.38** | 37.18 | 40.71 at N=8 | 0.333 prior | 5.37 prior |
| effect-aligned MoG | 35.45 | 35.61 | 33.62 | 0.409 prior | 5.10 prior |
| effect-aligned flow | 35.97 | 35.94 | **32.61** | 0.394 prior | 5.12 prior |

结论：

- held-seed 的 prior-center 单一 prediction 仍由 deterministic baseline 最好。
- independent posterior 最会重建已知未来，但 prior 无法复现。
- flow best-of-32 coverage 超过 deterministic，但需要 oracle sample selection。

### 3.2 Held-task

| 模型 | posterior EPE px | prior center EPE px | best-of-32 px | prior dCos | prior xyz cm |
|---|---:|---:|---:|---:|---:|
| zero motion | 38.95 | 38.95 | - | - | 7.12 |
| deterministic | 38.39 | 38.39 | - | 0.161 | 6.62 |
| independent local two-stage | 37.03 | 39.41 | 41.34 at N=8 | 0.182 | 6.72 |
| effect-aligned MoG | 37.09 | 37.17 | 35.73 | 0.264 | 6.49 |
| effect-aligned flow | **36.69** | **36.61** | **33.15** | 0.238 | 6.49 |

结论：

- effect-aligned conditional flow 是唯一在 held-task 上通过增加 sample 数持续改善的 prior。
- point-weighted best-of-32 比 zero-motion 降低约 14.9%。
- point-weighted best-of-32 比 deterministic 降低约 13.7%。
- point-weighted best-of-32 xyz EPE 为 6.15 cm，deterministic 为 6.62 cm。
- paired clip-weighted delta 相对 deterministic 为 `-5.24 +/- 0.23 px SE`。

### 3.3 Sampling coverage

held-task flow prior：

| Samples | clip EPE px | point EPE px | point xyz cm | paired delta vs det px |
|---:|---:|---:|---:|---:|
| 1 | 39.89 | 37.40 | 6.60 | -0.67 |
| 2 | 38.32 | 35.82 | 6.43 | -2.24 |
| 4 | 37.22 | 34.80 | 6.32 | -3.34 |
| 8 | 36.41 | 34.05 | 6.25 | -4.15 |
| 16 | 35.82 | 33.58 | 6.20 | -4.74 |
| 32 | 35.32 | 33.15 | 6.15 | -5.24 |

held-seed flow prior：

| Samples | clip EPE px | point EPE px | point xyz cm | paired delta vs det px |
|---:|---:|---:|---:|---:|
| 1 | 40.69 | 36.46 | 5.20 | +1.21 |
| 8 | 37.18 | 33.48 | 4.89 | -2.30 |
| 32 | 36.10 | 32.61 | 4.81 | -3.38 |

coverage 单调改善，说明 flow prior 确实产生不同 future，而不是重复 deterministic output。best-of-N 是 oracle coverage 指标，不等同于可部署的 sample ranking。

## 4. Latent transfer

正式结果：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_flow_aligned_twostage/latent_analysis.json`
- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_mixture_aligned_twostage/latent_analysis.json`

未对齐 MoG：

- held-seed latent nearest-neighbor effect error / random：0.720。
- held-task：0.826。
- held-task linear effect R2：0.163。
- 约 96% posterior codes 对应同一个 mixture component。

effect-aligned flow：

- held-seed nearest/random：0.661。
- held-task nearest/random：0.755。
- held-seed linear effect R2：0.289。
- held-task linear effect R2：0.198。

说明 direct effect orientation 改善了跨场景 latent 语义，但还远未形成完全场景无关的 action space。

## 5. 已验证失败

### 5.1 Joint CVAE

- posterior 与 prior prediction 几乎相同。
- sample diversity 只有约 0.3 到 0.5 px。
- decoder 忽略 latent。

结论：发生 posterior collapse，不能作为 latent world model 成果。

### 5.2 Independent particle stochastic prior

两阶段训练与 action-shuffle 能迫使 posterior 使用 latent，但：

- prior KL 很高。
- prior sample 破坏粒子相关性。
- best-of-N 不改善。

结论：适合 inverse-motion diagnostic，不适合主生成 prior。

### 5.3 MoG prior

- 4-component/12D 模型主要使用 1 到 2 个 component。
- 8-component/32D 模型也主要使用 2 个 component。
- 32D 没有优于 12D。

结论：问题不是 latent 容量不足，而是离散 mixture 退化与 prior 目标不匹配。

### 5.4 当前 flow 仍未解决

- sample variance 与真实 error 的相关性接近 0，未校准。
- magnitude ratio 仍约 0.44，继续低估位移。
- prior center 在 held-seed 仍不如 deterministic。
- best-of-N 依赖 oracle selection，当前没有 learned ranker。
- sample calibration 与 prior-center depth 仍未解决。
- `handover_block` held-task 仍出现错误方向。
- 当前 probe 只有窗口首帧到 5 个 horizon，不等于真正任意起止帧。
- 当前 probe 没有 Gaussian renderer，只验证了 particle motion latent 结构。
