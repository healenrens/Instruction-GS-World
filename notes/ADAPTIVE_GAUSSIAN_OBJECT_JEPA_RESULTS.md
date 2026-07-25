# Adaptive GPSToken Object-JEPA 实施与验证结果

日期：2026-07-18
状态：核心结构和 DDP 路径已实现并验证；完整方案未通过大规模训练 gate。

## 1. 最终判断

GPSToken 没有被废弃。

本轮实现证明了三件事：

1. learnable GPSToken 可以作为当前帧特征区域与密度分配器。
2. Object Slot Dynamics 可以只预测 latent feature，并在部分任务上优于 latent current-copy。
3. 严格 current-only prior、future-only target/posterior 和 DDP full-state checkpoint 可以正确工作。

但完整方案还不能启动两机八卡大训，原因是：

- 真实 held split 上，Object-JEPA 虽优于 current-copy，但仍弱于简单 flat feature baseline。
- GPSToken 有效数量与场景复杂度的相关性跨 seed 不稳定。
- Conditional Flow Prior 的输出经 Dynamics 后几乎没有有效样本多样性。
- 现有真实 causal pair 只有一帧历史和一个未来端点，无法验证真实三帧历史。

权威 gate：

- 本地：`/Users/hela/Instruct-GS-World/outputs/adaptive_gaussian_wm_feasibility/gate_summary.json`
- 远端：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/adaptive_gaussian_wm_feasibility/gate_summary.json`

`large_scale_ready=false`。

## 2. 实现范围

设计文档：

`/Users/hela/Instruct-GS-World/notes/ADAPTIVE_GAUSSIAN_OBJECT_JEPA.md`

新模型目录：

`/Users/hela/Instruct-GS-World/code/igsw/adaptive_gaussian_wm/`

主要模块：

| 文件 | 职责 |
|---|---|
| `config.py` | tiny/probe/full 配置与合法性检查 |
| `gpstoken.py` | learnable 2D seed、内容条件偏移、软分配、activation |
| `object_slots.py` | appearance+geometry Object Slot 聚合和身份锚定 |
| `scale.py` | signed physical gap 和 AdaLN/FiLM block |
| `dynamics.py` | 联合未来 Object latent Transformer |
| `latent_action.py` | future-conditioned Posterior 和 current-only Flow Prior |
| `decoder.py` | auxiliary Gaussian feature/attribute readout |
| `losses.py` | JEPA、latent change、flow、action-use、feature 和正则 |
| `pair_dataset.py` | 现有严格因果 DINO pair adapter |
| `model.py` | online/EMA target/posterior/prior 的边界 |

训练与评估入口：

- `/Users/hela/Instruct-GS-World/code/scripts/test_adaptive_gaussian_wm.py`
- `/Users/hela/Instruct-GS-World/code/scripts/experiment_adaptive_gaussian_wm.py`
- `/Users/hela/Instruct-GS-World/code/scripts/train_adaptive_gaussian_wm.py`
- `/Users/hela/Instruct-GS-World/code/scripts/evaluate_adaptive_gaussian_wm.py`
- `/Users/hela/Instruct-GS-World/code/scripts/train_adaptive_gaussian_wm_2n8g.sh`

旧目录 `/Users/hela/Instruct-GS-World/code/igsw/gpstoken_wm/` 和 `/Users/hela/Instruct-GS-World/code/igsw/latent_particle_wm/` 未被替换。

## 3. GPSToken 的保留与修正

保留：

- 当前/历史特征决定 token 区域。
- 不同区域由不同数量的微高斯覆盖。
- center、covariance、feature、opacity、activation 保留为可渲染 readout。

废弃：

- 手工熵递归切分作为最终 allocator。
- 固定 `active_count`。
- 使用未来 `traj[K]-traj[0]` 的 oracle placement。

新 allocator：

- `M` 是最大容量，不是固定有效数量。
- 可学习 2D seed 提供局部连续性。
- 当前特征产生 query offset 和内容分配。
- activation 由 occupancy、局部 feature dispersion 和 token latent 共同预测。
- future 不进入 allocator。

## 4. 结构和因果测试

远端环境：

- 代码根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`
- Python：`/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python`
- PyTorch：2.8.0+cu126
- 当前容器：2 张 NVIDIA A100-SXM4-80GB
- 未重装环境
- 未在本地运行测试

权威结果：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/adaptive_gaussian_wm_feasibility/structural_test_final.json`

通过项：

| 指标 | 结果 |
|---|---:|
| current encoder future-swap max diff | `0.0` |
| Flow Prior context future-swap max diff | `0.0` |
| history mask branch future-swap max diff | `0.0` |
| Posterior future-swap max diff | `1.91e-3` |
| grid permutation max diff | `7.15e-7` |
| finite gradient tensors | `149` |
| 单帧历史 | 通过 |
| 三帧历史 | 通过 |

这证明 future 只进入 EMA target 和 Action Posterior，没有进入 current encoder、prior context 或历史重建分支。

## 5. 合成多样化实验

覆盖：

- 一到三个对象。
- 单帧/三帧混合历史。
- 不规则时间间隔。
- 相同历史对应两个未来分支。
- 不同运动方向和局部遮挡。
- full、no-mask、no-scale、flat 对照。

权威结果：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/adaptive_gaussian_wm_feasibility/full_v6_seed17.json`
- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/adaptive_gaussian_wm_feasibility/full_v6_seed23.json`

### 5.1 通过

单帧 latent prediction：

| Seed | 模型 MSE | Current-copy MSE |
|---:|---:|---:|
| 17 | `4.03e-4` | `8.52e-3` |
| 23 | `1.24e-4` | `3.33e-3` |

三帧 latent prediction：

| Seed | 模型 MSE | Current-copy MSE |
|---:|---:|---:|
| 17 | `2.39e-4` | `2.41e-4` |
| 23 | `1.79e-4` | `2.02e-4` |

说明联合 latent Dynamics 可以学习未来 feature change，尤其单帧历史不是简单复制。

### 5.2 未通过

- token-count/complexity Pearson：seed17 `0.531`，seed23 `0.046`。
- prior rendered feature diversity：约 `2e-5` 到 `6e-5`。
- synthetic Gaussian feature readout 未稳定优于 current-copy。

自适应密度存在信号，但跨 seed 不稳定；Flow Prior 没有形成有意义的可部署多样性。

## 6. 真实严格因果 pair

数据：

- pair：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_pairs_e2e_20260717_v1/`
- DINO：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_pairs_e2e_20260717_v1_dino32/`
- train：384 pairs
- held-seed：128 pairs
- held-task：128 pairs

模型只读取：

- `dino0`
- `dino1` 作为 target
- 当前归一化网格坐标
- frame horizon

tracker、未来位移、对象 mask、机器人 mask、显式 action 均未进入输入。

### 6.1 Held-seed

| 方法 | Feature MSE |
|---|---:|
| Current-copy | `0.62683` |
| Object-JEPA Posterior | `0.58738` |
| Flat feature baseline | `0.48878` |

Object-JEPA 相对 current-copy 改善 `6.29%`，但落后 flat baseline。

latent MSE：

- Object-JEPA：`8.38e-4`
- Current-copy：`3.17e-3`

### 6.2 Held-task

| 方法 | Feature MSE |
|---|---:|
| Current-copy | `0.60382` |
| Object-JEPA Posterior | `0.58515` |
| Flat feature baseline | `0.47371` |

Object-JEPA 相对 current-copy 改善 `3.09%`，但落后 flat baseline。

latent MSE：

- Object-JEPA：`8.25e-4`
- Current-copy：`3.13e-3`

### 6.3 Prior

- 两个 held split 的 prior future-swap max diff 均为 `0.0`。
- prior feature diversity 约 `8e-5`。
- best-of-4 几乎不改善。

因此当前 prior 主要退化为接近确定性的预测，不能声称多可能性学习成功。

## 7. DDP、ZeRO-2 与 checkpoint

历史事实：

- `/Users/hela/Instruct-GS-World/code/scripts/train_gpstoken_wm.py` 使用 DDP。
- `/Users/hela/Instruct-GS-World/code/scripts/train_vla.py --deepspeed 1` 使用 DeepSpeed ZeRO-2。

新版使用 DDP，不使用 DeepSpeed：

- 两个 rank 的 representation+joint 测试已通过。
- bf16 Transformer/MLP 与 fp32 Gaussian Cholesky/inverse 已通过。
- rank 0 保存完整 `state_dict`。
- checkpoint 标记：`parallelism=ddp_full_state_dict`。
- 单卡 `strict=True` 加载：missing `0`，unexpected `0`。
- `--resume` 已验证。

DDP checkpoint 不需要 shard 合并。若未来切换 ZeRO-2，必须另行保存 optimizer/engine 状态，不能假设训练态 checkpoint 与当前 DDP 格式完全等价。

2×8 launcher：

`/Users/hela/Instruct-GS-World/code/scripts/train_adaptive_gaussian_wm_2n8g.sh`

launcher 使用：

`torchrun --nproc_per_node "$NPROC_PER_NODE" --nnodes "$WORLD_SIZE" --node_rank "$RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"`

当前 gate 未通过，launcher 默认拒绝正式训练；`DRY_RUN=1` 仅打印命令。

服务器最终核验：

- 新包和五个新入口脚本已通过 Python 编译检查。
- launcher 已通过 Bash 语法检查。
- `DRY_RUN=1` 正确展开为 2 节点、每节点 8 进程、共 16 个 DDP rank；默认配置有效 batch 为 `1 × 16 × 16 = 256`。
- 关闭 `DRY_RUN` 后，当前 `large_scale_ready=false` 会在创建训练进程前以退出码 `2` 拒绝启动。

## 8. Gate 总表

| Gate | 状态 |
|---|---|
| 完整前向、loss、反向 | 通过 |
| current/prior 严格因果 | 通过 |
| Posterior 未来敏感 | 通过 |
| 单帧/三帧合成结构 | 通过 |
| 两 rank DDP | 通过 |
| full-state checkpoint load/resume | 通过 |
| synthetic latent 优于 copy | 通过 |
| real posterior feature 优于 copy | 通过 |
| real Object-JEPA 优于 flat | 未通过 |
| adaptive density 跨 seed 稳定 | 未通过 |
| Flow Prior 有效多样性 | 未通过 |
| 真实三帧历史 | 无数据，不能确认 |

## 9. 下一步优先级

1. 先解决对象化价值：加入不使用外部 mask 的 slot temporal consistency / anti-collapse 目标，并要求真实 held split 超过 flat baseline。
2. 再解决多可能性：提高 target effect 的可辨识度，检查 posterior action 使用率，并要求 prior best-of-N 有实质且可校准的增益。
3. 构建严格因果三帧历史与多未来端点数据；在真实数据完成 mixed `1/3` history 和 scale ablation。

在这三个 gate 通过前，不启动两机八卡 full profile。
