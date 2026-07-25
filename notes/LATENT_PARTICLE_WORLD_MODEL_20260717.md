# Instruct-GS-World Latent Particle World Model
日期：2026-07-17
状态：完成代码/数据审计、严格因果 probe、三类 latent 结构初测；尚未完成像素级 world model 大训练。
## 1. 目标与边界

目标是只依赖视频学习一个可采样的 latent world model：

- 从视频中抽取当前帧 `I_t` 与未来帧 `I_{t+k}`。
- 当前帧产生 2.5D/3D Gaussian particle state。
- future-conditioned inverse model 学习实际发生的 latent action。
- current-only prior 学习可采样的未来 action 分布。
- dynamics 预测物体、机械臂、相机和可见性的未来变化。
- tracker/keypoint 只能作为训练辅助监督，不能参与当前状态、token identity 或 inference 输入。
- 语言、机器人 action、goal 可以作为可选条件，但不能成为核心视频学习成立的前提。

代码与实验仅位于：

- 本地权威副本：`/Users/hela/Instruct-GS-World/`
- 远端运行副本：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`
- 远端连接：`ssh -p 8600 root@106.13.104.32`

本研究与 XR-2 无关。

## 2. 参考 LPWM 的关键结构

已阅读：

- `/Users/hela/.codex/attachments/fd8b28fd-0f6a-40bb-8b5a-2d7a612da1a1/pasted-text-1.txt`
- `/Users/hela/.codex/attachments/fd8b28fd-0f6a-40bb-8b5a-2d7a612da1a1/pasted-text-2.txt`

LPWM 的关键不是“使用粒子”本身，而是以下完整概率结构：

1. `M` 个 proposal particle 全部进入 dynamics。
2. 只有 `L` 个 active particle 进入 renderer。
3. future-conditioned posterior `q(a_t | z_t, z_{t+k})` 提取实际发生的 latent action。
4. current/history-only prior `p(a_t | z_{\le t})` 在 inference 时采样。
5. posterior 与 prior 对齐，但 future 永远不进入 prior。
6. dynamics 使用 latent action 预测未来 particle state。
7. 图像重建与感知特征重建保证粒子状态仍对应真实视频。

官方材料：

- [LPWM project](https://taldatech.github.io/lpwm-web/)
- [LPWM code](https://github.com/taldatech/lpwm)
- [LPWM OpenReview](https://openreview.net/forum?id=lTaPtGiUUc)

## 3. 当前数据 pipeline 的事实

### 3.1 旧 world-model 数据

旧数据生成入口：

- `/Users/hela/Instruct-GS-World/code/scripts/robotwin_spatrack_clip.py`
- `/Users/hela/Instruct-GS-World/code/scripts/agibot_clip_stv2.py`
- 远端实际数据：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rtvid_multi_v2/`

旧 RoboTwin SpaTracker 数据的当前 `means/uv` 并不严格因果：

- VGGT/SpaTracker 读取完整 13 帧视频。
- `support_frame` 使用最后一帧。
- 当前点会按完整未来可见性过滤。
- 保存的 `means == traj[0]`，但二者都来自 full-video tracker。
- 训练时 oracle token placement 又使用 `traj[K] - traj[0]` 的未来位移。

因此，future 不只作为 target，还改变了当前 particle candidate set 和有限 token budget。

### 3.2 严格因果数据

严格因果生成入口：

- `/Users/hela/Instruct-GS-World/code/scripts/rt2_build_causal_dataset.py`
- `/Users/hela/Instruct-GS-World/code/igsw/causal_geometry.py`

远端数据：

- 当前输入：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_v1/`
- 原始 13 帧 RGB：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_joint_src/`
- tracker targets：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_joint/`
- 验证结果：`/mnt/pfs/public/xuhaoming/instruct_gs_world/logs/causal_v1/dataset_verification.json`

已验证：

- 总窗口：20,642。
- train：15,113。
- held-seed：3,922。
- held-task：1,607。
- 当前 state 始终由单帧 head RGB 经过 VGGT 得到固定 `48x48=2304` proposal。
- full-video SpaTracker 只写入 `traj/vis/geom_valid` target。
- `max_traj0_abs_error = 0`。
- `max_reprojection_px = 6.15e-05`。
- 3,032 个无 tracker 窗口仍保留，可训练 appearance/presence/action，不伪造 motion label。

这套数据可以作为正式 latent world model 的因果起点。

## 4. 当前 GPSToken-WM 实际在学什么

核心代码：

- `/Users/hela/Instruct-GS-World/code/scripts/train_gpstoken_wm.py`
- `/Users/hela/Instruct-GS-World/code/igsw/gpstoken_wm/wm_model.py`
- `/Users/hela/Instruct-GS-World/code/igsw/gpstoken_wm/tokens.py`
- `/Users/hela/Instruct-GS-World/code/igsw/gaussians/gpstoken.py`

当前模型是确定性 `frame0 -> frameK` predictor，不是 latent world model：

- 没有 future-conditioned inverse posterior。
- 没有 current-only learned prior。
- 没有 latent sampling。
- 固定 horizon，主要做 Smooth-L1 位置/flow 回归。
- ambiguous future 会被压成条件均值。
- 训练 token placement 可依赖未来 mover saliency。
- 旧 candidate identity 也可能已经被未来可见性改变。
- 默认 JEPA readout 接收 detached trunk hidden，不能保证 future feature objective 塑造主干。

历史实验已显示：

- DINOv2 dense features 明显优于 Qwen image patches。
- direct 3D 在早期实验优于 flow+depth。
- rotation 不能靠独立 translation token 自然学出。
- motion magnitude 系统性低估。
- constant-angle 数据可记忆，varied-angle 难以学习。
- held-task 明显崩塌。

这些现象与“确定性均值回归 + 不相干粒子运动 + 非因果 tokenization”一致。

## 5. 三个完整候选

### 5.1 Candidate A：Causal Particle CVAE

这是 GPSToken/Gaussian 版本的直接 LPWM 基线。

状态：

`g_i = [u_i, v_i, log z_i, sigma_i, alpha_i, appearance_i, presence_i]`

结构：

- 当前单帧产生固定 `M` proposal identity。
- current-only GPSToken entropy/relevance 只决定 active renderer mask `L`。
- posterior 为每个粒子预测 `q(a_i | g_t, g_{t+k})`。
- prior 为每个粒子预测 `p(a_i | g_t)`。
- interaction transformer 使用全部 `M` 粒子。
- decoder 预测 `delta u, delta v, delta log z, appearance, presence`。
- Gaussian renderer 重建 `I_{t+k}`。

完整性：

- 有合法 posterior/prior。
- 有 stochastic inference。
- 有像素与 geometry 闭环。
- 没有 future 输入泄露。

理论风险：

- 256 个独立随机 latent 很难形成刚体平移、旋转、抓取接触和多物体协调。
- current-only prior 必须同时拟合大量独立未来变量，容易产生不相干噪声。

Candidate A 应保留为必要基线，不应作为主方案。

### 5.2 Candidate B：Effect-Aligned Correlated Action Field

这是当前最推荐的主方案。

核心改变：

- 不独立采样每个粒子的 action。
- posterior 提取一个低维 correlated action set，例如 4 个 action token，每个 8 维。
- action token 通过 cross-attention 生成所有粒子的非均匀 motion field。
- 粒子仍有局部 deterministic residual，但不再有独立随机 prior。
- current-only prior 使用 conditional flow matching 生成整个 correlated action set。

posterior：

`q(A | G_t, G_{t+k}, I_t, I_{t+k})`

prior：

`z_0 ~ N(0,I)`
`d z / d tau = v_theta(z_tau, tau, G_{\le t})`
`A = Flow_theta(z_0, G_{\le t})`

effect alignment：

- latent 前若干维直接对齐相机归一化 effect：
  - mean `delta u, delta v, delta log z`
  - spatial std
  - mean speed
  - moving fraction
- 其余维度表达接触、遮挡、appearance change 等不能被低阶统计覆盖的 residual。
- 跨场景相似 effect 应在 latent 空间相近，避免 action code 记忆场景纹理。

优势：

- 保留随机、多模态 future。
- 从结构上保证粒子运动相关。
- flow prior 不要求离散 mode 数量，也不会像 MoG 一样依赖 component 使用率。
- 可以自然添加 language/action/goal conditioning，但视频模型本身仍成立。

### 5.3 Candidate C：Soft Part SE(3) + Residual Field

这是物理结构更强的第二研究轨。

结构：

- 当前粒子通过 soft assignment 形成 `P` 个动态 part。
- 每个 part 由 correlated latent action 预测 `SE(3)` transform。
- 每个粒子预测小的 non-rigid residual。
- 背景、相机 ego-motion、机械臂、可交互物体分别拥有 transform slot。
- contact graph 决定 part 间信息交互。

未来粒子：

`x'_{i} = sum_p w_{ip}(R_p x_i + t_p) + delta x_i`

完整 supervision：

- RGB/Gaussian rendering。
- DINO feature reconstruction。
- tracker displacement/visibility auxiliary。
- local rigidity。
- cycle/composition consistency。
- posterior/prior action alignment。

优势：

- rotation、刚体运动和接触关系成为显式结构。
- 比独立 translation head 更符合机器人场景。

风险：

- part discovery 可能把机械臂与物体错误合并。
- 需要防止所有粒子退化到一个 part。
- 应在 Candidate B 稳定后并行验证，而不是先替换全部 dynamics。

## 6. 验证摘要

详细实验设计、数据质量、完整指标和失败分析见：

- `/Users/hela/Instruct-GS-World/notes/LATENT_PARTICLE_WORLD_MODEL_RESULTS_20260717.md`
- `/Users/hela/Instruct-GS-World/outputs/latent_particle_probe_v2/summary.json`

关键结论：

- 所有模型 future-swap 后的 prior 最大变化均为 `0.0`，因果接口成立。
- current-only active particles 占约 35%，覆盖 68% 到 74% movers。
- joint CVAE 发生 posterior collapse。
- independent particle posterior 能提取未来，但 prior 不相干。
- MoG 只使用 1 到 2 个 component，扩大 latent 无效。
- effect alignment 改善跨场景 latent transfer。
- conditional flow 是唯一在 held-task 上随采样数单调改善的 prior。
- 同一 point-weighted 口径下，held-task best-of-32 为 `33.15 px`，deterministic 为 `38.39 px`。
- held-seed best-of-32 为 `32.61 px`，deterministic 为 `35.14 px`；但 prior center 仍不如 deterministic。
- paired clip-weighted flow-vs-deterministic delta 为 held-task `-5.24 +/- 0.23 px SE`、held-seed `-3.38 +/- 0.22 px SE`。
- flow 的 sample ranking、概率校准和 prior-center depth 仍未解决。

## 7. 主方案决策

主方案选择 Candidate B：

**Effect-Aligned Correlated Action Field + Conditional Flow Prior**

保留：

- 固定因果 proposal identity。
- 全 `M` dynamics。
- current-only GPSToken active renderer mask。
- future-conditioned inverse posterior。
- current-only flow prior。
- explicit 2.5D geometry head。
- tracker visibility/motion auxiliary。
- RGB 与 DINO reconstruction。

不采用：

- future visibility filtering current particles。
- oracle mover placement。
- independent per-particle random prior。
- 单一 deterministic Smooth-L1 作为完整 future 模型。
- 把语言/action 当作视频 world model 成立的必要输入。

Candidate C 的 Soft Part SE(3) head 作为第二结构轨，与 Candidate B 共用同一 posterior/prior，不重新建立另一套数据 pipeline。

## 8. 正式模型规格

### 12.1 Causal state encoder

输入仅为 `I_{\le t}`：

- 单帧/短历史 RGB。
- single-frame depth/geometry estimator。
- frozen DINO dense appearance。
- 固定 grid proposal identity。

每个 Gaussian：

`[u,v,logz,log sigma_x,log sigma_y,opacity,appearance,presence]`

所有 `M=256..1024` proposal 进入 dynamics。
current-only GPSToken 选择 `L=64..256` active renderer particles。

### 12.2 Inverse posterior

posterior 读取：

- current state。
- future frame target encoding。
- 可选 tracker pseudo-label。
- `delta t`。

输出 correlated action tokens：

`A = [a_1,...,a_K]`，建议 `K=4`，每个 8 维。

tracker 不能决定 particle identity，只提供：

- flow/3D displacement。
- visibility。
- local rigidity。
- contact/change proxy。

### 12.3 Current-only flow prior

prior context：

- current/history particle state。
- `delta t`。
- 可选 language/action/goal。

flow matching 目标为 stop-gradient posterior code。
inference 从高斯噪声积分得到多个 action samples。

### 12.4 Dynamics

两层结构：

1. action tokens 与 particle cross-attention，产生 correlated motion field。
2. particle interaction transformer/graph network，处理接触和遮挡。

输出：

- `delta u, delta v, delta log z`。
- scale/rotation residual。
- opacity/presence。
- appearance feature residual。
- optional soft-part assignment 与 `SE(3)` transform。

### 12.5 Decoder 与 losses

主 loss：

- Gaussian RGB reconstruction。
- DINO feature reconstruction。
- multi-scale perceptual loss。
- presence/visibility BCE。
- posterior action reconstruction。
- flow-matching prior loss。
- effect orientation。
- action-shuffle counterfactual usage loss。
- local rigidity 与 part transform loss。
- multi-horizon composition consistency。

tracker loss 是 auxiliary，不是唯一目标。
即使 tracker 缺失，RGB/DINO/rendering 仍能训练模型。

### 12.6 Camera motion

移动相机不能混入 object latent：

- 单独预测 ego-motion latent/transform。
- object motion 在 ego-motion 补偿后建模。
- static background 用于相机约束。
- dynamic particles 使用 residual motion。

## 9. 下一步三个最高优先级任务

### P0：真正任意帧 pair 数据

基于：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_joint_src/`
- `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_v1/`

生成每个视频多个随机 `(t,t+k)`：

- 每个 `t` 单独运行 single-frame geometry。
- 固定 proposal identity 只由 `I_t` 决定。
- tracker target 通过当前 grid matching 得到。
- 覆盖 `k=1..12`，而不是只用窗口首帧。

成功门槛：

- future-swap prior difference 严格为 0。
- 任意 pair current state hash 与 future 无关。
- tracker jump/visibility 分开记录。

### P1：4-action-token flow model + Gaussian renderer

把 probe 的 correlated flow prior 接入正式 GPSToken/Gaussian dynamics：

- 4x8 action tokens。
- 256/512 proposal particles。
- 96/192 active render particles。
- RGB + DINO + geometry 联合训练。
- posterior stage 与 prior stage 分离。

成功门槛：

- posterior 显著优于 deterministic mover EPE。
- prior best-of-16 held-seed 不低于 deterministic。
- prior best-of-16 held-task 优于 zero 与 deterministic。
- sample diversity 增加时 coverage 单调改善。
- 3D EPE 与 2D EPE 同时改善。

### P2：Soft Part SE(3) 物理头

在同一 action posterior/prior 上增加 soft part transform：

- 与 direct particle field 做严格对照。
- 单独评估 rotate、handover、stack、open 等任务。
- 评估 local distance preservation、rotation angle、contact transition。

成功门槛：

- varied-angle rotation 优于 direct translation field。
- local rigidity error 不恶化。
- held-task 不靠场景/任务标签记忆。

## 10. 真值入口

聚合结果：

- 本地：`/Users/hela/Instruct-GS-World/outputs/latent_particle_probe_v2/summary.json`
- 远端：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/summary.json`

缓存审计：

- 本地：`/Users/hela/Instruct-GS-World/outputs/latent_particle_probe_v2/cache_audit.json`
- 远端：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/cache_audit.json`

正式 probe cache：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/latent_particle_probe_v2_800_200_200.pt`

推荐 checkpoint：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_flow_aligned_twostage/model.pt`

推荐 metrics：

- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_flow_aligned_twostage/metrics.json`
- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_flow_aligned_twostage/prior_coverage.json`
- `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/latent_particle_probe_v2/global_flow_aligned_twostage/latent_analysis.json`

## 11. 状态分层

已完成：

- LPWM 文档与官方实现核对。
- 本地/远端数据、代码、环境、历史结果审计。
- 严格因果 particle probe。
- 确定性、independent latent、hierarchical latent、MoG prior、flow prior 对照。
- held-seed、held-task、5 horizon、best-of-N、latent transfer、局部几何与因果不变量评估。

已验证：

- 旧 pipeline 存在 future-conditioned input/tokenization。
- current-only fixed-grid 数据可直接使用。
- joint CVAE posterior collapse。
- independent particle stochastic prior 不相干。
- MoG component 退化。
- effect alignment 改善跨场景 latent transfer。
- conditional flow 改善 unseen-task stochastic coverage。

尝试中：

- 无。

待办：

- 任意起止帧数据。
- 正式 Gaussian renderer。
- 4 action token correlated field。
- camera ego-motion。
- Soft Part SE(3)。
- 更大视频数据上的 scale-up。

风险：

- tracker pseudo-label 长 horizon 仍有少量跳点。
- 视频中单一观察未来不足以直接校准完整概率分布。
- held-task 的任务意图在纯当前帧中不可辨识，prior 必须覆盖而非猜中唯一目标。
- 2D coverage 改善不代表 depth 正确。

不能确认：

- 当前小 probe 是否能在完整 RGB renderer 下保持优势。
- 4 action token 的最佳数量和维度。
- RoboTwin 外真实视频的跨域泛化。
- 移动相机下 ego/object disentanglement 的有效性。
