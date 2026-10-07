# Instruct-GS-World Object-Level World Model 永久主线与实验账本

> 更新日期：2026-10-05
> 本地权威代码：`/Users/hela/Instruct-GS-World-recovered-20260725/`  
> 当前开发分支：`codex/object-video-sequence-v69`
> 当前实现：V69 Stage 2 large，冻结6000步State，训练约356.7M Posterior/Dynamics；3s历史、5s未来
> 最新训练证据：SwanLab `00cxvd6i`，已读取step 1–6830，数据截至2026-10-05 13:22:56 CST；见15.56，不代表此刻实时状态
> 当前执行接口：`pretrained_query_object_video_sequence_v3_stage2_large`，执行提交`c073c3b`；历史单卡失败见15.45，不再代表当前运行状态
> State专用变化评测：第15.48–15.49节，6000步快照支持观测变化还原；Stage 2尚无本轮held结果
> 继承数据：V68 `991099e`，所有类别轨迹全局Top-75%；RoboTwin只复用旧轨迹，不重新追踪
> 最新研究规划：第15.58节，语言/历史视频条件下预测现有effect；15.57的分段effect降为后续备选，Stage 2执行见`OBJECT_VIDEO_STAGE2_V69_RUNBOOK.md`
> 最新人工反馈：第 15.32 节记录物体覆盖不足、跨数据集机械臂混淆和 RoboMIND 视频来源疑问；尚未证明 teacher 可用于物体级监督
> 历史 V66 验证提交：`2e513d9d0f3f1b37a24fe5af4f1df2c95ac141f4`（V66 四卡、六源、1536 held clips G0 audit）
> V62 E0/E1 实现提交：`6fa0d63e67daf85d24654aaa649e725eb5245bfe`
> V62 B/C/D structural audit 实现提交：`2ad1158084ff0e8b4070f43c6721261df1884485`（静态验证，待服务器执行）
> 当前证据边界：V69在用户批准的冻结State可行性实验内推进；历史teacher/object语义缺口未自动消失，但不再把V67逐例观看列为当前唯一任务
> 远端代码工作区：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/`  
> 远端运行与产物根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`  
> W&B：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`
> 当前训练SwanLab：`healenrens/instruct-gs-world`；仅同步训练标量，评测逐例产物保留本地

> 2026-10-08研究决策更新：主线定位为 **world dynamics representation**，不以机器人 action/VLA/RoboTwin 成绩定义主实验。新增独立 Physion OCP、SSv2 frozen representation probe；执行协议见 `WORLD_DYNAMICS_BENCHMARK_RUNBOOK.md`。本轮不修改 Stage 1/2/3，不将旧 tracker/object validity 缺口记为通过。

## 0. 唯一主线

### 2026-10-08主实验定位修订

**What changed**：核心表征由冻结 State 与 observed-transition effect 共同接受外部任务检验；Physion OCP 只观察官方 input boundary 内的 prefix，SSv2 使用全部方法相同的可见视频片段；对照为 DINOv3、明确版本的 V-JEPA2、State-only、State+z。RoboTwin/语言/控制是分离的应用，不再是主线必须达到的终点。

**Why**：旧训练/重建/teacher一致性不足以说明变化表征具有外部任务价值；先固定可闭环的外部判断任务，再决定模型迭代。

**Impact**：Physion per-scenario linear reference 与新 matched token-attention protocol 分开报告；首轮 pilot 和正式分区结果不能混称。MOVi 只保留独立 object validity 诊断规划；仍不声称16个query等于16个物体。历史 promotion chain 保留为旧实验证据边界，而非把本轮外部 frozen probe 阻断或误记为所有 gate 通过。

从本文档此次更新开始，版本号只表示实现迭代，不再表示研究方向重启。项目只保留以下一条主线：

> 从纯视频学习可部署的、query-conditioned、persistent object state；在该 state 通过独立 object validity 验证后，再学习 latent effect conditioned object dynamics，最终由 goal、language 或 policy 选择 object query 与 latent effect。

固定的研究数据流分为训练期 teacher 与可部署 student 两条边界清楚的路径：

```text
Training-only target path:
RGB clip -> frozen point tracks + frozen DINO/SigLIP fields
         -> continuous query-object observations -> teacher object-state target

Deployable student path:
Observed RGB history + current point/region query
  -> frozen perception fields -> query-conditioned persistent object state

Transition path:
matched source/target object states -> latent effect posterior (training only)
source object state + latent effect + delta-time -> future object state
```

训练期 point tracker 可以使用完整视频构造 correspondence、relation、visibility evidence；部署 student 只能读取已经观察到的 RGB history 与 query。DINO/SigLIP 的 patch grid 只允许作为 perception backbone 的内部 feature sampling，不得作为 object GT、object 边界或最终 reconstruction 单元。future RGB、future tracks、teacher state、instance annotation、机器人显式 action 都不得进入 student history path。

这条主线按以下 promotion chain 单向推进：

| Gate | 必须回答的问题 | 通过前禁止做的事情 |
|---|---|---|
| G0 Objective validity | 正确解是否比 all-scene、seed-only、merge、split、visibility collapse 等捷径更优？ | 禁止真实长训。 |
| G1 Query binding | 给定 query，student 是否找到了与其相关、并排除了无关 track 的区域？ | 禁止声称学到完整 object state。 |
| G2 Persistent state | identity、dynamic、geometry、visibility/unknown 是否无坍缩、低复杂度可读且形成充分状态？ | 禁止训练 Dynamics。 |
| G3 Independent object validity | 不使用训练 tracker 的 evaluator 是否确认 object coverage、leakage、reappearance 与 deletion locality？ | 禁止把 teacher agreement 写成 object semantics。 |
| G4 Latent effect | 从真实前后 object state 提取的连续 effect 是否必要、稳定且不含显式 action/center delta？ | 禁止训练 History-only Prior。 |
| G5 Object Dynamics | 正确 effect 是否显著优于 zero/shuffled，且 rollout 保持 object identity 与 lifecycle？ | 禁止进入任务级 claim。 |
| G6 Selector / task A | goal、language 或 policy 是否能选择 query/effect，并在 RoboTwin task A 上产生可用预测？ | 禁止与 XR-2 混淆。 |

任何实验只能推动当前最早未通过的 Gate。后面的 Gate 即使某个指标变好，也不能覆盖前面 Gate 的失败。

2026-10-02用户明确批准一次冻结Stage 1的Stage 2容量/可用性实验（15.51）。此为执行顺序的显式例外，不把G0/G2/G3自动记为通过；仅检验现有表示条件下的transition建模，不作独立object语义或部署selector成功声明。

## 1. 文档用途与证据规则

本文档是本项目唯一的研究决策账本。它记录每一轮实验回答了什么问题、怎样执行、观察到什么、为什么导向下一轮。后续开发按第 0 节的 promotion chain 和第 11 节的当前 TODO 推进；前一项未通过，不启动后一项长训。

证据分为五级：

1. **结构 gate**：shape、gradient、causal boundary 和 objective falsification 通过，只说明代码契约成立。
2. **训练 telemetry**：loss 或在线指标变化，只说明优化器正在拟合某个目标。
3. **representation sufficiency**：collapse、effective rank、retrieval、frozen probe 和 nuisance/Markov 测试，说明 latent 是否存在可读的任务信息。
4. **held teacher evaluation**：在未参与优化的 clip 上与 point-track teacher 对齐，仍不是独立 object truth。
5. **independent evaluation**：使用 simulator ground truth 或人工标注，且不使用训练 tracker，才可证明 object state 具有外部语义。

因此，loss 下降、readout 变好、固定 slot index 稳定或 tracker agreement 都不能单独写成“学到了 object”。
G2/G3 的统一评测定义见
`/Users/hela/Instruct-GS-World-recovered-20260725/docs/OBJECT_STATE_REPRESENTATION_EVALUATION.md`。

### 1.1 强制更新协议

从现在起，每次发生以下任一事件，都必须在同一提交或紧随其后的文档提交中更新本文档：

- 修改 object 定义、teacher/student 边界、loss、数据采样或 evaluator；
- 启动、停止或恢复一次具有新假设的训练；
- 从 W&B、checkpoint 或独立 evaluator 得到足以改变决策的新证据；
- 决定保留、否决、后置或重新开放一个模块。

每次记录必须包含：

```text
日期 / 版本 / branch / commit / W&B run / checkpoint
当前 Gate
假设：本实验只检验什么
单一主要改动：相对上一 accepted baseline 改了什么
冻结项：哪些结构、数据和指标保持不变
执行：数据、history、GPU、batch、steps、resume/init_from
结果：完整曲线、分 source/H 指标、运行终态
证伪：哪些 shortcut 或反事实测试通过/失败
判决：promote / iterate / reject / infrastructure-only
继承：保留什么，明确不继承什么
下一步：仍然只解决哪个最早失败 Gate
```

状态词固定为：`已完成`、`已验证`、`尝试中`、`待办`、`风险`、`不能确认`。`running`、单次 loss 下降、结构 gate 或复制成功不允许写成 `已验证`。

## 2. 最终研究目标

核心目标始终是学习 **object-level dynamics**，而不是用低分辨率 latent 复刻整张图像。

新主线将部署时的 object state 定义成一个 query-conditioned entity：

$$
O_t(q)=E_{\theta}(I_{\le t},q)
=\left(u_t,d_t,g_t,\mathcal{R}_t,v_t\right).
$$

- $q$：当前帧中的一个 point 或 region query。
- $u_t$：跨时间缓慢变化的 identity/appearance。
- $d_t$：可变化的 dynamic state。
- $g_t$：相对 center、support shape、relative scale 等 geometry。
- $\mathcal{R}_t$：该实体需要的可变数量 local region carriers。
- $v_t$：当前 observability，不等同于世界中的 existence。

第二阶段才学习 latent effect：

$$
z_{t:t+\Delta}\sim q_{\phi}\left(z\mid O_t(q),O_{t+\Delta}(q)\right),
$$

$$
\widehat O_{t+\Delta}(q)=F_{\xi}\left(O_t(q),z_{t:t+\Delta},\Delta\right).
$$

这里 $z$ 表示实体状态变化，不包含机器人显式 action、RGB difference 或真实 center delta。部署期由 goal、language 或 policy 选择 $q$ 与 $z$，不要求 world model 自己从无条件历史猜唯一未来。

## 3. 数据、运行方式与共同边界

### 3.1 数据演进

| 阶段 | 数据 | 作用与结论 |
|---|---|---|
| 初期 | AgiBot clip、ManiSkill GT、LIBERO/Pi3 | 验证 3DGS motion 与 language conditioning；属于早期显式 Gaussian dynamics 主线。 |
| v28-v37 | `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_visual_episodes_no_language_v1/` | 缓存 DINO/视觉 episode；manifest 标记约 16.67 Hz，后续确认不符合原生 30 Hz 目标。 |
| v38-v52 | `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_visual_episodes_rgb_native_30hz_v4/` | 原生 30 Hz RGB，DINO JIT；动态 history 与 point-track object state 的主要数据。 |
| v53-v56 | `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/multisource_real_robot_video_v53/index.json` | RobotWin、AgiBot、DROID、RoboMind、Bridge、HY 六源平衡数据。v53 最终看到约 4,073 万 samples。 |

六源数据不是预先重编码为统一 feature cache。训练按视频 index 读取原视频，运行冻结 DINO 与 CoTracker。HY 与部分 AV1/OBU 视频曾出现损坏、缺失或 decoder 不兼容，后续通过 index filtering、replacement pool、共享 quarantine 与 decode-frontier audit 处理。`decode_replacement_fraction` 必须作为数据质量指标报告。

### 3.2 共同执行方式

从 v28 开始，代码只在本地权威 checkout 修改，再推送 GitHub；服务器只拉取对应 branch。训练通常由版本 manager 完成：

```bash
export ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export NPROC_PER_NODE=auto

bash "${ROOT}/code/scripts/manage_<version>.sh" verify
bash "${ROOT}/code/scripts/manage_<version>.sh" foreground
```

卡数由 `torch.cuda.device_count()` 自动确定。v53-v56 在 8×A100-80GB 上把 `BATCH_PER_GPU` 提升到 64、effective batch 提升到 512；冻结 DINO 使用独立 `DINO_FRAME_BATCH`。W&B 记录训练和 evaluation，checkpoint、rank logs 与 W&B 是三个独立证据源。

## 4. 早期 3DGS 与语言条件实验

这一阶段的执行入口主要是 `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/train_sim.py` 与 `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/eval_langswap.py`。

| 实验 | 执行细节 | 结果 | 推导结论 |
|---|---|---|---|
| v1 / run1 | AgiBot 离线 clip；Qwen/Cosmos 条件；SC-GS 风格 control points 与 LBS | 相对 static 约 +2.5 dB | 预测 Gaussian motion 可行，但未证明 object abstraction。 |
| stream11 | AgiBot 流式训练；加入 InfoNCE 与 NaN gradient guard | language collapse 有缓解，训练可持续 | 条件信号必须有反事实或对比约束；仅拼接 language 不会被使用。 |
| ManiSkill GT | simulator 给解析 actor pose；per-control spatial grounding | 单 clip corr 约 0.95；held task corr 约 0.65-0.76 | 旧模型定位失败的主要原因之一是 noisy geometry/track target。 |
| LIBERO v5-v7 | St4R 后切换 Pi3；逐实体 Procrustes 与 2D tracks | Pi3 修复明显各向异性，方向指标改善约 +0.93 | 相对几何质量直接决定 motion supervision。 |
| v8 language | relevance head 与 counterfactual swapped instruction loss | swap/true 从约 1.00 降到约 0.10 | language 因果使用需要错误 instruction 的显式负例。 |
| v9 language | early/center 双窗口破除 gripper-nearest shortcut | held scene 泛化约 0.25 提升到 0.75 | 数据反事实比增加网络容量更有效。 |
| v9 open-vocab | GroundingDINO+SAM2 替代 GT mask | 2×2 检查约 0.75；OV 训练选择约 0.50 | segmentation 可替换，但显式 mask/instance 路线不满足纯视频 scale 目标。 |

这些实验解释了“条件、geometry 和监督质量”如何影响显式 3DGS motion，但没有解决纯视频 object state。因此后续主线转向无语言、无显式 action 的 Object JEPA。

## 5. v28-v42：Object Memory、Gaussian readout 与 carrier 审计

### 5.1 v28 Object Memory JEPA

**执行**：`train_object_memory_jepa_v28.sh`；persistent 16 object slots、relative geometry、continuous latent effect、Gaussian DINO readout。代表 run `k4b1ccxp` 的有效 batch 为 256。

**结果**：早期多次被 autocast BCE、non-finite gradient、launcher 与 rolling checkpoint 问题打断。修复后形成约 14,500-step parent checkpoint。Dynamics 在 object latent 上可优于 persistence，但 Gaussian dense readout 长期不能保留同等改进。

**结论**：模型可以优化 compact latent dynamics；“latent loss 下降”与“dense feature 可恢复”是两件不同的事。Gaussian readout 可能成为信息瓶颈，也可能暴露 object slots 根本没有保存局部信息。

### 5.2 v29-v36 carrier capacity 审计

这些版本大多不是重新长训，而是在固定 v28 parent checkpoint 上做 heldseed capacity audit。共同基线是 DINO token error 约 0.194。

| 版本 | 方法与执行入口 | 关键结果 | 判定与推导 |
|---|---|---|---|
| v29 | hierarchical Gaussian basis；`audit_hierarchical_gaussian_basis_v29.py` | child=2/4/8 gap recovery 约 0.145/0.275/0.354，低于 0.70 | 增加 Gaussian children 不能修复错误 object basis。 |
| v30 | dense object assignment readout；独立 readout 训练 | current error 0.183，优于 token 0.203；future relative gain vs persistence 为 -0.451 | current feature 可解码，但 future readout 破坏 Dynamics 改进；瓶颈不是单纯 loss 权重。 |
| v31 | adaptive object-centered carriers | current/future B64 error 0.438/0.442；gap recovery 约 0.414/0.416 | object center + local Gaussian 只有部分容量。 |
| v32 | attention carriers | current/future B64 error 0.617/0.617，比 v31 更差 | 让全局 attention 选择 256 carriers 会丢失 object-centered inductive bias。 |
| v33 | signed object residual field | geometric B256 recovery 约 0.863，oracle B256 约 0.999；系数幅度和 condition number 很高 | basis 有容量，但预测 geometry 驱动时数值不稳，不能部署。 |
| v34 | stable residual transport | B256 recovery 约 0.866，但 Dynamics error 约 persistence 的 11.45 倍 | 限制系数没有修复 transport 的结构错误。 |
| v35 | partition-of-unity experts | B256 recovery 降至约 0.434；full/persistence 约 1.004 | smooth partition 把独立局部变化再次平均掉。 |
| v36 | orthogonal residual basis | current/future B256 recovery 约 0.872/0.856；predicted rigid transport 约 persistence 的 7.60 倍 | orthogonalization 改善 conditioning，不会自动产生正确 object motion。 |

### 5.3 v37-v42 temporal 与 lifecycle

| 版本 | 修改 | 结果与推导 |
|---|---|---|
| v37-v38 | change-only loss、原生 30 Hz、RGB cache 与 JIT DINO、strict resume | 修正 temporal/data contract；证明旧 16.67 Hz cache 不应继续作为 30 Hz 实验依据。 |
| v39 | history $H\in[1,4]$、固定 +1 秒 short target、terminal goal、direct/rollout | 14,560 steps 时 object Dynamics relative gain 约 0.525，但 dense readout gain 约 -0.230 | compact state 能拟合 short transition，readout 仍差；无条件 long goal 不唯一。 |
| v40 | lifecycle 与 relative transport | 加入存在、可见、遮挡传播的明确状态契约 | 结构 gate 通过不代表 identity 语义成立。 |
| v41 | correspondence 与 track presence | 多次暴露 identity normalization、BF16/FP32 和 reappearance gate 问题 | 固定 slot 的“同 index”不能继续充当主要 correspondence。 |
| v42 | stabilized correspondence/presence | deterministic gate 可运行 | 仍未得到外部 object truth，继续调 slot loss 的收益有限。 |

## 6. v43-v49：Region capacity 与固定 slot Object State

### 6.1 v43 Object-Region JEPA

**结构**：DINO top layers、64-256 regions、16 roots、causal region transformer、hierarchical Dynamics，课程为 representation → short Dynamics → posterior dual horizon。

**执行**：`manage_object_region_jepa_v43.sh`，8 卡、effective batch 256。代表 run `f5ig7bsc` 到 19,620 steps。

**结果**：

- short relative gain over persistence 约 0.945；说明短期目标可被强力拟合。
- region effective-rank fraction 约 0.277；比早期约 0.11 的观察更好。
- region presence 约 0.996、active allocation 接近上限；capacity 没有真正自适应。
- history H1-H4 的 short root error 没有形成稳定“更多历史更好”的单调关系。

**推导**：action-free short prediction 很容易利用视频平滑性和 representation self-consistency，并不能证明 state 是 object。后续不再把它作为主成功标准。

### 6.2 v44-v49 固定 slot 路线

| 版本 | 方法 | 运行结果 | 推导 |
|---|---|---|---|
| v44 | pure-video temporal object set；50k 长训 | run `dlm5qfva` 完成 50k，末次 loss 约 0.458 | 可稳定长训，但稳定 recurrence 不等于稳定 object identity。 |
| v45 | predictive object tube | 加强 tube-level future prediction | 仍依赖模型自身 slot/tube 定义，闭环监督未打破。 |
| v46 | observation-complete state | 加强当前观测、geometry 与 FP32 稳定性 | 改善数值，不改变 object assignment 的监督来源。 |
| v47 | grounded object state | 增加 foreground/object margin | ground evidence 仍由模型内部 grouping 解释。 |
| v48 | VideoSAUR-like slot contrast；稳定 recurrence | run `l1eu5wr3` 完成 50k；H8/H16/H24/H32 assignment entropy 波动明显 | 模型能形成少数稳定 partitions，但未证明 partitions 对应语义对象。 |
| v49 | trajectory-anchored object state、zero-effect identity | 把轨迹引入 slot training | 方向正确，但如果只给旧 slot 加 tracker loss，仍会重现自洽闭环。 |

## 7. v50-v56：Point-track teacher、Objective-first 与六源训练

### 7.1 v50-v51

**v50 结构**：冻结 CoTracker 产生训练期 tracks；RGB-only student 仍输出固定 16 object slots；identity、dynamic、geometry、visibility、existence 拆开；compositional decoder 评估 deletion locality。

**v50 evaluation**：run `xo5ezwv5` 在 heldseed/heldtask、H8/16/24/32 上完成。track assignment correct cosine 多在 0.964-0.989，deletion locality ratio 最低约 2.94；但 dynamic motion probe relative gain 从约 -0.242 到 +0.213，不稳定。

**推导**：tracker relation 能让 assignment 与 tracker 自洽，decoder deletion 也更局部；但 dynamic state 没有稳定携带 motion。teacher 与 evaluator 共用 tracks，不能证明 object semantics。

**v51**：加入 lifecycle、memory/effect dimension 修复与 H schedule telemetry。run `ps611f8n` 到约 13,280 steps，loss 约 2.026。训练被运行问题中断，但更重要的是 objective 本身仍混合固定 root assignment 与 teacher evidence。

### 7.2 v52 Learning-Objective-First

v52 先定义 intended object：persistent、compositional、relation-constrained visual entity，并让 objective 对八种 shortcut corruption 做 falsification。禁止 Dynamics、latent effect、language 与旧 checkpoint。

run `pmhrm7c6` 到约 16,760 steps。H8-H32 target effective roots 约 2.62-3.02，但 maximum root share 约 0.87-0.92。loss 可下降，root 仍明显集中。

**结论**：可证伪的静态 synthetic objective 是必要条件，但真实数据上的 evidence distribution 仍可能让模型找到 dominant-root shortcut。

### 7.3 v53 六源数据与 semantic latent tokenizer

v53 参考 RepWAM/AdaWorld 思路，把 DINO semantic latent 与 VAE-style latent action tokenizer 放进同一版本，并把数据扩展到六个真实机器人视频源。

数据构建经历的主要问题：AgiBot 索引指向缺失 mp4、HY packed videos 与 episode 映射不一致、损坏 AV1/OBU、worker 间 quarantine 不共享、替换池耗尽。对应修复集中在 `build_multisource_video_index_v53.py`、`multisource_point_track_dataset.py` 和 `video_file_decoder.py`。

代表 run `3urwjcux`：effective batch 512，约 28,680 steps，loss 约 3.266，decode replacement fraction 约 1.62%。该 run 最终 crashed，因此不能写成完整 tokenizer 结果。

**结论**：数据规模与 source diversity 建立了，但模型还没有先证明 semantic object state。先同时训练 tokenizer/dynamics 会把 representation 问题隐藏进 latent action。

### 7.4 v54-v55 relation graph

| 版本 | 代表 run | 结果 | 推导 |
|---|---|---|---|
| v54 relation-semantic | `4w8yhdou`，约 16,020 steps | loss 约 3.162；effective roots 约 2.15；largest root 约 0.953；teacher batch fraction 仅 0.0156 | 明确 dominant-root collapse。loss 下降与 object factorization 相反。 |
| v55 relation-component | `uchb3ndj`，约 28,440 steps | loss 约 2.652；component effective roots 约 1.35；largest component share 约 0.923 | relation graph 聚类后仍会收缩到少数 components；数值修复不是结构修复。 |

### 7.5 v56 verified relation objective

v56 删除 legacy owner、track-cycle、root-count loss，只保留 signed relation partition、contrastive cycle、positive object support、identity、motion、lifecycle、geometry 和低权重 reconstruction。训练期 tracker 占每 batch 的 12.5%。

**训练执行**：`manage_verified_relation_object_state_v56.sh`；六源、8 卡、每卡 64、effective batch 512、20,000 steps。run `0v5skdc6` 正常 finished，最终：

- object-state loss 约 0.563；external target 约 0.551；aux reconstruction 约 0.414。
- verified effective roots 约 2.545；maximum root share 约 0.562。
- decode replacement fraction 约 1.49%。

**完整 source-balanced evaluation**：run `bmtxgsqt`，6 sources × H4/H8，共 12 conditions：

| 指标 | 聚合结果 | 含义 |
|---|---:|---|
| effective roots | mean 2.538，range 1.904-3.057 | 没有退回单 root，但 16 roots 只使用少数。 |
| maximum root share | mean 0.563，max 0.689 | dominant-root collapse 被缓解。 |
| relation root margin | mean 0.675，12/12 为正 | same/different relation 在固定 roots 上可分。 |
| track correspondence margin | mean 0.172，12/12 为正 | 对训练 tracker 的 held correspondence 有效。 |
| identity reappearance margin | mean 0.201，12/12 为正 | tracker 定义的遮挡重现 assignment 比 shuffled 好。 |
| deletion locality ratio | mean 11.99，12/12 为正 | 删除一个 component 的 decoder 影响较局部。 |
| active motion readout vs zero | mean +0.199，10/12 为正 | 专用 readout 在多数条件能拟合 relation-component motion。 |
| post-hoc motion probe gain | mean -14.826，2/12 为正 | state 本身没有稳定、通用地线性编码 motion。 |
| visibility balanced accuracy | mean 0.4999 | 等同常数猜测。 |
| occluded recall | 12/12 为 0 | 模型没有学到 occluded state。 |
| visibility Brier gain | mean -0.216，12/12 为负 | visibility prediction 比常数基线更差。 |
| presence negatives | 0，positives 209,890 | 数据/teacher 无法识别 absence，presence loss 不可学习。 |
| causal prefix difference | 12/12 为 0 | student 没有读取 future；因果边界成立。 |

**v56 总结**：模型学到了 relation-compatible partitions 和局部 decoder attribution；没有学到稳定的 dynamic state、occlusion lifecycle 或可部署语义对象。固定 16 roots 让“整个 scene 如何分桶”成为主要难题，而我们真正需要的是“给定目标，哪个实体会怎样变化”。

## 8. 跨版本根因归纳

1. **目标不唯一**：无条件 history-only future prediction 面对多可能未来，只会学 persistence 或 conditional mean。
2. **压缩对象错误**：把 256/1369 DINO patches 直接压进固定 16 roots，会同时混合背景、机械臂、多个物体和局部区域。
3. **监督闭环**：模型产生 assignment，再用 assignment 定义 object，再用同一 decoder 评价 object，loss 可在错误分解上下降。
4. **teacher 不等于 truth**：CoTracker 给 surface correspondence 和 visibility，不给 semantic instance truth；共同运动也可能来自机械臂、阴影或 camera motion。
5. **readout 不是唯一根因**：v29-v36 证明 basis capacity 可提高到 85%-99% oracle recovery，但 predicted transport 仍远差于 persistence。
6. **lifecycle 不可识别**：tracker 不可见只能说明 unknown/occluded candidate，不能推出 absent；v56 数据没有 negative presence。
7. **更多数据不会自动修复 objective**：v53-v56 已扩到六源和约 4,073 万 samples，结构问题仍然出现。
8. **DINO 不是当前首要瓶颈**：DINO token 可被较高容量 basis 重建；真正问题是 object grouping 与 dynamics target。v57 第一阶段继续冻结 DINO。

### 8.1 历史路线判决矩阵

这张表是防止后续迭代“倒回去”的依据。`保留` 表示已经形成可复用能力；`否决` 表示不能再次作为主线成功标准；`后置` 表示只有前置 Gate 通过后才能恢复。

| 历史路线 | 已经证明的能力 | 已经证明的失败 | 永久判决 |
|---|---|---|---|
| 显式 3DGS motion / RGB rendering | geometry target 足够好时可预测局部 motion；反事实可迫使模型使用 language | 依赖 geometry、mask 或 rendering，不等于纯视频 object abstraction | **退出主线**；只保留 geometry 与 counterfactual 经验。 |
| Gaussian readout / carrier v28-v36 | compact latent 可以被训练；oracle basis 有较高 dense capacity | predicted transport/readout 多次丢掉 latent Dynamics 改进 | **否决其作为主 state 或主评估**；dense readout 仅作 probe。 |
| Action-free short prediction v39/v43 | 能拟合 +1 秒视频平滑性；验证 causal temporal sampler | persistence/self-consistency 足以让指标很好，不能证明 object 或多可能 dynamics | **永久取消为 object-state 成功标准**。 |
| 固定 16-slot / root v40-v48 | recurrent set、relative geometry、数值稳定和长训工程可用 | identity 由固定 index 自证；background locking、fragmentation、capacity saturation 持续存在 | **否决固定 slot index 作为 identity truth**；不回到“调 slot 数/均衡权重”路线。 |
| Point-track + fixed roots v49-v56 | external correspondence、relation partition、reappearance 和 deletion locality 可测 | teacher/evaluator 闭环；dominant-root collapse；visibility 与 motion state 不稳定 | **保留 tracker teacher 与独立评测思想，否决 fixed-root student**。 |
| 六源 semantic tokenizer / Dynamics v53 | 六源数据、decode audit、balanced sampler 和大 batch 基础设施已建立 | object state 未通过时 joint tokenizer/Dynamics 无法解释失败来源 | **基础设施保留，tokenizer/Dynamics 后置到 G4/G5**。 |
| Query-conditioned binding v57 | held-out positive/negative track binding 明确可学，且不依赖固定 slot index | visibility collapse；完整 support、lifecycle、independent object semantics 与 Dynamics 均未成立 | **当前唯一结构主线**；只迭代其最早失败 Gate，不推倒重建。 |

### 8.2 不再改变的研究边界

后续版本不得重新引入以下替代目标：

- 以 whole-frame reconstruction、Gaussian rendering 或 dense DINO error 作为主要成功标准；
- 以固定数量 slots 的均匀使用率作为 object 定义；
- 以 action-free short prediction 优于 persistence 作为 object state 证据；
- 在 object state 尚未通过 G3 时同时训练 latent tokenizer、Dynamics、Prior 或 language；
- 因某个辅助 loss 失败而把主线切换回旧 readout、旧 roots 或显式 instance segmentation。

允许的迭代只包括：修正当前 Gate 的目标定义、teacher evidence、student state factorization、独立 evaluator 和必要的容量；每次必须相对最近 accepted baseline 做可归因对照。

## 9. v57 Query-Conditioned Object Dynamics

### 9.1 为什么改成 query-conditioned

固定 slots 要求模型在没有任务定义时枚举整张 scene 的“所有对象”，这个目标在纯视频中本身不确定。query-conditioned state 把问题改成：给定当前观测中的一个点，找出它所属的可持续实体，并编码它的变化。

训练期 teacher 可以看完整视频来挑选一个有 relation evidence 的当前点，但 student 只能读取 observed RGB history 与该点坐标：

$$
q_t=\left(x_t,y_t\right),\qquad
O_t(q_t)=E_{\theta}\left(S_{\le t},q_t\right),
$$

其中 $S_t=E_{\mathrm{DINO}}(I_t)$，DINO 冻结。

point tracks 被拆成两部分：

- prompt：query track 与同实体 alternate seed；
- held-out：未给 student 的 same/different relation tracks，用于验证 support 是否扩展到实体而不是记住一个点。

第一阶段 objective：

$$
L_{\mathrm{bind}}=
L_{\mathrm{heldout-relation}}
+\lambda_1L_{\mathrm{seed-id}}
+\lambda_2L_{\mathrm{seed-support}}
+\lambda_3L_{\mathrm{query-separation}}
+\lambda_4L_{\mathrm{semantic}}
+\lambda_5L_{\mathrm{compact}}.
$$

它不包含全图 RGB reconstruction、不包含固定 root count、不包含 future prediction。

### 9.2 当前已实现的第一项

当前分支已增加：

- `query_object_teacher_v57.py`：从现有 trajectory relation teacher 选择 query、alternate seed、negative seed，并严格拆分 held-out tracks。
- `query_object_state_v57.py`：无固定 slots 的 RGB-history-only single-query encoder，输出 support、identity、dynamic、center、covariance、visibility 和 pooled semantic。
- `query_objective_v57.py`：held-out relation、same-seed invariance、different-query separation、semantic consistency 与 compactness。
- `test_query_object_binding_v57.py`：synthetic causal contract、gradient contract 和 shortcut falsification。

这一项只证明“目标函数与接口没有立即奖励 all-scene、whole-frame、other-entity 或 seed-only shortcut”。它还不是实际视频上的成功结果。

2026-08-23 已在远端 `/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/` 使用提交 `4a349b3138ad099980a2d59e2fa0159d6bdcebaf` 完成 tensor gate：

- `status=passed`，architecture 与 checkpoint contract 分别为 `query_conditioned_object_state_v1`、`57`；
- student 不读取 point tracker，交换 teacher relation 不改变 student 输出，`student_teacher_swap_max_difference=0.0`；
- teacher 确实读取 future tracks，交换 relation 后 target 变化，`teacher_swap_max_difference=1.0`；
- prompt 与 held-out tracks 无重叠，`prompt_heldout_overlap=0.0`；
- 33 组 student parameter tensors 获得有限梯度；
- reasonable solution 的 objective 为 `0.03096`，低于 whole-frame `5.07872`、seed-only `14.17658`、all-scene `14.31552` 和 other-entity `19.34216`。

该结果验证了 synthetic interface、因果边界、梯度路径和已列 shortcut 的排序。它没有覆盖六源真实视频的 teacher coverage、真实 object binding、独立标注评测或 Dynamics，因此不能据此声称 object state 已学成。

## 10. v57 Gate 与实验设计

### Gate A：Single-query teacher coverage

在六源真实数据上统计：query valid、alternate seed、negative seed、held-out positive/negative coverage。每个 source × H 单独报告。coverage 不足时先修改 query sampling，不训练 student。

### Gate B：Single-query binding

从 scratch 训练，只使用 observed history。必须满足：

1. same-entity 不同 seed 的 support IoU 和 identity cosine 显著高于 different-query。
2. query shuffle 后 support 必须移动，不能保持同一 foreground template。
3. held-out same tracks 的 support probability 高于 held-out different tracks。
4. all-scene、whole-frame、seed-only 与只跟机械臂的解都比正常解差。
5. H 增加时，遮挡或运动 active subset 至少不退化。

### Gate C：Independent object binding

使用 RoboTwin simulator mask/object ID 或少量人工标注，禁止 evaluator 导入 CoTracker。检查 object coverage、cross-object leakage、gripper separation、reappearance 与 deletion locality。Teacher agreement 与 independent truth 必须分别报告。

### Gate D：Single-entity latent effect

冻结通过 Gate C 的 encoder，再训练 posterior $q_{\phi}$ 与 Dynamics $F_{\xi}$。正确 effect 相比 zero/shuffled 至少改善 10%；direct 与 composed rollout 同时优于 persistence。

### Gate E：Multi-query 与 selector

最后才增加多 query 去重、persistent queried memory，以及 goal/language/policy selector。Language 的作用是选择 query/effect，不进入底层 object binding 定义。

### 10.1 Step 6750 训练诊断与正式判决

V57 真实六源 coverage、startup gate 和 4-GPU strict-resume E2E 已通过。长训 run `q02tvfqq` 使用提交 `0c7cf6f204c2a57bb099d724fb655b9723d38159`，8 卡、每卡 batch 16、effective batch 256，原目标 30,000 steps。W&B 最终状态为 `killed`，最后 history step 为 6,760；恢复 checkpoint 为 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/query_object_binding_v57_seed17_0c7cf6f_long_foreground/v57_binding_recovery.pt`，记录 step 6,750。最后一条 summary 为：

- `heldout_support_positive=0.9771`，`heldout_support_negative=0.0149`，说明给定 query 后，student 能在训练 teacher 定义的 relation 上区分 related 与 different tracks；
- `heldout_relation=0.0898`，相比训练初期显著下降；
- `visibility_mean=4.82e-6`，已经完全塌缩；
- `support_probability_mean=0.3890`，说明未被明确 relation 标注的 patches 仍缺乏完整约束；
- `semantic_consistency=1.36e-11`、`compactness=4.24e-5`，不能解释为 temporal semantics 或 compact object support 已学成；
- `runtime/grad_norm=0.0941`，训练数值稳定，但这只证明错误目标仍可继续优化。

visibility collapse 来自当前 objective 的直接捷径：

$$
L_{semantic}=\frac{\sum \Delta s^2 v_t v_{t-1}}
{\max(\sum v_t v_{t-1},1)},\qquad
L_{compact}=\frac{\sum \operatorname{tr}(\Sigma_t)v_t}
{\max(\sum v_t,1)}.
$$

当前没有独立的 visibility target。令 $v_t\rightarrow 0$ 会同时把两个 loss 压到零。因此本轮证明的是 **G1 query-conditioned track binding 可优化**，不是 G2 persistent object state 已成立。继续训练到 30,000 steps 不会自行修复该目标漏洞。

正式判决为 `iterate at G2`：

- **保留**：single-query student、training-only tracker teacher、prompt/held-out track split、positive/negative relation binding、六源数据与运行基础设施；
- **否决**：v57 visibility、由 student visibility 加权的 semantic consistency/compactness，以及把这两个接近零解释为成功；
- **不重启旧路线**：不回到固定 roots、Gaussian readout、action-free prediction 或 joint tokenizer/Dynamics；
- **下一实验唯一问题**：在不削弱 v57 binding 的前提下，让 observability、unknown 和 object support 获得不可被 student 自己关闭的外部监督。

## 11. 当前 TODO 与单向推进规则

以下11.1–11.3保留原始决策上下文；2026-10-05之后的执行优先级以15.58为准。历史Gate未通过不等于已获授权的V69训练需要停止；工程运行、研究假设和成功声明分别记录。

### 11.1 当前 Gate 状态

| Gate | 状态 | 证据 | 下一动作 |
|---|---|---|---|
| G0 Objective validity | **dynamic target 待验证** | v52 的 synthetic corruption gate 只证明手工 objective 排序；V61 motion target 允许 near-zero shortcut | 先做 V62 continuous target codec 与 teacher-state oracle。 |
| G1 Query binding | **held-teacher 已验证，继续冻结** | v57 `q02tvfqq` positive 0.9771、negative 0.0149 | 复用 query/track graph，不再优化固定 slot binding。 |
| G2 Persistent state | **失败** | V61 dynamic effective rank 2.08/2.75，motion probe 全部低于均值 baseline，source identity 可读性 97.39%/99.38% | oracle target 成立后，从 RGB-only Student 重新学习 state；不继承 V61 state checkpoint。 |
| G3 Independent object validity | **未通过** | V61 复评 truth scope 仍是 training tracker teacher | Student state 通过后执行 RoboTwin truth/人工小集 evaluator。 |
| G4 Latent effect | **只允许诊断，不允许晋级** | v59/v60 证明 teacher compact target 上 posterior effect 可被使用，但 target 信息不足且 short/static calibration 不稳定 | 用 teacher-state oracle 隔离检验 effect target；结果不能覆盖 G2/G3。 |
| G5 Object Dynamics | **待办** | 现有 action-free 与 compact-target Dynamics 均不能证明 object-level future state | teacher oracle、Student state 与 independent validity 依次通过后才训练正式 Dynamics。 |
| G6 Selector / task A | **禁止提前** | 尚无可部署 state 与 effect | 等待 G5。 |

### 11.2 下一版本的唯一改动面

V62 是同一条主线下的一个完整 program，但严格按顺序运行：

1. 先建立不依赖 Student 的 continuous teacher object observation 与 state codec，确认学习目标本身保留 object-local semantic、support 和 lifecycle。
2. 在固定 $100\,\mathrm{ms}$ 上运行 teacher-state transition oracle，确认正确 transition effect 必须优于 persistence、zero 和 matched-shuffled effect。
3. oracle 通过后才把 deterministic effect 改为 Gaussian VAE posterior，并用 rate-distortion 选择容量。
4. 只有 target、oracle 与 effect bottleneck 都通过，才从 observed RGB history 训练单一路径 RGB-only Student；DINO/SigLIP 是同一 Student 的 frozen perception 输入，不拆成两个 Student 分支。
5. Student state 通过 G2/G3 后，才训练 Student-conditioned Dynamics、multi-horizon rollout 和部署期 Prior。

完整模型、数据、loss、执行预算和判决树见第 15 节。V62 的第一项实现只允许覆盖第 15.7 节的 E0 与 E1，禁止一次启动端到端长训。

### 11.3 严格晋级标准

晋级不再由单个 loss 或相对 baseline gain 决定。V62 必须同时报告：

- 对真实 held teacher observation 的 absolute semantic、continuous support、lifecycle distortion；
- correct、zero、matched-shuffled 和 persistence 四条路径的原始误差与样本数；
- static、motion-active、occlusion、六源、history length 和 delta-time 分解；
- effect 的 active units、effective rank、KL/rate、donor identity/source leakage；
- Student state 的 retrieval、nuisance probe、Markov sufficiency 与 independent truth。

每阶段的数值门槛见第 15.8 节。E0 或 E1 失败时，不允许通过增加 Student、Prior、语言、数据量或训练步数来掩盖目标失败。

## 12. W&B 运行索引

下表列出当前项目中与本主线直接相关的 run。`crashed/killed/failed` 只表示运行终态，不能自动解释为方法失败；对应方法结论以正文的 held metrics 为准。

| 版本 | Run ID | 名称缩写 | 状态 |
|---|---|---|---|
| v28 | `0qoe6cw8` | representation 2cd5d42 | failed |
| v28 | `bcb73gbj` | representation 6313872 | crashed |
| v28 | `k4b1ccxp` | representation 49d7a6e restart | killed |
| v28 | `ytasdoka` | representation restart | crashed |
| v28 | `3oaf56tx` | representation restart | crashed |
| v30 | `mswofu6u` | dense readout 33e34f0 | crashed |
| v39 | `0b84gd2y` | dual horizon 35cbbea | crashed |
| v39 | `qlfpo2i8` | dual horizon 70ce163 | crashed |
| v41 | `ushna3gz` | correspondence/presence | crashed |
| v42 | `8ayrujep` | stable correspondence | crashed |
| v43 | `f5ig7bsc` | object-region JEPA | crashed |
| v44 | `kruz2icx` | temporal object set | failed |
| v44 | `ifo82316` | temporal object set | crashed |
| v44 | `dlm5qfva` | temporal object set high-memory | finished |
| v45 | `1l6jq893` | predictive object tube | crashed |
| v46 | `1sh4farc` | observation-complete state | crashed |
| v48 | `wrf0crz0` | slot contrast | failed |
| v48 | `l1eu5wr3` | stable recurrence | finished |
| v48 eval | `ius318i7` | held evaluation | finished |
| v48 eval | `fvhm1ik4` | held evaluation | finished |
| v50 | `toofrk21` | point-track state | crashed |
| v50 | `vq82tha5` | point-track state | killed |
| v50 eval | `0qsm4f54` | comprehensive eval | failed |
| v50 eval | `xo5ezwv5` | comprehensive retry | finished |
| v51 | `k0d0ca9m` | lifecycle state | failed |
| v51 | `wd3yy14m` | lifecycle state | crashed |
| v51 | `ps611f8n` | lifecycle state ac84b98 | crashed |
| v52 | `pmhrm7c6` | objective-first state | crashed |
| v53 | `cdujav8i` | multisource tokenizer | crashed |
| v53 | `ljzsrwz7` | multisource tokenizer | crashed |
| v53 | `q9kdtgln` | multisource tokenizer | crashed |
| v53 | `8h7333jn` | multisource tokenizer | crashed |
| v53 | `jjeo8m4a` | multisource tokenizer | crashed |
| v53 | `3urwjcux` | multisource tokenizer e84fac0 | crashed |
| v54 | `4w8yhdou` | relation-semantic state | crashed |
| v55 | `h6kxfzjd` | relation-component state | failed |
| v55 | `uchb3ndj` | relation-component state | crashed |
| v56 | `0v5skdc6` | verified relation state | finished |
| v56 eval | `dy4q7hv7` | complete eval attempt 1 | failed |
| v56 eval | `o6bdk6g3` | complete eval attempt 2 | failed |
| v56 eval | `pu5rvd7o` | corrected eval | finished |
| v56 eval | `bmtxgsqt` | aligned final eval | finished |
| v57 gate | `pcafelg1` | six-source query coverage | finished |
| v57 E2E | `gscprsrp` | query binding 4-GPU E2E | finished |
| v57 | `q02tvfqq` | query binding long foreground | killed at history step 6,760; recovery step 6,750 |

## 13. 当前明确禁止的捷径

- 不再用固定 slot index 作为 object identity truth。
- 不用训练 tracker 指标冒充 independent object evaluation。
- 不用 action-free short prediction 下降证明 object state 成功。
- 不用 full-image RGB 或 dense DINO reconstruction 主导 object dynamics。
- 不从 tracker invisibility 推导 absence。
- 不在 Object State gate 通过前训练 latent effect、Prior、language 或控制接口。
- 不继承 v43-v56 model/optimizer checkpoint；只复用数据、冻结 DINO、冻结 point tracker、W&B 与运行基础设施。

## 14. 决策日志

### 2026-08-23：v57 Query Binding 长训判决

- **版本证据**：branch `codex/query-conditioned-object-dynamics-v57`；commit `0c7cf6f204c2a57bb099d724fb655b9723d38159`；W&B `q02tvfqq`；recovery checkpoint `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/query_object_binding_v57_seed17_0c7cf6f_long_foreground/v57_binding_recovery.pt`。
- **当前 Gate**：G1 已通过 held-teacher 标准；G2 失败。
- **假设**：用 query 与外部 trajectory relation 监督，可以避免固定 slot index，并从 observed RGB history 学到 query-conditioned object support。
- **主要改动**：从固定 16 roots 改为 single-query object encoder；teacher prompt tracks 与 held-out relation tracks 分离。
- **冻结项**：DINO 与 point tracker 冻结；dynamic head 冻结；无 language、显式 action、RGB reconstruction、Dynamics 或旧 checkpoint。
- **执行**：六源数据；8 卡；每卡 batch 16；effective batch 256；从 scratch；计划 30,000 steps，外部中断于 history step 6,760。
- **结果**：positive 0.9771、negative 0.0149，relation binding 成立；visibility 4.82e-6，persistent state 不成立。
- **证伪**：all-scene、whole-frame、seed-only 与 other-entity synthetic shortcut 已通过；all-visibility-zero 未被 objective 排除并在真实训练中发生。
- **判决**：`iterate`。v57 作为 G1 accepted baseline；不继续训练、不回到旧路线；下一版本只修 G0/G2。
- **继承**：继承 query encoder 接口、relation teacher、六源数据与执行基础设施；不继承当前 visibility/compactness objective，也不从 v57 checkpoint resume 新 objective。
- **下一步**：完成第 11.2 节的 G2 objective、falsification 和 evaluation；通过前禁止 Dynamics。

### 2026-08-23：v58 Persistent Query State 实现决策

- **代码版本**：branch `codex/query-persistent-object-state-v58`；核心实现提交 `2a8519c`。
- **当前 Gate**：只处理 G0/G2；G1 relation binding 保持不变；G3-G6 继续禁止提前实现。
- **核心假设**：v57 的失败来自错误 objective，而不是 query binding capacity。只要 observability、identity persistence 和 observed motion 都由不可被 student 关闭的 teacher evidence 监督，query-conditioned encoder 才有机会学习可部署的 persistent state。
- **不继承资产**：v58 从 scratch 训练，不允许 v57 `init_from`，也不允许用 v57 checkpoint `resume`；只复用六源数据、冻结 DINO、冻结 point tracker、relation teacher 和 DDP/W&B 基础设施。
- **新增状态**：student 输出逐帧 `identity_sequence`、`dynamic`、relative support geometry、`visibility_logits`，以及序列级 `identity`。student 输入仍只有 observed RGB 的 frozen DINO patches 与 current query coordinate。
- **teacher lifecycle**：训练期 tracks 为 query surface point 产生 `visible`、`occluded_candidate` 和 `unknown`。tracker invisible 不会被解释为 absent；query anchor 之前或无法确认的帧进入 unknown。
- **motion target**：只用 observed prefix 内相邻可见 track 的 camera-motion-corrected residual velocity；按 observed frame time 归一化，不读取 future RGB，不使用机器人 action，也不把真实 center delta 拼进 latent state。

v58 objective 为：

$$
L_{v58}=L_{bind}+L_{visibility}+L_{identity-persistence}
+L_{semantic}+L_{compact}+L_{dynamic-motion}+L_{geometry-motion}.
$$

其中：

$$
L_{visibility}=
\operatorname{BalancedBCEWithLogits}
\left(\hat v_t,v_t^{teacher};m_t^{known}\right).
$$

semantic、compactness 与 identity persistence 的 mask 全部来自 detached teacher lifecycle：

$$
L_{semantic}=
\frac{\sum_t m_t^{visible}m_{t-1}^{visible}
\left\|\bar s_t-\bar s_{t-1}\right\|_2^2}
{\max\left(\sum_t m_t^{visible}m_{t-1}^{visible},1\right)}.
$$

student 自己预测的 $\hat v_t$ 只进入 $L_{visibility}$，不再充当其他 loss 的开关。因此 $\hat v_t\rightarrow0$ 会增加 visibility loss，不能再把 semantic/compactness 人为压到零。

dynamic state 通过一个小型 readout 对 observed residual velocity 负责：

$$
\hat r_t=h_{motion}(d_t),\qquad
L_{dynamic-motion}=\operatorname{SmoothL1}(\hat r_t,r_t^{track}).
$$

support center 的变化只作为 relative geometry state 的辅助检查：

$$
L_{geometry-motion}=\operatorname{SmoothL1}
\left(\frac{c_t-c_{t-1}}{\Delta t},r_t^{track}\right).
$$

这不是 latent action，也不是要求用绝对坐标表示 action。它只检查 query support 的相对几何是否随 observed object evidence 移动。

**实现文件**：

- `query_object_teacher_v58.py`：observed lifecycle、unknown 和 residual velocity target。
- `query_persistent_state_v58.py`：逐帧 identity/dynamic/geometry/visibility state。
- `query_persistent_objective_v58.py`：teacher-masked objective 与 calibration/reappearance/motion diagnostics。
- `query_object_coverage_v58.py`：六源 × H coverage 与 occlusion evidence gate。
- `test_query_persistent_object_state_v58.py`：H1-H4 gradient contract 和 constant-visibility falsification。
- `verify_query_persistent_object_state_v58.py`：真实数据 causal、frozen teacher、query sensitivity、mask independence 和 backward gate。
- `evaluate_query_persistent_object_state_v58.py`：六源 × H held-teacher G2 promotion evaluator，并同步 W&B。
- `train_query_persistent_object_state_v58.py`、`v58_training_loop.py`、`v58_checkpointing.py`：fresh/strict-resume DDP 长训。

**必须先通过的执行顺序**：

1. `audit-coverage`：六源 × H=1,2,3,4 的 query/held relation coverage 必须继续通过，aggregate 必须实际包含 occluded candidate。
2. `verify`：student future swap 差异小于 $10^{-6}$；DINO/point tracker 冻结；H1-H4 所有 trainable parameters 均在反向图；collapsed visibility 必须被直接惩罚，且不得改变 semantic/compactness。
3. fresh 长训：默认 30,000 steps、effective batch 256、GPU 数量和单卡 batch 自动选择；不继承 v57。
4. held evaluation：六源 × H 分别检查 binding、visibility calibration、reappearance、query perturbation 与 motion readout。

**v58 晋级判据**：

- held support positive $\ge0.90$，negative $\le0.10$；
- 有 visible/occluded 两类证据的 condition 上，visibility balanced accuracy 和 F1 均 $\ge0.55$，Brier 优于常数 baseline，visible-rate relative error $\le20\%$；
- observed reappearance condition 的 identity margin 相对 batch-shuffled identity 至少 $0.02$；
- query perturbation 的 support mean difference 大于 $10^{-3}$；
- 有 observed motion evidence 的 condition 中，dynamic motion prediction 必须优于 zero-motion baseline；
- aggregate 中必须存在真实 occlusion evidence。

上述 evaluator 仍使用训练期 tracker teacher，只能决定 G2 是否成立，不能替代 G3 independent object truth。即使全部通过，下一步也只能进入 RoboTwin mask/object ID 或人工小集的 independent evaluator；不能直接恢复 latent effect 或 Dynamics。

**当前验证状态**：本地完成 Python 静态编译、Ruff、shell syntax 和 `git diff --check`。服务器
`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source` 已使用
`/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/bin/python` 通过独立 tensor gate：
`gradient_tensor_count=41`，H=1/2/3/4 的 gradient norm 分别为
`17.1617/19.9988/14.1236/12.6101`，`dynamic_head_trainable=true`，
`student_visibility_gates_other_losses=false`；合理状态 objective 为 `1.8163`，
`all_visibility_zero` 和 `all_visibility_one` 均为 `5.8163`。这只验证 objective 与梯度契约，
尚未验证六源 coverage、真实 GPU startup、长训收敛或 held G2 指标，因此当前状态仍是
“静态与 synthetic tensor gate 已通过，等待真实数据 Gate”，不是“v58 已通过 G2”。

### 2026-08-23：v58 Real Coverage 失败与 Tracker 时序契约修正

- **失败证据**：W&B run `01ueptqv`，名称 `query_object_coverage_v58_837ab37`，状态 `finished`，但 gate status 为 `failed`。对应服务器报告为 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/v58_gates/837ab37e4245ba0bd7143034d9545028582d6178_coverage.json`。
- **已通过部分**：六数据源乘以 H=1,2,3,4 的 24 个 condition 全部存在；每个 condition 都有 query coverage 和 trainable example；prompt/heldout disjoint、aggregate trainable coverage、future-track teacher sensitivity 均通过。aggregate trainable fraction 为 `0.9895833`，query valid fraction 为 `1.0`。
- **唯一失败项**：`aggregate_contains_occlusion_candidates=false`；所有 condition 的 `lifecycle_occluded_candidate_fraction` 都为 `0`，aggregate visible fraction 为 `1.0`，unknown fraction 为 `0`。后续 verifier 因 coverage report 的 `status=failed` 拒绝启动，这是正确的阻断行为。
- **根因**：旧 tracker 只做 forward tracking，且 query anchor 由完整 clip 的固定比例产生，没有保证在 observed current frame 发出 query。v58 却要求从当前 query 向历史维护 persistent state，因此原数据路径无法可靠观测“历史可见、中间不可见、当前重新可见”的 enclosed occlusion。这是 tracker temporal contract 与 state objective 不一致，不是 coverage 阈值过严。
- **修复**：CoTracker 显式启用 bidirectional tracking；额外在 observed current frame 建立 tracker query；occlusion 只定义为前后都有可见证据的中间不可见帧，边界处缺少证据的 invisibility 保持 unknown；若当前可见 query 中存在 observed reappearance track，teacher 优先选择该 track；默认 temporal steps 扩展为 `100,200,400,800` ms，real coverage 每个 condition 的样本数从 8 提高到 16。
- **防回归契约**：synthetic test 必须选中 reappearing query，并验证 enclosed occlusion 不被标成 unknown；coverage report 和 startup verifier 必须同时记录并要求 `tracker_bidirectional=true` 与 `tracker_include_observed_current_anchor=true`。旧 coverage JSON 不满足新契约，不能用于训练启动。
- **当前状态**：代码已实现并通过本地静态检查；新的六源 real coverage、GPU startup verifier、训练和 held evaluation 均尚未执行。只有新的 coverage report 中 `aggregate_contains_occlusion_candidates=true` 后，才允许重新启动 v58 fresh training。

**后续执行结果**：提交 `3fa9610` 的六源 real coverage 已返回 `COVERAGE_RC=0`，因此 bidirectional/current-anchor 修复已经通过真实 coverage gate。startup verifier 随后暴露独立实现错误：masking falsification 将 4 帧 student support 与 8 帧完整 teacher clip 的 feature-valid mask 相乘。修复要求 verifier 显式构造 4 帧 observed feature prefix，不在 objective 内静默裁剪。该 verifier 修复仍需服务器复跑；在 `VERIFY_RC=0` 之前不启动训练。

### 2026-08-24：v59 Dynamic Objective 优先诊断决策

- **代码版本**：branch `codex/object-transition-objective-v59`；核心实现提交 `a0304ae`。
- **用户决策**：当前最高优先级转为解决 dynamic objective。visibility、coverage、reappearance 和 independent object validity 暂不作为本轮主要优化项。该决策允许实现 G4/G5 的目标诊断代码，但不代表 G2/G3 已通过，也不允许将本轮 teacher-dependent 结果写成 object semantics 已成立。
- **历史问题**：v39 的 action-free short prediction 可以利用 persistence/video smoothness；v43-v53 在 Object State 未成立时联合训练 Dynamics，失败来源不可归因；v58 的 `dynamic` 只通过小 readout 回归单个 tracked point 的二维 residual velocity。三者都没有验证 latent effect 对 future object state 的 predictive sufficiency 和 necessity。
- **本轮唯一假设**：若一个连续 latent effect 真正表示 object transition，则在冻结同一个 source Object State 后，正确 posterior effect 必须稳定优于 zero effect、其他样本的 shuffled effect 和 persistence，并且这种优势应在 motion-active object 与多个时间跨度上同时出现。
- **继承资产**：六源视频 index、decode frontier、冻结 DINO、冻结 CoTracker、v58 query encoder checkpoint、group-balanced sampler、DDP、W&B 和 rolling checkpoint 基础设施。
- **不继承资产**：不恢复 v58 optimizer/scheduler；不使用 v58 `motion_readout`；不把 point velocity、RGB、dense DINO reconstruction、显式 center delta、机器人 action、language 或 History-only Prior 作为 latent effect。
- **冻结控制**：v58 encoder 全部冻结；DINO 与 CoTracker 全部冻结；只训练 Posterior、source-state adapter 与 effect-conditioned Dynamics。这样失败可以归因到 transition target/objective 或冻结 Object State 的信息不足，而不是 encoder 与 target 同时漂移。

训练期 target 由 query 对应的一组 positive-relation tracks 聚合，而不是一个 surface point：

$$
Y_t=\left(y_t^{semantic},y_t^{relative\ geometry},y_t^{visibility}\right).
$$

其中 $y_t^{semantic}$ 是 related tracks 上 frozen DINO feature 的加权聚合；$y_t^{relative\ geometry}$ 包含 object center 相对当前可见 track field center 的二维位置以及 object support 的对称 covariance；visibility 表示该 related-track object support 当前可观测的比例。完整 target 始终 detached，只进入 training-only teacher path。

Posterior 与 Dynamics 为：

$$
z_{t\rightarrow t+\Delta}=Q_{\phi}(Y_t,Y_{t+\Delta},\Delta),
$$

$$
\hat Y_{t+\Delta}=F_{\theta}(S_t,z_{t\rightarrow t+\Delta},\Delta),
$$

其中 $S_t$ 是冻结 v58 encoder 从 observed RGB/DINO prefix 和 current query 得到的 source state。$z$ 保持连续 `[4,32]`；4 表示 effect factors，不是四个时间点。

核心预测距离同时比较 semantic cosine、归一化 relative geometry 和 lifecycle BCE：

$$
D(\hat Y,Y)=D_{semantic}+D_{geometry}+0.25D_{lifecycle}.
$$

zero effect 被额外锚定为 source-state reconstruction，避免模型通过任意恶化 zero branch 伪造 gain：

$$
L_{zero}=D(F(S_t,0,\Delta),Y_t).
$$

motion-active 样本上的 intervention objective 要求正确 effect 相对三种 baseline 至少有 10% 相对优势：

$$
L_{rank}=\sum_{b\in\{zero,shuffle,persistence\}}
\max\left(0,D_{correct}-0.9\operatorname{stopgrad}(D_b)\right).
$$

完整 objective 为：

$$
L_{v59}=L_{prediction}+0.5L_{zero}+L_{rank}+0.05L_{effect\ variance}.
$$

- **因果边界**：student source encoder 只读取 observed prefix；future object target 只进入 training-only Posterior；zero-effect Dynamics 不得随 future target swap 改变；future target swap 必须改变 posterior effect 和 correct prediction。
- **执行契约**：默认六源数据、H=1/2/3/4、future horizons 1/2/4 个采样间隔、effective batch 256、10,000 steps。80GB GPU 默认每卡 batch 32；GPU 数量自动读取。每 1,000 steps 保存 milestone，每 100 steps保存 recovery；支持严格 v59 resume。
- **W&B 主指标**：`correct_active_error`、`zero_active_error`、`shuffled_active_error`、`persistence_active_error`，以及基于这些原始聚合误差重新计算的三项 gain。`point_velocity_probe` 不再进入核心 objective。
- **held 验收**：source-balanced evaluator 必须确认 aggregate correct effect 相对 zero、shuffled 和 persistence 都改善至少 10%；分别报告每个 source、H 和 horizon。该 evaluator 仍依赖训练 tracker 构造 target，所以只能判定 dynamic objective 是否成立，不能替代 G3 independent object validity。
- **当前结果**：代码、Python 静态编译、Ruff、shell syntax 和 `git diff --check` 已通过；真实 GPU startup verifier、训练曲线和 held evaluation 尚未执行。当前状态是 `implemented, awaiting real GPU falsification`，不是 objective 已通过。
- **首次真实 verifier 修复**：提交 `4f3495e` 将 persistence lifecycle baseline 从 autocast 不支持的 probability-space `binary_cross_entropy` 改为数学等价的 float32 logit 加 `binary_cross_entropy_with_logits`。该修复不改变 target、权重或 objective 含义；旧 revision 的 startup report 不可复用，必须在新 HEAD 上重新运行 verifier。
- **下一 Gate**：先运行 v59 startup verifier；随后只进行 10,000-step objective experiment。若 2k/5k/10k 的 correct effect 不能持续优于三种 baseline，不增加数据量或延长训练，直接判定当前 target/objective 失败并分析哪一项 baseline 未被超越。

### 2026-08-25：v59 长训完成与 Unseen-Window Evaluation 契约

- **训练证据**：W&B `1ud4ajuy`，run
  `object_transition_objective_v59_seed17_6193744`，代码提交
  `61937441b8a62d1db881d83b65b5a7fb6b14ab4d`，已正常完成 10,000 steps。
  最终 checkpoint 为
  `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_transition_objective_v59_seed17_6193744/v59_transition_0010000.pt`。
- **完整历史结果**：最后 500 steps 的 motion-active 原始误差为 correct
  `0.07978`、zero `0.14127`、shuffled `0.49548`、persistence `0.10001`；
  correct 分别改善 `43.53%`、`83.90%`、`20.23%`。rolling-500 aggregate 在
  step 4,400 首次超过 persistence，并在 step 5,700 达到 10% improvement。
- **时间尺度分解**：最后窗口中，`h1/h2/h4` 相对 persistence 分别为
  `-21.29%/+3.64%/+33.57%`。因此训练 aggregate 已通过，但变化较小的 short
  horizon 仍失败；不得把 aggregate 结果写成所有时间尺度的 dynamics 都成立。
- **收敛判决**：8k-10k correct error 每 1,000 steps 只下降约 `0.00085`，LR 已到
  `2e-5`，继续同配置训练不能作为修复 short-horizon failure 的主要方案。
- **旧 evaluator 问题**：旧脚本将 `train` split 上固定抽取的窗口称为 held
  evaluation，没有排除训练 sampler 已经访问的 base indices；同时按 batch mean
  聚合 active error，可能因各 batch 的 motion-active 数量不同产生偏差。W&B 只记录
  aggregate 与 gate，无法检查 source、history、temporal step 和 horizon failure。
- **新 evaluator 单一改动**：从 v59 checkpoint 的 batch、grad accumulation、seed、
  world size 和 global step 精确重建 DDP sampler，排除所有训练 base indices，再从相同
  train split 选择 source-balanced unseen windows。该集合是 `sampler-unseen windows`，
  不是 held episode、held task 或 independent object truth。
- **统计契约**：以真实 motion-active count 累积 correct/zero/shuffled/persistence error，
  同时报告 source-history macro、全样本 micro、六个 source、H=1-4、100/200/400/800ms
  temporal bins 和 h1/h2/h4。对 24 个 source-history condition 做 2,000 次 bootstrap，
  输出三项 aggregate gain 的 95% CI。
- **双层 Gate**：`aggregate_status` 要求 macro 与 micro 均对三种 baseline 改善至少
  10%、CI 下界为正且训练/评测 index overlap 为零；`temporal_status` 额外要求每个
  horizon 都达到三项 10% improvement。总 `status` 只有两层同时通过才为 passed。
- **W&B 契约**：完整写入 `eval/macro/*`、`eval/micro/*`、`eval/source/*`、
  `eval/history/*`、`eval/temporal/*`、`eval/condition/*`、`eval/bootstrap/*` 和
  `eval/gate/*`；每完成一个 source-history condition 即同步一次进度与该 condition
  指标，最终再写完整 summary。评测仍使用 frozen DINO、CoTracker 和 relation teacher，
  只能决定 v59 dynamic objective 是否成立，不能证明 independent object semantics 或
  部署期 Prior。
- **下一 Gate**：在 10,000-step checkpoint 上运行一次 unseen-window evaluation。
  先判断 aggregate 是否从训练曲线泛化，再明确 short-horizon failure 是否跨 source、H
  与 temporal step 普遍存在；结果返回前不修改 Dynamics 或继续训练。

### 2026-08-25：v59 Unseen-Window Evaluation 正式结果

- **执行证据**：W&B run `nayw8awc`，名称
  `object_transition_v59_step10000_unseen_eval_25f3951`，状态 `finished`；评测代码
  revision 为 `25f3951bdeaa069eb92b963c39dcbb0c59f6eb9d`，输入 checkpoint 为 v59
  step 10,000（训练 revision `61937441b8a62d1db881d83b65b5a7fb6b14ab4d`）。24/24
  source-history conditions 均完成，总运行时间约 410 秒。
- **数据隔离成立**：训练 sampler 共访问 `2,071,383` 个唯一 base indices；评测从六个
  source 各取 64 个、共 384 个 base indices，训练/评测 overlap 为 0。每个 base index
  分别评测 H=1/2/3/4，共 1,536 clips。该结果仍是 train split 内的 sampler-unseen
  window 泛化，不是 held episode、held task 或 independent object truth。
- **Aggregate 通过**：source-history macro correct error 为 `0.08314`，相对 zero、
  shuffled、persistence 分别改善 `43.46%`、`82.71%`、`19.61%`；micro 对应改善
  `43.94%`、`83.13%`、`20.84%`。persistence gain 的 95% bootstrap CI 为
  `[11.51%, 26.05%]`，因此 aggregate improvement 不是少数 condition 的偶然均值。
- **Horizon Gate 失败**：macro 的 h1/h2/h4 persistence gain 分别为
  `-29.96%/+5.88%/+38.05%`；micro 分别为 `-20.97%/+8.35%/+37.85%`。h1 明确比
  current-state copy 差，h2 尚未达到 10% 门槛，只有 h4 稳定通过。该模式与训练末段
  `-21.29%/+3.64%/+33.57%` 一致，排除了只由训练 rolling window 造成的假象。
- **Source 分解**：Bridge、HY、RoboMind、RoboTwin 的 aggregate persistence gain 分别为
  `+32.47%/+16.64%/+30.53%/+28.23%`；AgiBot 与 Droid 分别为
  `-8.66%/-16.83%`。24 个 condition 全部显著优于 zero 和 shuffled effect，但只有
  16/24 对 persistence 改善至少 10%；失败的 8 个 condition 正好是 AgiBot 与 Droid
  的全部 H=1/2/3/4。
- **时间间隔分解**：100/200/400/800ms aggregate persistence gain 为
  `+1.00%/+15.99%/+22.96%/+29.97%`。h1 在四个 temporal bins 上均为负；h2 只有
  800ms 明确通过。这说明失败随真实变化量减小而加剧，而不是 history 长度不足。
- **History 分解**：H=1/2/3/4 aggregate persistence gain 为
  `+13.56%/+19.97%/+23.76%/+24.84%`，更多 history 有稳定帮助；但每一种 history
  长度内部的 h1 仍为负，因此仅增加 history 不能修复 short-horizon calibration。
- **Latent effect 未塌缩**：macro effect std 为 `0.1959`，高于 `0.1` 门槛；所有
  24 个 condition 中 correct effect 都比 zero 和 shuffled effect 好至少 10%。这证明
  posterior effect 携带真实 future transition 信息，Dynamics 也实际读取了 effect。
- **正式判决**：`aggregate_status=passed`，`temporal_status=failed`，总
  `status=failed`。v59 证明了 teacher-conditioned latent effect 对中长时 object
  transition 有用并能泛化到未采样窗口，但没有形成统一、时间校准的 transition model；
  对短时或近 persistence 数据，模型会预测过量变化。当前最高优先级不是继续延长 v59，
  而是修正 transition target 与 Dynamics 的变化尺度/静止分解，并针对 AgiBot、Droid
  的短时窗口建立一般化的 no-change/small-change 表达。

### 2026-08-25：v60 Gated Residual Object Transition 实现决策

- **版本主线**：branch `codex/gated-residual-object-transition-v60`；
  `checkpoint_version=60`；`architecture=gated_residual_object_transition_v1`。
- **唯一问题定义**：v59 已证明 continuous posterior effect 携带 future transition
  信息，但 full-state Dynamics 对所有样本直接回归完整 future state，导致小变化和短时间
  间隔发生 systematic over-prediction。v60 不修改 object teacher、不增加数据特例，也不
  调整 source sampling 权重；它只修正 transition parameterization 和 objective。
- **继承资产**：冻结 v58 Object State encoder；从 v59 step 10,000 checkpoint 严格加载
  encoder 与 continuous `[4,32]` posterior；继续使用六源视频、冻结 DINO、冻结
  CoTracker、relation teacher、source-balanced sampler、DDP、W&B 和 checkpoint 基础设施。
- **明确不继承**：不加载 v59 full-state Dynamics，不恢复 v59 optimizer/scheduler，不把
  RGB、机器人 action、真实 center delta、language 或 History-only Prior 加入模型。

对每个有效 object transition，先用与 prediction error 相同的 semantic、relative geometry
和 lifecycle 距离定义 teacher change distance：

$$
d^*_{t,\Delta}=D_{semantic}(Y_t,Y_{t+\Delta})
+D_{geometry}(Y_t,Y_{t+\Delta})
+0.25D_{lifecycle}(Y_t,Y_{t+\Delta}).
$$

再去掉 frozen teacher 的小噪声区间，并映射为连续变化强度：

$$
g^*_{t,\Delta}
=1-\exp\left(
-\frac{\max(d^*_{t,\Delta}-\epsilon,0)}{\tau}
\right),
\qquad g^*_{t,\Delta}\in[0,1].
$$

默认 $\epsilon=0.02$、$\tau=0.08$。这不是 motion-active 的二值分类器；它明确区分
no-change、small change 和 large change，避免把 teacher noise 当成必须预测的动态。

模型先从 RGB-only source state 重建一个与 future 和时间无关的 source base：

$$
B_t=B_\omega(S_t).
$$

Posterior 仍解释真实 transition 的内容：

$$
z_{t,\Delta}=Q_\phi(Y_t,Y_{t+\Delta},\Delta),
\qquad z_{t,\Delta}\in\mathbb{R}^{4\times32}.
$$

新增 change gate 只预测变化幅度：

$$
\hat g_{t,\Delta}=\sigma(G_\eta(z_{t,\Delta})).
$$

Dynamics 不再生成完整 future state，而只生成相对 source base 的 residual：

$$
\Delta\hat Y_{t,\Delta}=R_\theta(S_t,z_{t,\Delta},\Delta),
$$

$$
\hat Y_{t+\Delta}=B_t+\hat g_{t,\Delta}\Delta\hat Y_{t,\Delta}.
$$

semantic 分量在 residual 相加后重新归一化；relative geometry 与 visibility logits 直接做
residual update。zero branch 在代码中直接返回 $B_t$，而不是再运行一个输入 zero effect
的神经网络。因此 zero/no-change 不能被模型任意恶化来伪造 intervention gain。

v60 objective 为：

$$
L_{v60}=L_{future}
+0.5L_{base-source}
+0.5L_{gate-calibration}
+0.5L_{no-change}
+L_{magnitude-rank}
+0.05L_{effect-variance}.
$$

其中 $L_{base-source}$ 直接把 $B_t$ 锚定到 teacher source object state；
$L_{gate-calibration}$ 使用 soft-target BCE 令 $\hat g$ 拟合 $g^*$；$L_{no-change}$ 按
$1-g^*$ 加权，要求低变化样本的 correct prediction 保持接近 source；intervention margin
按 $g^*$ 连续缩放，变化越小越不强迫模型制造相对 persistence 的固定 10% 优势。

- **优化器边界**：posterior 与 change gate 使用 `5e-5`；全新 source base 与 residual
  Dynamics 使用 `2e-4`；两组共享 5% warmup 和 0.1 cosine floor。默认 10,000 steps、
  effective batch 256；80GB GPU 默认每卡 batch 32，GPU 数量自动发现。
- **新增 W&B 指标**：除 v59 comparable active error/gain 外，记录 change-weighted、
  low-change、high-change error/gain，teacher change strength、predicted gate mean/MAE/
  correlation、low-change residual magnitude，以及每个 horizon 的 gate calibration。
- **因果契约**：source encoder 与 source base 在 future swap 下差异必须小于 $10^{-6}$；
  Posterior、gate 和 correct prediction必须变化；gate=0 的 zero branch必须与 source base
  完全一致；增大同一 effect 的 gate 必须单调增大 residual magnitude。
- **正式验收**：沿用 v59 sampler-unseen 24-condition evaluator。macro/micro 继续要求相对
  zero、shuffled、persistence 均改善至少 10%；h1/h2/h4 必须分别通过；low-change 不得比
  persistence 更差，high-change 必须保留至少 10% improvement；AgiBot 与 Droid 不得继续
  出现 aggregate negative persistence gain。
- **本地验证状态**：Python compile、Ruff、shell syntax、`git diff --check` 和 CPU synthetic
  contract 已通过。synthetic 中 no-change teacher strength 为 `0.0`，large-change 为
  `0.99999`；高 gate residual 大于低 gate，change gate 与 residual Dynamics 均有梯度。
  尚未执行真实 GPU verifier、六源训练或 unseen-window evaluation，因此当前只能记为
  `implemented, awaiting real GPU falsification`，不能记为 v60 已解决 temporal failure。

### 2026-08-27：v60 10k 训练结果与正式 Sampler-Unseen Gate

- **训练执行证据**：W&B run `4et02kkh` 已 `finished`，训练 revision 为
  `c2930c6dd71f6c70e6075d1ad1a436934b15749e`，world size 8、每卡 micro batch 32、
  gradient accumulation 1、effective batch 256，共完成 10,000 steps。最终 checkpoint 为
  `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/gated_residual_object_transition_v60_seed17_c2930c6/v60_transition_0010000.pt`。
- **训练内主要结果**：末 500 个记录点按有效 transition count 加权后，correct、zero、
  shuffled、persistence error 分别为 `0.05552/0.11672/0.43762/0.10001`；correct 相对
  zero、shuffled、persistence 的 gain 分别为 `+52.43%/+87.31%/+44.48%`。按 horizon
  汇总的 persistence gain 为 h1 `+26.05%`、h2 `+37.11%`、h4 `+50.25%`，相比 v59
  的 `-21.21%/+3.73%/+33.65%`，短时 over-prediction 在训练曲线上已明显修复。
- **Gate calibration 结果**：末段 predicted change gate mean 为 `0.2448`，teacher
  change strength mean 为 `0.2411`，MAE 为 `0.03917`，correlation 为 `0.98057`。
  high-change 对 persistence 改善约 `59.42%`；但 low-change 占约 `61.6%`，其 correct
  error `0.03076` 仍高于 persistence `0.02753`，即 persistence gain 为 `-11.73%`。
  因此 v60 不能仅依据总 loss、全体 gain 或 gate correlation 宣布成功。
- **当前瓶颈假设**：source base anchor error 在末段约 `0.048`，与 correct future error
  同量级。训练曲线不能区分剩余误差来自 source base reconstruction、change gate 还是
  residual content。继续延长同一训练只会混合这三项，缺少明确可证伪结论，故不延长 v60。

本次新增一个只读的 V60 sampler-unseen evaluator。它严格复现 checkpoint 内保存的
DDP sampler，排除 10,000-step 训练实际访问过的 base indices，再以六个 source、
H=1/2/3/4、100/200/400/800ms 和 h1/h2/h4 分解结果。每个有效 transition 同时计算：

$$
\hat Y_{standard}=B_{student}+\hat g\,\Delta\hat Y,
$$

$$
\hat Y_{oracle\ gate}=B_{student}+g^*\,\Delta\hat Y,
$$

$$
\hat Y_{teacher\ base}=Y_t^{teacher}+\hat g\,\Delta\hat Y,
$$

$$
\hat Y_{joint\ oracle}=Y_t^{teacher}+g^*\,\Delta\hat Y.
$$

四条路径分别表示标准部署 student、仅替换 oracle change magnitude、仅替换 teacher source
base、同时替换 source base 与 gate。所有路径共享同一个 learned raw residual，因而：

- oracle gate 相对 standard 的 improvement 衡量 gate calibration 的剩余误差；
- teacher base 相对 standard 的 improvement 衡量 additive source-base anchor 的剩余误差；
- joint oracle 仍不能超过 persistence 时，说明 residual content 本身没有解释 transition；
- joint oracle 明显有效但 standard 无效时，才可把失败定位到 deployable base/gate 接口。

这里替换的只是 Dynamics 最终相加的 base；raw residual 仍由 student source hidden state
条件化。因此该反事实能识别 additive base-anchor bottleneck，但不能独立测量整个 student
encoder 的误差，也不能把 teacher-base gain 直接写成 object representation 已被修复。

正式 Gate 同时要求：训练/评测 base-index overlap 为零；macro 与 micro 对 zero、shuffled、
persistence 均改善至少 10%，且 2,000 次 condition bootstrap 的 95% CI 下界为正；每个
horizon 均达到 10%；low-change 不弱于 persistence；high-change 至少改善 10%；AgiBot 与
Droid 不再为负；gate correlation 至少 `0.80` 且 MAE 不超过 `0.10`。结果还会同步
W&B 的 condition/source/history/temporal/bootstrap/factorization 全量指标。

- **边界**：评测仍使用 frozen DINO、CoTracker 和 trajectory-relation teacher，只验证
  teacher-defined dynamic objective 在 sampler-unseen windows 上是否成立。它不能证明
  independent object identity、部署期 latent-effect selection 或机器人控制能力。
- **当前状态**：V60 训练已完成；正式 sampler-unseen evaluator 已实现并通过 CPU synthetic
  contract、Ruff、Python compile 和 shell syntax，真实六源 GPU evaluation 尚未运行。
  下一步唯一任务是执行该 evaluator，并依据四路 factorization 选择下一版修改对象；在结果
  返回前不增加数据、不继续训练，也不改 Dynamics 结构。

### 2026-08-27：暂停 Dynamics 优化并建立 Observation-Grounded 复评

- **触发原因**：V60 sampler-unseen 结果证明 learned effect 对 teacher-defined compact
  transition 有明显作用，但现有绝对误差仍只比较预测状态与 CoTracker+DINO 构造的
  pseudo-GT：一个 1024 维 object-average semantic、五个 relative geometry 数值和一个
  visibility。该误差不是 future RGB 或完整 future DINO patch field reconstruction。
  baseline gain 只能回答模型是否优于 copy/zero/shuffled，不能回答 compact state 是否保留了
  足够的真实 observation 信息。
- **代码证据**：branch `codex/v60-observation-reconstruction-eval`；复评实现提交
  `c4b2de4af6fae4f3a155ce375d22aee95bc3c5a5`。本轮不修改 V60 model、checkpoint、训练 loss
  或 optimizer，只增加只读 evaluator、W&B 输出和 CPU 数学契约测试。
- **Observation GT**：对真实 future RGB 运行同一个 frozen DINOv2-L，直接使用其
  `16x16x1024` future patch field。query-object 的实际评测区域由可见 related tracks 在
  patch grid 上做固定 kernel splat 得到。当前版本不声称 RGB pixel reconstruction，因为
  V60 state 没有 RGB、scene renderer 或 camera/background state；DINO patch field 是第一层
  真实 observation 复评，后续 RGB probe 必须单独报告 decoder floor。

复评固定比较以下四条主要路径：

$$
R_{direct}=R\left(Y_{t+\Delta}^{teacher},M_{t+\Delta}^{track}\right),
$$

$$
R_{state}=R\left(Y_{t+\Delta}^{teacher},M(Y_{t+\Delta}^{teacher})\right),
$$

$$
R_{pred}=R\left(\widehat Y_{t+\Delta},M(\widehat Y_{t+\Delta})\right),
$$

$$
R_{persist}=R\left(Y_t^{teacher},M(Y_t^{teacher})\right).
$$

其中 $M_{t+\Delta}^{track}$ 是 future related tracks 直接形成的 observation support；
$M(Y)$ 是由 state 中 relative center、二维 covariance 和 visibility 构造的 Gaussian moment
support；$R$ 将单一 semantic vector 填入 object support，并把其余区域保留为 source-frame
feature persistence。所有路径最终都与真实 future DINO patch field 比较，而不是互相比较。

- `semantic_compression_floor`：teacher future semantic 与真实 object-region future patches
  的 absolute cosine error。它高说明把 object appearance 压成一个平均 vector 已经丢失关键
  局部结构，继续改 Dynamics 无法修复。
- `direct_observation_floor`：使用 teacher semantic 和 actual track support 后仍剩余的 object
  reconstruction error，反映单向量 semantic 与最小 compositor 的共同下限。
- `teacher_moment_support_iou` / `geometry_state_penalty`：actual track support 换成 teacher
  的 center+covariance moment support 后损失多少。它差说明五维 geometry 无法表达 object
  shape、articulation 或多区域遮挡，应该扩展 object-local carriers，而不是调 loss weight。
- `teacher_state_observation_error`：完整 teacher compact state 对真实 observation 的可恢复
  上限。若该值本身很差，则当前学习目标不充分，即便 state loss 降到零也不能完成目标。
- `dynamics_observation_gap`：V60 prediction 相比 teacher-state ceiling 额外增加的 error。
  只有 teacher-state ceiling 已经足够好而该 gap 明显时，下一轮才应继续修改 Dynamics。
- `prediction_gain_over_persistence`：在相同真实 future observation metric 上比较 V60 与
  persistence。它继续保留，但只是相对价值指标，不再代替 absolute reconstruction。
- `composite_full/object/background_error`：分别报告全 patch field、query-object support 和
  其余区域，避免静态背景数量压低 full-frame error，也避免把 V60 缺失的 scene/camera branch
  误判成 object Dynamics 失败。

本轮不设置事先拍脑袋的 pass threshold。先在与 V60 正式评测完全相同的六 source、
H=1/2/3/4、100/200/400/800ms、sampler-unseen windows 上收集 macro、micro、source、history、
temporal 和 2,000 次 condition bootstrap。结果按以下单向决策解释：

1. `semantic_compression_floor` 或 `direct_observation_floor` 高：下一版优先把单一 semantic
   average 改成可变数量 object-local semantic carriers；禁止继续调 Dynamics。
2. direct floor 可接受，但 `teacher_moment_support_iou` 低或 `geometry_state_penalty` 高：保留
   semantic，扩展 geometry/support representation；禁止用更大 Dynamics 掩盖 shape loss。
3. teacher-state observation error 低，但 `dynamics_observation_gap` 高：当前 state target
   足够，下一版只修改 effect-conditioned Dynamics。
4. correct 接近 teacher-state ceiling 且 absolute observation error 低：V60 compact dynamics
   得到 observation-grounded 支持，再进入 independent tracker/人工子集与 evaluation-only RGB
   probe；在此之前不做 selector、language 或 policy。
5. 只有 background error 高：单独增加 scene/camera state，不改变 object state 定义。

- **本地验证**：新增 CPU contract 共 9 项，验证 actual-support oracle IoU、teacher/prediction
  等价、错误 persistence semantic 可分辨、sample mask 与 accumulator 有限；Ruff、format、
  Python compile、shell syntax 和 `git diff --check` 均通过。
- **当前判决**：`已完成，拒绝继续优化当前 compact-state Dynamics`。服务器已使用 V60
  step 10,000 checkpoint 完成只读复评；W&B run 为
  `mxejelzj / v60_step10000_observation_reeval_005c076`，状态 `finished`。完整 history 包含
  `6 sources x H=1..4 = 24` 个条件，没有只读取部分 steps。

宏平均结果如下。所有 error 均为相对真实 future frozen-DINO patch field 的 cosine error，
不是 RGB pixel error：

| 指标 | 结果 | 结论 |
| --- | ---: | --- |
| `semantic_compression_floor` | 0.43782 | 单一 object-average semantic 丢失大量 object-local feature 结构 |
| `direct_observation_floor` | 0.17697 | 即使用真实 future support，最小 compositor 仍有较高绝对误差 |
| `teacher_moment_support_iou` | 0.62990 | center+covariance 只能粗略覆盖真实 track support |
| `geometry_state_penalty` | 0.02513 | 五维 geometry 额外贡献约 11.69% 的 correct error |
| `teacher_state_observation_error` | 0.20209 | 当前 compact teacher state 本身不是充分 observation target |
| `dynamics_observation_gap` | 0.01287 | Dynamics 只贡献约 5.99% 的 correct error，不是主瓶颈 |
| `correct_composite_object_error` | 0.21497 | V60 对真实 future object observation 的绝对误差 |
| `persistence_composite_object_error` | 0.21983 | correct 仅相对改善约 2.04% |

2,000 次 condition bootstrap 给出：`semantic_compression_floor` 95% CI
`[0.42360, 0.45179]`，`direct_observation_floor` 为 `[0.17005, 0.18463]`，
`teacher_state_observation_error` 为 `[0.19628, 0.20854]`，
`prediction_gain_over_persistence` 为 `[0.01240, 0.02887]`。因此较高的 absolute floor 和很小的
persistence gain 都不是少数 condition 的偶然值。

source 分解进一步显示：RobotWin 和 Bridge 的 prediction gain 分别为 5.19% 与 3.90%，
HY-Embodied 为 2.47%，RoboMind 为 1.00%；Droid 仅 0.03%，AgiBot 为 -0.34%。24 个条件中
20 个优于 persistence，但 AgiBot 的 H1/H3/H4 和 Droid H1 为负。history 从 H1 增加到 H4
没有一致改善，说明当前模型没有稳定利用更长 history。

还发现一个 evaluator 级 shortcut：`shuffled_composite_object_error=0.20484` 虽低于 correct，
但 shuffled 的 `dense_object_semantic_error=0.56757`、`track_semantic_error=0.47013` 和
`support_iou=0.32746` 均显著差于 correct 的 `0.45236 / 0.32538 / 0.62006`。原因是 predicted
support 缩小时 compositor 会复制 source feature，从而以 persistence 掩盖错误 prediction。
因此 composite error 不能单独用于 effect intervention 排名；下一版 evaluator 必须增加固定
actual support 上的 prediction-only semantic/geometry error，并把 support miss 单独计罚。

最终归因可写成：

$$
0.21497\approx 0.17697_{\text{semantic/compositor floor}}
+0.02513_{\text{geometry state}}
+0.01287_{\text{Dynamics}}.
$$

约 82.32% 的 correct absolute error 已存在于 direct observation floor，约 11.69% 来自 compact
geometry，只有约 5.99% 是 Dynamics 相对 teacher state 的新增误差。下一项主线不是增大
Dynamics、调 loss weight 或继续 V60 长训，而是先把 object state 从“单一 semantic vector +
单 Gaussian moment”改成能够表达 object-local appearance 与多区域 support 的紧凑组合状态；
Dynamics 必须等新 state 的 observation-grounded ceiling 明显改善后再训练。

### 2026-08-27：V61 单 Student SigLIP2 Continuous-Carrier Object State

**What changed**

1. 主线从 V60 的单一 semantic vector 与单 Gaussian support，改为一个 RGB-only Student
   Encoder 产生 token field，再由 128 个连续 carriers 与 16 个 persistent object roots
   表达 object-local appearance、support、identity、dynamic、visibility 和 presence。carrier
   center 来自对 Student token field 的可学习连续加权；patch 只作为 encoder 的内部采样，
   不再作为 object GT、重建单元或主要评测对象。
2. Student 只有一个视觉 backbone。实现四个共享下游状态头的对照：`dino`、`siglip`、
   `siglip_dino`、`siglip_dino_object`。最后两组分别增加 frozen DINO continuous-point
   alignment，以及 frozen SigLIP2 object-crop semantic alignment；它们都是 training-only
   teacher，不构成第二条 Student 分支，部署路径只保留一个 Student。
3. Object State 监督改为 continuous CoTracker points 和 soft trajectory relations：同一 track
   的 carrier/root assignment、连续坐标、motion、visibility/presence、same-object relation 和
   different-motion relation分别监督。SigLIP2 object target 由 soft relation component 的首尾帧
   object-centered crops 得到，并增加 batch 内 semantic retrieval；不使用 instance annotation、
   hard component pseudo-label 或固定 slot index 作为 GT。

**Why**

V60 observation-grounded 复评显示约 82.32% 的绝对误差已经存在于“单一 semantic average +
最小 compositor”下限，Dynamics 只占约 5.99%。因此继续优化 Dynamics 或 patch-field decoder
无法解决 object state 不充分的问题；V61 必须先比较 DINO 与 SigLIP2 对 object semantics 的
贡献，并验证 continuous carriers 是否能在外部轨迹坐标上形成稳定、可分离、可重现的对象状态。

**Impact**

- V61 是 `object_state`-only 阶段，明确不含 Dynamics、latent effect、language 或 explicit
  action；V60 及更早 checkpoint 不允许 warm-start，四组实验都从各自 foundation model 初始化。
- 核心 W&B 指标改为 `track_coordinate_error`、`carrier_track_cycle_error`、
  `object_root_track_cycle_error`、`relation_same_error`、`relation_different_error`、
  `visibility_error`、`presence_error`、`track_reappearance_identity_error`、
  `dino_local_alignment_error`、`object_semantic_retrieval_accuracy`、effective carriers/roots 和
  scene owner fraction。成功标准是 held video 上这些外部目标共同改善，不再以 dense patch
  reconstruction 或 action-free persistence prediction 作为 Object State 成功标准。
- 实现分支为 `codex/siglip-continuous-carrier-v61`。本地已完成 Python compile、Ruff、shell
  syntax 和 `git diff --check`；本机 Python 无 `torch`，所以 CPU tensor-contract 尚未执行。
  真实 DINO/SigLIP2/CoTracker GPU verifier、四组 probe、held evaluator 和 W&B 结论仍属于
  `待服务器验证`，在结果返回前不能声称 SigLIP2 或 V61 已经有效。

### 2026-08-27：V61 Teacher Space、Held Split 与 Object-Bound Dynamics 补全

**What changed**

1. Frozen DINO 的 1024 维 local descriptor 与 frozen SigLIP2 的 768 维 object descriptor
   改为参数固定的 grouped projection，统一进入 256 维 teacher space；Student carrier/root
   只预测该 256 维目标，不再通过可学习 head 追逐 teacher 原始维度。
2. multisource 数据增加 source-local task-group held partition。V61 state 与 Dynamics 训练排除
   held groups，评测只读取 held groups；四个 encoder variant、四种 effect capacity 和所有 resume
   必须使用相同 `held_group_stride`。
3. Object State 训练和验收保持独立。只有选定的 DINO-aligned state checkpoint 通过后，才冻结
   Student 与 state encoder，训练 posterior-conditioned Dynamics。latent effect 容量对照为
   `4x32_global`、`4x32_bound`、`8x64_bound` 和 `16x32_root`；默认 `8x64_bound`
   同时预测 effect value、activation 和对 16 个 object roots + scene 的 owner distribution。
   前两组隔离 object binding，`4x32_bound` 与 `8x64_bound` 隔离总容量，后两组保持
   512 个标量维度并比较 learned sparse factors 与 root-wise factorization。

**Why**

全局 `[4,32]` effect 不仅可能容量不足，更缺少 object binding，无法区分同时发生的机械臂、
物体、遮挡与 scene 变化；原维度可学习 teacher head 也会让监督坐标系随 Student 漂移。

**Impact**

- Dynamics 不做 history-only deterministic regression。Posterior 训练期读取 source/target state，
  Dynamics 分别运行 correct、zero 和 shuffled effect；部署期的 Prior、language 与 explicit action
  仍不在 V61 范围内。
- zero/shuffled route 只提供 detached intervention reference，不能通过故意恶化对照路径满足
  margin；梯度只推动 correct posterior-effect route 降低真实 future observation error。
- held evaluator 的单样本 shuffled route 会分别打乱 latent channel 与 owner binding。完整
  `(value, activation, owner)` factor tuple 的顺序置换对 set-valued Dynamics 不可见，不能作为
  有效反事实。
- effect variance 使用可反传的 cross-rank gather；它衡量同一 optimizer forward 中所有 rank 的
  latent channel 变化，不依赖 gradient accumulation 补足统计样本。
- final audit 将 carrier/root 更新改为真正的 causal predict-correct：下一帧 carrier attention 以
  上一帧 center 为空间先验，下一帧 owner assignment 以持久 root feature 为 query；固定 seed 和
  fixed root query 只用于第一帧初始化。
- reappearance identity 使用遮挡前最后一个 visible identity 作为 reference，跨过任意长度的
  lifecycle-known occlusion，并只在第一次重新 visible 时计分；同时报告有效 reappearance 数量。
- Dynamics 只在 held task groups 上验收：future continuous-track coordinate/appearance/lifecycle
  error 必须相对 persistence、zero effect 和 shuffled effect 都改善至少 10%。
- `object_state` checkpoint 与 `dynamics` checkpoint 具有不同 architecture/stage contract；旧 V61
  首次提交的 checkpoint 因 teacher space 与 held split 已变化，不能 resume 到补全后的版本。

### 2026-08-30：VAE-style Object State 表征充分性评测协议

**What changed**

1. G2/G3 新增统一的 `Distortion + Rate/Capacity + Utility` 评测地图；deterministic Object State
   使用 active units、effective rank、owner/carrier usage 代替不存在的 VAE KL，只有未来实际存在
   Gaussian posterior 时才报告 KL/Rate 和 sampling stability。
2. Object State 必须通过 frozen kNN/retrieval、linear/MLP probe、低数据量曲线、nuisance probe、
   absolute observation error 和 Markov sufficiency；training loss、teacher agreement、temporal
   consistency 或 reconstruction 不再具有单项晋级权。
3. 正式协议落地到
   `/Users/hela/Instruct-GS-World-recovered-20260725/docs/OBJECT_STATE_REPRESENTATION_EVALUATION.md`，
   并记录 V61 `dino` run `fcak90ya` 的当前证据边界。

**Why**

V61 `dino` probe 可以把训练 objective 稳定降到较低水平，但 held temporal identity error 很低时，
identity retrieval 仍只有约 `0.36%`；这说明“前后相似”可以由 identity homogenization 获得，不能
证明 latent 已形成可区分的 object feature。

**Impact**

- V61 `dino` 只被接受为数值稳定的 optimization baseline，不被接受为 G2 Object State。
- `siglip`、`siglip_dino`、`siglip_dino_object` 必须在同一预算下完成，并使用同一套表征充分性
  评测后才能选择 checkpoint。
- 在 representation sufficiency 与 independent object validity 通过前，继续禁止启动 latent-effect
  Dynamics、Prior 或 task-level claim。

### 2026-08-30：V61 四组 Encoder Ablation 正式结果

**执行**

四组 run 均使用提交 `82bbd8d4873fb3f6552a132ce666d01f39d43ee9`、seed 17、8 GPU、
effective batch 256、3,000 steps 和相同 held-group split。W&B run 分别为：

- `dino`: `fcak90ya`
- `siglip`: `9ttfq8h4`
- `siglip_dino`: `60mczuva`
- `siglip_dino_object`: `vaos6ix0`

四组状态均为 `finished`，最终 step 均为 3,000，未出现 non-finite。以下为最后 480 steps
的均值；不同 variant 含有不同附加 objective，因此 `object_state_loss` 总值不可横向排序，必须比较
共享指标。

| 指标 | dino | siglip | siglip_dino | siglip_dino_object |
|---|---:|---:|---:|---:|
| held identity retrieval | 0.378% | 0.345% | 7.173% | 9.927% |
| held temporal identity error | 0.001144 | 0.000940 | 0.001553 | 0.003692 |
| coordinate error | 0.02344 | 0.03029 | 0.03234 | 0.04276 |
| motion error | 0.01393 | 0.01393 | 0.01393 | 0.01392 |
| visibility error | 0.03923 | 0.03911 | 0.03922 | 0.04440 |
| presence error | 0.01181 | 0.01178 | 0.01255 | 0.01780 |
| scene owner fraction | 5.44% | 9.48% | 6.10% | 0.0168% |
| effective owner categories / 17 | 3.45 | 3.47 | 3.17 | 2.87 |
| effective carriers / 128 | 127.82 | 127.76 | 127.61 | 127.42 |
| held DINO alignment error | N/A | N/A | 0.3540 | 0.3516 |
| object semantic retrieval | N/A | N/A | N/A | 16.01% |

**推导结论**

1. `dino` 与 `siglip` 的 held identity retrieval 都低于 0.4%，因此单独更换视觉 backbone 不能消除
   identity homogenization。极低 temporal error 主要说明向量彼此接近，不等于对象可区分。
2. `siglip_dino` 的 retrieval 从约 0.35% 提升到 7.17%，且从 step 500 的 2.97% 持续增长到
   step 3,000 的约 7.32%。frozen DINO local alignment 是本轮第一个被实验证实有效的 identity
   anchor，而不是 SigLIP backbone 本身。
3. `siglip_dino_object` 将 held identity retrieval 进一步提高到 9.93%，object-semantic retrieval
   持续增长到约 16.17%。object-level semantic target 确实增加了身份区分信息。
4. 完整 variant 同时出现新的 shortcut：scene owner fraction 几乎降为零，effective owner categories
   只有 2.87，128 个 carriers 仍几乎全部 active；coordinate、visibility、presence 和 relation 指标也
   比 `dino` 更差。它倾向于把大量区域压到少数 object owners 来满足 semantic retrieval，尚未形成
   compositional Object State。
5. 四组 motion error 几乎相同，说明本轮没有获得 dynamic-state 增益。该实验只验证了 semantic
   identity signal，不构成 Dynamics 或 Markov sufficiency 的证据。

**判决**

- `siglip_dino_object` 是当前 identity discrimination 最强的 probe，但因 scene/owner collapse，不能
  直接晋级 G2。
- `siglip_dino` 是当前较合理的 Pareto candidate：它保留了大部分 identity 增益，结构退化弱于完整
  variant；它仍未通过 capacity、owner decomposition 与 independent object validity。
- 当前 Gate 保持在 G2/G3。禁止因为任一训练 loss 已收敛而启动 latent-effect Dynamics。
- 下一项唯一工作是对 `siglip_dino` 与 `siglip_dino_object` 执行统一 representation sufficiency
  evaluator：identity/dynamic active units、effective rank、Recall@K、frozen probes、nuisance probes、
  occlusion/reappearance、deletion locality、scene leakage 和 Markov sufficiency。结果返回前不再增加
  backbone variant，也不延长当前 3,000-step probe。

### 2026-08-30：V61 Representation Sufficiency Evaluator 实现

**What changed**

新增一个不修改训练 checkpoint 的 held evaluator，同时读取 `siglip_dino` 与
`siglip_dino_object`。它共享 frozen DINO、CoTracker 和 object-component teacher 的每个 clip 输出，
分别收集两个 Student 的 compact state，并将 capacity、Recall@K、reappearance、frozen probes、
nuisance、scene leakage、external-track deletion locality 与 Markov diagnostics 写到同一个 W&B run。

**Why**

四组 probe 已经证明 semantic target 可以提高 retrieval，但训练 telemetry 无法区分“identity feature
更可分”与“owner assignment 更集中”的收益。新 evaluator 直接检查 latent rank、低复杂度可读性、
跨遮挡检索和 history 增量，避免继续依靠总 loss 或单个 temporal error 选择 checkpoint。

**Impact**

- 评测入口为
  `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/evaluate_representation_sufficiency_v61.sh`。
- 评测在前台单进程运行，不启动训练、不修改 checkpoint、不访问 GitHub。
- truth scope 明确为 held training teacher，只能裁决 G2 candidate；独立 simulator/object-mask G3
  evaluator 仍然待办。
- 当前状态为 `代码已实现，服务器 GPU 复评待执行`。W&B 结果返回前，两个 checkpoint 都不晋级
  Dynamics。

### 2026-08-30：V61 Representation Sufficiency 正式复评结果

**执行**

- W&B run `t33taede`（`v61_representation_sufficiency_0cd9a40`）状态为 `finished`；evaluator
  提交为 `0cd9a40a23bde8d8b07c1756c66e4d022cb822c0`。
- 对比 `siglip_dino` 与 `siglip_dino_object` 两个 seed 17、step 3,000 checkpoint；共 144 个 held
  conditions，覆盖 6 个 source、3 个 clip length，decode replacement rate 为 `2.083%`。
- truth scope 是 `held_training_teacher_not_independent_object_truth`，只裁决 G2 candidate，不替代 G3。

| 指标 | `siglip_dino` | `siglip_dino_object` |
|---|---:|---:|
| identity active units / 128 | 29 | 88 |
| identity effective rank / 128 | 5.04 | 2.67 |
| dynamic effective rank / 128 | 2.08 | 2.75 |
| first-to-last Recall@1 | 40.29% | 34.09% |
| reappearance Recall@1 | 32.95% | 23.80% |
| external-track deletion locality | 6.17 | 4.27 |
| identity same-different margin | 0.0123 | 0.0284 |
| effective owner categories / 17 | 3.16 | 2.90 |
| scene owner fraction | 5.78% | 0.0022% |
| source-from-identity linear accuracy | 97.39% | 99.38% |
| coordinate-from-identity linear gain | 0.941 | 0.862 |
| motion-from-dynamic linear gain | -0.445 | -0.503 |
| motion-from-dynamic MLP gain | -0.037 | -0.059 |
| visibility MLP balanced accuracy | 50.0% | 53.1% |

**推导结论**

1. `siglip_dino` 的 first-to-last 与 reappearance retrieval 明显高于 chance（2.38% 与 1.05%），
   并接近 frozen DINO teacher（49.27% 与 34.32%），说明它保留了跨帧 appearance correspondence。
   但 identity effective rank 只有 `5.04 / 128`，source 可读性达到 `97.39%`，因此该 retrieval 同时
   混入 dataset style、background 和绝对位置 shortcut，不能直接解释为稳定 object identity。
2. `siglip_dino_object` 虽将 same-different margin 提高到 `0.0284`，正式 held Recall@1、遮挡重现和
   deletion locality 却全部下降。88 个 active identity dimensions 只形成 2.67 的 effective rank，
   scene fraction 又接近零，证明 semantic objective 形成了 owner/feature shortcut，而非更好的
   persistent object。
3. dynamic state 没有学成。两者 128 维 dynamic feature 的 effective rank 都低于 3；linear/MLP
   motion probe 均不如训练集 motion 均值基线，visibility 也接近 chance。`siglip_dino` 加入 previous
   dynamic 后 motion gain 从 `-0.401` 改善到 `-0.102`，说明当前 state 丢失历史信息；但两条路径都
   未超过均值基线，所以当前首先失败的是 dynamic representation，而非已经证明了 Markov sufficiency。
4. `siglip_dino_object` 的 first-to-last Recall@1 在六个 source 和 clip length 4/6/8 上全部低于
   `siglip_dino`，退化不是单一数据源或时间跨度偶然现象。

**判决与主线**

- 两个 checkpoint 都未通过 G2；不启动 latent-effect Dynamics、Prior 或 task-level claim。
- `siglip_dino` 仅保留为下一版诊断 baseline，`siglip_dino_object` 不晋级。
- 延长当前训练不能解决主要问题。下一版必须修正 representation objective：identity 去除 source 和
  absolute-position shortcut，dynamic 接受可读的 relative motion/lifecycle 监督，owner decomposition
  阻止 scene 消失和少数 owner 集中。只有 frozen probes 同时恢复 capacity、dynamic utility 和跨 source
  compositional validity 后，才重新进入 Dynamics。

## 15. 2026-08-31：V62 Object Transition Learning 完整实验方案

### 15.1 What changed / Why / Impact

**What changed**

1. 当前最高优先级从继续优化 V61 identity/owner 指标，改为先证明一个可学习、可证伪的
   object transition target。V62 不把 point residual flow 直接叫作 dynamic state，也不把
   action-free short prediction 当作 object dynamics。
2. V62 采用同一套代码中的阶段化训练：continuous teacher object codec、teacher-state oracle、
   variational effect、RGB-only Student、Student-conditioned Dynamics、multi-horizon rollout、Prior。
   阶段之间只能按 Gate 单向推进。
3. DINO/SigLIP patch grid 只保留为 frozen perception field 的内部实现。训练和评测的 object target
   改成 continuous query-object observations；不以 patch、instance mask、固定 slot index 或整图 RGB
   reconstruction 定义 object。

**Why**

V61 的 `identity` 与 `dynamic` 都是同一 recurrent carrier feature 的线性投影；motion target 又是减去
全局平均后、按真实时间归一化的 point residual flow，而 Student 没有接收对应 `delta-time`。这使
`dynamic` 接近常数、motion readout 接近零成为低成本解。V61 复评中 dynamic effective rank 只有
`2.08/2.75`，linear 与 MLP motion probe 都不如均值 baseline，说明继续增加维度、数据或训练步数不会
自动修正学习目标。

**Impact**

- V61 checkpoint、optimizer 和 carrier state 不进入 V62 初始化；它们只作为失败 baseline。
- 六源视频 index、冻结 DINO、冻结 SigLIP、冻结 CoTracker、source-balanced sampler、W&B 与前台 DDP
  运行基础设施继续复用。
- V62 E0/E1 可以在 G2/G3 未通过时执行，因为它们只诊断 target 与 transition objective，不构成
  Student 或 deployable world model 晋级。E2 以后仍严格受 G2-G5 约束。

### 15.2 本轮只解决的科学问题

V62 需要依次回答四个问题：

1. **Target validity**：纯视频 teacher 能否定义一个足够丰富、非 patch-level、非单向量平均的
   object observation？
2. **Transition sufficiency**：给定正确 source object state，从真实 source/target 提取的 effect
   是否是预测 target object state 所必需的变量？
3. **Deployable state estimation**：只读 observed RGB history 的 Student 能否估计同一种 object state，
   而不是复制 dataset、background 或 absolute coordinate shortcut？
4. **Deployable dynamics**：Student source state 与正确 posterior effect 能否预测未来 object state，
   并支持多步 composition？

这四个问题不能在一个端到端 loss 中同时回答。若 teacher-state oracle 都失败，则 Student、backbone、
Prior 和语言均不是当前根因；若 oracle 通过而 Student 失败，才把问题定位到视觉状态估计。

### 15.3 数据与因果契约

正式数据继续使用：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/multisource_real_robot_video_v53/index.json`

E0/E1 第一版继续使用现有 source/task-balanced sampler，六个 source 均有确定 target budget。teacher
object change magnitude 记录为连续指标，并在 held evaluation 中分解 `near-static / medium / active`；在
完成真实数据 change distribution audit 前不创建离线 motion pseudo-label 或声称 motion-balanced sampling。
change magnitude 不作为模型输入。

第一轮固定真实时间间隔：

$$
\Delta t = 100\,\mathrm{ms}.
$$

所有 source 使用时间戳选择最近帧，不用统一 frame index 冒充统一帧率。固定 $100\,\mathrm{ms}$ 的目的
是先消除 V61 的多时间尺度歧义，并直接检查最容易被 persistence 掩盖的小变化。E5 才扩展到
$100/200/400/800\,\mathrm{ms}$，且 `delta-time` 必须显式进入 Posterior 与 Dynamics。

因果边界固定为：

- teacher codec 可以在训练期读取当前 object 的 tracks 与当前帧 frozen perception field；
- transition Posterior 可以读取 matched source/target teacher states；
- Student 只能读取 observed RGB history、历史时间戳和当前 query；
- future RGB、future tracks、future object state 不得进入 Student、zero branch 或部署期 Prior；
- query-object geometry 表达在去除 robust background/global flow 后的相对坐标系中，避免把相机运动
  当成 object effect；
- tracker invisibility 只表示 `occluded/unknown`，不能直接监督 `absent`。

### 15.4 Continuous teacher object observation

对当前 query $q$，训练期 teacher 构造可变长度集合：

$$
\mathcal P_t(q)=
\left\{
(x_t^i,f_{D,t}^i,f_{S,t}^i,r_t^i,v_t^i,w_t^i)
\right\}_{i=1}^{N_t}.
$$

各变量含义为：

- $x_t^i\in[-1,1]^2$：continuous track coordinate；第一版以 source frame 为原点累计 CoTracker
  residual flow，去除每个相邻帧的全局平均 flow，同时保留 object 相对位移；
- $f_{D,t}^i$：在 $x_t^i$ 双线性采样的 frozen DINO feature；
- $f_{S,t}^i$：在同一连续位置采样的 frozen SigLIP feature；
- $r_t^i$：该 track 与 query 的 soft same-object relation evidence；
- $v_t^i$：visible、occluded 或 unknown lifecycle evidence；
- $w_t^i$：由 relation confidence、track quality 与可见性组成的 detached 置信度。

patch grid 在这里仅用于从 frozen feature field 做连续插值。object support、positive/negative query、
reconstruction 和评测都不按 patch 单元定义。teacher 不产生 hard instance mask，也不要求每个视频先做
instance annotation。

### 15.5 Teacher object-state codec

训练期 codec 把 $\mathcal P_t(q)$ 压缩成一个 query-object state：

$$
Y_t(q)=C_{\tau}(\mathcal P_t(q))
=\left(u_t,H_t,G_t,l_t\right).
$$

第一版固定状态容量为：

```text
identity root u_t       [B, 256]
local latent carriers H [B, 16, 256]
relative centers        [B, 16, 2]
support covariance      [B, 16, 2, 2]
carrier visibility      [B, 16]
object lifecycle l_t    [B, 3]     # visible / occluded / unknown logits
```

这里的 16 个 carrier 是**一个 query object 内部的 local support components**，不是 16 个全图 object
slots，也不使用固定 index 表示跨视频 identity。codec 使用 set cross-attention 读取可变数量 teacher
observations；decoder 在任意 continuous coordinate $x$ 上查询：

$$
(\hat m_t(x),\hat f_{D,t}(x),\hat f_{S,t}(x),\hat v_t(x))
=D_{\tau}(Y_t(q),x).
$$

$\hat m_t(x)$ 是 query object 的 soft support probability；两个 semantic feature 是 object-local
perception target；$\hat v_t(x)$ 是 lifecycle observation。它不生成 RGB，不在规则 patch grid 上重建
整张图，也不允许不同 object state 通过全局 decoder 相互补偿。

codec loss 为：

$$
L_{codec}=L_{support}+L_{DINO}+L_{SigLIP}
+0.5L_{identity}+0.25L_{visibility}+0.25L_{lifecycle}+0.02L_{capacity}.
$$

所有项同时记录 absolute error。`pooled semantic + single Gaussian support + all-visible lifecycle` 只作为
detached compact baseline 计算 held gap recovery，不进入 target，也不通过调权替代方法判断。positive
coordinate 来自 query-related tracks；negative coordinate 来自同一 clip 的 unrelated tracks。第一版不把
缺少 external relation evidence 的连续随机点强行标成 background，以免制造 false negative。

### 15.6 Latent effect 与 Object Dynamics

E1 先使用 deterministic posterior 验证目标；E2 再改成 Gaussian VAE posterior。Posterior 只读取
matched local state change、relative geometry 与 lifecycle change，不读取 identity root $u_t$：

$$
z_{t\rightarrow t+\Delta}
=q_{\phi}(H_t,G_t,l_t,H_{t+\Delta},G_{t+\Delta},l_{t+\Delta},\Delta t).
$$

默认 effect 为 `[B,8,32]`，总维度 256。`[4,32]`、`[8,32]`、`[8,64]` 只在 E2 做受控
rate-distortion sweep，不能凭直觉提前断言 128 维或 256 维足够。

正式 VAE Posterior 为：

$$
q_{\phi}(z\mid Y_t,Y_{t+\Delta},\Delta t)
=\mathcal N(\mu_{\phi},\operatorname{diag}(\sigma_{\phi}^2)),
$$

$$
z=\mu_{\phi}+\sigma_{\phi}\odot\epsilon.
$$

Dynamics 采用 transport + residual，而不是从零生成完整 future state：

$$
\widetilde H_{t+\Delta}=A_{\theta}(z,\Delta t)H_t,
$$

$$
\widehat H_{t+\Delta}=\widetilde H_{t+\Delta}
+R_{\theta}(H_t,z,\Delta t).
$$

$A_{\theta}$ 是对 local carriers 的 row-stochastic soft transport；它负责 component 重排与运动。
$R_{\theta}$ 只负责 transport 不能解释的 deformation、semantic state 与 lifecycle residual。identity root
默认从 source copy，只允许小的 gated residual；这样 donor effect 不能覆盖 recipient object identity。
真实 center delta、point flow、RGB difference 或机器人 action 都不会拼进 $z$。

这个分工吸收了 RepWAM 的 semantic visual latent 与 soft transport + residual 思路、AdaWorld 的
source/target-conditioned variational action tokenizer，以及 PlaySlot 先在 object state 上验证 latent action
再做 future prediction 的顺序；但 V62 的 effect 表示纯视频中的 object transition，不等同于机器人 action。

预测距离在 continuous held coordinates 上计算：

$$
D_{obj}(\widehat Y,Y)
=\bar D_{support}+\bar D_{semantic}+\bar D_{geometry}+\bar D_{lifecycle}.
$$

训练同时比较：

```text
correct:         F(Y_t, z_correct, delta-time)
zero:            F(Y_t, 0, delta-time)
matched-shuffle: F(Y_t, z_other, delta-time)
persistence:     Y_t
```

`matched-shuffle` 必须在相同 source、相同 $\Delta t$ 和相同 change-magnitude bin 内抽取，避免用明显不同
的 donor effect 制造过于容易的负例。核心 objective 为：

$$
L_{transition}=D_{obj}(\widehat Y_{correct},Y_{t+\Delta})
+L_{intervention}+L_{static}+0.1L_{identity-copy}+L_{rate}.
$$

- $L_{intervention}$ 要求 correct 比 zero、matched-shuffle 和 persistence 更好；
- $L_{static}$ 要求 near-static 样本不被迫制造变化；
- $L_{identity-copy}$ 阻止 effect 携带 donor identity；
- $L_{rate}$ 只在 E2 以后启用，并以 KL/rate-distortion 曲线选择容量，而不是只凭重建 loss 选最大 latent。

### 15.7 实验顺序与训练预算

所有阶段使用同一份六源 index、同一 source/motion 分层 sampler、同一 held episode 划分和同一组
absolute metrics。版本号不随每个阶段重启；它们都属于 V62。

| 阶段 | 唯一问题 | 训练模块 | 预算 | 通过后才允许 |
|---|---|---|---:|---|
| E0 Target/Codec | continuous teacher target 是否保留 object observation？ | teacher codec/decoder | 20k steps | E1 |
| E1 Deterministic Oracle | effect 是否对 future object state 必要？ | deterministic posterior + transport/residual Dynamics | 20k steps | E2 |
| E2 Variational Effect | VAE bottleneck 是否在有限 rate 下保留 transition？ | Gaussian posterior；128/256/512 dim sweep | 每组 10k，selected 20k | E3 |
| E3 RGB-only Student | observed RGB history 是否能估计同一 object state？ | 单一 fused DINO+SigLIP Student | 30k steps | G3 independent eval |
| E4 Student Dynamics | Student source + posterior effect 是否仍可预测？ | freeze Student first，训练 Posterior/Dynamics；随后低 LR joint tune | 30k steps | E5 |
| E5 Multi-horizon | transition 是否时间校准且可组合？ | explicit time conditioning；direct/rollout consistency | 30k steps | E6 |
| E6 Prior/Selector | 部署期如何从 goal/language/history 选择 effect？ | conditional stochastic Prior/selector | 50k steps | task A |

共同运行契约：

- GPU 数量自动发现；单机 DDP；目标 global batch 256，80GB 默认单卡 micro batch 32；effect statistics
  使用本次 forward 的 cross-rank gather，gradient accumulation 只决定 optimizer update；
- DINO/SigLIP frozen frame encoding 使用独立 frame batch，避免 object model batch 被 perception runtime
  人为限制；
- 所有训练前台运行，不使用 `nohup`、`&` 或 detached launcher；
- runtime 不访问 GitHub；代码同步、verify 与训练是三个独立命令；
- 每 2,500 steps 保存 milestone，每 250 steps 保存 rolling recovery；resume 必须恢复 sampler、optimizer、
  scheduler、stage、global step 与 W&B run ID；
- W&B group 固定为 `object-transition-v62`，每个阶段独立 run，但共享 dataset split revision 与 target
  normalization statistics。

### 15.8 每阶段 Gate

#### E0 Target/Codec Gate

定义 direct teacher interpolation 为 oracle floor，`pooled semantic + single Gaussian support` 为 compact
baseline。对每个误差项计算：

$$
R_{gap}=1-\frac{E_{codec}-E_{oracle}}{E_{baseline}-E_{oracle}}.
$$

E0 需要：

- semantic、support 与 lifecycle 的 held $R_{gap}$ 均不低于 0.80；
- 六个 source 的 codec 都优于 compact baseline，不能由一个 source 拉高 aggregate；
- local carrier effective rank 不低于可用 rank 的 20%，active carrier 数应随 support complexity 改变，
  不能在所有 source 与所有 object 上长期固定为 1 或 16；
- query swap、all-scene、merge-all、split-by-time 与 lifecycle corruption 均使对应 absolute error 上升；
- held coordinate 使用未进入 encoder 的 continuous samples，不能在训练 coordinate 上自评。

E0 失败表示 object target/codec 不成立。此时只允许修正 query relation、continuous support 或 codec，
禁止进入 effect/Dynamics。

#### E1 Deterministic Oracle Gate

在 source-condition macro 与 transition micro 两种聚合下同时要求：

- correct 相对 zero、matched-shuffle、persistence 均改善至少 10%，2,000 次 bootstrap 的 95% CI
  下界大于 0；
- motion-active subset 相对 persistence 改善至少 20%；
- near-static subset 的 correct error 不得比 persistence 高超过 2%；
- 六个 source 各自 correct 都优于 persistence，不能靠 source averaging 掩盖失败；
- effect active rank 不低于 20%，zero/shuffle intervention 改变 prediction，source-only path 对 future
  swap 的差异小于 $10^{-6}$；
- 将 donor effect 施加到 matched recipient source 时，recipient identity 保持，donor identity retrieval
  不得显著高于 chance。

E1 是整个方案最关键的 falsification。它失败时不增加 Student/backbone，也不把 total steps 从 20k
延长到 50k；先定位 target、transport 或 effect bypass。

#### E2 Variational Effect Gate

- VAE correct absolute distortion 不得比 deterministic oracle 恶化超过 5%；
- correct-vs-zero/shuffle/persistence 继续满足 E1；
- active KL dimensions 与 effect effective rank均不低于可用维度的 20%；
- posterior sampling 的 prediction variance 与 teacher transition uncertainty 同方向变化；
- 在 128/256/512 三个容量中选择 rate-distortion Pareto 最小者，不选择单纯 reconstruction 最低者。

#### E3 Student State Gate

- 令 $E_{codec}$ 为 teacher codec error、$E_{compact}$ 为 compact baseline error；RGB-only Student 必须满足
  $E_{student}\le E_{codec}+0.2(E_{compact}-E_{codec})$，即至少保留 codec 相对 compact baseline 的
  80% 改进；
- identity retrieval、reappearance、continuous support、visibility calibration 与 motion/lifecycle frozen
  probe 全部优于 V61 `siglip_dino`；
- source、camera、absolute coordinate nuisance probe 不得以 object utility 提升为代价继续恶化；
- H=1/2/3/4 分别报告，motion-active 与 occlusion subset 上增加 observed history 必须带来正收益；
- 使用不导入训练 CoTracker 的 RoboTwin truth 或人工小集通过 G3，才可进入 E4。

#### E4/E5 Dynamics Gate

- 使用 Student source 后仍满足 E1 的 correct/zero/matched-shuffle/persistence 门槛；
- 100/200/400/800ms 每个 horizon 单独通过，不能只报 aggregate；
- 四次 100ms rollout 与 direct 400ms prediction 都优于 persistence，rollout error 不得超过 direct
  400ms error 的 1.5 倍；
- identity、support、visibility/existence 的 rollout error 分开报告；
- reversed effect 与 forward effect 的 composition 应接近 identity transition；否则 effect 不具备可组合性。

#### E6 Prior/Selector Gate

- Prior 不使用 future state/tracks；只读取 observed state 与 goal、language 或 policy condition；
- 多模态 future 使用 best-of-N coverage 与 calibration，单样本均值不能写成部署性能；
- correct goal/instruction 相比 matched wrong condition 的 object-state future error至少改善 5%；
- 最终只在 RoboTwin task A 上验证，与 XR-2 无关。

### 15.9 必须保留的对照实验

1. **旧 motion target 对照**：V61 mean-subtracted point residual flow，证明新 target 的收益不是网络增大。
2. **Dynamics parameterization**：transport-only、residual-only、transport+residual，判断 object motion 与
   deformation 分别需要什么。
3. **Effect capacity**：128、256、512 dimensions，以 rate-distortion 选容量。
4. **Time contract**：fixed 100ms、mixed horizon without time（负对照）、explicit delta-time。
5. **Perception semantics**：frozen DINO only 与 fused frozen DINO+SigLIP；只在 E0/E3 选定 Gate 上比较，
   不再建立两个 Student 分支。
6. **Target information**：semantic-only、support-only、semantic+support+lifecycle，验证哪部分使 effect
   对 object transition 必要。

这些对照共享 sampler、steps、batch、seed、target normalization 和 evaluator。一次实验只改变表中一个
因素，禁止同时修改 backbone、latent dimension、loss 与 sampler 后再归因。

### 15.10 W&B 指标与报告方式

每个 condition 都必须记录 raw numerator、denominator 与 sample count，最终重新聚合，不能平均 batch
mean。最少包含：

```text
absolute/semantic_error
absolute/support_error
absolute/geometry_error
absolute/lifecycle_brier
transition/correct_error
transition/zero_error
transition/matched_shuffle_error
transition/persistence_error
transition/gain_over_{zero,shuffle,persistence}
effect/active_units
effect/effective_rank
effect/kl_total
effect/identity_leakage
effect/source_leakage
codec/active_carriers
codec/effective_rank
codec/gap_recovery
runtime/decode_replacement_fraction
```

所有 transition 指标按 source、change bin、history length、delta-time、visibility state 和 direct/rollout
分解。训练曲线、held teacher evaluation、independent truth 与 task A 是四个不同 W&B run type，不能在
同一个 summary 中用一个 `passed` 覆盖。

### 15.11 失败判决树

```text
E0 codec fails
  -> target/grouping/continuous support 错；不看 Dynamics。

E0 passes, E1 fails
  -> transition target、transport/residual 或 effect bypass 错；不训练 Student。

E1 passes, E2 fails
  -> bottleneck/rate contract 错；不扩大 Student/backbone。

E2 passes, E3 fails
  -> RGB-only state estimation 错；只修 Student/Object State。

E3/G3 passes, E4 fails
  -> Student state 与 Dynamics 接口或 effect conditioning 错。

E4 passes, E5 fails
  -> time calibration/composition 错；不训练 Prior。

E5 passes, E6 fails
  -> condition/selector/multimodality 错；不否定已验证 Object Dynamics。
```

### 15.12 预期代码结构与第一项 TODO

V62 计划使用以下解耦模块，单文件保持小于 550 行：

```text
adaptive_gaussian_wm/
  continuous_object_observation_v62.py
  teacher_object_codec_v62.py
  teacher_object_autoencoder_v62.py
  teacher_object_codec_objective_v62.py
  continuous_object_decoder_v62.py
  object_effect_posterior_v62.py
  object_transport_dynamics_v62.py
  object_transition_objective_v62.py
  object_transition_metrics_v62.py
  object_transition_teacher_runtime_v62.py
  teacher_transition_oracle_v62.py
  v62_config.py
  v62_checkpointing.py
  v62_training_loop.py
scripts/
  verify_object_transition_v62.py
  train_object_transition_v62.py
  train_object_transition_v62.sh
  evaluate_object_transition_v62.py
  manage_object_transition_v62.sh
  test_object_transition_v62.py
```

第一项实现固定为 **E0 continuous teacher target/codec + E1 deterministic teacher-state oracle**。本轮不实现
RGB-only Student、Prior、language、task A、RGB decoder 或旧 checkpoint warm-start。E0/E1 的 held report
返回前，不开始后续模块的长训。

### 15.13 E0/E1 实现记录

2026-08-31 已在 `/Users/hela/Instruct-GS-World-recovered-20260725/` 的
`codex/object-transition-v62` 分支完成 E0/E1 代码，尚未启动服务器训练。

**E0 已实现**

- frozen DINOv2-L、frozen SigLIP 与 frozen CoTracker 只存在于 training-only teacher runtime；三者不进入
  checkpoint trainable state；
- relation teacher 从真实视频轨迹选择一个 query object，并生成 continuous coordinates、soft support、
  visibility 与 visible/occluded/unknown lifecycle；
- DINO 1024 维和 SigLIP 768 维 feature 分别通过 parameter-free grouped projection 形成 256 维 target；
- query object 被压缩为 16 个 `[256]` local carriers、一个 `[256]` identity root、relative center、2D
  covariance、presence、visibility 和 lifecycle logits；
- compositional decoder 只在请求坐标上输出 support、DINO/SigLIP semantic 与 visibility，不做 RGB 或
  full patch-grid reconstruction；
- loss 与 W&B 同时记录 absolute errors、single-Gaussian/pooled compact baseline、gap recovery、carrier
  overlap/effective count 与 object-valid fraction。

**E1 已实现**

- E1 只允许读取本版本 E0 checkpoint；codec 与 decoder 全冻结，V61 及更早 checkpoint 禁止 warm-start；
- deterministic Posterior 输入 source/target local state 与显式 `delta_seconds`，不读取 identity root，输出
  `[B,8,32]` bounded effect；
- Dynamics 使用 row-stochastic 16×16 soft transport 加 feature/geometry/lifecycle residual，identity root
  从 recipient source 精确复制；
- 同一次 forward 计算 correct、zero、same-source/change-bin matched-shuffle 与 persistence；训练 objective
  只推动 correct prediction，并用 detached intervention baselines 防止通过主动恶化负例取巧；
- W&B 记录四条路径 absolute error、gain、effect variance/effective rank、transport entropy、target/predicted
  change magnitude、identity-copy error 和 matched-shuffle quality。

**运行与评测接口**

- `verify_object_transition_v62.py` 使用真实六源 RGB、DINO、SigLIP 与 CoTracker 执行 forward/backward contract；
- `train_object_transition_v62.py` 和 `train_object_transition_v62.sh` 支持 E0/E1、自动卡数、单机 DDP、W&B、
  milestone/recovery checkpoint 与 strict resume；launcher 只前台 `exec torchrun`；
- `evaluate_object_transition_v62.py` 在 held groups 上逐 source 记录 E0 absolute/gap 指标和 E1 intervention
  指标，并同步 W&B；
- `manage_object_transition_v62.sh` 只编排 foreground verify/train/resume/eval，不提供后台启动。

**当前验证边界**

- 已完成 Ruff、Python compile、Bash syntax、单文件行数与 `git diff --check`；
- 本地 Python 环境没有 PyTorch，因此 `test_object_transition_v62.py` 的 tensor/backward 测试未在本地执行；
- 服务器已在 `f00082d7678f24dd7323ebcb789ef40fca7c0654` 完成 E0 real-teacher GPU verifier：六源覆盖、
  finite gradients、无历史 checkpoint、无 RGB/patch-grid target，`status=passed`；该结果只证明启动路径，不是 E0
  训练或 held quality 结果；
- E0 训练、E0 held Gate、E1 verifier/训练/held Gate 仍待按顺序执行；
- 在 E0 held report 达到第 15.8 节门槛前，不得启动 E1；E1 通过前不得实现或启动 E2/E3。

**E0 首次长训状态（2026-08-31）**

- W&B run `24jo2k1m`（`object_transition_v62_e0_seed17_f00082d`）使用单机 8 卡、每卡 32、
  effective batch 256；目标为 20,000 steps；
- 该 run 的 W&B terminal state 为 `crashed`，最后训练记录为 step 3,340，最近 rolling recovery
  checkpoint 为 step 3,250，因此不能记录为 E0 完成；
- step 3,340 的 train-batch 指标为：loss 0.4254、DINO cosine error 0.1431、SigLIP cosine error
  0.1585、semantic gap recovery 0.3399、support BCE 0.0996、support gap recovery 0.0785、
  support soft-IoU 0.2826、lifecycle gap recovery 0.4891、visibility BCE 0.0250；
- 从前 20 个日志点到最后 20 个日志点，loss 下降约 75.8%，semantic gap recovery 从负值升到约
  0.325，说明 codec 确实在学习；但 support gap 从约 step 1,000 起长期停留在约 0.07，远低于
  E0 held gate 的 0.80；
- 以上全部是 train-batch telemetry，不是 source-balanced held evaluation。当前只允许从 recovery
  checkpoint 继续 E0，同时运行 B/C/D 结构审计；不得把 step 3,340 曲线解释为 E0 Gate 通过或进入
  E1 长训的依据。
- 首次中断的直接异常为 `continuous_object_decoder_v62.py` 对奇异 `2x2 covariance` 执行
  `torch.linalg.inv`。根因是二阶矩处于 BF16 autocast：接近 rank-1 的 covariance 会把 `1e-3` floor
  舍入掉。提交 `3a3f7c671f2881831d67e5ccb4039532675ddc30` 将 codec 与 compact baseline 的 spatial
  moments 固定为 FP32，并以 Cholesky solve 计算 Mahalanobis distance；同时记录 covariance minimum
  eigenvalue 与 condition number。该修复不改变参数、optimizer 或 checkpoint shape，step 3,250
  recovery checkpoint 可继续使用。

### 15.14 B/C/D 并行结构审计

2026-08-31 在 `codex/v62-structural-audits` 增加三个**不改训练参数、不写训练 checkpoint**的独立入口。
它们只共享只读六源 index、frozen teacher 权重和指定 E0 checkpoint；输出、W&B group 与 run ID 完全隔离。
A/E0 训练固定为单机 8 卡，B/C/D 审计固定为单机 4 卡；B/C 按 rank 切分每个 source 的 held 样本，rank 0
汇总各 rank 的原始 numerator/denominator 或 sum/count 后再写 JSON 和 W&B，不重复计算样本。

#### B：Teacher target structural audit

- 入口：`run_teacher_target_audit_v62b.sh`；
- 不读取任何训练 checkpoint，直接在 held 六源视频上运行 frozen DINO、SigLIP、CoTracker 与 relation teacher；
- 保持 membership 质量分布不变，将 track membership 确定性平移一半作为 corruption；
- 比较 selected 与 shuffled 的 DINO/SigLIP group dispersion、motion dispersion、relative-geometry
  instability、same/different relation evidence、persistence、motion salience 与 effective track count；
- 每项记录 numerator、denominator、每源均值和全局 micro aggregate；
- 独立输出根：`outputs/v62_parallel/b_teacher_target/<RUN_ID>/`；
- W&B group：`object-transition-v62b-teacher-audit`。

#### C：E0 codec structural counterfactual

- 入口：`run_object_codec_structural_eval_v62c.sh`；
- 只读一个 E0 checkpoint，在 held 六源数据上执行 normal、continuous-coordinate holdout、query swap、
  all-scene、merge-all、carrier deletion、carrier swap 与 split-by-time；
- 报告 absolute codec/compact/oracle error、gap recovery、各 corruption 的 error increase、carrier deletion
  inside/outside external support change ratio，以及 normal/split temporal identity error；
- continuous holdout 使用一半 tracks 编码、另一半 tracks 解码，避免在输入坐标上自评；
- 独立输出根：`outputs/v62_parallel/c_codec_structure/<RUN_ID>/`；
- W&B group：`object-transition-v62c-codec-structure`。

#### D：E1 DDP/runtime probe

- 入口：`run_transition_runtime_probe_v62d.sh`；
- 只读指定 E0 checkpoint，在 held data 上临时构造 Posterior + transport/residual Dynamics；
- 通过 `torchrun --nproc_per_node 4` 执行三次真实 DDP forward/backward/optimizer step，不保存模型；
- 报告 unused parameter count、gradient norm、rank parameter sync、correct/zero/shuffle/persistence、future-swap
  posterior sensitivity、zero-effect causal isolation 与 identity exact copy；
- 独立输出根：`outputs/v62_parallel/d_e1_runtime/<RUN_ID>/`；
- W&B group：`object-transition-v62d-ddp-runtime`。

#### 隔离与证据边界

- 三个 launcher 都在前台运行，不使用 `nohup`、`&` 或 detached manager；
- `RUN_ID` 默认包含时间与 PID；三类任务没有共享 `latest`、log、report 或 checkpoint 路径；
- B 可以与 E0 训练立即并行；C/D 只有指定 E0 immutable milestone checkpoint 存在后才能执行，禁止读取
  训练过程中持续替换的 recovery/latest 路径；
- D 的 optimizer step 只发生在进程内临时模型上，不写入 E0/E1 输出，也不构成 E1 训练结果；
- B/C/D 只回答 target、binding 与 runtime 的结构性问题，不替代第 15.8 节 E0/E1 held Gate。

#### 在线任务源码兼容修正

- 首次提交 B/C/D 后，在线任务在 Python 启动前报
  `code/scripts/run_teacher_target_audit_v62b.sh: No such file or directory`。仓库提交内入口存在，失败原因是
  在线任务挂载了旧的源码快照，而在线运行环境又不能访问 GitHub；
- 新增 `deploy_object_transition_v62_runtime.sh`。代码同步阶段把当前提交的完整 `code/igsw` 与
  `code/scripts` 复制到
  `/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/object_transition_v62/releases/<SOURCE_REVISION>/`；
- 在线 B/C/D 只从该 immutable release 运行，不读取
  `/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/`，也不调用 Git 或 GitHub；
- release 根写入 `SOURCE_REVISION`，B/C/D launcher 在没有 `.git` 的运行快照中从该文件读取 provenance；
- 该修正只改变代码分发边界；当前 B/C/D 审计统一使用 4 卡 DDP，不改变数据、模型、指标或输出隔离契约。

#### E0 第二次中断与 checkpoint 迁移

- E0 恢复运行到 step 3,940 后再次在 `continuous_object_decoder_v62.py` 的
  `torch.linalg.inv(state.covariance.float())` 中断；当前修复提交已使用 Cholesky solve，因此该 traceback
  直接证明任务仍从 `/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/` 的旧代码启动，而不是数值修复失效；
- recovery checkpoint 每 250 steps 保存，因此本次可恢复边界为 step 3,750，而不是最后打印的 step 3,940；
- E0 也改为从 immutable release 运行。新增显式
  `RESUME_COMPATIBLE_SOURCE_REVISION`，只允许 checkpoint 中记录的指定旧提交或当前 release 提交通过；
- 该迁移不放宽 architecture、stage、world size、model/config、batch、optimizer、scheduler、训练步数或数据契约；
  covariance 修复没有新增参数或改变 tensor shape；
- 迁移后保存的 checkpoint 使用当前 release 的真实 `git_commit`，并额外记录旧 checkpoint revision，后续
  resume 不需要伪装成旧代码。

#### D mixed-precision runtime 修正

- D 在 immutable release `4afe969d6dd2ed05622817a7bd1d8438271e9077` 进入真实 8 卡 forward 后，
  intervention diagnostics 将 autocast 内产生的 BF16 latent effect 在 autocast 外送入 FP32
  `effect_input`，触发 `mat1 and mat2 must have the same dtype`；
- 根因是 Posterior、Dynamics 与 Decoder 的公共接口默认调用者始终位于同一个 autocast context，导致主训练
  forward 正常而离线 intervention/probe 失败；
- v62 现在在三个模块边界将 state、effect、coordinate 与 time inputs 转为对应模块 parameter dtype。
  autocast 内仍由 PyTorch 选择 BF16 kernel，autocast 外则使用 FP32，不改变模型参数、checkpoint shape 或 loss；
- `test_object_transition_v62.py` 增加 BF16 state/effect 离开 autocast 后依次调用 Posterior、Dynamics 和
  Decoder 的回归路径，覆盖 D 本次实际失败方式。

### 15.15 E0/B/C/D 联合结果（2026-09-01）

本节以 W&B 项目 `healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world` 的完整 run
history/summary 为依据。E0 是在线 train telemetry；B/C/D 是 held structural audit/runtime probe，证据层级不同。

#### E0：继续学习 semantic，但 support 已平台化

- run `24jo2k1m` 在 2026-09-01 12:18 CST 仍有 heartbeat；读取时最新 step 7,640，最新 checkpoint
  step 7,500，checkpoint 已迁移到提交 `311bc28919ac2ccea620b98d7f1fe4b8c7d7d349`；
- step 3k--4k 到 7k--7.64k：loss `0.4246 -> 0.3629`，semantic gap recovery
  `0.3453 -> 0.4642`，DINO cosine error `0.1424 -> 0.1169`，说明 appearance/semantic compression
  仍在缓慢改善；
- 同期 support gap recovery 只从 `0.0772` 到 `0.0860`，support soft-IoU 约
  `0.2824 -> 0.2864`。最近 25 个点的 support gap slope 为每 1k steps `-0.0606`，已经不是持续改善；
- lifecycle gap recovery 约 `0.49`，identity cosine error 已降到约 `9e-4`；carrier effective count
  升到约 `15/16`，但这只证明 capacity 被使用，不能证明 carrier 是 semantic object；
- covariance minimum eigenvalue 从 3k--4k 的 `0.0444` 降到 7k 后的 `0.0131`，condition number
  从 `6.80` 升到 `9.76`；当前 finite，但 carrier support 正持续变尖，需要继续记录而不能解释为 object quality。

#### B：teacher membership 的 object coherence 很弱

- run `5j5b72qs`，六源各 32 个 held items，全部 object-valid；teacher 选择的 support 平均只占 point
  queries 的 `5.66%`，effective tracks 为 `14.77`，track persistence 为 `0.9898`；
- 相比保持 membership 规模不变的 half-track-roll corruption，teacher 的 DINO/SigLIP dispersion margin
  仅 `0.00153/0.00152`，same-relation margin `0.0165`，motion dispersion margin `0.00057`；
- 这些总体正 margin 主要由 Bridge 拉动。RoboTwin 的 motion 与 same-relation margin 为负，HY 的
  DINO/SigLIP margin 为负，RoboMind 的 DINO/SigLIP margin 也接近零或为负；
- 因而 teacher 具有高 visibility/persistence，但在多数来源上没有稳定证明“选中的 tracks 比同规模错配 tracks
  更像同一个 object”。E0 当前最多能声称学习 teacher-selected local track group，不能声称 semantic object。

#### C：codec 使用了 object/query，但未形成可靠 continuous object field

- run `1ksunj26` 使用 E0 step 2,500 checkpoint；held normal absolute error `0.4617`，compact baseline
  `0.6011`，gap recovery `22.4%`；semantic gap recovery `23.8%`，support gap recovery仅 `7.0%`；
- query swap、all-scene、merge-all、carrier swap 分别使 absolute error 增加
  `0.6543/1.4772/1.3194/0.7214`；split-by-time 使 identity error 从 `0.0349` 升到 `0.3103`。
  这证明 codec 没有忽略 teacher query、carrier 或时间 identity；
- 删除 object-support mass 最大的一个 carrier 后，inside change `0.0674`、outside change `0.00110`，
  locality ratio `69.5`。这是 compositional locality 的正证据，但单 carrier 删除只使总 error 增加 `0.0191`；
- 用一半 tracks 编码、在未输入的另一半 tracks 上解码时，error 是 normal 的 `1.85x`，且六个来源均为
  `1.59x--2.11x`。因此当前 decoder 主要拟合已观察 query/support，尚不能稳定表示连续 object extent。

#### D：runtime 正确，但当前 Dynamics 的结构语义不成立

- run `j4igr1h4` 使用 8 卡、E0 step 2,500 checkpoint 完成 3 次真实 optimizer steps；unused parameters
  为 `0`，rank parameter sync difference 为 `0`，future swap 使 Posterior RMS 改变 `0.0682`，zero-effect
  future-swap difference 为 `0`，identity exact-copy difference 为 `0`；
- effect effective rank `13.31`、batch variance `0.00819`，Posterior 在该 probe 中没有立即 collapse；
- 但 correct/zero/shuffled/persistence absolute error 分别为
  `1.7524/1.7495/1.7534/0.5173`。correct 相对 persistence 的 gain 为 `-1.2350`，相对 zero 为
  `-0.00285`，相对 shuffled 只有 `0.00105`；
- target change magnitude 只有 `0.1156`，correct 与 zero prediction change magnitude 却分别为
  `1.0979/1.0945`，约放大 `9.5x`；transport entropy `2.77197` 几乎等于 16-way uniform 的
  `ln(16)=2.77259`；
- 这不是“三步没训好”可以完全解释的问题：zero effect 仍触发大幅变化且 transport 初始为近均匀混合，说明
  当前 Dynamics 没有结构性满足 `zero effect = persistence`。在修复该契约前不得启动 E1 长训。

#### 联合决策

1. E0 只继续到 step 10,000 milestone，用于完成学习曲线和新版 held C；没有证据支持把 20k 当作
   object-state promotion run。
2. 第一优先级是重构 teacher membership，使 selected tracks 在至少 5/6 来源上同时优于 corruption 的
   appearance、motion 与 geometry coherence；否则 student 只能精确拟合错误 target。
3. E0 objective 必须加入 encode-query/decode-query 分离的 continuous holdout supervision；不能继续只在输入
   point coordinates 上训练 support field。
4. E1 Dynamics 改成 identity/persistence base 加 effect-gated residual，并强制 `z=0` 时 transport 为 identity、
   feature/geometry/lifecycle residual 全为零；完成新的 D probe 前不进行 E1 长训。

### 15.16 V62 后续改造顺序与 C10k 契约

**What changed**

1. C 的正式复评 checkpoint 固定为 E0 step 10,000 immutable milestone。continuous-query 评测使用偶数
   point tracks 编码、奇数 point tracks 解码，并增加“全部 tracks 编码、只在奇数 tracks 解码”的
   full-context reference。
2. C 分别报告 held-query 的 support BCE/soft-IoU、DINO/SigLIP cosine error、visibility BCE、lifecycle
   cross-entropy，以及它们相对 compact baseline 和 full-context reference 的 recovery/ratio；不再用单个
   `continuous_holdout_absolute_error` 代替 semantic 与 support 结论。
3. 后续主线固定为：C10k 冻结 V62 结论；独立重构 teacher membership 并完成六源 corruption audit；teacher
   通过后训练 encode-query/decode-query 分离的 codec；codec 通过后实现 persistence base 加
   effect-gated residual Dynamics；以上全部通过后才整合新的完整预训练入口。

**Why**

step 2,500 的 C 已证明模型使用 query/carrier，但不能区分 unseen-query 失败究竟来自 semantic field 还是
support/lifecycle；同时 B 和 D 已分别否定当前 teacher coherence 与 zero-effect Dynamics 契约，因此不能把三个
未验证模块一起放进一次长训后再依赖总 loss 排障。

**Impact**

- item 2（旧 E1 长训）保持关闭；
- 每一级只生成独立 audit/训练产物，不覆盖当前 E0 checkpoint；
- 完整预训练代码只在 teacher、codec、Dynamics 三个结构门槛分别获得真实六源证据后交付。

### 15.17 C10k 结果与 consensus teacher 实验

**What changed**

1. W&B run `wdpiq9fa` 使用 E0 step 10,000 完成新版 C。normal absolute error 从 step 2,500 的
   `0.4617` 降至 `0.3397`，normal semantic gap recovery 从 `0.2378` 升至 `0.4938`；但 half-track
   unseen-query absolute error 从 `0.8401` 升至 `0.9360`。
2. 在完全相同的 odd track queries 上，half-track encoding 相对 all-track full-context encoding 的总误差、
   semantic error、support BCE 分别为 `2.86x/3.00x/2.05x`。unseen semantic/support gap recovery
   分别为 `-0.6819/-0.6716`，六个数据源全部同向失败。
3. 新 teacher candidate 不再使用旧 relation row 作为 object。它在前半段视频上用 frozen DINO、SigLIP
   appearance 共识与 CoTracker motion/relative-geometry/covisibility 构建 signed affinity graph，从动态 seed
   做两跳 soft diffusion；后半段视频只用于独立 coherence audit。

**Why**

C10k 证明继续优化原 objective 会提高输入 track reconstruction，同时恶化未输入位置的 object field；因此下一步
必须先改变 membership target，而不是增加 E0 steps 或调整 reconstruction loss 权重。

**Impact**

- consensus teacher 在每个来源上与保持 membership 权重分布不变的 half-track roll corruption 比较 DINO、
  SigLIP、motion 与 relative geometry 四项 held-suffix dispersion；
- 单来源只有在至少 50% 样本可形成多-track membership 且四项 corruption margin 全为正时才通过；六源至少
  `5/6` 通过才允许进入新 codec；
- old same-seed one-hop teacher 同时作为只读对照，但不进入 candidate 构图或 held-suffix target。

### 15.18 V63 consensus teacher 审计结果

**What changed**

1. W&B run `jssoxex6` 在 8 卡上正常完成六源 held audit，运行状态为 `finished`，程序报告状态为
   `completed`；这不是启动失败或运行时崩溃。
2. candidate 在六个来源上都能构造有效 membership，总体 `candidate_valid=1.0`，audit valid fraction 为
   `0.9583`；prefix/suffix membership cosine error 为 `0.00446`，说明两段独立时间窗口生成的 membership
   数值上稳定。
3. 但 candidate 相比 old same-seed one-hop teacher 的 DINO、SigLIP、motion、relative geometry error
   improvement 分别为 `-0.04775/-0.01889/-0.00326/-0.00100`。四项全部为负，说明两跳 consensus
   diffusion 扩张后的 component 比旧 target 更不一致。
4. 对 half-track roll corruption，candidate 的总体 DINO、SigLIP、geometry margin 为正，但 motion margin 为
   `-0.000134`；按单来源严格判据，只有 RoboTwin 四项 margin 全为正，最终 `passing_source_count=1/6`，低于
   要求的 `5/6`。

**Why**

低 prefix/suffix disagreement 只证明 membership 构造可重复，不证明它对应 object。两跳 diffusion 把 seed
邻域稳定地扩张到了 appearance 或共见相似、但 motion/geometry 不属于同一 persistent object 的 tracks；Droid
甚至在四项 corruption margin 上全部失败。由于 audit coverage 充足，失败不能归因于样本不足或 tracker 无输出。

**Impact**

- V63 决策固定为 `reject_consensus_teacher`，不重跑、不延长、不通过放宽阈值进入 codec；
- 第 3 项 teacher membership 继续保持进行中，第 4 项 codec 与第 5 项 Dynamics 保持关闭；
- 下一版 teacher 必须避免无约束 graph diffusion，以可证伪的 object-bound transition consistency 作为 component
  合并依据，并继续使用六源 held corruption audit 决定是否晋级。

### 15.19 V64 object-bound transition teacher 实现契约

**What changed**

1. V64 删除 pairwise graph diffusion。每个 component 从 persistent motion-active seed 出发，其他 track 必须与
   seed 存在直接 DINO、SigLIP、covisibility、locality、rigidity 和 motion 一致性，不能通过中间 track 传递加入。
2. Membership 在前半段视频上分别拟合 `100/200/400ms` 的 component-level affine residual-flow model，并只保留
   在所有可用 horizon 上共同成立、且至少包含四个可见 tracks 的 components；object components、scene 与
   unknown 显式分流，单个 track 最多属于一个 object component。
3. 后半段独立报告 absolute shared-transition residual、persistence gain、DINO/SigLIP/motion/geometry dispersion，
   并执行 half-track roll、same-source sample swap、alternate-component merge 和 old one-hop 对照。

**Why**

V63 已证明两跳 affinity 可以生成跨时间稳定但 motion/geometry 错误的 membership；V64 改为要求一个 component
能够被同一个紧凑 transition model 联合解释，使 teacher 的定义与后续 object Dynamics 使用的变化单位一致。

**Impact**

- V64 是六源 teacher 晋级审计，不训练 codec、Student 或 Dynamics，也不读取历史模型 checkpoint；
- 资源契约固定为：所有审计使用单机 4 卡，只有参数训练使用单机 8 卡；V64 launcher 因此固定启动 4 个 rank；
- 只有至少 `5/6` source 同时通过 corruption、persistence 和 old one-hop 对照才进入 cross-query compositional
  codec；
- 若 V64 失败，停止继续设计 hard pseudo-object teacher，转向由 held-track prediction 决定 assignment 的 latent
  cross-fitted binding。

### 15.20 V64 四卡 held audit 结果

**Execution**

- W&B run `2sa91010` 使用提交 `f052fed17c6f50d285c7b088368eb735fb8ababe`、单机 4 卡和每源 32 个
  held clips 正常完成；run state 为 `finished`，程序状态为 `completed`，实际 `world_size=4`；
- 最终决策为 `reject_object_bound_teacher`，`passing_source_count=0/6`，不得进入 codec 或 Dynamics 训练。

**Evidence**

1. 在 audit-valid 子集上，component-level shared transition 确实包含变化信号：总体 residual 为 `0.20764`，
   persistence error 为 `0.66593`，即 residual 相对 persistence 降低约 `68.8%`；old one-hop residual 为
   `0.35922`，V64 相对降低约 `42.2%`。DINO、SigLIP、motion 和 relative-geometry dispersion 也都优于
   old one-hop。
2. 该信号覆盖面不足：总体 candidate-valid fraction 只有 `0.25`，最终 audit-valid fraction 只有 `0.1875`，
   低于 `0.5` 门槛。各 source audit-valid fraction 为 RoboTwin `0.375`、Bridge `0.34375`、AgiBot
   `0.1875`、RoboMind `0.15625`、Droid `0.0625`、HY `0`；HY 的 32 个样本均未形成有效 component。
3. 有效 component 平均约 `5.41` 个 tracks，但总体 unknown fraction 为 `0.4661`。这说明 direct affinity 与
   三个 horizon 的共同 affine consistency 只保留了少量容易解释的局部轨迹，尚不能形成广泛 object state。
4. half-track roll corruption 基本没有被稳定区分：总体 shared-transition residual margin 为 `-0.00064`，
   SigLIP margin 为 `-0.00005`；RoboTwin、Droid 等 source 也出现负 margin。因此即使在有效子集上，当前
   membership 仍未证明是 object-specific，而可能只是易于同一局部 affine model 拟合的小轨迹集合。
5. same-source sample-swap 指标存在独立实现缺陷：swap 可能把另一个样本的 invalid/zero membership 移入当前
   audit-valid 样本，但 accumulator 只使用原 candidate 的 valid mask，导致 zero residual 被当作更好结果。
   所以总体 swap residual margin `-0.07069` 不能解释为错误 sample 的 transition 更准确；后续 evaluator 必须
   对 candidate 与 corruption 的联合有效集聚合，并单独报告 corruption coverage。

**Decision**

- V64 证明“shared transition residual 可以作为 object binding 的训练信号”，但否定了“先用 hard direct-seed
  规则产生通用 pseudo-object teacher”的路线；不能通过降低最少 track 数、放宽 `0.5` coverage gate 或增加
  样本数来晋级；
- 下一主线是 latent cross-fitted binding：prefix-only encoder 产生 soft assignment，prefix transition 拟合与
  held-suffix prediction 直接优化 assignment；scene/unknown 保留独立出口，hard V64 membership 只作为诊断，
  不再作为训练 GT；
- 在实现新 binding 前，先修正 corruption 的 joint-valid aggregation。该修正只保证评测语义正确，不改变 V64
  的低 coverage 与 roll falsification 失败结论。

### 15.21 V65 native reliable multi-track transition objective

**Record**

- 日期：2026-09-02；branch：`codex/reliable-native-object-transition-v65`；实现 commit：
  `a1375111e44f4edf63a1839581f4b47c4b8b7dce`；W&B run：待执行；checkpoint：无；
- 当前 Gate：G0 Objective validity。V65 是冻结 teacher/runtime 的六源 held audit，不训练 Student、codec、
  latent effect 或 Dynamics，不读取任何历史模型 checkpoint；
- 单一假设：一个 object transition 必须由多条可靠 track 在 prefix 中共同支持，并且只用这些 core tracks 拟合的
  transition 应在未参与拟合的 tracks 上，命中 future native image 中更相符的局部视觉内容。若它不能优于
  persistence、错误 core 和其他样本 transition，则该 dynamic target 不成立。

**What changed**

1. 数据增加 `preserve_native_rgb` 路径。原始 RGB 不再进入旧的整帧 `224/518` square resize；异分辨率样本只在
   batch 中做右侧和底部 zero padding，并保留每个样本自己的 `native_image_hw`。tracker、tile token 和 local
   pooling 均在每个样本自己的 native 坐标系内计算，padding 不改变运动尺度。
2. Frozen DINOv2-L 与 SigLIP 不再对整张高分辨率图像做一次大压缩，而是在原图上使用 `224` pixel overlapping
   tiles、`168` pixel stride。point observation 使用半径 `14/28/56` pixel 的离散近邻 token attention；不使用
   单点 bilinear feature sampling，也不把 patch index 当作 object GT。
3. CoTracker 只提供 noisy geometry measurement。每张图使用 `16x16` grid，并从 clip 起点和中点分别发出 queries；
   每条 primary track 又在后续 relay frame 重新查询。primary/relay disagreement、joint visibility 与跨时
   DINO/SigLIP appearance consistency 共同形成 reliability。低 reliability track 不能进入 component。
4. V64 的 single-seed direct component 被替换为 multi-track core：proposal 只用于找到候选邻域，最终 core 由候选
   tracks 间的 mutual affinity 共同选择。默认 candidate pool 为 `12`、core 为 `6`；component membership 必须
   同时通过 absolute affinity floor 和 core consensus，不能因为所有 affinity 同样接近零而被相对阈值误收。
5. `100/200/400ms` transition 使用 reliability-weighted Huber IRLS 拟合完整 2D affine map。fit 只读取 prefix
   core tracks；每个 component 至少保留 `2` 条非 core holdout tracks，不能把参与拟合的点重新作为验证点。
6. 主要 evaluator 不再把 CoTracker future flow 当 GT。prefix identity 与预测坐标处的 future native-image
   DINO/SigLIP local features 比较；future tracker coordinate error 只保留为 tracker-dependent auxiliary
   diagnostic。persistence、spatially rolled core 和 same-batch shuffled transition 都在各自 joint-valid 子集上
   聚合，修复 V64 invalid corruption 被当作零误差的问题。

**Execution contract**

- 数据：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/multisource_real_robot_video_v53/index.json`，六个来源，
  held group partition，默认每源 `32` clips、每 clip `10` 帧、固定 `100ms` temporal step；
- 资源：所有审计固定单机 `4` 卡；本版本没有训练任务，因此不存在 8 卡 launcher；
- 运行：代码先部署到
  `/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/native_reliable_transition_v65/releases/<commit>/`，之后 audit
  在该 immutable release 前台运行，不使用 `nohup`、`&`、Git 或 GitHub；
- 产物：独立 report、W&B run 和 output directory，不覆盖 V62--V64 任何结果。

**Local structural evidence**

- `python3 -m py_compile`、Ruff、shell syntax 与 `git diff --check` 已通过；
- synthetic verifier 检查了 heterogeneous native padding、relay corruption、multi-track core、fit/holdout 分离、
  component purity 和 future visual falsification，状态为 `passed`；
- 合成样本中 good/bad relay reliability 为 `1.0/0.13137`，selected core count 为 `4/4`，holdout count 为
  `3/4`，object purity 为 `1.0/1.0`；correct transition 相对 persistence 的 future visual gain 为
  `0.23564/0.39859`，相对错误 core 的 margin 为 `0.24895/0.42133`。

这些数字只证明实现可以在已知构造数据上识别预期反事实，不是 Robot data 上的方法结果，也不证明 CoTracker
measurement 正确。

**Promotion / rejection rule**

- 每个 source 必须同时满足：audit-valid fraction 至少 `0.5`、rolled/shuffled corruption joint coverage 至少
  `0.25`、component reliability 达标、correct future visual error 分别优于 persistence、rolled core 与 shuffled
  sample transition；
- 六源至少 `5/6` 同时通过，才允许把 V65 objective 晋级为下一轮可学习 soft binding 的监督；否则直接拒绝该
  target，不通过放宽阈值、增加 steps 或调 loss weight 补救；
- 真实四卡 audit 返回前，状态固定为 `implemented, awaiting real-data objective falsification`。下一步仅执行
  structural verifier 与四卡六源 audit；不得启动 Student、codec 或 Dynamics 长训。

### 15.22 V65 四卡六源结果与 shared-affine 判决

**Execution**

- W&B run：`dhfr98xu`，名称
  `reliable_native_transition_v65_seed17_20260902_120848`；代码提交
  `ba000d46e8d356bf723fcfcd49e66a43053a7e6d`；run state 为 `finished`，程序状态为 `completed`；
- 单机 `4` 卡、六个来源、每源 `32` 个 held clips，共 `192` 个样本；没有 checkpoint、训练 step 或历史模型；
- 最终程序判决：`reject_native_reliable_transition_objective`；通过来源为 AgiBot、RoboMind，共 `2/6`，低于
  `5/6` 晋级门槛。

**Evidence**

1. V65 明显修复了 V64 的 coverage 问题。总体 audit-valid fraction 从 V64 的 `0.1875` 提高到
   `0.57292`；AgiBot、Droid、HY、RoboMind 分别达到 `0.75/0.6875/0.78125/0.5`。Bridge 和 RoboTwin
   仍只有 `0.40625/0.3125`。有效 component 平均包含 `11.95` effective tracks，并保留 `6.32` 条
   holdout tracks；这证明 native-resolution multi-anchor evidence 和 multi-track core 能在真实数据上形成更广的
   可审计 component。
2. Tracker reliability 不是主要失败项：有效 component 的平均 reliability 为 `0.7580`，primary/relay
   normalized disagreement 为 `0.00233`。这些数字只代表 tracker 内部一致，不代表外部位置 GT。
3. Component 与 transition 具有 specificity。正确 core 相比 spatially rolled core 的 overall future visual
   margin 为 `+0.02710`；正确样本 transition 相比 shuffled sample 的 margin 为 `+0.04413`。除 Droid 的
   rolled-core margin 为 `-0.00023` 外，有 joint coverage 的 source 基本均为正。这说明 multi-track binding 没有
   完全退化为任意局部点集，且不同 clip 的 transition 不是可互换常数。
4. 但核心目标失败：correct/persistence/oracle-track future visual error 分别为
   `0.09117/0.08814/0.07261`。correct 相比 persistence 的 gain 为 `-0.00303`，即 shared affine transition
   整体不如不移动。AgiBot、RoboMind 和 Bridge 有正 gain `+0.00176/+0.00048/+0.00969`；Droid、HY 和
   RoboTwin 为 `-0.00850/-0.00358/-0.02323`。
5. RoboTwin 是最明确的反例：oracle track 相比 persistence 留有 `0.01864` visual headroom，但 affine correct
   比 persistence 还差 `0.02323`，coordinate gain 也为 `-0.01680`。这不能解释成“视频基本静止”，而是当前
   shared affine parameterization 对该来源的 object motion 给出了错误外推。
6. Droid 与 HY 的 oracle headroom 只有 `0.00225/0.00092`，说明它们的大量有效样本在当前 `100--400ms`
   尺度上几乎没有可由 local semantic feature 观察到的变化。把这些样本与 motion-active transition 混合平均，
   会让 persistence gate 同时测量动态建模能力和数据中的静态比例；下一轮必须同时报告全量与 image-derived
   motion-active strata，不能只替换总体均值。
7. Raw-track oracle 明显优于 correct affine，说明被 reliability 选中的 raw trajectories 含有可用 future visual
   correspondence；但 oracle 仍受 full-clip appearance reliability 的选择影响，只能作为 model-class ceiling，
   不能写成 CoTracker 是真实运动 GT。

**Decision**

- 保留：native RGB 数据路径、overlapping tiled DINO/SigLIP、multi-anchor relay reliability、multi-track core、
  fit/holdout 分离、future-image visual evaluator 和 joint-valid corruption aggregation；
- 拒绝：一个 object 只用单个 shared 2D affine map 表示 `100/200/400ms` transition。该参数化即使 coverage
  足够，也不能把 raw trajectory 的视觉对应关系转换为优于 persistence 的 dynamic target；
- 不启动 Student、codec、latent effect 或 Dynamics 训练，也不通过增加样本、延长训练或调整 loss weight继续
  V65；
- 下一项仍属于 G0 model-class audit：在完全相同的 component、prefix/holdout 和 future-image evaluator 下，
  比较 translation、shared affine 与“global motion + 少量 object-local residual modes”的可组合 motion field。
  residual modes 必须由 prefix core tracks 共同估计，并在未参与拟合的 holdout tracks 上预测；禁止退化为每条
  track 独立外推。评测同时报告 all-valid 与 image-derived motion-active strata。只有新的 object-level motion
  field 在至少 `5/6` 来源上优于 persistence 和两类 corruption，才允许进入可学习 soft binding。

### 15.23 V66 G0 大样本 object motion-field 方法审计

**Record**

- 日期：2026-09-02；branch：`codex/object-motion-field-g0-v66`；实现提交：
  `293482aad349195a642f61d87b909b77d61787ed`；W&B run：待执行；checkpoint：无；
- 当前仍是 G0 Objective validity，不训练 Student、codec、latent effect 或 Dynamics，也不继承 V62--V65
  checkpoint；V65 的 native image、reliable multi-track binding 和 core/holdout evaluator 是冻结的实验边界；
- 单一问题：V65 的失败究竟来自“object motion 根本不能由 prefix tracks 预测”，还是来自“一个 shared affine
  容量不足”。V66 只改变 transition model class，不改变数据、object candidate 或 primary target。

**Method**

1. 对每个样本和每个 `100/200/400ms` horizon 同时拟合四个有严格包含关系的模型：persistence、shared
   translation、shared affine、以及 `shared affine + 3 object-local RBF residual modes`。三个 local modes 由
   prefix core tracks 的 object-relative positions 决定，系数由所有 core tracks 共同估计；不存在 per-track
   head、track ID embedding 或 future-conditioned parameter。
2. 拟合只读取 clip 前半段的 core tracks。主评估只使用没有参与拟合、且在预测起点实际可见的 holdout tracks；
   真实 future tracker coordinates 只计算 auxiliary coordinate error，主判据仍是预测位置处的 future native-image
   DINO/SigLIP local feature error。
3. 反事实包含 spatially rolled core 和 same-batch shuffled sample field。每个 margin 都使用 candidate 与
   corruption 的联合有效集；如果某个 corruption 没有足够有效样本，该项明确判为证据不足和失败，不会以零误差
   或缺失字段进入平均。

**Sampling and statistics**

- held clips 从每个来源 `32` 提高到 `256`，六源共 `1536` 个样本；固定 `10` 帧、`100ms` temporal step，
  使用单机 `4` 卡前台执行；这不是训练 batch 或超参数搜索；
- 每个 source 同时报告 all-valid 与 high-change strata。high-change 定义为该 source 中、在 held future native
  image 上 persistence visual error 的上半区；选择只依赖 baseline，不读取 proposed motion-field error。该集合
  表示“图像观测上 persistence 明显失配”，不能被解释成物理运动 GT；
- translation/affine/field 相对 persistence，以及 field 相对 translation/affine/rolled/shuffled 的 paired
  difference 均使用 `2000` 次 bootstrap 给出 95% CI。每个 high-change 判据至少需要 `32` 个 joint-valid
  observations；总样本增加与 CI 同时用于降低 V65 每源 32 条可能造成的方差和 source 偏差。

**Pre-registered decision**

- 单个 source 必须满足：audit-valid fraction 至少 `0.5`；high-change joint-valid 样本至少 `32`；all-valid
  motion field 不差于 persistence；high-change 上 field 相对 persistence、affine、rolled core 和 shuffled
  sample 的 95% CI 下界全部大于零；
- 六源至少 `5/6` 通过才晋级 `object_motion_field_g0`。失败后不通过增加 mode 数、调 ridge、改变 strata 或延长
  运行补救；结果将直接决定是进入可学习 soft object binding，还是拒绝当前 tracker-derived object-transition
  objective；
- translation 与 affine 保留为解释性 nested baselines。只有 field 同时超过它们和错误 object/sample
  corruption，才能说明增益来自 object-local dynamic structure，而不是更大的坐标回归器。

**Implementation and current evidence**

- 新增独立 V66 config、compact motion-field fitter、held audit、分层 bootstrap 汇总、synthetic structural
  verifier、immutable deploy 和四卡前台 launcher；runtime release 不访问 Git/GitHub，不产生训练 checkpoint；
- `python3 -m py_compile`、Ruff、shell syntax 与 `git diff --check` 已通过；本机没有安装 PyTorch，且本轮到
  `10.66.0.39:8600` 的连接在认证前被远端关闭，因此 synthetic numerical verifier 与真实四卡 audit 尚未执行；
- 当前状态固定为 `implemented, statically verified, awaiting four-GPU structural and real-data audit`。服务器
  返回 synthetic verifier 与 W&B 六源结果前，不声称 motion field 有效，也不启动任何下游训练。

### 15.24 V66 四卡六源结果与 G0 判决

**Execution**

- 正式结果使用 W&B run `39byxraf`，名称
  `object_motion_field_g0_v66_seed17_2e513d9_256ps`；run state 为 `finished`，程序状态为
  `completed`，代码 revision 为 `2e513d9d0f3f1b37a24fe5af4f1df2c95ac141f4`；
- W&B run `b739j6qn` 与 `39byxraf` 的 config、六源数值和最终判决完全相同，是同一份 report 的重复上传，
  不能计作第二个 seed 或独立重复实验；
- 单机 `4` 卡、每源 `256` 个 held clips、六源共 `1536` 个样本、每 clip `10` 帧、固定 `100ms`
  temporal step；本轮没有训练、optimizer step 或 checkpoint；
- 高变化集合只按各来源 held future native-image persistence error 的上半区选取，共 `467` 个有效样本；
  每个 paired comparison 使用 `2000` 次 bootstrap 估计 95% CI。

**Aggregate evidence**

1. all-valid 上 persistence、translation、shared affine、motion field 与 raw-track oracle 的 visual error 分别为
   `0.085442/0.085973/0.085319/0.085121/0.073020`。motion field 相比 persistence 的平均 gain 只有
   `+0.000260`，95% CI 为 `[-0.001349,+0.001900]`，不能确认优于不移动。
2. high-change 上对应误差为 `0.126789/0.123662/0.122058/0.122043/0.101313`。motion field 相比
   persistence 的 gain 为 `+0.004802`，95% CI `[+0.001837,+0.008109]`；但相比 shared affine 只改善
   `+0.000063`，95% CI `[-0.000219,+0.000355]`。因此 high-change 的主要收益已经由 shared affine
   提供，三个 object-local RBF residual modes 没有得到统计支持。
3. high-change motion field 相比 rolled core 和 shuffled sample 的 margin 分别为 `+0.023395`
   `[+0.018632,+0.028444]` 与 `+0.053212` `[+0.043151,+0.064425]`。这说明 component 与 sample
   specificity 确实存在；失败不能简化为“所有 track target 都是随机的”。但是 specificity 不等于可预测性：
   正确 object/sample field 仍未稳定超过 persistence 或较简单的 affine。
4. high-change raw-track oracle 相比 persistence 留有 `0.025476` 的平均 visual headroom。future-image
   evaluator 能检测到真实轨迹位置与静止位置的差异，但 prefix-fitted field 没有恢复这部分 headroom。
   raw-track oracle 仍依赖 tracker，只是 model-class ceiling，不是独立运动真值。

**Per-source decision**

| Source | audit coverage | high-change N | all-valid field gain vs persistence | high-change field gain vs persistence, 95% CI | high-change field margin vs affine, 95% CI | 判决 |
|---|---:|---:|---:|---:|---:|---|
| RoboTwin | `0.426` | `55` | `-0.00632` | `-0.01077 [-0.02070,-0.00140]` | `-0.00003 [-0.00073,+0.00067]` | fail：coverage 不足，且在 high-change 上显著差于 persistence |
| AgiBot | `0.734` | `94` | `+0.00786` | `+0.01975 [+0.01192,+0.02828]` | `+0.00002 [-0.00068,+0.00071]` | fail：有效运动来自 affine，local modes 无可靠增益 |
| Droid | `0.773` | `100` | `-0.00179` | `-0.00178 [-0.00570,+0.00170]` | `-0.00004 [-0.00043,+0.00037]` | fail：不优于 persistence、affine，rolled margin 也不确定 |
| RoboMind | `0.555` | `71` | `+0.00201` | `+0.00875 [+0.00046,+0.01949]` | `+0.00063 [+0.00010,+0.00131]` | pass：唯一满足全部预注册条件的来源 |
| Bridge | `0.395` | `51` | `-0.00212` | `+0.00117 [-0.00700,+0.01045]` | `+0.00038 [-0.00046,+0.00123]` | fail：coverage 不足；persistence、affine、shuffled 证据均不足 |
| HY-Embodied | `0.746` | `96` | `-0.00138` | `+0.00489 [+0.00005,+0.01060]` | `-0.00033 [-0.00130,+0.00056]` | fail：all-valid 退化，local modes 不优于 affine |

**Decision**

- 最终程序判决为 `reject_object_motion_field_g0`：仅 RoboMind 通过 `1/6`，远低于预注册的 `5/6`；
- V65 到 V66 已依次检验 shared affine 与 affine 加 object-local residual modes。大样本结果排除了“只是每源
  32 条方差太大”以及“只需给 affine 增加少量局部容量”这两个解释；不再增加 mode 数、调整 ridge、降低
  coverage gate 或继续运行同类 audit；
- 保留 native-resolution visual evaluator、DINO/SigLIP local features、multi-anchor reliability、multi-track
  core、fit/holdout 分离、all-valid/high-change 分层以及 rolled/shuffled 反事实。这些组件证明了可观测的
  correspondence specificity，但不能继续把 prefix track 的几何外推当作通用 dynamic target；
- 明确拒绝的是“从 prefix tracker geometry 拟合确定性 kinematic field，并把它作为六源通用 object transition
  监督”的目标，不是否决 tracker 作为 correspondence/visibility measurement，也不否决 future-conditioned
  latent effect 或 object-level world model；
- 下一项仍是 G0 objective redesign：dynamic target 必须由成对的 current/future object observations 解释
  已发生的 semantic state change，tracker 只提供对应关系与可见性权重；不得再要求 prefix-only kinematic
  extrapolation 直接预测 future，也不得在新 target 通过 persistence、swap 和 held-future falsification 前启动
  Student、codec、latent-effect 或 Dynamics 长训。

### 15.25 V67 continuous predictive object field 预注册

**What changed**

1. 世界状态从固定数量的 slot、carrier 或 tracker component 改成连续可查询函数。给定只来自历史的锚点
   $q=(t_q,x_q,\sigma_q)$，模型输出
   $S_t^q(x,\sigma)=[\Pi_t^q(x,\sigma),A_t^q(x,\sigma),R_t^q(x,\sigma),V_t^q(x,\sigma),U_t^q(x,\sigma)]$。
   其中 $\Pi$ 是与锚点属于同一 persistent entity 的软概率，$A$ 是语义/appearance，$R$ 是可变化的
   response state，$V$ 是 visibility，$U$ 是 uncertainty。对象不再由 slot index、hard mask、patch index、
   Gaussian center/covariance 或 tracker 聚类定义。
2. dynamic target 从 prefix tracker geometry extrapolation 改成真实 current/future observations 之间已经发生的
   transition。连续 posterior $Q_\phi$ 解释 source field 与 target field 的差异，连续 operator
   $\mathcal T_\xi$ 用该 effect 预测 target field；tracker 只给 correspondence、visibility 和 reliability 权重，
   DINO/SigLIP 只给 semantic projection target，二者都不是 object truth。
3. 压缩标准从 RGB reconstruction 或固定 latent 容量改成 predictive rate-distortion：共享一个 object code/effect
   必须在未参与编码的坐标和未来时刻上，以更低 rate 达到不差于 separate encoding 的 distortion。训练和评测
   明确拆分 context coordinates 与 held-out coordinates，避免用模型自己的 support 证明自己是 object。

**Why**

V61--V66 已反复表明，固定 slot/carrier 的自洽 reconstruction 和 tracker prefix kinematic target 都不能可靠产生
跨来源的 dynamic object state；V67 只保留纯视频、连续表征和 object-level dynamics 目标，改用 future observation
可证伪的 predictive sufficiency 来定义对象和变化。

**Block contract**

1. `NativeContinuousSampler`：输入原生分辨率 RGB clip、任意归一化坐标与连续 scale，输出局部多尺度 RGB
   observations。坐标采样只是数值积分点，不是 object token；同一坐标的小扰动必须得到连续变化的 feature。
2. `ContinuousScaleFieldEncoder`：用共享高分辨率 local encoder 和历史 temporal mixer 将 observations 映射为
   $F_t(x,\sigma)$。small scale 保留局部细节，large scale 提供 context；输出不是 patch grid，也没有固定 object
   count。它的目标是预测 frozen DINO/SigLIP 的连续局部投影，并保留可被 future objective 使用的 response feature。
3. `QueryRelationField`：从历史 anchor descriptor 与任意 target descriptor 计算 $\Pi,A,R,V,U$。同一 track 的
   correspondence 是正 evidence，可靠的跨轨迹 negatives 是负 evidence；reflexivity、symmetry 和 soft
   transitivity 是函数约束。unknown/occluded 通过 uncertainty 与 visibility 表示，不被误写成 absent。
4. `PredictiveObjectCode`：只使用 context coordinates，将 query-conditioned field 压缩为连续 stochastic code
   $c_t^q$；KL 是 rate，held-out object observables 的预测误差是 distortion。code 不是 object ID，而是当前历史
   对这个 query object 的最小 predictive sufficient state。
5. `ContinuousEffectPosterior`：训练期读取 source code 与真实 target code，输出每个 query 的连续 stochastic
   effect $e_{t\to t'}^q$。它解释实际发生的变化，不拼接真实 center delta、RGB change、机器人 action，也不承担
   deployment-time effect selection。
6. `ObjectFieldOperator`：输入 source code、effect、$\Delta t$ 和任意 output coordinate/scale，预测 future
   $\Pi,A,R,V,U$。它采用 branch/trunk operator 结构，使参数不依赖采样分辨率，并同时训练 direct prediction 与
   two-step rollout consistency。
7. `PredictiveRateDistortionObjective`：在 held-out coordinates/times 上计算 semantic、relation、visibility、
   response 和 uncertainty-calibrated distortion；比较 shared-object 与 separate-object coding。若错误合并两个
   entity 不能提高 rate 或 distortion，$\Pi$ 就没有 object semantics。
8. `IndependentEvaluator`：只在未参与编码的坐标、future frames 和 query swaps 上评分；分别报告 absolute target
   error、persistence error、correct-vs-zero/shuffled effect、shared-vs-separate rate-distortion 和 coordinate/scale
   continuity。RGB 只作为可视化 probe，不进入晋级判据。

**Training contract**

- E0 `predictive_state`：训练连续 local field、query relation 和 stochastic predictive code；不训练
  action-free future regression，不训练 language/Prior，不继承历史 checkpoint。
- E1 `posterior_dynamics`：从通过 E0 held-out predictive sufficiency 的 checkpoint 初始化，训练
  future-conditioned effect posterior 与 object field operator。正确 effect 必须同时优于 zero effect、batch-shuffled
  effect 和 persistence；否则不能进入 Prior、语言或控制。
- 数据继续使用六源原生 RGB index；训练只读取过去/当前 frame，target encoder、future observations 和 tracker
  future correspondence 只进入 training target/posterior。deployment student 只需要 RGB history、query 和 scale。

**Falsification**

- Future swap 后 history encoder、source code 和任何 deployment path 的最大差异必须小于 $10^{-6}$；posterior
  与 target 必须变化。
- held-out semantic/visibility/relation absolute error 必须随训练下降，且不能只通过增大 uncertainty 改善 NLL；
  standardized residual 应接近单位尺度。
- 同一 query 的坐标/scale 小扰动应产生连续输出；交换 query 或打乱 track correspondence 必须显著恶化
  relation 与 held-future prediction。
- shared code 相比 separate code 只有在 rate 明显更低且 held-out distortion 不升高时才算形成 object；all-scene、
  all-same 和 one-coordinate-per-object 都必须被 rate-distortion 反事实拒绝。
- E1 在 motion-active held samples 上 correct posterior effect 相比 persistence、zero 和 shuffled effect 至少改善
  `10%`，并在六个来源至少 `5/6` 成立，才允许继续学习 History/Language Prior。

理论依据是 predictive state representation、predictive rate-distortion、conditional neural process 与 neural
operator；其共同点是用 future observables 定义 state、用 context/target split 验证压缩、并让函数映射独立于数值
采样网格。DINO/SigLIP alignment 仅提供 semantic observables，不能替代上述 object 与 dynamics 判据。

### 15.26 V67 实现落地与第一轮工程验证

**实现边界**

- 权威代码根为 `/Users/hela/Instruct-GS-World-recovered-20260725/`，交付分支为
  `codex/continuous-predictive-object-field-v67`。V67 不读取 V61--V66 的 model/optimizer checkpoint；六源
  RGB index、冻结视觉 teacher 和 CoTracker 权重是允许复用的数据资产。
- 架构固定为 `continuous_predictive_object_field_v1`，checkpoint version 为 `67`。实现没有固定 object
  count、hard instance mask、Gaussian carrier 或 patch-grid object state，也没有 RGB reconstruction、显式
  robot action、language 和 History Prior。
- 当前数值配置使用每段 `8` 帧、source/midpoint/goal 索引 `3/5/7`；先从 $16\times16=256$ 个候选连续坐标中
  选择 `32` 个 query anchor，并把其余坐标分成 context/held-out quadrature points。`32` 和 `256` 只控制一次
  Monte Carlo 估计的成本，不定义图像中必须存在多少对象，也不是部署状态的固定空间网格。

**逐 block 的实际输入、输出和唯一职责**

1. `continuous_field_sampling_v67.py` 接收原生 RGB $[B,T,3,H,W]$、像素有效区、归一化坐标
   $X\in[-1,1]^{B\times T\times P\times2}$ 和连续尺度 $\sigma\in\mathbb R^{B\times T\times P}$，在原始图像
   上采集多尺度 local crops。它只负责把任意坐标处的视觉邻域变成可微 observation，不进行 object grouping。
2. `continuous_scale_field_v67.py` 将每个 crop 编码成 `field_dim=256` 的 local feature，并用 `4` 层、`8` 头
   causal temporal mixer 聚合 source 以前的历史。它输出 $F_t(x,\sigma)\in\mathbb R^{256}$；目标是保留局部
   细节、尺度 context 和历史变化，同时严格不读 target frame。
3. `continuous_predictive_teacher_v67.py` 在训练期提取冻结 DINOv2-L `1024D`、SigLIP `768D` 和 CoTracker
   relation/visibility/reliability，再投影为两个 `256D` semantic observables。DINO/SigLIP 描述“这里看起来是
   什么”，CoTracker 只描述“哪些观测可对应以及是否可信”；它们都不直接生成 dynamic state。
4. `QueryRelationFieldNetworkV67` 对每个 query 与每个 quadrature point 计算 soft support
   $\Pi\in[0,1]^{B\times Q\times P}$、`192D` response、visibility logit 和 uncertainty。其职责是学习
   query-conditioned persistent entity relation，而不是给 point 分配永久 slot ID。
5. `PredictiveObjectCodeNetworkV67` 只在 context points 上按 $\Pi$ 加权汇聚 field，产生每个 query 的 Gaussian
   code $c\in\mathbb R^{320}$：前 `128D` 是 normalized identity，后 `192D` 是 dynamic state。mean/log-variance
   定义 stochastic code 与 KL rate；同时存在逐 point code 作为 separate-encoding rate/distortion 对照，它不被
   当作 object representation。
6. `PredictiveObjectFieldDecoderV67` 是 branch/trunk continuous decoder。branch 接收 query code
   $[B,Q,320]$，trunk 接收任意相对坐标、距离和 log-scale，输出该 query 在 $P$ 个位置上的 support、DINO/
   SigLIP semantic、response、visibility 和 uncertainty。它的职责是检验一个共享 code 能否解释未参与编码的
   空间，而不是还原整张 RGB 图。
7. `ContinuousEffectPosteriorV67` 在 E1 中读取 source/target object codes 及 $\Delta t$，经 query interaction
   Transformer 输出每个 query 的 `256D` stochastic effect。effect 只解释真实观测到的 object-field transition；
   它不是离散 codebook，也不包含 tracker flow、center delta 或 robot action。
8. `ObjectFieldOperatorV67` 把 source code、effect 和 log-time 送入 `4` 层、`8` 头 interaction operator，分别
   更新 identity、dynamic state 和 uncertainty，再调用同一个 continuous decoder 预测 future field。identity
   更新被限制为较小 residual，而 dynamic state 可以完整更新；同一个 operator 同时承担 short、direct 和
   rollout 路径，避免为每个 horizon 学独立捷径。
9. `predictive_rate_distortion_v67.py` 是唯一训练判据集合。E0 比较 held-out semantic/relation/visibility/
   response distortion、共享 object-code rate、逐 point rate、continuity、symmetry/transitivity 和 anti-collapse；
   E1 比较 correct posterior、zero、shuffled、persistence、short、direct 和 rollout。任何单项 reconstruction
   下降都不能替代 correct-effect intervention 和 held-out target error。
10. `v67_checkpointing.py`、`v67_training_loop.py` 和 `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/`
    下的 V67 entrypoints 负责 strict stage/checkpoint、EMA、DDP、W&B 和独立 evaluation。E1 必须从 E0
    checkpoint 初始化，且冻结 E0 state decoder；它只学习 transition，不允许重写 object semantics。

**已完成的验证层级**

1. 本地静态检查已通过：V67 Python compilation、Ruff、所有 V67 shell 的 `bash -n` 以及 `git diff --check`。
2. 服务器 synthetic CUDA forward/backward 已通过。E0 loss 为 `5.17696`，E1 loss 为 `11.14788`；E0/E1
   分别有 `126/82` 个 trainable gradient tensors。future swap 时 source path 最大差为 `0`，target 最大差为
   `0.31970`，synthetic effect intervention difference 为 `0.09154`。该结果证明 tensor、gradient、stage freeze
   和 causal wiring 可运行，不证明方法效果。
3. 六源真实 index 上的单 GPU E0 admission 已通过，实际样本包含 `518x640` 原生 RGB。loss 为 `5.46875`，
   finite gradient norm 为 `8.55056`；future swap 时 source path 最大差为 `0`，target path 最大差为
   `0.37771`。这证明 native-resolution data/teacher/student 路径和 causal boundary 可执行。
4. 完整训练入口真实执行 `1` step 并生成
   `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/v67_e0_code_validation_c65f5e5/v67_state_0000001.pt`；manifest
   记录 architecture/version/stage/world-size，checkpoint 大小为 `134766351` bytes。该 checkpoint 只用于验证
   save/load/E0-to-E1 contract，绝不能作为训练质量证据。
5. 使用上述 E0 checkpoint 的真实 E1 admission 已通过：loss 为 `13.32370`，finite gradient norm 为
   `55.45940`；source future-swap 差为 `0`，posterior 和 target 差分别为 `0.37996`、`0.24129`。这证明 E1
   确实读取 future 来解释 transition，而 deployment source path 没有 future leakage。
6. 正式四 GPU admission 尚未执行：测试时服务器只有物理 GPU 2 空闲，GPU 0/1/3 被其他任务占用。单 GPU
   admission 的 `world_size=1` 不能替代 DDP=4 结论；本轮也没有启动任何长训。

**当前判决与下一证据**

- 工程判决是 `ready_for_four_gpu_admission`，不是 `object_state_validated`。V67 已把错误问题从“能否从 prefix
  tracker 外推运动”改成“shared continuous object state 是否以更低 rate 预测 held-out future observables”。
- 下一步只能先运行正式四 GPU admission，再运行 E0 长训与 held-out evaluator。只有 E0 同时满足 absolute
  target error 下降、shared-vs-separate rate-distortion 优势、query/correspondence falsification 和 uncertainty
  calibration，才允许开始 E1 长训。
- E1 的核心判决仍是 motion-active held samples 上 correct posterior 相比 persistence、zero、shuffled 至少
  改善 `10%`，并覆盖六源至少 `5/6`。若失败，优先否决/修改 state 与 dynamic objective，不进入 Prior、语言、
  控制或更多容量调参。

### 15.27 V67 指标与 teacher contract 重新审查

此前文档把 `320D` Gaussian code 的前 `128D` 直接称为 `identity`、后 `192D` 称为 `dynamic`，并用
`object_code_std`、同 query 的 future cosine、teacher relation 均值等指标解释其语义。代码审查确认，这些名称和
指标只能描述实现切片、数值健康或模型内部一致性，不能证明 object identity、dynamic state、same-object relation
或真实视觉 visibility。`response` 同样只是 relation MLP feature，而现有 reconstruction target 是该 feature 自身的
stop-gradient 副本，并不是外部 dynamic supervision。此前所有依赖这种语义升级得到的结论均降级为待验证，不再
作为架构晋级依据。

从本节开始执行以下永久准则：

1. **名称不是语义。** 任意 latent partition 在获得可证伪定义、监督来源和独立 held evidence 前，只能按张量位置或
   计算来源命名；MLP 输出的一个切片不能因为变量名而成为 identity/dynamic/object state。
2. **proxy 不是 correctness。** variance、effective rank、reconstruction、同 query cosine、query-shuffle margin
   只能回答各自的数值问题；不得代替 object identity、motion semantics、visibility correctness 或 world-model
   capability。
3. **contract 必须先校准再入 loss。** CoTracker visibility、relay reliability、DINO/SigLIP validity、motion
   coherence 和 relation weighting 必须拆开报告，并与独立人工/外部标签对照；未经验证时只能标为 teacher proxy。
4. **先保留 case/query，再聚合。** 每个样本、query、候选点和 failure mode 必须可追溯；均值只能在完整分布、分位数
   和失败样本之后报告，不能用一个均值替代异质来源与局部对象分析。
5. **禁止循环论证。** 用 teacher 训练出的 student 与同一个 teacher 更一致，只能证明 teacher imitation；用模型
   自己的 support/mask 定义 object 再评估 object 不构成独立证据。

新增 `audit_continuous_predictive_contracts_v67.py`，在六源 held partition 上执行以下审计而不修改训练目标：

- 输出每个 case/query 的 code、response、relation、motion、visibility 和 reliability 分布，不再只保存均值；
- 对 `identity`、`dynamic` 和完整 code 分别做 held-task-group motion/position probe 与 fixed-query leakage probe；
- 将进入 code 的 `aggregated_response_192` 作为第四个匿名 partition 做同样 probe，并对 code 前 128/后 192 维执行
  zero/query-roll decoder intervention，判断每个切片实际控制哪些输出，而不是根据变量名推断；
- 将 shared/separate distortion、KL rate、effective point count 和 saving 保留到每个 query；现有 separate rate 是
  relation-weighted point KL 之和再除以最大 support weight，不是真实 bitrate，因此均值 saving 不再被解释为已经形成
  object compression；
- 把 raw CoTracker visibility、in-bounds、DINO valid、SigLIP valid、relay agreement 和最终 teacher visibility 分开；
- 对 motion sigma `0.02/0.04/0.08/0.16/0.32` 报告 relation 密度敏感性及对应像素尺度，不从 proxy 数据中反向挑选
  “最佳”超参数；
- 生成八帧可视化和 `human_review_manifest.jsonl`。只有补充独立 visibility/same-object 标签后，才计算 teacher
  confusion、Brier 和 AUROC；无标注时结果必须明确写为 `unverified_no_independent_annotations`；
- W&B 同步分布、分位数、held probes、case table 和 review images，但程序不输出 promote/reject 决策。

在该 audit 完成并解释 teacher/object contract 之前，暂停根据 V67 的 latent 命名、relation 均值或 visibility proxy
设计新 Dynamics，也不以调 loss weight、latent size 或训练步数作为修复手段。

### 15.28 2026-09-19 原始 CoTracker 可视化与抽帧对照

**What changed**

1. 当前最早未解决的问题仍是 G0 的 target/measurement validity。本轮独立查看 point tracking，而不是增加
   world-model loss：六源各 5 个 held clips，比较相同物理 anchor、query points、起止时刻下的连续帧与
   8 帧输入；请求间隔为 100/200/400 ms，实际间隔由各 source 原生 fps 决定。不将这 30 个探索样本作为
   总体准确率统计，也不声称已经覆盖所有小物体/遮挡/快速运动场景。
2. 原样保留 predictor 输出的轨迹和 visibility，独立记录 in-bounds、实际模型类、内部 resize 尺寸和数据
   replacement。颜色只表示 point ID，不是 object；query 帧是给定位置，不是预测成功。冻结现有训练代码、
   数据和权重，不加载 DINO、SigLIP、world-model checkpoint，不增加 relation 公式。
3. 新入口 `review_point_tracker_v67.py` / `review_point_tracker_v67.sh` 提供单 GPU 前台执行、逐 case 复用、
   原视频/轨迹视频/固定原图 crop/同时间戳对照、逐点 CSV、离线 gallery 的人工点选及 W&B 视频表/ZIP。
   原始 tracker 默认是 CoTracker3 offline，不是 CoTracker2。native/sampled 分歧不作为 GT error。

**Why**

必须先区分“没有采到目标”“时间抽样过稀”“点追错或 visibility 错”“点正确但 object grouping 错”，才能决定
哪些 teacher 证据可以进入后续学习目标；不能继续用 teacher 与 student 的一致性证明真实 object semantics。

**Impact**

继承六源 native RGB index、既有 decoder 和 CoTracker 权重；拒绝把旧 object-code/response 的命名与内部指标
当作正确性依据。当前结果状态为实现交付、待用户服务器执行，没有新的真实 tracking 结果或晋级判决。
下一证据是逐 case 人工观看及必要的独立点/visibility 标记，不是新一轮 Dynamics 长训。
完整操作和输出说明在
`/Users/hela/Instruct-GS-World-recovered-20260725/docs/POINT_TRACKER_VISUAL_REVIEW_V67.md`。

### 15.29 2026-09-20 十秒非腕部片段与变化区域密集采点

**What changed**

1. 用户观看上一批结果后反馈：点追踪看起来较准确，但八帧/400 ms 只覆盖约三秒，移动区域上的点不足。
   这是用户的逐例视觉反馈，不是已标注的总体 accuracy。新 review 固定至少十秒连续区间，抽帧数量由区间和
   fps 决定；30 Hz 时 native=301 帧、400 ms=26 帧。排除 wrist/未知相机与过短 episode，不重建原 index。
2. 全图 jitter queries 改为多时刻 motion-region masks 内的确定性密集采点。先估计 dense flow、扣除 robust
   partial-affine 相机运动、阈值分割连通区域，再按区域面积平方根分配采点；每段最多 1024 点，每次 tracker
   处理 256 点且看完整十秒。mask 与 raw query 坐标/时间公开展示，不将 connected component 宣称为 object。
3. 保留所有已选 case，包括空 mask；分别记录选样排除原因、相机依据、mask 阈值/面积、point query 帧和全部
   visibility/轨迹。HTML 和 W&B 增加 mask+采点视图与人工误选/漏选记录。多时刻 query 属于 offline review，
   使用了未来视频信息，不接入部署 student，不修改当前任何 loss、模型或训练采样。

**Why**

先区分短时间窗口与移动区域覆盖不足造成的观测缺口，再评价 tracker 的长时漂移、遮挡和重现，不能把缺少动态
证据直接解释为追踪能力或学习目标已经正确。

**Impact**

继承六源 index、原视频 decoder、原 CoTracker 权重和可视化入口；拒绝旧八帧时长与全图随机点作为本轮默认值。
Motion mask 不是 instance segmentation/GT，阴影、机械臂、视差、低纹理/静止小物体和前景主导相机拟合仍可能
误选或漏选。0.75 px、3 MAD、9 px 最小面积是显式记录的 proposal 参数，未获得语义校准。旧 cache 的预处理及
CoTracker 内部 resize 不会被本修改消除。本轮只完成本地代码与静态核对，未运行服务器推理，尚无新质量结论。
下一证据仍是用户生成样本后的逐 case/mask/point 复查；G0 未晋级，不启动 Dynamics 或长训。

### 15.30 2026-09-20 明显移动点与完整轨迹的独立显示

**What changed**

1. 用户反馈原 mask 区域大于实际运动区域，并要求只看明显移动的点且画出轨迹。代码中的 Farneback 局部窗口、
   前后两次残差取最大值以及 3x3 closing 都可能扩大变化响应；尚未逐例确认各因素的贡献。该 mask 仍只是采点
   proposal，不是运动物体的准确边界。本轮不通过提高 mask 阈值或强行收缩区域宣称解决 segmentation。
2. 新 CPU 入口读取已完成 review 的 `source.mp4` 与 `tracks.pt`，不重新解码训练原数据、不加载 CoTracker，
   不改变 query 或轨迹。每点排除自身 query 帧，只使用 tracker 判为 visible、in-bounds 且坐标有限的位置。
   至少 6 帧有效位置后，以 x/y 的 5%-95% 分位范围构成的包围盒对角线作为移动幅度，默认超过
   `max(12 px, 原图短边的 2%)` 才显示。这是显示阈值，不是已标定的 tracking/object 判据。
3. native 结果选出的同一组原始 point IDs 同时用于两个分支，避免分别选出更好看的样本。完整 raw tracks、
   CSV、mask 与 proposal 点保留；每点选择原因和阈值单独写入 `motion_filter.json`。原有 sampling consistency
   统计仍覆盖所有原始点，不能解释成已筛选子集的 accuracy。全部未入选时保留该 case，显示无点视频。
4. 原 0.25 秒 trail 在 400 ms 抽帧下几乎无法画线，改为显示截至当前帧的完整历史轨迹；可见性、越界或非有限
   坐标处断开，不插值跨遮挡。另保存最终帧上的完整轨迹 PNG。主视图只展示点与线，旧橙色 proposal 默认折叠，
   人工反馈增加 oversized-region/background-spill 选项。重绘使用有损归档 RGB 仅作背景，不用于计算新误差。

**Why**

先分离“宽 proposal 选入静止背景点”和“点确实移动但 tracker 漂移/相机整体运动”，同时让长时轨迹可以被人直接
检查。删掉显示上的静止点不等于确认真实物体边界，也不能推导出有效的训练目标。

**Impact**

复用已生成的十秒 review，输出独立目录并前台 CPU 重绘；原输入目录、训练代码、数据和权重均不变。图像空间移动
仍可能来自相机、机械臂、阴影或 tracker 漂移；缓慢微小运动可能被隐藏，须通过保存的逐点记录检查。新阈值未做
独立校准。本轮只做静态语法/接口核对，未在本地或服务器执行推理/重绘。下一证据是用户观看重绘结果并定位区域
过大样本，再决定 proposal 边界应如何改进；G0 仍未晋级。

### 15.31 2026-09-20 Grounded-SAM-2 区域采样与物体优先数据导出

**What changed**

1. 用户认可移动点+完整轨迹 baseline 的展示，要求保留；同时反馈少量阴影伪运动、机械臂与物体未分开、DROID
   远处小物体漏选。用户提出阈值筛选可能是漏点原因，决定不先单独定位，而直接实现 Grounded-SAM-2 方案。
   这些是当前样本的人工反馈，不是总体准确率结论。baseline `6866816` 保留，新流程用独立分支与输出目录。
2. 新流程继承至少十秒、非腕部、原生连续 RGB 和相同 held cases。在每两秒的 query frame 上，用冻结
   GroundingDINO 的 `robot arm / robot gripper / robot hand` boxes 提示 SAM2 得到机械臂上下文 mask；
   同时用 SAM2 全图+四个重叠原图 crop 产生 class-agnostic masks。SAM 点网格只是分割提示，真正的 CoTracker
   queries 在去重后的 mask 内部按区域分配，并作确定性 farthest-point 采样；不再先以 optical-flow 阈值淘汰区域。
3. 默认每段 2048 点预算：物体候选 80%、机械臂上下文 15%、unknown/scene 5%；不存在的上下文角色预算还给
   物体候选。先给每个候选区域轮流分配四点，再按面积平方根分配剩余量，每区域最多 96 点。容量不足明确记录；
   小 mask 在区域数量上限处优先保留。机械臂重叠区域标 unknown，不直接删除，不假定与夹爪接触的物体属于机械臂。
4. 原生 CoTracker 追踪后，按每点 5%-95% 位置范围和区域尺度筛选运动候选：阈值为
   `max(1.5 px, 0.08 * query区域包围盒对角线, 3 * 二阶差分尺度)`，至少六个非 query 有效帧。
   二阶差分尺度是轨迹变化统计，不是已校准的 tracking uncertainty；这些参数仍是可见、待人工复查的启发式。
   机械臂/unknown/scene 保留为上下文，不进入 `object_motion_target_mask`。所有原始点和逐点数值仍保存。
5. 主视频用绿色显示物体运动候选、橙色显示机械臂、紫色显示 unknown、蓝色显示 scene，保留完整轨迹和遮挡断线；
   另有 `all_queries.mp4` 展示包括静止/微动候选的全部点。每个分支导出 `training_candidates.pt`，其中 target、
   context 和 display IDs 使用同一份选择结果；`training_manifest.json` 索引全部分支，不能只保留好看的 case。

**Why**

目标是物体运动成为主要学习证据，机械臂保留为交互与遮挡上下文。把区域发现与运动强度分开，避免远处小物体在获得
任何查询点前就被整图位移阈值淘汰；保留 raw/all-query 对照，才能看清是 SAM 没分到、没有点预算、tracker 漂移，
还是最终筛选隐藏了真实微动。展示和导出采用相同点集，避免只修饰展示却继续用另一套数据训练。

**Impact / 当前证据边界**

- 已实现单卡前台 sampling/tracking/render/export 和中断复用、独立权重下载步骤、W&B 表及完整 ZIP。使用 HF 原生
  GroundingDINO/SAM2 接口，不重装环境，不改现有训练模型、loss 或训练入口。
- SAM region 是 query-frame proposal，region ID 只在该 anchor 有效；不是跨帧 object ID。CoTracker 传播点，
  本版没有传播 SAM mask。GroundingDINO 没检测到机械臂不等于机械臂不存在。SAM 也可能分出部件、阴影或漏掉小物体。
- 采样/运动选择使用完整片段，导出是 training-only teacher candidate，不能作为因果 student 的未来输入。
  输入按原尺寸解码并增加 crop，但 SAM/CoTracker 自身内部 resize 仍然存在。
- 此次静态审查已完成；尚未执行新版本服务器推理，未证明机械臂分离准确率或小物体召回提升。G0 不晋级，
  下一步只运行这批可视化与数据生成，人工检查 held-object 被误归机械臂、DROID 小物体、阴影和遮挡轨迹。
- 完整执行/下载说明在 `/Users/hela/Instruct-GS-World-recovered-20260725/docs/POINT_TRACKER_VISUAL_REVIEW_V67.md`
  的 Grounded-SAM-2 小节；旧 moving baseline 入口和产物保持不变。

### 15.32 2026-09-20 Grounded 采样人工复查：覆盖、角色与来源分开判断

**What changed**

1. 用户观看新结果后的逐源反馈如下。未提供具体 case ID/逐点标注，本轮未重新读取服务器视频，因此只记录用户观察，
   不把它们写成总体准确率或已定位的根因。

   | 数据源 | 用户观察 |
   |---|---|
   | AgiBot | 整体较好，但操作物体覆盖不完整；例如瓶盖有点、瓶身无点 |
   | DROID | 操作物体标记错误较多，物体与机械臂混淆；分辨率/数据质量为待确认假设 |
   | RoboMIND | 疑似使用剪辑版而非原版视频；开头无机械臂时，操作物体被误标为机械臂 |
   | Bridge | 与 DROID 类似的操作物体标记和角色混淆问题 |
   | HY | 多例操作物与机械臂混在一起 |

2. 本地代码核对发现，SAM 每个 prompt 只取最高分 mask，容量截断前又按面积从小到大排序；这不保证整个物体覆盖。
   瓶身缺点也可能发生在后续运动/visibility 筛选，未查看具体 raw/mask case 前不能归因于单一环节。
   机械臂 detector 的输出直接生成 robot_context；与其重叠的候选归 unknown，近重复候选被去重，二者均不进入
   object_motion_target。该规则会放大机械臂误检对操作物体目标的影响，尤其需要检查抓取接触区域。
3. RoboMIND 本地默认来源为 `/mnt/pfs/public/RoboMINDv2_LeRobot`，index builder 从 LeRobot episode metadata
   读取 camera、from_timestamp 和 length，review 再截取连续十秒。尚未确认用户看到的是源视频剪辑/拼接、转换问题、
   episode 边界问题，还是预期内的十秒截取；未猜测替换原始数据路径。后续按“来源连续性 -> 物体覆盖 -> 角色分离”
   顺序推进，保留现有 baseline 和原始未筛选轨迹；不先通过降低 DROID/Bridge 配比解释或掩盖共有的 teacher 问题。

**Why**

可跟踪的局部点不等于完整 object-level target；错误角色可能系统性排除真正被操作的物体，而剪辑/错误时间边界
不能作为连续物理变化监督。这三类问题应分别归因，不能合并成一个数据集平均质量或追踪成功指标。

**Impact**

本次只更新证据与优先级，不修改训练权重、采样参数、模型或数据路径。RoboMIND 修正需要具体 case 的源文件和帧范围；
角色修正应允许“机械臂未出现/证据不足”，物体覆盖修正不能通过简单膨胀 mask 或只增加瓶盖上的点数冒充完成。

### 15.33 2026-09-20 覆盖优先采点与跨时刻角色证据

**What changed**

1. 主修改是把单帧部件/角色判定改为保留多种支持范围、再沿轨迹积累角色证据。SAM 自动 point prompts 保留全部
   multimask alternatives；去除近重复后，按面积排序分三档轮换选 mask，同档优先尚未覆盖的区域。
   Farthest-point 采样同时避开该时刻其他 mask 已采的坐标，避免瓶盖和瓶身重叠部分反复消耗预算。
   不再减掉 robot mask 或因 robot 重叠提前拒绝 object proposal。保持既有质量/稳定性过滤、总点预算2048、
   原尺寸全图+crop、至少十秒非腕部片段和相同 held cases；不改变源配比、模型、loss、权重与 tracker。
2. Query-frame 的 robot mask 只是假设。追踪后，在多个 anchor 上取点对应的 robot-mask 证据；排除给定 query
   帧本身，至少两个其他可观测 anchor 落入 robot core、且占可观测 anchor 的比例不小于2/3，才标 robot_context。
   至少两个其他 anchor 在 mask 外且总重叠比例不超过1/4，作为 object candidate（原大场景 proposal 仍为 scene）；
   冲突或证据不足标 unknown。Core 默认距边界超过2 px。这些仍是未校准启发式，不是正确性概率，重复误检仍可能
   通过；“未被检测为 robot”也不是物体语义证明。逐点投票全部保存，unknown 运动另存
   `uncertain_motion_candidate_mask`，不当作 robot/background GT。运动阈值默认去掉区域面积项，保留
   `max(1.5 px, 3 * 二阶差分尺度)`，避免同一运动因 whole-object mask 比部件大而被隐藏；所有 raw 点仍导出。
3. 新导出契约为 `grounded_object_motion_teacher_v2`，主视频使用同一 resolved role/target/context 选择。
   每 case 添加 `source_review.json`（实际索引文件、episode 与片段文件帧范围、container 元数据）、整段 indexed
   episode 的24帧概览、`role_resolution.json`（逐点跨 anchor 证据）和 `resolved_queries.pt`。
   概览均匀抽样仅供检查源视频，不进入 tracker。RoboMIND 不猜测替换原路径；用户明确本轮只能交付代码，
   由用户访问服务器/原始数据并反馈，未访问远端。可按现有 CASE_MANIFEST 接口提供修正后的完整 case record。

**Why**

避免局部 SAM mask 与单帧机械臂误检过早丢掉操作物体证据，同时将“源片段是否连续”和“采点/角色是否正确”分开查看。

**Impact / 待验证结论**

- 保留 baseline `9c11b3e6398d50d308b0e917c5163db1da64cc73`；新分支、新输出目录，前台单 GPU运行，运行阶段无
  GitHub/模型下载依赖。复用完成 case/anchor/native tracks。W&B 新增整段来源概览与 uncertain-moving 计数。
- 本次只做本地静态审查，不跑本地模型或远端推理。未证明覆盖率/角色准确率提升，更未证明 RoboMIND 原始映射修复。
  新实现不做 source-specific 降权，不把轮换三个尺度档、2/3/1/4 比例或二阶差分解释为学习理论或语义真值。
- 可证伪检查：同 case 瓶身仍无 raw query 则 proposal/预算问题未解决；raw 有点、主视频消失则查 motion_filter；
  操作物体仍被持续标 robot 则跨时刻检测证据不足以分离角色；源概览有剪辑/错误边界则须先由用户确认原文件与映射。
- 下一项仍是 G0 逐例用户复查，不启动 world-model 长训，不晋级 object-level teacher。新的独立运行/下载命令见
  `/Users/hela/Instruct-GS-World-recovered-20260725/docs/POINT_TRACKER_VISUAL_REVIEW_V67.md` 最后一节。

### 15.34 2026-09-20 复用轨迹的运动 Top-K 候选筛选

**What changed**

1. 用户反馈整体追踪尚可，物体与机械臂粘连仍未解决，暂时保留；背景点增多且有错误移动，提出训练数据优先选择
   移动点前50%。本轮冻结 SAM proposals、queries、CoTracker 输出、角色判断、源配比和模型，不继续增加 robot
   分类规则。用户未找到来源概览，gallery 将其默认展开并增加 PNG/JSON 直接链接；不声称已确认 RoboMIND 原始数据。
2. 在每个连续 clip 内，仅对已经通过现有 visibility/in-bounds/有限坐标/运动门槛的 object_candidate 排名。
   排名分数仍是原生轨迹的非 query 有效位置 x/y 5%-95% 跨度的对角线长度，不引入新的语义评分。
   默认保留 ceil(候选数*0.5)，同分按原 point ID 升序；空候选保持为空。Robot、unknown、scene 不参与名额竞争，
   原 context 和 uncertain masks 不变。导出改为 `grounded_object_motion_teacher_v3`，同时保存 Top-K 前候选 mask、
   排名、cutoff、每区域筛选前后数量与最终 target mask；本版未接入任何旧训练 loss。
3. 新 CPU 前台入口读取上一版输出中的真实 tracks 与 teacher candidate export，重新筛选、导出和画图；不读取源数据、
   不运行 SAM/CoTracker、无需 GPU。归档 source.mp4 只作绘图背景，不用于计算运动量。新增物体候选筛选前/后视频，
   不叠加 context；原主视图仍展示最终 target+完整context，全点视频和原始数据不变。W&B 同步前后视频和数量。

**Why**

先降低低幅运动候选占用主要监督的比例，并直接观察被排除的点；避免为少量显示噪声重跑追踪或再改模型结构。

**Impact / 证据边界**

- 原输出不覆盖，新目录可同配置续做重筛选；既有完整 GPU采样入口也采用同一选择函数，默认 MOTION_TOP_FRACTION=0.5，
  设1.0可取消排名截断，仍保留原运动门槛。复用 baseline `36dc6bea17caac3cb253a08277f62c1df2221d95` 的轨迹。
- 50% 是用户提出的实验性采样预算，不是有效性的证明。相机运动、阴影或大幅漂移可能得高分；较慢/较远物体可能被
  排除。Top-K 不能证明剩下的点更准确，不能替代 visibility/role/GT 验证，也不能用更干净的展示宣称 G0 通过。
- 本次仅完成本地实现与静态检查，没有服务器访问或新效果数据。下一步由用户查看同 case 的 before/after 和逐点
  筛选记录，特别检查真实小幅物体运动是否丢失、伪运动是否反而占据名额；机械臂粘连仍为未解决问题。

### 15.35 2026-09-20 训练数据接入讨论：先修正视角与背景，再学习可测量的物体变化

**What changed**

1. 用户将运动候选保留比例调整为70%-80%，下一版方案取75%，并排除 DROID；HY/RoboMIND 背景误标和
   RoboMIND 腕部视角必须列为实际修正工作，不能仅登记风险。本轮核对代码发现：复用 CASE_MANIFEST 时不重新
   执行 camera_view；相机判断仅依据路径名称；SAM 面积小于阈值的 proposal 被初标为 object_candidate，跨 anchor
   判定主要排除 robot，不能证明其为物体；Top-K 使用未经相机补偿的像素跨度。尚未查看用户报告的具体源片段，
   不把这些代码缺口写成所有错误 case 的已确认根因。
2. 拟统一新采样、旧清单复用和训练 loader 的 source/view policy。按采集配置读取实际 LeRobot camera metadata，
   输出同 episode 多视角概览，记录经确认的外部视角映射；换相机时使用该相机的 episode 时间/文件映射，不简单
   替换路径字符串。未确认、只有腕部或存在剪辑跳变的片段不进入强 object-motion target，不用腕部作为缺失外部
   视角的替代。背景修正采用稀疏空间均衡参考轨迹估计背景主导运动，再结合跨时刻区域支持、局部轨迹一致性和
   独立重查询的漂移检查，区分相对背景运动与整幅画面移动；可用可靠运动点/邻近背景点作为 SAM 正负 prompts
   收紧混合 mask，再密集采点。运动一致性不能单独证明 objectness，单平面补偿不能解释所有视差；证据冲突的
   点保留为 unknown/context，不作为背景或不存在的真值。75% 在合格运动区域内部选取并保留空间覆盖，避免全图
   排名让大幅运动区域挤掉小物体；同时保留 raw、背景参考、被排除点及原始/补偿坐标，供逐例复查。
3. 模型主线保持单 RGB-only history student、连续区域聚合和 compact object-level latent-effect Dynamics。
   当前 train_continuous_predictive_object_field_v67.py 仍实例化旧在线 teacher；新的 teacher_v3 导出尚未接入。
   当前 _operator 在真实未来 track_coordinates 上调用 decoder，因此该路径不能证明预测了未来空间位置。
   拟新增源坐标/区域局部坐标到未来坐标、support 与可观测性的读出，由共享 object latent 驱动；未来坐标仅作
   teacher target，不作为 decoder 查询输入。对应点位置误差是主要可解释监督，DINO/SigLIP 对应区域特征为辅助，
   不把任意投影的 cosine、std 或独立点跟踪成功改名为 object identity/物理状态成功。静态区域保留为 state/context，
   动态权重用于 transition，不将对象定义为始终移动。部署输入中的区域/采点仅依赖已观察内容；全视频75%筛选
   只影响训练目标权重，不能反过来决定 student 看见的区域、密度或历史长度。

**Why**

下一步需要直接学习和评估物体如何改变，而不是继续用背景主导的弱标签、未来位置提示或自洽特征距离替代这个目标。

**Impact / 执行方案与证据边界**

- 本节是讨论方案和用户反馈记录，尚未修改或推送可执行代码；当前脚本默认仍为50%，既有 index 仍含 DROID。
  未访问远端、未读取 RoboMIND 原始数据、未运行模型或证明背景/视角问题已经解决。
- 数据生成采用独立离线 teacher 作业，先产出可复查的固定版本 shard，再扩展；不在每个训练 step 在线运行
  SAM/CoTracker，也不交替用尚未验证的 student 重标 teacher。剩余 shard 可与训练并行生成，但每次训练固定
  manifest，新增数据显式开启后续数据版本，strict resume 不偷偷改变样本集合。
- 缓存原生连续至少十秒的轨迹、query 时间、mask/role 证据、实际时间戳和视频映射；原图继续按需读取，
  DINO/SigLIP 可沿用冻结 JIT 编码。Track ID 只代表表面点 correspondence，SAM region ID 只代表局部 proposal，
  两者不直接等同于持久 object ID；tracker visibility 仍是弱观测，不等同于存在性真值。
- 保留当前被用户认可的 baseline。先用同 case 的原始轨迹、背景补偿、最终目标和多视角映射区分误差来源，
  同时检查小/远物体的点没有被清掉；按源记录误保留背景与误删真实运动。最早未完成项仍为 G0 teacher 有效性，
  不从“展示更干净”或结构检查通过直接跳到 object-level 能力结论。通过后再接入共享 teacher reader，先学习
  state/correspondence，再用 posterior-conditioned transport 学习变化；固定时间差、正确/zero/shuffled effect、
  held 逐例位置误差和区域删除干预分别检验预测精度、effect 使用与对象独立性。

### 15.36 2026-09-20 V68：数据和模型同步实现，先交付造数据指令

**What changed**

用户明确“同步改，我只是告诉你先给我什么指令”：本轮不是只做可视化，也不是先启动训练。
分支 `codex/grounded-motion-data-v68` 实现离线 teacher、共享 reader、RGB-only student、posterior transport 和独立前台 launcher。
原 V67 baseline、源 index 和旧输出不修改。详细模块、目标、限制与独立执行命令见
`/Users/hela/Instruct-GS-World-recovered-20260725/docs/GROUNDED_MOTION_DATA_AND_TRANSPORT_V68.md`。

1. 数据统一排除 DROID；native 至少10秒；默认512 pilot、2048总点预算、每次256点，采用75%区域内保留。
   新采样/旧 CASE_MANIFEST/reader 共用 source-camera policy。RoboMIND 腕部映射按真实 LeRobot metadata 换到同 episode
   外部 camera；无对应外部文件的记录排除，不沿用旧视角。整段 episode、多相机截图和原始路径/offset 默认显示。
   HY 默认使用 cam_high；实际 camera key 是否错误命名、视频是否已经剪辑，仍要看用户运行的 case，不能声称自动修复。
2. 独立空间参考点拟合背景主导 homography，保存原/补偿轨迹和逐帧拟合支持；pilot 运动正提示与背景负提示重新生成 SAM
   支持，再密集采点。另一时刻重新查询检验轨迹漂移；角色/运动/mask/重查询条件合格后，每区域空间格轮换保留75%。
   导出 `grounded_object_motion_teacher_v4`。最终 `target_valid` 同时用于训练监督和主可视化，raw/context 全保留。
   homography 是二维主导运动模型，SAM/role 是 pseudo label，relay 是同一 tracker 的一致性证据，均不等于独立 GT。
3. 独立 GPU worker 离线造 shard，按 case 和中间阶段保存，重复同配置命令复用；全量 manifest 只在完成时集中写入，
   中途进度保存在各 shard 的 progress.json。W&B 同步案例表和产物；全量表为明确标记的前64例/源/worker展示，
   不是随机 held 质量估计。原图按需读取，构建期和训练期都不重建 RGB cache。
4. Student 单路径输入1–4张历史 native RGB，stride-4 CNN 保留空间场，18个 recurrent slots 输出 `[B,18,256]`：
   16个 object hypotheses、robot/scene 各1个。不把 teacher 的未来密集查询当作 encoder 输入，也不把名字当作物理语义证明。
   冻结原生 tile 的 DINO1024/SigLIP768 局部特征作 appearance 辅助；本版不调用旧 arbitrary fixed-group feature 压缩。
   State 以 track ownership correspondence、角色证据、区域 binding/不同运动分离和 appearance 训练。弱分组证据可能有误，
   同区域并不被提升为真实持久 object ID；没有预训练成功或 capacity 恢复的实验结论。
5. Dynamics 从 V68 state checkpoint 显式初始化并冻结 encoder/target，用训练期 posterior 得到每 component32维连续 effect。
   共享18-token Transformer 产生变化，由当前坐标/当前 ownership 读出未来坐标。真实未来点只进 loss，不进 transport query。
   固定1秒、3秒目标和 direct/rollout alignment；3秒不是 episode 终局。Zero effect 持续当前状态是结构约束，不是学习成绩。
   主 loss 直接比较轨迹位置，辅以 path、shuffled intervention、KL、tracker observability；没有显式 action、Prior、语言、RGB loss。
   W&B 保存 native-pixel EPE 分布与逐案例误差，名称明确为相对 tracker 目标而不是人工准确率。
6. 发布到独立服务器目录 `/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/grounded_motion_v68/releases/`。
   数据四卡、训练八卡均前台运行，无 GitHub/模型下载依赖。state/dynamics分开；checkpoint v68 保存完整模型/EMA、优化器、
   scheduler、rank RNG、数据 manifest、epoch/cursor、W&B ID；只在显式 RESUME 时恢复，不根据输出目录自动猜测。

**Why**

用户认可的是逐点追踪观感，但指出目标区域与视角仍错。先明确生产出的实际监督，并把模型训练绑定到同一导出；
同时替换旧“在真实未来位置解码”的测量路径，让新模型必须从当前位置预测对应点去了哪里。
这解决接口与预测任务错位，不等于已经解决 objectness、机械臂粘连、背景视差、阴影或 tracker 系统性误差。

**Impact / 已验证与未验证**

- 已实现并静态审查模块调用、shape、mask、AMP 使用、DDP accumulation、断点恢复与离线运行接口。
  本地未运行任何模型/训练；服务器未在本轮执行，RoboMIND 原始文件未访问。
- 先执行每源8个 episode 的 train-partition 数据检查批，查看实际 target、原始/context 点、mask refinement 与相机映射。
  这不是 held 能力测试。确认数据后再扩大固定离线版本，state 学好后再启动 dynamics，不自动提交两阶段训练。
- 首批目标仍为 G0 teacher 有效性。新工程默认阈值不是经标注校准的结论；camera-key whitelist 也不是视觉内容证明。
  不为“脚本完成”晋级主线，后续记录必须逐例区分误保留背景、误删小运动、机械臂接触、未知可见性以及真实物体范围。

### 15.37 2026-09-20 用户调整首批规模：400例全部可视化，核验后训练

- 15.36 的40例执行安排被用户明确替换：五源各80个 episode，各一个连续10秒窗口，总目标400例，全部可视化。
  DROID 继续排除，75%区域内采样保持不变，模型同步实现保留，但不启动训练。
- Python/shell 默认 cases_per_source 改为80；外部视角/时长不合格的 episode 不计入名额，从同源后续候选补足。
  不按 tracker 成功与否补样；零目标和错误追踪案例必须保留展示。数据不足时记录实际 per-source 数量，不复制样本凑400。
- 新目录采用 grounded_motion_v68_review400 前缀；完整独立命令和下载方式已同步到
  `/Users/hela/Instruct-GS-World-recovered-20260725/docs/GROUNDED_MOTION_DATA_AND_TRANSPORT_V68.md`。
  本轮仍只有代码/静态检查证据，400例尚未由用户在服务器生成，尚无质量结果。

### 15.38 2026-09-21 离线生产20,000条，固定400例展示，八卡每卡双 worker

**What changed**

1. 用户将生产规模扩到20,000条，仍只展示400例：五源各4,000个 train episode，各取一个原生连续10秒窗口；
   每源在确定性、按group轮换的全局 case plan 中预选前80条展示，再分配给 worker。展示集不依赖追踪成功与否。
   DROID排除、外部相机策略、75%区域内目标筛选、teacher字段和模型结构保持不变，不把扩量视为监督质量改进。
2. `DATA_GPUS=8`、`WORKERS_PER_GPU=2` 启动16个独立数据进程，逻辑设备为 local_rank整除2。
   每个进程独立模型、缓存和 shard，让CPU解码/写入与同卡另一进程推理重叠；保留平台可见设备映射，不启动16卡DDP。
   非展示样本不渲染媒体；gallery、W&B案例表和zip只展示预选400例，完整训练manifest包含全部完成片段。
3. 使用新的 `grounded_motion_v68_train20k_review400_<revision>_8g2w` 目录；不覆盖或停止旧单卡400例任务。
   相同revision/配置/worker拓扑重跑复用完成case和中间阶段；更换拓扑使用新目录。全部前台，不自动启动模型训练。

**Why**

先离线产数据，避免每轮训练重复昂贵teacher计算；将可视化开销限制在固定检查集，利用同卡并发覆盖部分解码等待。

**Impact / 证据与下一步**

当前只完成实现和静态检查，尚无新服务器吞吐、双worker峰值显存或20,000条完成证据。并发复制模型会增加显存和主机内存，
不承诺线性提速。现有格式保留中间缓存，301帧/2048点下主要tensor约41MB/clip，20,000条约830GB，另计mask/JSON/展示媒体。
源不足时报告实际数量，不复制凑数；400条是确定性检查样本，不是随机总体质量估计。最早未完成项仍为G0 teacher有效性：
检查HY/RoboMIND背景误标、实际外部相机、小物体覆盖与robot/object接触边界，通过后再固定数据版本用于训练。

### 15.39 2026-09-21 媒体失败隔离与已完成样本的独立展示

**What changed**

1. 用户上传的旧单卡400条日志显示：第83条已保存teacher，第84条 `agibot_ep124302_f402` 在
   `agibot_world_beta_lerobot_3.0/task_666/videos/observation.images.head/chunk-000/file-000.mp4`
   遇到libdav1d OBU解析失败，worker退出。日志支持“这一读取请求被解码器拒绝”，不能单独证明整文件损坏或编码库缺陷。
   原入口未完成最终合并/打包，不能将其写成400条数据已经完成。
2. 离线构建改为显式记录已有解码器识别的媒体失败，并从同源预排reserve补位；原分片、数据源/相机策略、采样和teacher保持不变。
   每个失败case保留路径/帧范围/错误，失败文件在worker内隔离；reserve按worker分开，默认每源额外25%且至少32个episode。
   只因无法读到要求帧而补位，不因追踪质量、无motion target或模型loss补位；reserve不足报告incomplete，不造数据凑配额。
   episode/多相机概览失败只标记为不可用，不用虚构图片作为teacher，也不丢掉可正常读取的10秒片段。
3. 增加读取原 `workers.json` 的显式跨修订续跑入口，保持原采样和worker拓扑，仅授权复用teacher配置不变的旧结果。
   增加不加载模型的partial打包入口，从原子complete记录收集已有案例，独立写partial HTML/manifest/zip，
   不修改正在构建的进度或完整训练索引。8卡每卡双worker的20,000条生产同样适用。

**Why**

已有多源视频会遇到局部解码失败，不能让一次媒体失败抹掉所有已完成工作的访问入口，也不能让补样隐藏失败或被误当作质量提升。

**Impact**

本轮是代码实现和静态检查，没有重跑服务器或确认完整400条。旧83条的实际可复用数量由complete记录决定；
partial展示不代表完整数据集，失败率与补位必须同质量检查一起阅读。G0仍未通过，不启动模型训练。
用户最终明确：有多少已完成就打包下载多少，不继续凑满400条；20,000条共用入口同步修复，但不在本轮提交其生成任务。

### 15.40 2026-09-21 所有点全局75%与真实训练样本展示

**What changed**

1. 用户指出之前绿点不能可靠区分夹爪与物体；原意是所有点前75%，不是物体候选或每区域75%。
   新契约`all_point_motion_teacher_v5`将所有类别的可测量轨迹按背景补偿后span全局排名，保留ceil(0.75*N)。
   不使用role/refined-mask/固定运动门槛/区域配额；无法获得两个有效位置的点单列。逐帧仍保留背景可用/relay有效性，
   不将不可观测位置制造成GT。机械臂、夹爪、背景与物体都可能入选，颜色不再表示已识别的物体。
2. 新reader只从该75%池抽取当前可见的至多256点；role与SAM region标签置unknown，关闭新数据的角色CE及同mask binding。
   RGB-only路径、1–4历史帧、1秒/3秒请求、appearance辅助、track correspondence和transport结构保留。
   新构建的pilot预算与refinement也不再按物体/机械臂类别限制。旧轨迹重筛无法补回原来未采到的表面点。
3. CPU重筛入口只快照已完成案例，复用原轨迹/背景/relay，写独立新teacher和manifest，不继续400构建，不自动启动20,000或训练。
   展示直接使用训练Dataset的相同index/seed/epoch：原始输入帧、当前查询、+1秒/+3秒坐标及有效mask逐点导出。
   另列全clip入选池视频，避免“导出的75%候选”与“单次训练256点”混淆；新20,000合并仍只展示预选400例。

**Why / Impact**

未经验证的robot/object分类不能决定主要监督或反过来充当类别真值；用户需要检查实际训练接收的数据，而非另一套展示采样。
旧v4teacher、原生RGB、tracker和背景配准资产保留；不将重筛解释为追踪变准、成功区分物体或学到了object semantics。
最早未完成项仍是G0 teacher有效性，下一步只查看已完成案例的新排名和实际训练样本，特别记录夹爪/背景占比、小物体漏采和无效目标。
本轮完成实现、静态检查并提供一个CPU回归入口；未本地运行模型/数据测试，也未在服务器运行或得到用户的新质量结论。

### 15.41 2026-09-22 跳过坏视频并继承旧生产任务的完整轨迹

**What changed**

1. 用户新日志确认20k生产任务实际运行的是旧release `6cd5c380377b17a2490cddd8eac13eabb3411a30`，
   不是15.39后的解码跳过版本。worker14进入AgiBot `task_666` 的head视频时被libdav1d拒绝，torchrun结束其余worker；
   当时仍有worker处理RoboTwin，日志不能证明RoboTwin全部完成或所有其他来源均未完成。用户明确坏视频直接跳过，
   不修复视频、不转码、不重建索引；仅从未使用的同源候选补位，继续使用原来源/外部相机策略和展示配额。
2. 新增隔离恢复路径：读取旧workers/cases和已完成teacher，在独立输出目录复用完整raw轨迹、背景与relay，按当前
   所有点全局75%重筛；不重跑这些案例的SAM/CoTracker。未完成案例使用当前全类别采点。记录每个case的原轨迹revision
   和parent_teacher，原目录不修改。旧轨迹缺失的采样覆盖不能靠重筛补回，因此不把二者伪称为相同raw采样过程。
3. 每worker改为来源轮换处理，保持原case集合、配额和预选review IDs；progress同时记录每源计划、完成、跳过和继承量。
   这消除按来源串行造成的早期展示偏向，不改变最终训练采样权重。再次执行同一恢复命令复用新目录已完成结果。

**Why / Impact**

旧任务应继续积累可读取的数据，不应因单个坏媒体终止，也不应为更新目标筛选重复昂贵追踪。
本轮保留RGB、teacher原轨迹和已选片段；不继承旧角色筛选标签，不更改模型或宣布G0通过。仍需用户检查新的实际训练样本，
并以最终manifest的实际完成/跳过/来源计数判断生产进度，不能将20k目录名称当完成数量。本轮仅静态检查，未重新运行服务器生产。

### 15.42 2026-09-22 RoboTwin只重处理已有轨迹，其余来源继续生成

**What changed**

用户确认RoboTwin2已完成，授权重处理但禁止重新追踪。续跑新增 `REPROCESS_ONLY_SOURCES=robotwin`：
复用旧完整teacher，包括teacher已原子写入但展示尚未完成的case；只重筛75%及重新展示，不进入SAM/CoTracker。
若没有旧轨迹则记录 `reprocess_only_skip`，不补采RoboTwin、不阻断其他来源。AgiBot、RoboMIND、Bridge、HY继续原计划，
坏视频直接跳过并同源补位。来源限制随workers配置保存，重复续跑仍生效；兼容已在新恢复目录生成的结果。

**Why / Impact**

已有轨迹能满足重筛所需信息，无需再次支付采点与追踪开销。原输出目录和raw轨迹不修改，其他来源、teacher筛选及模型不变；
缺失旧轨迹的实际数量单列，不用新RoboTwin补齐或把缺失写成完成。最早未完成项仍为G0；本轮只有静态检查，没有服务器新结果。

### 15.43 2026-09-27 预训练视觉输入与3s/5s object sequence计划

**What changed**

用户明确history最长3秒、future最长5秒，借鉴video-gen的连续时间段、多帧条件与序列目标，不从头训练视觉基础encoder。
审阅V68发现：视觉先验只作为teacher，Student仍用浅层CNN；固定18个global slots与query-conditioned主线不一致；
角色/region监督关闭后归属JS仍有uniform解；future只有两个端点；当前query仍依赖full-video tracker，visibility混合unknown。
计划用冻结DINOv3 ViT-L/16作为主要视觉输入，V-JEPA2.1-L作同量级替换对照，不叠加SigLIP；恢复current-query条件的
Object Memory和object-token序列Dynamics。默认参考采样16帧history覆盖3秒、25个future时刻覆盖5秒，原生轨迹不降采样。
新/未来episode按视频均匀采样，motion-richness先用原视频与逐例材料验证再启用；旧配额数据不伪称全库均匀。

**Why / Impact**

应复用预训练局部视觉能力，并用连续未来监督实际变化，而不是把更大的随机ViT接到未证实的slot目标上。
保持原RGB、10秒raw轨迹、RoboTwin不重新追踪和坏视频跳过；这次只更新设计，不更改运行中的生产或提交训练。
最早未完成项仍为G0。encoder适配、query绑定和独立object有效性先取得证据；不由公开benchmark或loss下降直接晋级。
详细问题依据、模块输入输出、loss、证据包与执行顺序见 `docs/OBJECT_VIDEO_SEQUENCE_PLAN.md`。

### 15.44 2026-09-28 V69完整Object Video Sequence实现

**What changed**

1. 主链路改为本地预训练且冻结的DINOv3-L原生分辨率输入，V-JEPA2.1-L作prefix-causal受控对照。
   默认3秒16帧history、5秒25帧future；历史visual queries不读tracker或未来筛选。每query anchor+8 carriers、宽512，
   共享27.09M Object Memory和2.12M readout；训练期12.69M Posterior提取每query4x64连续effect，51.28M Dynamics展开未来。
   Anchor固定只是输入条件，不当identity成绩；support centers不自称物体GT。State/Dynamics分阶段，不新增action-free主线。
2. V68已完成轨迹只读复用，按episode划分held与均匀采样；保留原始位置、75% transport池、unknown观测和独立pixel mask。
   背景排名失效不抹除全部原始轨迹，均匀75%替代明确标记。主transport改为原生pixel距离，relative transport与latent为辅助，
   shuffled/zero只作干预对照，不给相似motion制造负类。无RGB主loss、显式robot action、语言或部署effect selector。
3. 完整实现单卡两阶段训练/strict resume/未中断对照/evaluator视频；held逐点逐时间绝对误差、独立人工点binding与删除干预、
   uniform/merge-all/track-per-object评测反例、motion窗口盲评和relay/visibility校准、两encoder冻结对照。
   Runtime只加载本地release/权重，前台执行，W&B在线；下载与Git同步独立。Checkpoint保存backbone/EMA/优化器/时间采样/RNG/W&B ID。

**Why**

复用视觉预训练能力并把连续object变化放回可测量的序列位置目标，同时移除future驱动query及未经证实的角色标签。

**Impact / 冻结控制与下一项**

继承原RGB、已有raw tracker、relay、V68来源映射；不继承旧CNN、18-slot optimizer或任一旧模型checkpoint。
新/旧数据目录隔离，不重新追踪RoboTwin，不改正在运行的数据构建。旧V68代码仅共享decoder新增可选PTS返回，旧调用返回值保持不变。
当前结果是静态实现与审查，尚无GPU通过、显存、学习曲线或held结论。用户尚未确认DINOv3官方源码及权重在服务器的位置。
最早未完成项仍G0；visual affinity不升级成object GT，tracker自评不升级成真实visibility，反例评测也不等于训练loss已经排除退化解。
下一步用户单卡跑完整工程测试并查看逐例数据证据；确认后八卡State重训，独立state证据通过后才启动Dynamics。
完整输入输出、参数量、loss语义和所有前台独立命令见 `docs/OBJECT_VIDEO_SEQUENCE_V69_RUNBOOK.md`。

### 15.45 2026-09-28 V69运行复现契约与失败证据

**What changed**

1. 用户反馈`ca2db53`单卡测试到Dynamics step10，`posterior.queries`未满足resume/未中断比较；日志无数值差异幅度。
   这证明两条Dynamics路径完成了有限步运行，不证明恢复正确，更不证明object学习成功。按脚本顺序State比较已越过，完整测试失败。
2. 数值比较改为显式deterministic执行：固定cuBLAS workspace、确定性算子、任务attention显式QKV。
   模型/数据/loss不改，浮点容差不放宽；生产fast模式与确定性复现模式明确区分并记录到checkpoint。
3. 测试保存逐microbatch数据、RNG、posterior noise、source与梯度；同时比较optimizer/scheduler/cursor/final RNG。
   差异报告在断言前写盘并上传W&B。增加完整脚本重跑入口，测试机同步发布，排队八卡任务仍只读取共享release。

**Why / Impact**

恢复同一随机数状态不自动保证CUDA数值确定性，旧断言无法定位数据、噪声与数值计算的首次分叉。
尚未读取服务器旧checkpoint，也未运行修订测试，因此不能把非确定性提前认定为唯一根因。
G0及teacher语义状态不变，本次不更改架构或实验学习目标；下一步仍是用户完整单卡反馈，而不是放宽条件宣布通过。

### 15.46 2026-09-28 独立数据迁移契约

**What changed**

1. 新增V69 portable exporter，按manifest固定快照复制原始媒体和完整teacher，去重同一路径媒体，保留frame offsets、时间、分辨率、
   raw轨迹、relay、Top-75%及train/held划分；不转码、不重新追踪，不更改原数据。缺失文件案例明确记录并跳过。
2. 包内`root="."`、teacher及操作用media路径相对manifest；V69训练/测试/评测统一解析相对路径，旧绝对路径输入保持兼容。
   训练run复制manifest时先解析root，避免把相对路径错误地解释为训练输出目录。代码/权重/环境/checkpoint不属于数据包。
3. 文件级可续跑导出、实际字节数与缺失报告、前台进度和tar归档；原打包MP4整文件复制可能带上未使用episode，体积如实报告。

**Why / Impact**

迁移后的数据不应依赖原始多源数据挂载，也不能通过再次压缩或重追踪静默改变学习目标。
本次是数据I/O契约改造，未在服务器生成实际包，未运行模型或改变G0判断；原八卡入口与默认数据路径不变。
执行与迁移路径见`OBJECT_VIDEO_SEQUENCE_V69_RUNBOOK.md`第14节，实际可迁移规模以export_report为准，不用20k目录名称替代结果。

### 15.47 2026-09-29 V69r2两层关联审阅与修正

**What changed**

1. 用户的结构反例成立：相同anchor下，加anchor后展平的effect集合仍对归属交换不敏感。撤回此前“加anchor即可保证绑定”表述。
   每个block先query-local effect conditioning，再跨query interaction；不把query index提升为物体ID，也不禁止交互后的跨物体影响。
2. Posterior除tokens/time外加入估计root相对source的位移与帧内carrier布局，避免仅local坐标丢掉translation。
   保留`POSTERIOR_GEOMETRY=on/off`同State起点的训练对照。Grouping先合并人工entity标签下的query概率，unbound不作共同物体。
3. 新增同物体多query共同位置读出、region/additional测点、单query替换和双effect交换、per-query/clip/entity KL及采样诊断。
   真实object指标依赖独立标签，无标签明确未测；不强制z相等、同物体各点同位移或zero vector静止。原KL均值loss尺度保留，KL计算改FP32。

**Why / 控制与证据**

要分开query–object关联与effect–query关联，不能由appearance hint代替后者的接口保证，也不能让评测偏好单query集中。
冻结encoder、native分辨率、3s/5s、latent4x64、State学习目标及数据保持不变；新增Posterior geometry为18,944参数。
本次已实现但未运行GPU测试，无学习效果结果。第一层真实object关联仍是未解决的科学问题，只补独立诊断而不制造新pseudo-instance GT。
当前State阶段不训练Posterior/Dynamics，因此不能把这两项修改宣称为State语义已经提升。

**运行与下一步**

架构名升级v2但继续使用V69脚本；旧checkpoint按原release恢复，不隐式混入新架构。未启动的八卡任务无需改命令，
测试机发布完整release后原子更新共享`DEPLOYED_REVISION`。单卡整链路增加同anchor、tuple重排、geometry-only和multi-query数学反例，
不放宽原resume比较。下一证据是服务器完整测试与独立少量多query标注评测。详见`OBJECT_ASSOCIATION_REVIEW_V69R2.md`。

### 15.48 2026-10-01 第一阶段State是否保留物体变化

**What changed / hypothesis**

第一阶段的主问题是已观测视频压缩成State后，能否还原物体变化。新增`evaluate_object_state_change_v69.sh`，
冻结同一checkpoint、视觉backbone、reference point、query归属和readout，只干预continuation State。
比较observed、同时冻结tokens/centers、打乱continuation时间、仅冻结tokens、仅冻结centers及reference copy。
位置绝对误差与相对t=0的位移误差分别报告；按真实位移区间、时间和来源分组，先逐案例统计再汇总分位数。

**Execution / inherited assets**

默认各源最多80个独立held episode、每episode一个固定seed窗口，每源8个视频；最多400 clips与40组可视化。
单卡/四卡独立前台入口，评测开始复制正在更新的latest.pt到本次独立OUT，所有rank使用同一快照。
严格加载encoder/target_encoder/readout及冻结backbone；支持旧v1 State，不继承其optimizer或使用新版Posterior/Dynamics。
使用既有原分辨率RGB、3s/5s、V68轨迹与原75%池，不重新追踪。W&B记录逐案例、成对干预、各源分布、视频与完整证据artifact。

**Evidence / falsification / next Gate**

当前W&B训练run `0oko1yhr`为v1 `40e4586`；读取到6020步。4500–5000至5500–6020的transport均值5.58px降至4.83px，
外观0.331升至0.354、对应JS约0.0064升至0.0069；这些只是训练拟合，不能证明object validity或泛化。
若不更新State或打乱State仍可同样还原明显运动，则“State承载变化”假设不成立。
tokens/centers干预定位变化的存储位置，不要求只有tokens承担变化，也不把分位数或熵提升为语义分数。
tracker仍为伪测量，raw displacement可能含相机运动；独立object grouping/删除局部性仅在人工/模拟器标签提供时测量，缺失明确未测。
本次新增数学回归与静态审查，真实held结果待用户执行；G0 teacher/object独立有效性仍未通过，不由重建成绩自动晋级Dynamics。

### 15.49 2026-10-02 State变化held结果：几何可用，object语义未测

**Evidence / execution**

从W&B run `gaiahhsw`实际读取完整report artifact、12个case Tables（57,600行）、paired Table（4,000行）及400行case inventory，
不是训练曲线或summary截图。评测revision `aed14bb57e71932965f0bf5d9e55b985cb385882`；checkpoint为V69 v1 `40e4586`、
State step 6000；单卡前台完成，五源各80个held episode、400 clips、decode failure为0。独立annotation未提供。
有386 clips提供有效t=0与continuation配对测量，motion-active门槛5px；共557,523个运动point/frame测量。
运行地址：https://wandb.ai/healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world/runs/gaiahhsw

**Results / interpretation**

下表是各clip先计算motion-active平均误差，再取跨clip的p50/p90；normalized列先逐测量除以真实位移，再在clip内平均，最后取跨clipp50。

| Condition | Change error p50 / p90 (px) | Normalized change error p50 |
| --- | --- | --- |
| observed State | 8.825 / 29.261 | 0.300 |
| frozen State | 44.231 / 111.132 | 1.000 |
| shuffled State | 36.835 / 100.986 | 1.160 |
| frozen tokens | 11.126 / 34.570 | 0.381 |
| frozen centers | 39.771 / 100.219 | 0.958 |

逐clip配对后，observed优于frozen State、shuffled State、frozen tokens的比例分别为96.63%、97.93%、87.82%。
相对frozen的逐clip误差降幅中位数79.33%；4000次clip bootstrap CI95为77.65%--81.20%，不涵盖teacher系统偏差。
冻结centers时还原位移幅度的clip中位数仅4.02px，observed为38.73px，真实为44.23px；运动还原主要依赖更新的geometry。
这与readout显式使用root center、relative carrier centers及feature residual一致：冻结tokens仍改变geometry与carrier混合，
不能把干预损失直接解释为“多少百分比信息存在feature中”。tokens更新在多数clip改善还原，但尚未验证独立pose/articulation/interaction属性。
shuffled干预只证明State内容须对应正确观测帧，不证明history的额外价值或未观测未来预测能力。

| Source | Motion clips | Observed change p50 / p90 (px) | Normalized p50 |
| --- | --- | --- | --- |
| Agibot | 79 | 12.81 / 28.32 | 0.390 |
| Bridge | 71 | 6.54 / 15.72 | 0.193 |
| HY | 78 | 3.62 / 5.59 | 0.194 |
| RoboMIND | 79 | 14.86 / 50.45 | 0.633 |
| RoboTwin | 79 | 11.61 / 30.29 | 0.309 |

HY为424x240、Bridge为256x256，其余数据为640x480或518x518；不能仅凭pixel误差排序源质量。
RoboMIND有10/79个motion clips的逐测量normalized误差均值超过1，observed优于frozen的比例87.34%；
其失败不能在未看视频/独立标签前归因为数据质量或teacher错误。典型候选`robomind_ep50078_f303`，
实际移动平均34.48px、还原误差72.74px、2,532个运动测量；不是只有一个点的统计异常。
小运动仍差：真实位移<1px时observed clip误差中位数1.26px，freeze为0.394px；1--5px时2.322px对2.072px。
5--20px normalized为0.527，>=50px为0.159；不能用整体运动提升掩盖静止抖动与小运动问题。
absolute position与change误差接近相同（最大clip均值差约2.1e-5px），因有效t=0测点就是history校准reference，
不算两条独立证据。独立object grouping、真实visibility、遮挡重现及非几何dynamic feature均未证明。

**Impact / next experiment**

保留当前State/readout作为“观测变化可还原”的accepted能力，不据此换encoder或回退dense reconstruction，也不宣布object State完整成立。
下一步在同一held集合比较后续checkpoint，检查geometry改善是否保留且小运动抖动是否减轻；同时用少量独立多物体标注验证query/entity对应、
跨部位一致运动、背景/机械臂混合及遮挡重现。现有annotation template可复用，不先增加整库标签或重训新架构。
这是一张6000-step截面，不是饱和判定；仍不以State拟合自动晋级Dynamics。G0 teacher/object独立有效性保持未通过。

### 15.50 2026-10-02 Stage handoff标准与梯度趋势

**What changed**

- 将“可启动Stage 2可行性实验”与“可宣称object State成立/扩展长训”分开。6000步held结果支持前者的候选资格：观测变化能还原且State干预有效；后者仍需独立entity对应证据。Stage 1不以固定步数、feature独自承担运动或gradient norm低于某常数作为毕业条件。
- Stage 2候选应补测实际冻结EMA target/readout路径，并在同一held集合与后续checkpoint比较小运动漂移和尾部误差。冻结表示的Dynamics实验再检验正确posterior相对zero/shuffled的收益、绝对误差以及距observed-state解码上限的差距；不要求先在Stage 1证明尚未训练的预测能力。
- W&B `0oko1yhr`本轮显示crashed，最后同步7540步（2026-10-02 15:30:26 CST）；读取3500步后全部203条记录。四个窗口3500–4500、4500–5500、5500–6500、6540–7540的preclip gradient norm中位数为8.37、18.82、69.91、155.48，最后窗口50条记录、最大364.88。对应transport均值5.851、5.313、4.918、4.772；appearance为0.329、0.337、0.371、0.392。代码裁剪阈值5，LR持续下降；这是持续梯度放大且目标改善不一致，不证明梯度导致中断。W&B文件未提供异常console日志，当前也无逐模块/逐loss梯度归因。

**Why**

State的任务是为后续变化建模保留足够且对应正确实体的信息，而不是无限压低训练拟合；必须用冻结表示下的实际预测检验最后一段可用性，避免把观测还原等同于未来预测。

**Impact**

本轮仅分析与记录，未停止、恢复或启动训练。当前r2 Stage 2为每query4x64连续effect、144个512维state tokens、8轮local effect cross-attention加global interaction，Dynamics约51.3M、Posterior约12.7M参数。现有入口全模型strict加载旧v1 State会缺少新增`posterior.geometry`参数；正式启动前需明确迁移State/EMA/readout并新建Posterior/Dynamics，不能把旧v1 checkpoint直接宣称为新版兼容resume。

### 15.51 2026-10-02 冻结Stage 1，扩大Stage 2容量并明确迁移

**Hypothesis / decision**

用户批准开启Stage 2，不再无限延长State训练。主实验扩大transition网络内部容量，而不增加State token数、不改变teacher、不重训视觉encoder。由Sol 6.1 high实现，父任务负责设计、代码审阅、静态核对和交付。此实验回答冻结表示能否支持连续effect驱动的未来预测，不宣称已通过独立object validity。

**Frozen controls / inherited assets**

- 固定使用W&B `gaiahhsw`已评测的6000步State快照：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_video_v69_state_change_held_aed14bb_20261001_235841/checkpoint_snapshot.pt`。路径来自W&B运行配置与评测代码的快照命名，本轮未直接访问服务器文件。
- 继承原State训练目录的`dataset.json`、冻结DINO、online State encoder、EMA encoder和readout；State仍为16 queries x 9 tokens x 512维、原生空间坐标、3秒16帧历史与5秒25帧未来。
- 不继承旧Posterior/Dynamics、optimizer、scheduler、W&B run或数据游标。新阶段是独立初始化；只有新Stage 2自身中断后才恢复完整训练状态。

**Primary change / execution**

Stage 2内部1024维、16 heads，Dynamics 12轮local effect cross-attention与global object interaction；Posterior 4层，effect仍为每query4x64。独立输入/输出投影连接冻结的512维State；静态参数形状计算Posterior 51,103,872、Dynamics 305,594,882，共356,698,754，服务器model inventory将记录实际计数。保留小模型默认值以兼容旧调用，新入口显式选择large配置。长序列采用activation checkpointing控制反向显存，不裁短时间跨度。

八卡前台、每卡B=2、global batch256（accum16），30000 steps、peak LR2e-4、5% warmup、floor0.1xpeak；每250步恢复checkpoint、每2500步编号保存。训练不访问GitHub、不下载模型；W&B online同步。拉取部署、单卡测试、八卡训练为三个可独立执行的步骤。

**Evidence / falsification / next step**

增加真实State初始化后的全容量BF16训练与保存恢复对照，测试不重新训练Stage 1。训练记录Posterior/Dynamics裁剪前梯度、实际参数量、direct/rollout/zero/shuffled误差以及冻结EMA State与current-copy还原误差。绝对误差、逐案例/来源/时间分布仍为科学评测重点；均值、工程通过或短测试不提升为语义证据。

实现已完成，9个Python文件AST解析、3个新Shell及runbook中4组独立命令语法检查和git diff --check通过。测试入口默认真实6000步快照，完整16+25帧、B1、两次更新，比较恢复与连续运行；报告`test_report.json`、`dynamics_resume_comparison.json`同步W&B。新增`observed_target_transport`、`current_state_copy_transport`、`last_observation_copy_transport`明确区分EMA观测还原、冻结当前State和最后可见测点，不进入loss。

本轮没有本地模型前向或GPU训练；服务器端到端测试和八卡长训由用户执行，尚无新训练结果。指令与严格resume见`OBJECT_VIDEO_STAGE2_V69_RUNBOOK.md`。若correct effect不优于zero/shuffled，或更大网络不能接近观测State的可还原水平，则容量扩展假设不成立，按实际误差定位，不据模型大小宣布成功。

### 15.52 2026-10-02 V69 原生 SwanLab 接入与恢复迁移

**Decision / scope**

用户要求将当前实验记录从 W&B 切换为 SwanLab。修改 V69 State/Stage 2 训练、单卡端到端测试、held evaluator、encoder/motion audit 和报告上传入口；不更改模型结构、loss、teacher、数据采样或数值恢复状态。历史 W&B run 保留，不据日志后端切换生成任何新的科学结论。API key 仅由环境变量或 SDK 登录提供，不写入仓库或运行文档。

**Implementation / execution**

使用原生 SwanLab 0.10.1 的 init/log/id/resume；标量名称和显式训练 step 保持。逐 case 数据通过原生 ECharts Table 同步，完整 JSON/JSONL/CSV 报告按内容分段上传为 Text，HTML/图片使用对应 media；公网服务不支持 W&B Artifact 式 save。MP4 原文件保留在共享存储，上传展示转换为 GIF，不把缩放后的展示当作训练数据。

checkpoint 新增 tracking 元数据，包括 backend、experiment ID、project/workspace。SwanLab checkpoint 严格恢复到同一 experiment；旧 W&B checkpoint 保留模型、optimizer、scheduler、per-rank RNG、cursor 和 batch/accum，首次迁移新建 SwanLab experiment 并记录 previous_wandb_id。原训练 source_revision 保留，新 execution_revision 单独记录实际执行的日志接入版本；resume 不返回旧 W&B release。

新训练使用独立 `_swanlab` OUT，已提交任务使用的旧 immutable release 不修改。Stage 2 新启动默认每卡 B=4、八卡 global batch256、accum8；resume 仍恢复 checkpoint 的实际数值配置，不擅自改变旧 B=2/accum16。在线训练命令不拉取 GitHub 或下载模型，SDK 安装与账号登录为独立准备步骤。

**Evidence / boundary**

在隔离临时环境实际执行 SDK-only offline integration：标量、nullable Table、长文本分段、HTML、MP4-to-GIF 和同 experiment ID 恢复均完成。报告 `model_executed=false`；没有调用本地模型前向、训练或 GPU smoke test。云端鉴权、服务器全容量 Stage 2 端到端恢复测试和八卡长训仍需用户执行，不能用 SDK 测试代替模型验证。操作入口及独立启动/resume 指令见 `OBJECT_VIDEO_STAGE2_V69_RUNBOOK.md`。

### 15.53 2026-10-02 将 SwanLab 上传范围收窄为训练指标

用户进一步要求“只上传训练数据”，本轮按只同步训练过程指标执行：State/Stage 2 仅上传数值型 loss、梯度、LR、显存、时间指标和训练配置；不上传逐 case Table、原始 RGB、视频、报告、数据集或 checkpoint。逐 case 测量与测试/评测结果继续正常计算并写入本地，不删减科学评测；非训练入口不创建在线 experiment，测试/评测 Shell 显式关闭 tracking。

训练去掉仅为云端 Table 服务的 case-row all_gather；标量 DDP reduction、loss、模型与 optimizer/RNG/cursor 的 checkpoint 恢复保持不变。SDK regression 已通过纯标量 offline 同 ID 恢复与非训练 online 不上传检查；3 个 Python、5 个 Shell 和 7 个运行文档 Bash 块通过静态检查，不执行模型、不验证云端鉴权。这一范围覆盖并替代 15.52 的全报告/媒体上传安排；当前训练、恢复指令见 `OBJECT_VIDEO_STAGE2_V69_RUNBOOK.md`。

### 15.54 2026-10-03 SwanLab 项目环境变量契约

- **What changed**：项目名使用官方 `SWANLAB_PROJ_NAME`；未指定 workspace 时不导出空字符串。V69 先从 CLI/environment 或 checkpoint 取得 project/workspace/id/resume，再移除 SDK 环境解析中的旧 project alias 和重复 workspace/id/resume，通过显式 `init` 参数传入；认证继续使用 `SWANLAB_API_KEY`。
- **Why**：SwanLab 0.10.1 的 `SWANLAB_PROJECT` 是结构化 `ProjectSettings`，workspace 若指定则必须是非空字符串；普通项目名和空 workspace 的环境变量与该契约冲突。
- **Impact**：兼容已提交脚本中的旧项目名、空 workspace；非空 workspace 和 checkpoint experiment ID 不变。仅修改日志配置入口与运行指令，不改变模型、loss、数据、batch 或 checkpoint 数值恢复；训练仍只上传标量指标。

### 15.55 2026-10-03 Stage 2 默认 microbatch 放大四倍

- **What changed**：Stage 2 新启动默认每卡 batch 由 4 改为 16，DINO frame batch 由 2 改为 8；八卡 global batch 保持 256，梯度累积由 8 改为 2。
- **Why**：用户反馈显存占用低、训练慢，要求放大四倍，以更大的实际 microbatch 和 encoder frame batch 执行现有计算。
- **Impact**：模型、loss、原生分辨率、16 帧历史与 25 帧未来不变。严格 resume 继续恢复 checkpoint 原 batch/accum 和 encoder frame batch；这不是原实验的自动 rebatch。未进行本地模型前向或八卡显存/吞吐测试，不声明四倍加速。

### 15.56 2026-10-05 SwanLab Stage 2 全历史读数：effect 有效，位置梯度主导

- **What changed / evidence**：直接从 SwanLab API 读取 `object_video_v69_stage2_large_b32_seed17_20261003_215438`（run `00cxvd6i`）完整标量记录，共 1353 个 loss 记录点，step 1–6830；最后同步时间 2026-10-05 13:22:56 CST，状态 RUNNING。实际执行版本 `c073c3b`，每卡 B32、global batch256、DINO frame batch16、workers4；冻结6000步State快照，训练约356.7M Posterior/Dynamics参数。这里记录的是训练读数，不是 held 结果。
- **Trend**：500–1500、3000–4000、5000–6000、6000–6830 四个窗口的 direct transport 均值分别为14.25、9.34、7.69、7.20原生像素；rollout为14.22、9.48、7.94、7.41；future latent auxiliary为0.2260、0.1982、0.1932、0.1904。最后两窗 direct 改善6.34%，不能宣布已到平台。
- **Effect evidence**：6000–6830窗口 current-state-copy24.44px、zero-effect29.60px、shuffled41.00px、correct direct7.20px，按窗口均值分别降低70.54%、75.68%、82.44%。该窗口152个已记录batch全部correct优于这三种对照；这不是逐case胜率。Observed-target readout5.65px，只是读取真实未来后经冻结State/readout得到的比较路径，不是数学上不可突破的下界。
- **Optimization / Why**：最后窗口裁剪前总梯度norm中位数2706.99，center_delta的平方norm占总梯度平方norm的平均99.9906%；feature_delta梯度中位数7.48、Posterior18.68。全局clip阈值5，缩放系数中位数0.001847；所有读取的finite标记均为1。早期总norm中位数4555.58，未见持续上升，不能归因为数值爆炸；但几何输出头主导更新方向，feature改善较弱。加权latent项约占loss0.30%，加权KL项约0.024%，该比例只说明数值量级，不等于梯度占比或直接调权重依据。
- **Operational observation**：最近约20.81秒/optimizer step，rank0 PyTorch累计峰值allocated显存73.28GiB；不是瞬时NVML显存或全卡利用率。B4旧run约64.02秒/step、B16约24.42秒/step，均为短run且workers/log频率不同，只作观测，不能作为严格受控速度对比。
- **Impact / next question**：训练支持“真实未来经Posterior压成effect后，Dynamics能还原明显优于静态复制的运动”。Posterior仍读取整个未来序列；尚不证明history-only预测、effect选择、语义可组合性或held泛化。下一次冻结checkpoint评测优先区分geometry/feature贡献、按未来时间与运动幅度的轨迹误差、held correct/zero/shuffled对照，不凭训练均值或模型大小宣布成功。本轮不改模型、不停止训练；建议8k–10k附近开展held评测。

### 15.57 2026-10-05 后续研究计划：从片段编码到可复用的object transition

**What changed / research claim**

主线不变：从纯视频学习与当前实体对应的紧凑状态和变化条件，在object state上完成Dynamics，并减少反复预测的计算成本。下一轮不以更低训练像素误差、更大的feature梯度或更大网络作为目标。必须区分三个问题：未见视频上的变化还原、变化条件的实体归属与复用、达到同等误差时的真实成本。

当前证据来自15.56的训练窗口，而非最终checkpoint评测。本轮检查当前代码，并由子任务只读审阅评测入口；没有修改模型、loss或当前训练。Geometry承载平移和支持形状是合理设计，center梯度主导不是feature无用的证明。15.49已显示State在小于5px运动上可能不如复制，而大运动表现较好，因此motion-relative与静止漂移是已有证据指向的缺口，不是新增形式指标。

**Why / current structural question**

`object_sequence_dynamics_v69.py`中的Posterior按query读取整个未来序列，得到每query四个64维effect tokens。Dynamics将同一组effect反复用于所有未来时间。这允许编码一段五秒轨迹，尚未要求编码具有局部时间含义或可跨上下文复用。正确effect优于zero/shuffled证明模型使用了该信息，不足以证明它表示通用变化。此处是待检验风险，不是已经发现作弊或训练失败。

训练时B>1的shuffled是跨clip打乱；现有held入口B1时则在同一clip内跨query打乱。两者必须分别命名和评测。当前16个query也不等于16个已验证物体；一个实体可能有多个query，实体级干预需要对query group操作。

**P0 / 固定checkpoint的变化能力评测，下一项实现任务**

1. 固定当前数据、State、readout和Stage 2 checkpoint，复用400条、五源各80条的诊断集合，先确认episode与训练划分隔离。该集合用于诊断与方法选择；另留未用于选方法的episode集合做最终确认，不能反复用同一held集合选方案后仍称其为独立test。
2. 逐track记录完整可见路径ADE、固定5s终点FDE、运动幅度、预测幅度、方向及覆盖率。每条track的相对路径误差是该track平均位置误差除以同一可见时刻、同一参考点下的平均真实位移；先track内计算，再track到clip、episode/source汇总，报告中位数、尾部及配对分布，不用整体误差均值除以整体运动均值替代。
3. 运动低于可分辨噪声时，不用极小epsilon制造比值；单列静止漂移和绝对误差，并报告分组边界敏感性。往返轨迹终点位移接近零不等于静止；终点ratio不可定义时保留绝对FDE和全路径ratio。不可见段不插值成GT，5s不可见不以更早终点冒充5s。
4. 参考点是每track最后有效历史观测，不保证恰在t=0；记录reference age，单列t=0可见子集。位置误差保留初始化偏差，位移变化误差从模型自己的source readout起算。像素与图像尺度归一化同时报告；CoTracker agreement仍不是独立物理真值。
5. 在同一case比较direct/rollout的correct、zero、query-shuffled、cross-clip-shuffled，加current-state-copy、last-observation-copy和observed-target。Posterior mean与sample明确分开。冻结tokens/centers的四格干预区分readout依赖与递推依赖，不能将其直接命名为几何或语义的百分比贡献。

实现复用`object_sequence_evaluation_v69.py`、`state_change_evaluation_v69.py`、`evaluate_object_video_sequence_v69.py`，不另写一套数据加载和指标定义。评测不改变训练目标，也不自动重启训练。

**P1 / Object归属与effect复用，可与P0并行准备**

使用一个按来源、运动幅度、遮挡与接触分层的独立实体审阅子集；只用于评测，不要求整库instance segmentation。检查默认history queries对应的实体，不能用人工指定query成绩替代自动query成绩。既测同实体多个query，也测不同实体；干预同一entity的query group，输出被干预实体到其他实体的响应矩阵。

选择预先规定的、变化相容的donor/recipient pairs，比较recipient自身effect、相容donor effect、不相容donor effect和zero。保持recipient初始状态、其他实体effects及时间不变。相容性依赖独立实体与变化定义，不按评测分数挑选配对；不要求不同尺寸或接触条件下出现完全相同的像素位移。无接触场景检验无关响应，接触场景允许有效交互传播。没有配对真实后果的effect交换只能报告响应，不能称正确反事实。

复用`object_association_diagnostics_v69.py`和现有annotation模板，新增跨clip配对输入。短期保持连续latent，不因可视化聚类就命名动作语义，也不强制latent向量相加具有物理意义。

**P2 / 下一轮训练的首选结构假设：分段object effects**

若P0显示held变化还原成立，而P1暴露effect只适用于原clip，则下一轮仅改变effect的时间组织方式：保留3s历史、5s序列目标、冻结State/readout、现有Dynamics容量与同一数据。对比A为当前整段effect；B为多个局部transition effects，连续rollout仍覆盖完整5s，不回到action-free短期预测。

B的首个受控设计沿用每query共4x64的code预算，把25个未来目标分成四段，每段一个64维effect，并传入真实段时间；共享Posterior只读取该段起点与该段目标状态，共享Dynamics顺序消费各段effect。训练期Posterior可用真实局部source/target，rollout的状态输入必须来自前一段预测，不能用真实中间state重置轨迹。分段是一个待验证的时间组织假设，不代表已发现四种动作，也不宣称64维足够。总latent标量数相同不保证信息率相同，需同时比较KL分布、失真和实际时间成本。

预期作用是使每个effect描述有限时间内、相对于source state的变化，而不是同一编码与绝对未来时间共同解出整条视频。时间分段本身不保证跨对象可迁移；必须由P1配对复用和未见组合的完整rollout证明。PlaySlot的逐transition inverse dynamics与object-conditioned autoregression提供先例，不提供本项目成功保证：<https://arxiv.org/html/2502.07600v2>。不直接搬用其VQ或RGB监督。

若P0反而显示observed-target本身在同一批case失效，优先修正State的证据获取、query binding或小运动稳定性，不先训练B。若direct好而rollout差，优先处理预测state分布下的递推训练，不用更多posterior容量掩盖。若A已经能复用，则保留A，转向成本与selector，不为分段而分段。

**P3 / 成本证据与后续部署**

与P0并行建立benchmark：分开统计history perception/State编码、训练期future/Posterior、给定effect后的Dynamics、readout。比较单次预测与多次候选rollout，报告同步时间分布、吞吐、显存和实际tensor规模；144个State tokens不是端到端加速结论。

先做一个小型、独立拟合的geometry-only transition对照，共享effect信息权限、时间跨度及readout，检验动态feature更新带来的精度与成本。source视觉features仍用于readout，这不是证明所有视觉信息可删。论文层面的dense-state对照需要独立训练并匹配条件信息和预算，不能仅比较token数量。

当object-conditioned变化可复用且成本有收益后，下一项才是goal-conditioned effect selector，先用图像目标或演示选择effect，再讨论language和机器人控制接口。部署预测路径不读真实未来；Posterior仍是训练和演示编码工具。无需先把latent命名成自然语言动作，也不需要以RGB图像质量定义world model成功。

**Impact / 执行边界**

本轮完成代码审查与研究规划记录，未实现P0–P3新增代码、未推送、未修改正在运行的实验。下一实现包优先P0，P1配对规范与P3计时可并行；下一次大规模训练依照这些结果选择，不同时更换encoder、teacher、State、effect和loss。SwanLab继续仅上传已授权训练标量，评测逐例文件/视频留在共享存储。现有Gate作为证据边界，不新增启动阻断或任意单一分数晋级标准。

### 15.58 2026-10-05 Stage 3规划：语言与历史视频条件下预测现有latent effect

**What changed**

1. 用户明确下一阶段应学习预测z，而不是先要求z跨场景复用或改成分段effect。当前Stage 2的任务仍是history-only S0加真实未来经Posterior得到的z，还原S1:25。下一阶段训练p(z | observed history, instruction)，冻结当前State、Posterior、Dynamics、readout，保持每query4x64的连续effect和3s历史/5s未来不变。15.57的motion-relative评测继续使用；分段effect不再是下一轮主改动。
2. 推荐预训练Qwen3-VL-4B-Instruct作为视频/语言conditioner，新增约300M量级的object-conditioned flow expert。4B是已公开权重的模型规格，expert参数量只是实现预算而非实测。首版保持VLM原视觉输入接口，不把任意DINO投影假称为预训练的视觉语言对齐；DINO/State继续提供当前高分辨率object states。两个视觉计算路径是复用旧world model与引入已有语言能力的成本，不能隐去，也不代表训练两个互相竞争的object State。
3. 补齐语言数据契约和posterior target导出，最终用预测z经过同一个冻结Dynamics后的结果评价，而不是只比较latent MSE。原始无语言视频仍可参与明确标记的无语言条件训练，但不能算作语言grounding数据。

**Why / source basis**

LAPA明确采用先学习latent action tokenizer，再用预训练VLM根据观测和任务描述预测latent actions的顺序；其离散VQ/分类头不直接套用于我们当前连续z。来源：<https://arxiv.org/html/2410.11758v2>。

pi0采用3B PaliGemma加约300M action expert，并通过flow matching生成连续action chunks；其缓存视觉语言条件、由较小expert重复采样的结构支持本方案，但我们的输出是object effects而不是机器人关节动作。来源：<https://arxiv.org/html/2410.24164v1>。

RepWAM先学视觉/变化表示，再在语言条件下生成visual/action chunks；借鉴语言条件与连续latent生成，不重新实现dense video generation。来源：<https://arxiv.org/html/2606.13674v2>。

Qwen3-VL-4B-Instruct官方提供图像/视频输入与时间位置机制，可作为conditioner初始化；这不是宣称其已会预测本项目的latent。来源：<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>。以上资料已在本轮查阅；不依据参数规模自动声明语义能力或八卡吞吐。

**Impact / proposed interfaces**

- 训练标签路径：固定时间窗口与history queries，冻结Posterior读取S0及真实未来states，导出Gaussian mean/logvar、query坐标/有效性、帧时间、episode/window标识、teacher checkpoint版本。目标不能跨query顺序、时间窗口或增强变换错配；只缓存小型effect标签，不要求重造RGB或缓存所有DINO patches。
- 条件路径：历史16帧/3s及相对时间、instruction输入VLM；历史State序列和query几何输入object adapter。VLM hidden tokens与object tokens构成expert context，不先生成文字解释或chain-of-thought。历史视觉采样/预算只依赖历史，不能由未来运动点筛选。首版VLM保留原processor的动态分辨率机制并显式记录视觉token预算；原State路径分辨率不随VLM改动。
- 输出路径：64个带query归属的effect tokens，以每query当前State为local条件，以其他queries和VLM context为交互条件，预测[B,16,4,64]连续latent。保持history query到teacher effect的对应，不将slot编号当语义类别。模型按输入query排列输出相应effect，padding query不参与损失。
- 首版生成对象为Posterior的pre-tanh mean u=mu，再以z=tanh(u)连接现有Dynamics，与当前deterministic posterior评测一致。它学习条件下mean-code的经验分布，不宣称拟合完整Gaussian posterior；logvar保留用于比较posterior采样影响，不把encoder噪声直接等同于多种任务未来。数据归一化统计仅从训练集得到，推理做对应逆变换。
- Flow训练对u_target与Gaussian噪声epsilon做线性插值u_tau=(1-tau)epsilon+tau*u_target，expert根据u_tau、tau与条件预测u_target-epsilon。tau是去噪时间，不是视频时间。推理从噪声积分生成u，不输入真实u_target或真实未来。VLM/object条件每次决策编码一次，缓存供多步expert使用；跨新观测需要更新条件。

**Impact / training and data plan**

先完成固定Stage 2 checkpoint的posterior reference评测和标签生成，再训练conditioner adapters及effect expert。首版冻结VLM视觉塔，对语言/多模态Transformer做LoRA适配，expert完整训练；不是从零训练4B，也不因VLM冻结大部分参数而把其预训练能力说成不存在。现有约16k独立episode规模来自上一轮manifest，不能当作足够全量微调4B的证据。全量微调或8B扩展仅在语言失败定位与数据覆盖支持后考虑；八卡显存/吞吐待具体实现测量，旧Stage 2 B32不直接沿用。

主训练先仅使用flow objective。暂不同时解冻Posterior/Dynamics，也不对每个随机生成的未来都强制拟合同一条GT轨迹，避免压掉合法多样性；是否增加decoder-aware细调在预测latent经Dynamics的误差定位后决定。用同一conditioner的确定性mean-regression头作一项对照，检验连续分布建模是否实际获益，不并行堆叠多个无关模型。

本地代码确认`ObjectVideoSequenceDatasetV69.__getitem__`返回RGB、时间、teacher、case/source等字段，但没有标准instruction输出；case原始metadata是否保留源语言、各源完整性尚未核验。需按source/episode/task/timestamp回连源instruction，记录instruction来源、时间范围、类型和缺失状态。episode级任务指令是允许输入，不能假称每个5s窗口都有精确子任务。不得用task ID代替可理解语言，也不得把未来路径细节写进输入描述。仅有整段目标与具有时间标注的子任务分别报告；若后续从完整视频生成伪描述，应明确是合成监督而非部署时已知意图，独立指令评测不使用它作为唯一真值。

按episode划分训练/验证/最终test，另设未见指令措辞、object-action组合与场景；同episode多窗口或语言改写不能跨集合泄漏。教师标签生成可离线并行，Stage 3训练不读取未来RGB。保留原始数据与现有采样轨迹，不先追加重型tracker/SAM流程。

**Impact / evaluation and delivery scope**

核心对比在相同case和同一冻结Dynamics下进行：posterior mean z、视频+正确语言生成z、视频无语言生成z、视频+明确冲突语言生成z、静态复制。绝对/运动归一化轨迹误差、分时间/来源/幅度分布沿用15.57；latent误差只作辅助。无语言与错误语言都只能作为解释性对照，若历史已唯一决定运动，语言收益不一定明显，需含真实指令歧义的场景。

主成绩报告单样本或固定采样规则下的期望表现；best-of-N仅说明覆盖能力，必须标为使用真实后果挑选的oracle指标，不得代替部署精度。instruction干预通过独立entity/目标关系评价，不把冲突指令的合理输出与原视频GT距离变大当作失败。视频历史顺序、对象归属与语言改写对照共同检查条件是否被使用。还需报告包括VLM视频编码、effect采样与Dynamics的完整延迟，不能只报小expert的速度。

本轮仅制定方案并更新本节，尚未实现Stage 3、未推送、未修改当前Stage 2任务。下一开发包是language manifest连接、固定posterior标签导出、VLM conditioner、object effect flow expert，以及复用Dynamics的paired evaluator。原stage2保留为独立可复现实验；分段effect和机器人action decoder均后置。

### 15.59 2026-10-06 V70实现与真实语言覆盖审计

**Decision / 覆盖此前建议**

用户批准V70：Qwen3-VL-4B-Instruct冻结视觉塔及merger，完整训练语言Transformer、embedding及final norm，不使用LoRA。新增12层、1024维、16 heads、SwiGLU4096的连续effect expert，实例化参数计数308,660,288。目标为固定Stage 2 Posterior的pre-tanh mean `[B,16,4,64]`；以原tanh和同一个冻结Dynamics/readout解码。四个effect tokens共同编码五秒，不改成四个时间段。只用已有片段和可靠原始语言，不引入机器人动作或重造tracking。

用户随后将工程测试从四卡改成**单卡**；正式训练仍为八卡FSDP FULL_SHARD。默认每卡4、累积8、global batch256、10k optimizer steps。默认值尚不是80GB显存实测结论。SwanLab仅上传训练标量；评测与视频留本地。同步/下载与前台训练分开，训练不访问GitHub或模型下载。

**Observed / 先审计实际集合，再使用语言**

经用户授权，在SWXC现有Codex对话“配置 port 8600 的 HF checkpoint”中，通过port8600读取服务器元数据。没有启动训练、占用GPU或修改现有数据。当前Stage 2 `object_video_v69_stage2_large_b32_seed17_20261003_215438/run.json`实际指向`outputs/object_video_v69_state_seed17_40e4586_20260929_003931/dataset.json`。集合为17,816条clip/独立episode，而非目录名暗示的20,000条；train16,034、held1,782。

| 来源 | 实际clip | 原始文本及来源回连数量 | 其中train / held | 本步不计入 |
|---|---:|---:|---:|---:|
| Agibot | 4,000 | 4,000 | 3,600 / 400 | 0 |
| Bridge | 2,075 | 2,002 | 1,800 / 202 | 73 |
| HY | 4,000 | 3,393 | 3,053 / 340 | 607 |
| RoboMIND | 4,000 | 3,960 | 3,565 / 395 | 40 |
| RoboTwin | 3,741 | 3,741 | 3,367 / 374 | 0 |
| 合计 | 17,816 | 17,096 | 15,385 / 1,711 | 720 |

9,962条已在当前group携带文本且与原始episodes.tasks精确一致；HY/RoboTwin另7,134条从原始元数据回连。Bridge排除unknown task67、人工复核的明显乱码5、过短take1；RoboMIND排除xxxx20和无法确认完整指令的putegg20。HY591条无相同视频起始偏移、16条视频偏移匹配但episode长度等不一致；table_002共512、table_004共95。没有使用最近episode、任务编号或文件名生成描述来补齐。

RoboTwin从缓存episode_source_index.json回连原始video_path、video_from_timestamp、原episode及dataset范围；3,741条均有原始任务文本。HY通过回连的3,393条又核对clip范围data Parquet的frame_index/task_index和tasks表，帧记录完整。

**Important limit / 有文本不等于有正确的五秒指令**

HY中889条clip含多个frame task IDs（797 train、92 held）。一个具体例子`hy_embodied_ep10026_f648`的episode文本为“把收纳盒的盖子盖上”，帧级task295文本为“Put the ring back into the 3rd slot of the storage box.”，两者不是同一句话的翻译。因此17,096只表示文本存在并有metadata来源，不是已视觉核验的训练指令数量。V70准备使用精确history anchor的帧级任务文本，记录连续有效帧段和原episode文本；不强行合并这些描述，不声称该子任务覆盖全部五秒未来。其余来源首版保留明确的episode级目标粒度。Agibot3,986条还有与clip重叠的action_text，但“重叠”不等于整个窗口被该动作标注覆盖。

原始报告位于运行根目录下`outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.json`和`.jsonl`，后者为17,816条逐clip来源证据。最终可用窗口数量需进一步经过8秒窗口、语言选择、token长度和实际decode，不能直接用覆盖数量乘4声称训练样本量。

**Implementation / 链路与测试证据**

新增独立V70语言manifest、离线label exporter、冻结history runtime、官方Qwen conditioner、query-local effect expert、FSDP训练/DCP恢复与paired evaluator。训练只读取16帧history；offline Posterior才读取未来；query次序与mean/logvar绑定同一窗口。保留每query每帧9个State tokens，不先全局平均。最多4个均匀合法窗口，按episode采样；原held按episode固定拆diagnostic/test，未使用评测成绩选划分。

固定教师示例选已列出的`step_0007500.pt`，准备时剥离optimizer并保存独立teacher.pt；不跟随latest。实际还列出了2500、5000步快照，各原文件约5.72GB；本次只检查文件元信息，没有重新评价这些checkpoint能力。修改teacher或语言策略应使用新数据目录，不能把旧标签复用于新窗口。

本地小型CPU整链路测试使用真实训练循环、AdamW、DCP、episode sampler及scheduler，恢复后最终参数最大差0、sample/noise/tau/数值trace一致、冻结视觉参数未变、语言和expert参数更新。官方小配置Qwen接口在Transformers4.57.1及5.18.0运行；expert参数计数为实现测得。以上不是完整4B CUDA或八卡测试。完整单卡测试脚本执行真实窗口、future swap、反向更新、保存恢复和更新记录；服务器GPU部分由用户后续执行，不宣称已经通过。

操作入口及独立命令见`docs/OBJECT_EFFECT_PREDICTION_V70_RUNBOOK.md`。下一项是用已审计来源生成语言窗口与固定teacher标签，再做单卡完整测试；通过真实执行反馈确定八卡batch与吞吐。语言条件预测是否有效，最终仍以同一冻结Dynamics下的逐轨迹绝对误差、运动归一化误差、静止漂移及语言干预结果判断，不以flow loss或参数数量宣布成功。

**Delivery / 已提交代码的服务器执行记录**

V70实现已推送至`codex/language-object-effect-v70`，代码提交`5dc062a3e917461a48317de9738ed4a6d6aef760`。SWXC从GitHub获取该提交，在8600临时目录`/tmp/v70-language-audit-5dc062a.VYql8T`原样执行`inspect_language_sources_v70.py --verified_trace ...`，exit=0、firsterror=null。实际审计17,816条、已有文本9,962、可回连7,134、缺失/排除720；teacher_files_read=0。结果写入`outputs/v70_language_audits/current_collection.json`和`.csv`。未部署runtime、未启动GPU、未导出labels或freeze checkpoint。单卡完整4B测试仍待用户执行，不能用这个CPU元数据结果代替。

新diagnostic/test拆分保证V70内episode隔离，但本次尚未逐一排除历代实验已经查看过的held case。最终独立test的科学声明仍需核对既往诊断case清单；目前先在diagnostic上开发，不把重新随机分组称为洗掉历史使用记录。

**Teacher选择更新（2026-10-06）**：用户要求由7,500改为8,500。SWXC只读检查确认没有`step_0008500.pt`；CPU以mmap读取`latest.pt["step"]`实际为8,750，与progress.json一致。编号快照仍只有2,500、5,000、7,500。建议改用实际8,750步，但尚未得到用户确认；未复制、覆盖或重命名checkpoint，未生成新teacher标签。此前7,500准备示例不再代表当前用户选择，待可用快照确定后更新执行命令。

**VLM资产复用（2026-10-06）**：V70默认`MODEL_PATH`改为服务器现有的`/mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct`，不重复下载。SWXC只读确认配置为Qwen3-VL、语言hidden size 2560/36层，processor/tokenizer配套文件可读，权重索引引用的两个分片均存在。此项只确认文件资产，不代表完整模型已通过GPU训练测试；不改变待确认的teacher选择。

**固定teacher与集合准备入口（2026-10-06）**：用户要求全部处理后交付除长训以外的集合命令，采用前述现存8,750步。SWXC用已推送的冻结脚本生成`data/language_object_effect_v70_step8750/teacher.pt`，CPU读回step=8750，文件2,865,329,694字节，不含optimizer；原checkpoint未修改。此项替代上文待确认状态。V70默认数据目录与全部命令同步到`language_object_effect_v70_step8750`，不追随latest。新增`prepare_and_test_language_object_effect_v70.sh`顺序执行语言审计、窗口生成、单卡标签导出与单卡完整保存/恢复测试，无八卡长训调用。已有标签可续导出，测试目录每次独立；完整GPU测试仍待执行。

**环境管理约定（2026-10-06）**：按用户要求，V70准备命令使用`uv pip install --python .../.venv/bin/python`管理现有环境，不依赖环境内的pip模块、不重建共享环境。运行阶段仍直接使用该解释器，不联网安装依赖。

**32卡执行与无机器人action评测（2026-10-06）**

启动契约修订：实际四节点任务未注入`NODE_RANK`，旧launcher在进入Python前退出。用户提供百舸官方启动接口：父进程`WORLD_SIZE`为节点数、`RANK`为节点编号；launcher映射到torchrun的`--nnodes`和`--node_rank`，继续使用平台`MASTER_ADDR/MASTER_PORT`，兼容显式`NNODES/NODE_RANK`。训练子进程的`WORLD_SIZE/RANK/LOCAL_RANK`由torchrun重新注入，不与平台节点变量混用。模型、数据、初始化和global batch不变；本次失败不产生新的训练结果，G0及后续科学结论不变。

- **What changed**：用户确认4节点各8卡、标准torchrun变量。保持已有teacher8750、manifest、模型、flow目标、每卡batch4及实际学习率text=`1e-5`/expert=`2e-5`；累积8改2，global batch仍256。新增独立32卡前台入口，支持从旧run已保存step500以原生DCP仅加载模型权重，optimizer、schedule、cursor、RNG及SwanLab重新开始，不称为跨卡数严格resume；保留旧资产。执行revision进入run及checkpoint记录。
- **Why**：本轮唯一训练侧假设是扩大并行可缩短相同global batch的墙钟时间，不以增加batch或改loss混淆吞吐与学习效果。多机通信和共享I/O可能限制收益；不预报四倍加速。
- **Impact / evidence**：旧八卡正式run已记录到step510、保存step500，独立八卡测试的optimizer恢复仍失败。本地CPU完整训练/DCP测试中model-only加载最大参数差0、重启step/cursor正确，普通CPU严格resume差0；原生torchrun两agent、四CPU worker rendezvous/all-reduce通过，全局rank为0--3。不能据此宣称32卡FSDP硬件或恢复通过。32卡尚未运行，科学结果为空，G0独立teacher/object有效性和语言条件能力不晋级。下一项是确认新job真实world_size=32与吞吐，以及使用同一冻结Dynamics做正确语言/无语言/冲突语言的held比较。
- **Evaluation boundary**：参考PlaySlot与FLAM时明确区分“从真实未来取得latent再重建”和“仅历史+语言生成latent”；SlotFormer的外部object/event评测比图像美观更贴合目标。LAPA/RepWAM的下游机器人成功率含action适配，不能直接搬来声称本系统可控制机器人。逐轨迹绝对/运动归一化误差为主，语言关系目标、独立人工复核和单样本/期望/best-of-N分栏为辅；具体来源、未实现项及命令记于`OBJECT_EFFECT_PREDICTION_V70_RUNBOOK.md`。不重新生成标签、不改变旧Stage 2、不为FVD另加RGB decoder。

**2026-10-07 / 5k配对评测决策**

- **Observed**：直接读取SwanLab `9618wzrs`全部455个已上传日志点，截至北京时间10:09/step4540。实际配置batch8、accum1、global256、text LR峰值1e-5、expert峰值5e-5，使用daca36e及step500模型warm start。loss均值在3001--3500/3501--4000/4001--4500为0.24043/0.22973/0.21959；末区间grad norm均值0.5035，最新expert LR4.9506e-5，上传数值无NaN/Inf。结论是收益递减但未停止优化，不能据此宣称生成未来有效或已过拟合。
- **What changed / Why**：新增2500/5000固定编号checkpoint配对评测，用相同诊断episode、窗口、测量点、噪声和冻结Dynamics判断loss下降是否转化为未来运动改善。默认256 episode、4卡按case分片；正确语言/无语言/跨episode打乱语言/Posterior mean/静态reference分别报告，固定单样本为主，4样本期望和oracle独立标注。主要测量是逐轨迹绝对/运动归一化误差、1/3/5s位置误差、静止漂移；按episode配对bootstrap，不把所有轨迹当作独立样本。
- **Impact / next**：不改训练入口、数据或旧release，评测不上传SwanLab。报告、逐例预测和视频保存本地。本地CPU小模型集成已通过实际flow采样、只读DCP恢复、固定计划/噪声、像素与相对误差、不可见终点、视频写入和比较报告；不是完整Qwen/DINO/CUDA结果。tau分桶flow loss及effect MSE仅辅助定位，不替代解码后指标。G0独立teacher/object有效性仍未通过；当前tracker测量不能变成独立object真值。先等待5k快照并执行此评测，再决定继续训练是否有科学收益；不是自动停止或自动晋级。单卡/四卡正式模型执行结果待用户运行。

**八卡导出与测试（2026-10-06）**：用户要求将标签生成和测试并行到八卡。标签及future-swap按窗口分片；完整测试支持`--nproc_per_node 8`，比较所有rank的采样、噪声、loss和恢复记录，而非仅rank0。语言审计与窗口枚举仍由单个CPU进程完成。准备入口不再覆盖GPU可见列表，单卡默认兼容；八卡任务命令直接用Python/torchrun，不安装依赖或访问GitHub。完整八卡CUDA执行结果待服务器测试，不将本地CPU结果当作FSDP实测。

**标签导出吞吐（2026-10-06）**：导出改为每rank批量窗口推理、CPU多worker解码和预取；默认每卡4窗口、4 workers、prefetch=1，DINO每次32帧。同原生分辨率窗口组批，DINO帧batch可跨clip，State/Posterior同步批处理；不缩小图像、不改变窗口或teacher。旧Stage 2感知入口默认路径不变。输出仍一窗口一标签并复用已完成文件，写入时复制单样本张量，避免保存整个batch底层storage。新增实际batch与解码等待/推理/写盘分项时间，GPU吞吐与显存占用待实测。
