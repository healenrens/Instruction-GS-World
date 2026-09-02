# Instruct-GS-World Object-Level World Model 永久主线与实验账本

> 更新日期：2026-09-02
> 本地权威代码：`/Users/hela/Instruct-GS-World-recovered-20260725/`  
> 当前开发分支：`codex/reliable-native-object-transition-v65`
> 当前实现提交：`a1375111e44f4edf63a1839581f4b47c4b8b7dce`（V65 native reliable transition audit；待真实四卡审计）
> 当前已验证代码提交：`f00082d7678f24dd7323ebcb789ef40fca7c0654`（E0 real-teacher GPU verifier）
> V62 E0/E1 实现提交：`6fa0d63e67daf85d24654aaa649e725eb5245bfe`
> V62 B/C/D structural audit 实现提交：`2ad1158084ff0e8b4070f43c6721261df1884485`（静态验证，待服务器执行）
> 上次账本提交：`3f677c5e4cc59b5a1169fcaa8e3611b258952f96`
> 当前实验：V65 native multi-track audit 已拒绝 shared-affine transition；下一项仍在 G0，只比较可组合 object motion-field parameterization
> 远端代码工作区：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/`  
> 远端运行与产物根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`  
> W&B：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`

## 0. 唯一主线

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
