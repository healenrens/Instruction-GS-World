# Instruct-GS-World Object-Level World Model 永久主线与实验账本

> 更新日期：2026-08-23  
> 本地权威代码：`/Users/hela/Instruct-GS-World-recovered-20260725/`  
> 当前开发分支：`codex/object-transition-objective-v59`
> 当前代码提交：`0c7cf6f204c2a57bb099d724fb655b9723d38159`
> 当前实验：W&B `q02tvfqq`，终态 `killed`，最后 history step 6,760，恢复 checkpoint step 6,750
> 远端代码工作区：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/`  
> 远端运行与产物根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`  
> W&B：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`

## 0. 唯一主线

从本文档此次更新开始，版本号只表示实现迭代，不再表示研究方向重启。项目只保留以下一条主线：

> 从纯视频学习可部署的、query-conditioned、persistent object state；在该 state 通过独立 object validity 验证后，再学习 latent effect conditioned object dynamics，最终由 goal、language 或 policy 选择 object query 与 latent effect。

固定的数据流是：

```text
Observed RGB history + current point/region query
  -> frozen perception patches
  -> query-conditioned object binding
  -> persistent object state
  -> latent effect posterior (training only)
  -> object-level Dynamics
  -> future object state
```

训练期 point tracker 可以使用完整视频构造 correspondence、relation、visibility evidence；部署 student 只能读取已经观察到的 RGB history 与 query。future RGB、future tracks、teacher state、instance annotation、机器人显式 action 都不得进入 student history path。

这条主线按以下 promotion chain 单向推进：

| Gate | 必须回答的问题 | 通过前禁止做的事情 |
|---|---|---|
| G0 Objective validity | 正确解是否比 all-scene、seed-only、merge、split、visibility collapse 等捷径更优？ | 禁止真实长训。 |
| G1 Query binding | 给定 query，student 是否找到了与其相关、并排除了无关 track 的区域？ | 禁止声称学到完整 object state。 |
| G2 Persistent state | identity、dynamic、geometry、visibility/unknown 是否各自学到正确含义？ | 禁止训练 Dynamics。 |
| G3 Independent object validity | 不使用训练 tracker 的 evaluator 是否确认 object coverage、leakage、reappearance 与 deletion locality？ | 禁止把 teacher agreement 写成 object semantics。 |
| G4 Latent effect | 从真实前后 object state 提取的连续 effect 是否必要、稳定且不含显式 action/center delta？ | 禁止训练 History-only Prior。 |
| G5 Object Dynamics | 正确 effect 是否显著优于 zero/shuffled，且 rollout 保持 object identity 与 lifecycle？ | 禁止进入任务级 claim。 |
| G6 Selector / task A | goal、language 或 policy 是否能选择 query/effect，并在 RoboTwin task A 上产生可用预测？ | 禁止与 XR-2 混淆。 |

任何实验只能推动当前最早未通过的 Gate。后面的 Gate 即使某个指标变好，也不能覆盖前面 Gate 的失败。

## 1. 文档用途与证据规则

本文档是本项目唯一的研究决策账本。它记录每一轮实验回答了什么问题、怎样执行、观察到什么、为什么导向下一轮。后续开发按第 0 节的 promotion chain 和第 11 节的当前 TODO 推进；前一项未通过，不启动后一项长训。

证据分为四级：

1. **结构 gate**：shape、gradient、causal boundary 和 objective falsification 通过，只说明代码契约成立。
2. **训练 telemetry**：loss 或在线指标变化，只说明优化器正在拟合某个目标。
3. **held teacher evaluation**：在未参与优化的 clip 上与 point-track teacher 对齐，仍不是独立 object truth。
4. **independent evaluation**：使用 simulator ground truth 或人工标注，且不使用训练 tracker，才可证明 object state 具有外部语义。

因此，loss 下降、readout 变好、固定 slot index 稳定或 tracker agreement 都不能单独写成“学到了 object”。

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
| G0 Objective validity | **部分通过** | v57 已证伪 all-scene、whole-frame、seed-only、other-entity；未证伪 all-visibility-zero | 为 visibility/unknown 增加 external target，并加入 visibility corruption attribution。 |
| G1 Query binding | **held-teacher 已验证** | `q02tvfqq` positive 0.9771、negative 0.0149；六源 coverage 与 E2E 通过 | 新版本必须回归保持，不重新设计 binding。 |
| G2 Persistent state | **失败** | visibility 4.82e-6；dynamic head 冻结；compactness/semantic 可被关闭 | 当前唯一实现任务。 |
| G3 Independent object validity | **待办** | 当前没有与训练 tracker 独立的完整结果 | G2 通过后执行 RoboTwin truth/人工小集 evaluator。 |
| G4 Latent effect | **禁止提前** | v53 joint tokenizer 不能证明 effect 建立在有效 object state 上 | 等待 G3。 |
| G5 Object Dynamics | **禁止提前** | 历史 action-free/compact dynamics 结果不能替代 effect-conditioned object dynamics | 等待 G4。 |
| G6 Selector / task A | **禁止提前** | 尚无可部署 state 与 effect | 等待 G5。 |

### 11.2 下一版本的唯一改动面

下一版本只修复 G0/G2，不修改 v57 relation binding 主体：

1. teacher 从 observed-history tracks 产生 `visible / occluded-candidate / unknown` target；invisible 不得自动成为 absent。
2. `L_visibility` 直接监督 student observability；class balance 和 calibration 由 teacher mask 决定。
3. semantic consistency 与 compactness 使用 stop-gradient teacher-valid mask，禁止使用 student visibility 作为 loss 开关。
4. identity 只在 teacher 确认 same-entity 且可比较的时刻保持；dynamic 和 geometry 必须允许变化。
5. unknown patches 单独报告，不强迫它们成为 object 或 background。
6. 增加 `all_visibility_zero`、`all_visibility_one`、identity swap、merge、split、background lock、shuffled track ID 和 occlusion reset falsification。
7. W&B 必须按六个 source 与 $H=1,2,3,4$ 分开记录 binding、visibility calibration、reappearance、support area 和 unknown activation。

### 11.3 严格晋级标准

下一版本只有同时满足以下条件才能从 G2 晋级 G3：

- heldout support positive 不低于 0.90，negative 不高于 0.10；
- visibility 不得坍缩到常数，balanced accuracy 与 F1 必须优于同分布常数 baseline；
- predicted visible rate 与 teacher visible rate 的相对误差不高于 20%；
- occlusion/reappearance subset 的 identity retrieval 显著优于 shuffled track ID；
- query perturbation 后 support 随 query 移动，不形成固定 foreground template；
- identity 稳定时，dynamic 与 geometry 在 motion-active clips 上保持非零变化；
- 上述标准在六源和各个 $H$ 上分别报告，不能只报混合均值。

G2 通过后才运行 independent evaluator。G3 使用不导入训练 CoTracker 的 RoboTwin object ID/mask 或人工标注小集；只有它通过，才允许实现 latent effect 与 Dynamics。

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
