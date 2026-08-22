# Instruct-GS-World Object-Level World Model 完整进度与 v57 路线

> 更新日期：2026-08-23  
> 本地权威代码：`/Users/hela/Instruct-GS-World-recovered-20260725/`  
> 当前开发分支：`codex/query-conditioned-object-dynamics-v57`  
> 第一阶段实现提交：`bb70757`
> 远端代码工作区：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/`  
> 远端运行与产物根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/`  
> W&B：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`

## 1. 文档用途与证据规则

本文档不是按版本罗列代码功能，而是记录每一轮实验回答了什么问题、怎样执行、观察到什么、为什么导向下一轮。后续开发按第 11 节的 gate 顺序推进；前一项未通过，不启动后一项长训。

证据分为四级：

1. **结构 gate**：shape、gradient、causal boundary 和 objective falsification 通过，只说明代码契约成立。
2. **训练 telemetry**：loss 或在线指标变化，只说明优化器正在拟合某个目标。
3. **held teacher evaluation**：在未参与优化的 clip 上与 point-track teacher 对齐，仍不是独立 object truth。
4. **independent evaluation**：使用 simulator ground truth 或人工标注，且不使用训练 tracker，才可证明 object state 具有外部语义。

因此，loss 下降、readout 变好、固定 slot index 稳定或 tracker agreement 都不能单独写成“学到了 object”。

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

## 11. 后续执行清单

| 顺序 | 工作 | 当前状态 | 晋级条件 |
|---:|---|---|---|
| 1 | single-query teacher、student interface、objective falsification | **远端 tensor contract 已通过；真实六源 coverage 待执行** | tensor contract 全通过；真实六源 teacher coverage 报告完整。 |
| 2 | 六源 query coverage audit + W&B evaluator | 待办 | 所有 source/H 有足够 positive、negative、held-out tracks。 |
| 3 | single-query binding 训练入口、checkpoint、resume、W&B | 待办 | held teacher gate 通过。 |
| 4 | RoboTwin independent truth evaluator | 待办 | independent binding gate 通过。 |
| 5 | single-entity latent effect posterior + Dynamics | 禁止提前 | correct effect 相对 zero/shuffled 改善至少 10%。 |
| 6 | multi-query dedup 与 persistent memory | 禁止提前 | 单 query object state 已独立验证。 |
| 7 | goal/language selector 与任务 A | 禁止提前 | object state 与 effect 均通过。 |

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

## 13. 当前明确禁止的捷径

- 不再用固定 slot index 作为 object identity truth。
- 不用训练 tracker 指标冒充 independent object evaluation。
- 不用 action-free short prediction 下降证明 object state 成功。
- 不用 full-image RGB 或 dense DINO reconstruction 主导 object dynamics。
- 不从 tracker invisibility 推导 absence。
- 不在 Object State gate 通过前训练 latent effect、Prior、language 或控制接口。
- 不继承 v43-v56 model/optimizer checkpoint；只复用数据、冻结 DINO、冻结 point tracker、W&B 与运行基础设施。
