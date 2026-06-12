# Instruct-GS-World 阶段性研究报告

> 整理自 `agent.md` §1–§90 全部实验记录。截至 **2026-06-12**（v15 共训运行中）。
> 配套文档：[`docs/EXPERIMENTS.md`](EXPERIMENTS.md)（训练/评估命令手册）、[`agent.md`](../agent.md)（逐日决策日志，本文所有 § 引用指向它）。

---

## 1. 项目目标与方法

**目标**：语言条件的 3DGS 动力学/世界模型——输入一个 3D Gaussian Splatting 场景 + 一句自然语言指令，预测每个高斯球的 3D 运动，自回归 rollout。

**最终形态的方法栈**（多轮淘汰后存活的设计）：
- **条件骨干**：冻结 Qwen3-VL-2B（Cosmos-Reason2-2B），special-token 空间聚合头蒸馏 28 层特征；
- **动力学**：1.66B SC-GS 风格控制点 transformer + 实体感知 LBS 到稠密高斯；**逐控制点平移场**是唯一运动载体（见 §3 结论 3）；
- **语言因果机制**：relevance 头（控制点 patch 特征 × 指令 token 交叉注意力）+ **反事实损失**（错误指令下命名物体的门必须关闭）；
- **几何一致性**：推理期逐实体加权 Kabsch 刚性投影（`rigid_agg`，旋转取 supervised omega 均值）；
- **数据**：sim（LIBERO，Pi3 抬升 + GT/openvocab 分割）+ **真实视频**（AgiBot，SpatialTrackerV2 伪 GT + openvocab + EEF 度量尺度校准）共训；
- **评估**：3D-first 指标（EPE3D / Acc3DS/R / 5°5cm / 幅度比中位+P10 / coherence / langswap 选择与方向）。

**当前生产模型**：`checkpoints/libero_v12_rigid`（sim+real 共训 + 推理刚性投影，§85）。

---

## 2. 实验尝试全景（按阶段，含失败）

### 阶段 1：可行性与规模化（§1–§37）
- v1 离线 clip 训练：+2.5dB vs 静态基线，**可行性通过**（§20）。
- AgiBot 流式训练（13.7 万集 JIT 解码、零 clip 存储）；语言塌缩用 **InfoNCE + MoCo 队列**替换饱和 hinge 修复（§29）；一次 NaN 事故毁掉全部权重 → **有限梯度范数守卫**成为标配（§30）。
- 教训：`clip_grad_norm_` 不是 NaN 防护；bf16 attention 必需、gsplat 必须 fp32。

### 阶段 2：数据转折——仿真干净 GT（§38–§44）
- **失败诊断**：真实数据伪 GT（Pi3+CoTracker）遮挡噪声大到模型连单 clip 都无法定位运动。
- **转折**：ManiSkill3 解析 GT（RGB-D 建 3DGS + sim 逐 actor 位姿解析移动）。**决定性 A/B**：干净 GT + per-control spatial-grounding → 过拟合 corr 0.95；缺任一 → 0.03（§38）。多 clip 泛化确认：heldseed≈train、从未训过的 StackCube corr 0.76（§39）。
- dyngate1-7（§44）：**dyn-gate**（动/静门，免费 mover 标签）修背景泄漏 0.08→0.01；**sem→gate**（3D 语义嵌入喂门）修遮挡叠加塌陷；**mover_magnitude_loss** 破 L1 中位数欠射（cube ratio 0.46→0.81）。

### 阶段 3：LIBERO + Pi3（§45–§53）
- 换 LIBERO（语言-物体绑定更丰富）；发现 St4RTrack 几何 x 轴压缩 2.4× → **换 Pi3**（各向同性内参、逐帧深度+相机、ego-ready）。
- 数据审计：LIBERO ~12% 失败演示需过滤；ooi mask 63% 是篮子（不能直接用）→ 运动仲裁挑目标。

### 阶段 4：语言因果化（§52a–§57）——本项目第一个核心贡献
- **§52a 关键发现：模型完全忽略语言**（langswap swap/true=1.00）。根因：夹爪邻近捷径（23/24 clip 目标=离夹爪最近物体）+ 控制点特征对指令盲。
- **v8-lang**：relevance 头 + **反事实损失**（同一画面、错误指令 → 命名物体门必须关；视觉捷径不可满足）→ **swap/true 1.00→0.10**，语言成为因果输入（§54）。
- **v8-ent 失败（重要）**：实体槽位 SE(3) 头（特征池化→逐实体刚体）把方向 +0.99→**−0.18**。教训：**特征空间池化丢方向**（§55）。同时暴露指标盲区①：corr/ratio 全是范数量、方向盲 → 补 **dir-cos** 指标。
- **v9-lang**：双窗数据（早窗=接触前起步，破坏邻近捷径）→ **场景泛化 sel 0.25→0.75**（§57）。

### 阶段 5：开放词汇分割（§58–§64）
- GroundingDINO+SAM2 替代 GT mask：SAM2 point-grid segment-everything + 机器人/篮子 box 归类 + SAM2 地板 mask 剔除 + 目标点 prompt。自验 IoU：目标 0.95-0.97、篮子 0.97、臂 0.65。
- **2×2 隔离实验（§64）**：GT 训练的模型在 openvocab 分割数据上得分**完全相同**（0.75=0.75）→ **openvocab 推理就绪**（真实视频无 GT mask 不损失性能）；OV mask 训练伤选择（0.75→0.50）不伤方向 → 训练侧需质量门。

### 阶段 6：刚体一致性（§65–§69）——用户目检驱动
- **用户发现**：预测物体高斯球散开（extent ×3.73）而非刚体移动；所有聚合指标对此盲视（盲区②）。
- 调研四家族 + 16 clip 数据核查 → **v10-rigid**：批量加权 Kabsch 实体聚合层。
- **V2 推理投影成功**：散开消除（刚性残差 2.15→0.01cm），sel/dir/endpoint **逐字节不变**（零重训）（§67）。
- **V3 训练在环失败**：detach-SVD 梯度配置下重训 800 步，方向 +0.81→+0.28（§68）。**结论：刚性投影放推理后处理，不进训练环。**

### 阶段 7：v11 计划——3D 诚实指标与幅度修复（§70–§75）
- **用户判断**：效果未达标、2D/视频指标无意义。**R0 重基线**（eval_3d.py：EPE3D/Acc3DS/R/5°5cm/幅度比中位+P10）全面暴露 2D langswap 掩盖的问题（盲区③）：**幅度塌缩成片**（P10=0.10×）、EPE3D≈物体运动一半、**5°5cm=0.00**、旋转误差 27-28°（§71）。
- **R1 幅度修复**：诊断定位塌缩根因 = **dyn-gate 把 25-32cm 的真 mover 误判静止**（gate 0.08-0.15，非 head 欠预测）（§72）。修法 = `mover_magnitude_loss` 入配方（梯度经 gate 流回顶开门）→ **mag 0.63→0.91×、sel 0.75→1.00、EPE3D P90 23.5→11.4cm**（§73）。
- **R2 旋转源修复**：发现 rigid_agg 旋转取自速度场 Kabsch（拟合噪声），而受监督的 omega 被丢弃 → 改 **omega-mean**（去 SVD）→ rot-err 31.9→19.0°；**GT 物体真的在转**（中位 17.5°，"伪旋转"框架错误）（§74）。in-loop 训练再次失败（梯度爆炸）→ 推理投影定为底线（§75）。

### 阶段 8：真实视频管线 R3（§76–§84）
- AgiBot（已在服务器，用户否决 DROID）逐里程碑打通：
  - **m1**：真实视频 → Pi3 3DGS + 真实 EEF 6-DOF GT（§77）；
  - **m2**：CoTracker+Pi3 伪 GT 噪声大（max 186.8cm 离群）→ **SpatialTrackerV2** 安装（torch 2.8 主 venv 直跑）→ 干净世界系 3D 轨迹（静止中位 0.1cm、max 44.7cm 合理）（§78, §81）;
  - **m3**：openvocab 在真实超市场景成立（§79）；
  - **m4**：**EEF 度量尺度校准**（逐 track Umeyama+RANSAC 共识：scale=0.697、残差 1.1cm）→ Pi3 重建获得米制尺度（§79）；
  - **m5**：零样本基线（LIBERO sim → 真实超市）：方向 +0.51、EPE3D 16.3cm、幅度 0.35×——不可用但非零（§80）；
  - **clip-builder v1.1–v1.4**：持握物仲裁（mover ∩ 名词框）、臂非刚性轨迹转移、名词解析、Retrieve 段尾取窗 → **50 集生成、31 过质量门（62%）**，词汇 cucumber/pear/carambola/corn、位移 5.6-60cm（§82–§84）。

### 阶段 9：sim+real 共训与旋转攻坚 R4（§84–§90）
- **v12 共训 = 模型首次在真实开放世界视频上 work**（§85）：真实 heldreal 幅度 0.35→**0.93×**、EPE3D 16.3→**10.8cm**（追平 sim）、3/6 clip genuinely 好；sim 守卫基本守住（sel 1.00→0.88）。**生产模型 = libero_v12_rigid**。
- **v13 词汇扩展失败（数据毒化）**：libero_90 是 fps20，book 批量沿用 WIN=48 帧=2.4s wall-clock（一半）→ 运动域错位污染训练（sim EPE 12.4→24.7cm）→ **异 fps 数据必须按 wall-clock 对齐窗口**（WIN=96 重生成）（§88）。
- **v14 旋转架构 A/B 双判负**（§89）：erot 残差头全面差于基线；bases-pure 灾难（方向 −0.14 = v8-ent 同款签名）。**至此结构化运动 6 次尝试全部失败**（v8-ent 池化、V3 在环、R2 omega 在环、w_rot↑、erot、bases-pure）→ 定律成形（§3 结论 3）。
- **v15（运行中）**：v12 配方 + 词汇数据（book WIN=96）+ **旋转丰富数据**（libero_goal 抽屉/旋钮 90° 弧——让位置 L1 自己携带旋转梯度，不改架构）；四 held 划分评估（heldgoal 5°5cm / held90 词汇 / heldreal / heldseed 守卫）（§90）。

---

## 3. 实验结论（固化的可复用知识）

1. **语言因果化的充分配方**＝relevance 头 + **反事实损失** + 早窗数据。反事实损失是承重墙：同画面、不同指令必须翻转门 → 纯视觉解不可满足（swap/true 1.00→0.10，§54）。数据捷径（夹爪邻近）不破坏，语言头再好也会被绕过（§52a/§57）。
2. **干净 GT 与 per-control spatial-grounding 二者缺一不可**：缺任何一个，模型连单 clip 过拟合都无法定位运动（corr 0.95 vs 0.03，§38）。
3. **"逐控制点平移场不可替代"定律**（六次独立失败归纳，§89）：本架构里结构化/低秩/实体级运动表示——特征池化（v8-ent）、刚性聚合训练在环（V3、R2）、强旋转权重、erot 残差头、SE(3) bases——要么训练爆炸、要么杀方向、要么注入噪声。**几何结构唯一稳定的施加点是推理期投影**（v12+omega-mean 投影：29.7→19.7°）。聚合要在**输出（投票）空间**做，不能在特征空间做。
4. **软先验不泛化，硬结构才行**：`w_rigid` 软 Kabsch 惩罚训练集有效、held-out 失效（散开 ×3.73）；推理期投影 by-construction 解决且零代价（§65–§69）。
5. **指标盲区三连的方法论教训**：方向盲（corr/ratio 范数量，§55）→ 散开盲（聚合量平均掉 spread，§65）→ 幅度盲（corr 对全局 0.1× 缩放不敏感、rPSNR 被背景主导，§71）。**凡聚合指标必有盲区；2D/渲染指标在动区占画面 ~5% 时无意义；3D-first（EPE3D/Acc/5°5cm/P10）+ 中位与分位数 + 目检并行**。
6. **openvocab 分割是 GT mask 的完美推理替代**（2×2：0.75=0.75，§64）→ 真实视频与无 mask 数据集解锁；训练侧用需 mask 质量门。
7. **sim→real 不是零样本问题而是少量共训问题**：31 个真实 clip 共训就把真实幅度 0.35→0.93×、EPE 16.3→10.8cm（§85）。
8. **异源数据按 wall-clock 对齐**：fps20 数据用 fps10 的帧数窗 = 运动域减半，毒化全部守卫（§88）。
9. **诊断方法**：失败先挖现有日志做独立失败分析（R1 的 gate 根因 §72 纯靠 dump 对照定位）；隔离渲染/2×2 交叉评估比猜测快；伪 GT 的病态（小位移 Kabsch 旋转）要在指标里门控。

---

## 4. 当前生产模型与最佳数字

**`checkpoints/libero_v12_rigid`**（v11mag 权重 + sim/real 共训 + 推理刚性投影 omega-mean）：

| 评估域 | 指标 | 数值 |
|---|---|---|
| sim heldseed | langswap sel / dir | 0.88 / +0.76 |
| sim heldseed | 幅度比中位 / EPE3D | 0.91× / 12.4cm |
| sim heldseed | coherence（刚性残差） | 0.06cm（刚体 ✓） |
| **real heldreal** | 幅度比中位 / P10 | **0.93× / 0.77×** |
| **real heldreal** | EPE3D 中位 | **10.8cm**（最佳 clip 4.8-8.7cm） |
| 全域 | 旋转 5°5cm | **0.00（未解）**，rot-err ~19.7°（投影后） |

---

## 5. 依然存在的问题（按优先级）

1. **旋转（核心未解）**：5°5cm 全域=0；架构路线 6 次失败已关闭；当前赌注=**数据杠杆**（libero_goal 大旋转弧让位置 L1 自带旋转梯度，v15 在测）。备选：bases-residual（已实现未测）、更长训练。真实场景 GT-rot 48.7° 而模型零捕捉（§85/§89）。
2. **词汇泛化未证实**：heldtask（未见名词）sel 一直 =0；v13 因 fps 毒化判负、修复后（WIN=96）并入 v15 重测（held90 划分）。这是"开放词汇"主张的最后一块缺口。
3. **真实数据规模与质量**：仅 31 个可用真实 clip、62% 通过率；同类实例渗漏未修（v1.5 候选：运动一致性过滤）；**MoSca 黄金子集校验伪 GT 自身从未做**（m7）——目前对 StV2 伪 GT 的信任仅来自目检与统计合理性。
4. **统计功效**：held 划分 n=6-8，±0.09 二项噪声——所有 held-out 结论都在噪声区边缘；scale-up（R4 后半）前不可过度解读单次数字。
5. **幅度塌缩残留**：train P10 0.29（少数 train clip 仍塌）；heldreal 个别 clip 过冲。
6. **长时域**：v8 之后未回测 10s 漂移（eval_longhorizon 未复活）。
7. **早窗方向**（+0.25~0.63）：pre-contact 时机歧义，正交于其它问题、未专项处理。
8. **工程债**：Phase C 逐帧 viewmats 已 pilot 验证但未在移动相机 clip 上端到端训过（AgiBot clip 已带 viewmats，v12 共训实际已消费——长时域移动相机渲染监督仍未压力测试）；HANDOFF.md/agent.md 含服务器 IP（仓库若公开需脱敏）。

---

## 6. 正在进行与下一步

- **运行中**：v15 共训（v12 配方 + book WIN=96 + libero_goal 旋转数据，**不改架构**）。判据：heldgoal 5°5cm 首次 >0、held90 未见名词 sel >0、heldreal/heldseed 守卫不回退（§90）。
- v15 之后的分叉：
  - 旋转若动 → 继续数据杠杆（更多大旋转任务）；若仍 0 → bases-residual 单 flag 测试；
  - 词汇若动 → R4 规模化（libero_90 全量 3921 集 + spatial、batch>1、≥20k 步长训）；
  - 真实线 → clip-builder v1.5（同类渗漏）+ MoSca 校验 + 扩 AgiBot 任务面。

---

## 附录 A：模型谱系

```
stream11c (AgiBot 流式) → sim_gen (ManiSkill 泛化) → dyngate7 (门控+语义+幅度)
  → libero v5/v6 (St4R) → v7_pi3 (Pi3 后端)
  → v8-lang (语言因果) [v8-ent ✗封存]
  → v9-lang (双窗, 场景泛化 0.75) [v9-lang-ov = openvocab 诚实性对照]
  → v9lang_rigid (推理刚性投影) [V3 in-loop ✗]
  → v11mag (R1 幅度修复) → v11_rigid (R2 omega-mean 投影)
  → ★ v12_rigid (sim+real 共训, 生产) [v13 ✗fps毒化; v14erot/bases ✗架构]
  → v15 (词汇+旋转数据, 运行中)
```

## 附录 B：关键脚本索引

| 用途 | 脚本 |
|---|---|
| sim 数据 | `gen_sim_dataset.py`, `pi3_video_gt.py`, `gen_libero_pi3_v2{,_ov}.sh` |
| 真实数据 | `agibot_video_gt.py`, `agibot_clip_stv2.py`（StV2+openvocab+EEF 校准） |
| 开放词汇 | `openvocab_seg.py`（segment_frame_amg / _sam2_point） |
| 训练 | `train_sim.py`（DDP+NaN 守卫+全 flag）, `orchestrate_v9ov/v12/v15.sh` |
| 评估 | `eval_langswap.py`（sel/dir/coherence）, `eval_3d.py`（EPE3D/5°5cm/幅度分位） |
| 诊断 | `_diag_collapse.py`（gate 根因）, `_diag_scatter.py`, `_viz_pred.py`, `_viz_rigidfix.py`, `test_rigid_agg.py` |
| 刚性投影 | `igsw/dynamics/rigid_agg.py`（batched weighted-Kabsch, omega-mean） |
