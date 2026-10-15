# Object Dynamics: 研究定位、模型改造与实验总计划

日期：2026-10-15。本文是研究与实施计划，不是新模型的实验结果。

阅读顺序：先看第 11 节的调研后取舍，再看第 3-7 节的模型/训练/数据工作包；第 9-10 节是逐篇证据。初始候选计划与最终收窄方向保留在同一文档，避免把调研前的假设当成调研后的结论。

用户决策：重新审视主干，不要求继承旧版本的架构、冻结方式或 latent 形状；先给全局计划，再由六个 Sol/Luna agent 并行阅读论文的方法、实验与必要代码。调研后用户批准进入实施阶段；当前先锚定迁移数据与评测，模型训练尚未开始。

历史入口：[长期实验账本](OBJECT_WORLD_MODEL_PROGRESS_V57.md)。旧实验事实保留，但不把旧路线的阶段门槛当成此次重新设计的前提。

## 1. 要回答的论文问题

**把视频变化表示成对象绑定、具有时间顺序的 transition sequence，能否改善未见组合上的变化生成，并降低达到同等变化精度所需的预测成本？**

这里有三个不同概念：

1. **重建**：真实后续视频可以被 posterior 编码；模型从初始状态和编码重现实际变化。
2. **生成**：只输入已观察历史及可选语言，由 prior 产生变化序列，再生成后续状态和视频。
3. **组合**：已经学过的变化可以用于新的对象、布局和执行顺序；不能仅用训练目标内的 latent 加法残差证明。

论文重点不是重新命名一个 embedding，也不是声称有了 slots 就发现了物理对象。重点是预测行为：指定实体的变化能否被保持、重新绑定、组合和执行。

### 1.1 已有经验的用途

- 用户对 V70 视频的审阅显示 posterior 路径相对较好，语言/no-language/shuffled-language 路径差别较小，并存在主体与发生阶段混合。这是定性观察，提示变化表达与条件选择需要分别研究。
- SSv2/Physion 首轮 frozen-probe 结果没有展示优势，但它们没有调用原 Dynamics 来检验条件变化重建。分类结果保留为迁移证据，不替代主任务。
- 旧 V69 的 root feature 在观测更新和 Dynamics 中都被显式保留；整个五秒的 effect 会重复用于每个预测时刻。新版不要求保持这两项结构。
- 旧 readout 输出轨迹、observation logit 和 DINO feature，不输出 RGB。新实验若比较视频重建，需要真正的可成像路径。

代码定位：`code/igsw/adaptive_gaussian_wm/query_object_video_encoder_v69.py`、`object_sequence_dynamics_v69.py`、`object_sequence_readout_v69.py`、`language_effect_model_v70.py`。

## 2. 调研后的创新边界

下面是当前已查到的直接重合；完整逐篇证据在后文汇总。

| 候选卖点 | 已有直接工作 | 当前判断 |
| --- | --- | --- |
| Object states 加 latent actions 做视频预测 | PlaySlot、LPWM | 不是新点 |
| Encoder、decoder、Dynamics 联合训练 | LPWM 等 | 不是新点 |
| 逐 transition effects 替代整段编码 | 多种 latent-action 模型 | 是对旧项目的改进，不等于论文创新 |
| 外观和变化分离，使变化可迁移 | DiLA 等 | 有直接重合 |
| Latent transition 组合与反向约束 | WLA、DLAM 等 | 有直接重合 |
| 从历史预测 motion tokens，再推进 state | LaMo 等 | 有直接重合，LaMo 当前仅部分正文可读 |
| 语言条件下生成 object video | TextOCVP、LPWM 等 | 必须区分已有能力与新贡献 |
| 真实交互下，主体正确且次序敏感的变化复用 | 需跨论文核对 | 候选研究问题，不是已证实的领域空白 |

**拟收窄的方向**：学习依赖对象当前状态与交互上下文的变化表示，在新对象/关系组合中保留主体、开始时间与执行顺序。组合通过状态递推完成，不预设所有变化可交换、可逆或能够线性相加。

## 3. 主干模型设计

```text
Observed RGB history
  -> pretrained visual frontend
  -> recurrent object-state encoder
  -> source object states + scene context

Training target RGB -> observation states -> transition posterior -> local effects
Observed history + instruction -> conditional effect generator -> local effects

source states + effect sequence
  -> shared interaction-aware dynamics
  -> future object states
  -> spatial renderer
  -> future RGB sequence
```

### 3.1 Visual frontend 与 Object State

- 用已有预训练视觉模型或已匹配的 image-autoencoder 组件启动，不从零重做通用视觉基础模型。
- 每个对象候选保存一组局部 tokens、空间支持和可更新状态；scene context 保存不适合归入对象候选的视觉信息。
- 历史聚合提供时间证据。对象特征不被强制恒定，外观、姿态、遮挡都可以导致状态更新。
- 空间 patch 是视觉计算单元，不作为 object 标签；query 编号不作为语义身份。
- 第一版不同时追求 metric 3D、完整实例分割和 embodied control。它们不是当前变化表示的必要接口。

### 3.2 Renderer 与表征共同学习

Renderer 要把对象状态转换成可观察结果，而不是用一个强视频模型自行补出 Dynamics。

- 主候选采用对象局部成像与场景合成，或已匹配的 image decoder 加可训练对象到空间映射。
- Renderer 逐帧工作；未来时间关系由 Dynamics 承担。mask 是可学习的成像权重，不自动等于独立物体标注。
- 表征和可训练的空间映射接受同一重建目标。可以复用冻结的预训练 image decoder，也可以联合更新 decoder；这两种训练方式都已有先例，不把其中一种写成唯一正确方案。
- 若保存初始图像的外观 context，应将它作为明确的模型输入和存储成本。目标是减小运动建模成本，不必要求紧凑 motion code 独自保存全部纹理。
- Renderer 的外观 context 与运动输入要能分别替换，以检查输出到底是否响应预测状态。

### 3.3 Transition Posterior

Posterior 读取一个局部时间段起点及后续观测状态，产生该段、该对象候选的连续 effect。输入包含当前状态和上下文，不能只将两个对象的几何差命名成完整动作。

- 将旧 `[query,4,64]` 整段编码替换为显式带时序的 `[segment,query,effect]` 序列。
- 第一版使用明确时间间隔和共享局部编码器；自动事件分段是独立的后续选择，不作为必须先解决的新模块。
- 各 effect 在自身源状态上定义。跨对象使用时，由 recipient state 和交互上下文决定具体后果。
- 学习随机 posterior 可以提供有限信息预算，但 Gaussian/VQ/连续 deterministic 的名称本身不构成语义证据。

### 3.4 Shared Dynamics

Dynamics 接收当前对象状态、该段 effects 和相对时间，输出下一段状态。先进行对象局部更新，再通过共享交互模块传播接触或关系影响；具体 attention 实现以已有代码和训练效率为基础。

长序列顺序执行各段 effect。训练包含真实 source 的局部预测与预测 source 的多步 rollout，使网络学会在自身输出上继续运行。不同次序可以产生不同结果，不强制 `z1+z2=z2+z1`。

### 3.5 Conditional Effect Generator

读取历史对象状态、历史变化与语言，预测下一段或多段 effects。采用连续分布建模时，flow 是候选实现而非贡献本身。

不把 V70 的全量 Qwen4B 微调当作起点。先使用可复用的语言/视觉条件模块与 effect generator；模块大小由序列任务需要决定。Posterior 仍用于训练标签，生成链路由 prior 自己推进。

## 4. 要训练什么

| 工作包 | 训练模块 | 核心目标 | 交付结果 |
| --- | --- | --- | --- |
| T0 可成像状态 | object encoder、空间映射/renderer；预训练视觉部分按选定方案参与 | 观测视频重建、空间与时间信息保留 | 可解码状态与视频 |
| T1 变化模型 | posterior、Dynamics，并与 T0 主干联合优化 | 给定真实变化编码的局部及多步重建 | posterior 条件的视频 rollout |
| T2 条件生成 | history/language conditioner、effect prior | 预测 T1 定义的局部变化分布 | 不读真实未来的条件生成 |
| T3 长序列训练 | 以自身预测状态为输入的 Dynamics/prior；必要时联合微调 | 多段递推、顺序和条件响应 | 长序列与新组合生成 |

T0 是优化准备，不要求单独进行一次昂贵长训。T0/T1 可在同一训练课程内先建立可成像状态，再增加 rollout。T2 初期固定变化表示，避免目标不停漂移；T3 才研究联合调整是否带来收益。没有按旧版本的任意分数设置启动门禁。

### 4.1 目标函数只保留直接服务主任务的部分

1. **观测与 rollout 重建**：预测输出与真实 RGB/已匹配视觉空间的差距，训练成像与变化链路。
2. **动态信息监督**：如果已有轨迹/flow 确实提供有效运动证据，作为空间变化的辅助监督；不把老师预测自动升级为真实物体标签。
3. **变化信息预算**：限制 effect 把目标画面原样搬过去；预算与还原质量一起报告。
4. **条件分布学习**：历史和语言到 effects 的生成损失，最终通过同一 Dynamics 评价。

不先增加大量 identity、orthogonality、prototype、cycle 和 ranking 项。若某个项承担论文机制，必须明确其输入、作用和相应实验，而不是用指标名替代解释。

## 5. 数据工作包

### D0 复用现有资产

保留原始 RGB、episode/camera/time 映射、语言来源、视频解码与分布式训练基础设施。旧 CoTracker 点可用于辅助训练或诊断；不要求重新为全部视频做一次 segmentation/tracking。

旧 checkpoint 用作参考模型，而不是新版必须恢复的初始化。当前已知数据集合曾为 17,816 个 clips/episodes，语言连接 17,096 条；这些是历史审计记录，不是本轮重新读取服务器后的数量。

### D1 受控组合实验

构造或选用能够独立控制对象、初始布局和变化序列的视频。训练与测试的区别要直接对应论文问题：新对象-变化组合、相同变化的新布局、已见局部变化的新顺序。

主体区分与顺序区分各有对照。例如同时存在 A/B 时只要求 A 移动；先推再挡与先挡再推可以不同。模拟器真值提供评价，不要求主训练必须读取它。

### D2 真实视频实验

优先检查 Sketchy 的共同重建协议，以及 LanguageTable/Bridge 的条件生成协议。项目多源视频用于扩展，而不是在论文机制未明确前直接用更大数据掩盖结构差异。

长视频保留自然时间与完整阶段；不要将 episode 级指令冒充每一个短窗口的精确子任务。语言不足以唯一指定未来的样本，用条件分布评价，不以单条录制路径定义唯一正确答案。

### D3 数据输出

统一视频、真实时间、初始/目标条件、训练划分及组合类别。可视化展示真正进入训练的样本和预测，不另做一套较好看的采样。

## 6. 主实验与对照

| 实验 | 核心问题 | 对照与观察 |
| --- | --- | --- |
| E0 可成像性与变化还原 | 状态及 effect 是否足够解释观测变化 | 观察 State、posterior rollout、静态复制分别显示 |
| E1 对象结构收益 | 对象组织是否比整体表示有用 | 相同输入与训练预算的 object model 和 non-object temporal model |
| E2 时间组织收益 | 分段是否改善主体/阶段混合 | 同容量整段 effect 与 transition sequence |
| E3 组合泛化主实验 | 新对象、布局与次序能否正确生成 | PlaySlot、LPWM、直接重合方法与我们的同任务比较 |
| E4 条件生成 | 历史和语言是否能选择正确变化 | 相同场景下不同合法指令及 effect samples |
| E5 成本与质量 | 紧凑 Dynamics 是否有实际收益 | 相同质量下的吞吐、显存、rollout 与完整编码-生成成本 |

E1/E2 是解释性消融，不能代替 E3/E4 主实验。E0 不是只看训练集重建。SSv2 分类和 Physion 接触可作为补充用途，不预设为主结果。

主要输出：标准视频质量、对象/点运动、发生时间与顺序、目标关系、计算成本。逐例错误先分类，再做来源/时长/运动幅度汇总。FVD 衡量分布，不单独证明语言遵循；PSNR/LPIPS 也不单独证明物理正确。

未来 posterior、语言 prior、真实 action 条件的结果分开。原方法没有提供某种输入接口时，不把另加适配器后的系统当作原论文原样结果。

## 7. 代码实施工作包

1. 统一视频序列与实验划分接口，复用旧 loader 与 camera/time 元数据。
2. 实现可成像 object state 与 renderer，移除旧 root 永不更新的硬限制。
3. 实现局部 posterior、effect sequence 和共享递推 Dynamics。
4. 实现联合重建/rollout 训练与冻结阶段切换，复用保存、恢复、训练日志。
5. 实现 history/language effect prior，复用已经可用的条件编码与连续生成模块。
6. 接入 matched non-object baseline、PlaySlot/LPWM 的可比评测入口。
7. 输出同一 case 的观测重建、posterior rollout、prior generation、干预结果和耗时。

本轮只做计划与文献审查。具体层数、latent 大小、训练步数和设备配置在选定主干后给出，不把尚未测量的默认值当作性能事实。

## 8. 六组调研的证据格式

每篇检查：原始论文链接与版本；方法输入输出；更新/冻结范围；数据与划分；测试是否读真实未来；具体实验、表号与数字；作者主张与结果实际能支持的区别；与本项目的重合。

| 组 | Agent | 范围 | 当前状态 |
| --- | --- | --- | --- |
| A | Sol / Boyle | PlaySlot、LPWM、DDLP、SlotFormer、OCVP、Object-Centric Latent Action Learning | 6 篇方法/实验深读已返回 |
| B | Sol / Confucius | C-SWM、OP3、G-SWM、COBRA、SCOFF、HOWM | 6 篇方法/实验深读已返回 |
| C | Sol / Leibniz | LAPA、LAPO、AdaWorld、RepWAM、Genie、CADDY | 6 篇方法/实验深读已返回 |
| D | Sol / Franklin | TextOCVP、HWM、VTA、LOVE、CompILE、KeyIn | 6 篇方法/实验深读已返回 |
| E | Sol / Erdos | SAVi、STEVE、VideoSAUR、DINOSAUR、REPA、RAE | 6 篇方法/实验深读已返回 |
| F | Luna / Raman | DynaMo、DINO-WM、V-JEPA2、TRAJAN、Physion、Kubric | 6 篇方法/实验深读已返回 |
| 主线程 | 本线程 | WLA、DiLA、DLAM；整合模型与训练计划 | 原文方法/实验及部分代码已核对 |

六组共 36 篇（含对象组补充的直接同题工作），主线程另核对 WLA、DiLA、DLAM，共 39 篇方法/实验审查；LaMo 另列为部分阅读，不计入 39 篇。不是 39 篇代码复现，也不是对全部文献的新颖性保证。主线程复查直接重合论文及关键条件，逐篇条目保留 primary-source 链接与实验位置。

## 9. 主线程已核对的直接重合

### P1. WLA: Inter-environmental World Modeling for Continuous and Compositional Dynamics (2025)

[原文](https://arxiv.org/html/2503.09911)。方法 §3-4：ViT slots 与逐 slot RGB/alpha decoder；将变化参数化为 Lie 旋转/缩放算子，联合前后向重建与稀疏目标。真实 action 到变化算子的 controller adapter 另行训练。

实验 §6：PHYRE 的 1 FPS 到 8 FPS 插值与 effect 相加主要由 Figures 3-4 展示。ProcGen Table 1 的 unseen ActionACC 为 14.62，对照 Genie 实现为 8.30；seen 为 21.07 对 10.25。Android Table 3 的 FVD 为 131.02 对 393.85，但 PSNR 为 20.82 对 21.16，并非所有指标都更好。

判断：对象级连续可组合变化已经是其明确方法目标。它假设变换可交换，组合示例不等于已经解决顺序敏感、接触依赖的操作。控制适配使用真实 action；Genie 对照是论文采用的开源实现，不是自动等同官方大规模模型。

### P2. DiLA: Disentangled Latent Action World Models (2026)

[原文](https://arxiv.org/html/2605.15725)、[官方实现](https://github.com/senngadaisuki/disentangled-latent-action-world-models)。§3：冻结 DINOv2/RAE；structure 路径预测变化，Mamba content memory 保存外观，fusion decoder 读取初始视觉特征，训练目标在 DINO 特征空间，不是 RGB 联训。30k teacher-forcing 后加 1k latent rollout。

Table 1：RT-1 LPIPS 0.206，对比 AdaWorld 0.429；SSv2 LPIPS 0.356 对 0.521，但 SSIM 0.660 低于 AdaWorld 0.674。Table 2：cycle-transfer LPIPS 0.343 对无 content 的 0.451。无 content 变体 646.17M trainable，完整模型约 123M；不是严格同参数消融。

判断：外观/变化分离与跨场景 effect reuse 有直接重合。cycle transfer 返回原视频，不要求目标场景的独立真实后果，因此不是正确反事实的证明。§5 还讨论了把 throwing 转移到空夹爪时可能把夹爪本身当成被抛物体，说明主体与可执行上下文仍是实际问题。

代码/文字差异：附录 D.2 描述按真实环境误差选轨迹；公开 `mpc/mppi.py` 使用 model predictions 计算 objective，`run_control.py` 记录执行 score。仅凭这两处不能确认主表实际采用哪种挑选协议，不据此宣称规划优势已完全可比。

### P3. DLAM: Distributional Latent Actions with Temporal Constraints (2026)

[原文](https://arxiv.org/html/2607.27138)。Methods：帧对编码 diagonal-Gaussian transition，source-conditioned decoder 使用 mean 重建；等间隔三帧上约束 normalized composition 与 reversal。实验用 11 个视频源及受控同容量 variants；decoder 后续丢弃，mean 序列作为 policy 联合生成目标。

Table 3：no temporal relations / mean-only / full 的 cumulative PSNR 分别为 21.236 / 22.109 / 22.400，MetaWorld macro success 为 76.6 / 82.1 / 87.6。Fig.5 与正文报告 3k-5k 跨度 direct/cumulative PSNR 29.14/22.40。说明时间关系目标有收益，不能把全部收益归于 variance。

判断：时间组合约束不是新点。其 common probe 使用与训练相同的长度归一化求和，正文明确 h>2 测的是 per-length consistency 而非 associativity；它不验证对象绑定或交互次序。variance 未进入 policy，也没有被证明是校准的不确定性。

### P4. LaMo: A Latent Motion World Model for Long-Horizon Prediction (2026 workshop; 部分核对)

[原始论文入口](https://openreview.net/forum?id=Pz9sdChr10)。OpenReview 直接正文访问返回浏览器验证页；搜索索引中的原始 PDF 已提供 Methods §4 与 Experiments §5，但未获取完整 PDF，因此本项只计为部分阅读。

可核对内容：冻结 Midway encoder/IDM/FDM，使用当前 state 与三步 motion history 预测下一 motion token；flow/GMM 两种分布头，再由独立训练的 decoder 成像。BDD100K 的 matched state-prediction baseline 与 motion-history model 使用相同 Transformer 容量；正文报告 1.6s FID 从 227 到 188。正文不同位置还出现 1.2s/1.6s 的 horizon 表述差异，需复现时确认。

判断：预测 motion tokens 而非重复预测 dense states 的主张已经存在。当前证据不涉及对象绑定、语言或复杂交互组合，不能据此说这些方向已经被解决。

## 10. 并行调研证据矩阵

本节数值都是论文报告，不是本项目重新运行的分数。每篇后面的判断限定于所列实验，不照搬其 abstract 的概括。

### A. 对象世界模型

**A1. PlaySlot (ICML 2025)** — [论文](https://arxiv.org/html/2502.07600v2)、[训练代码](https://github.com/angelvillar96/PlaySlot/blob/master/src/04_train_PlaySlot.py)。§3.2 既有 global InvDyn-S，也有共享 MLP 的 per-object InvDyn-M；latent action 是逐 transition 的离散 prototype 加连续 residual。先 RGB-MSE 联训 SAVi，再冻结 encoder/decoder，以 RGB、slot 和 VQ loss 联训 InvDyn/cOCVP；RGB 梯度穿过冻结 decoder。§4.2 的 6 帧 context、15 帧预测使用真实序列推断 actions，属于 posterior 条件重建。Table 1 BlockPush LPIPS 0.066 对 CADDY 0.090。GridShapes 1-5 物体分别训练，不能称未见数量外推。对象绑定时序 effect 已有，语言和严格事件组合不是该主表证明的内容。

**A2. LPWM (ICLR 2026)** — [论文](https://arxiv.org/html/2603.04553)、[代码](https://github.com/taldatech/lpwm)。§4 联训 encoder、RGBA decoder、Context、Dynamics；每粒子每步 continuous latent action，posterior 读取后续状态，prior 使用历史及可选 action/language/image goal。Table 2 LanguageTable 语言 prior FVD 15.96，对 DVAE 26.78；附录 Table 10 的 pixel metrics 使用 posterior，而非同一 prior 样本。A.4.4 粒子绑定局部网格、跨区域转交特征，并非稳定的物理实体身份。A.10.4 的短原子任务训练和更长 OGBench 任务有相关组合证据，但未隔离未见主体-动作绑定、次序及 onset。整体新架构主张与它高度重合。

**A3. DDLP (2023 preprint / TMLR 2024)** — [论文](https://arxiv.org/html/2306.05957)。§4-5 显式位置、尺度、深度次序、透明度、外观粒子，以可微跟踪维持对应；PINT 预测粒子分布，RGB 重建与动态 prior KL 端到端优化，没有单独 effect 序列。OBJ3D train/val/test=2920/200/200，给 6 帧续写 44 帧。Table 3 LPIPS 0.088，对重训 SlotFormer 0.135。初始位置编辑是定性干预，不等于系统组合留出。附录 F 显存 9GB 对 SlotFormer 第二阶段 22GB，但 GPU 推理 250ms 对 222ms，不能概括为所有成本更低。

**A4. SlotFormer (ICLR 2023)** — [论文](https://arxiv.org/html/2210.05861)。§3 预训练可渲染 slots 后冻结视觉模型，Transformer 自回归预测 states，不使用独立 effect。SAVi 设置用 slot 与 RGB-MSE；Physion 的 STEVE 设置不使用 RGB loss。CLEVRER 排除预测期新对象进入的序列。Table 2 FG-mIoU 29.81%，G-SWM 24.44%；Table 6 去掉 RGB loss 后 OBJ3D LPIPS 从 0.080 升到 0.093。已有可解码状态上的多步 dynamics，长时 rollout 不能自动证明新组合生成。

**A5. OCVP (ICIP 2023)** — [论文](https://arxiv.org/html/2302.11850)。同对象的 temporal attention 和同时间的 relational attention 分开，逐步预测 slots，无独立 effect。冻结已重建训练的 SAVi，通过 slot/RGB loss 训练 predictor。MOVi-A 9703/250 train/test，训练 6->8 帧、评估到 18 帧。Table 1 OCVP LPIPS 0.127，对普通 Transformer 0.134；Table 2 mIoU 0.511 对 0.508。对象-时间 attention 因子化是已有结构，增益不能被夸大为已证明高效的可组合程序。

**A6. Object-Centric Latent Action Learning (2025 preprint / AAAI 2026)** — [正式版](https://ojs.aaai.org/index.php/AAAI/article/view/39423/43384)、[扩展版 v3](https://arxiv.org/html/2502.09680v3)。先训练 VideoSAUR，用少量真实 action 的线性 probe 选择 slots，再固定表示，以未来 slot-MSE 联训 inverse/forward models，另有 masked RGB 变体。正文相邻步描述与附录 Table 6 的十步偏移应区分。下游 BC 以不超过2.5% action labels 适配。Table 1 DCS normalized return 0.24 对 LAPO0.13，DMW normalized success0.48 对0.31；不是直接的无action视频生成指标。正文报告机械臂与操作物可能同槽，未测独立实体 effect 或次序重组。官方代码本轮未读到，时间配置依据附录。这是“对象表示学局部 latent action”最直接的同题先例之一。

### B. 组合与规则复用

**B1. C-SWM (ICLR 2020)** — [论文](https://arxiv.org/pdf/1911.12247)。CNN object slots、共享编码器与 GNN 残差模型联合 contrastive training；逐环境步输入真实 action。decoder 在附录 A.2 单独拟合，只作像素评价。Table 1 的 2D Shapes 十步 H@1 99.9%，AE 6.5%；训练 1000x100 步，测试 10000x10 步，五个物体身份固定。它证明结构化 dynamics 的预测收益，不是未知对象/行为组合迁移。

**B2. OP3: Entity Abstraction in Visual Model-Based Reinforcement Learning (CoRL 2019)** — [论文](https://arxiv.org/html/1910.12827v2)。持续实体状态、对称交互、共享 RGB/mask decoder 联合训练，目标含重建与 posterior/prior KL。落块任务一步可对应落下直到稳定，不等同固定帧差；预测输入候选 action，规划输入目标图。附录 G 60000 条前后图、1-5 块训练，最多 9 块及新桥梁/金字塔测试；Table 1 成功率 OP3 82%、O2P2 76%、SAVP 24%。已证明一定数量/结构泛化，但未见新 dynamics 或新 action primitive 的证据。

**B3. G-SWM (ICML 2020)** — [论文](https://arxiv.org/pdf/2010.02054)、[实现](https://github.com/zhixuan-lin/G-SWM/blob/master/src/model/gswm/fg.py)。每对象有外观、位置、深度、存在与逐帧随机变量；对象 RNN、交互和背景形成历史 prior，以 ELBO 联训生成。给 10 帧生成 90 帧，不需 action/语言。Table 2 交互球前十步位置误差和 0.242，对 STOVE 0.411、SCALOR 1.004。3/6/16 球分别训练；生成时强制保留对象、关闭新对象发现，确定性球用潜变量均值。已覆盖历史条件的对象随机潜序列，不证明未知数量或事件序列泛化。

**B4. COBRA (2019)** — [论文](https://arxiv.org/pdf/1905.09275)。200万静态帧训练 MONet 后冻结；逐槽共享 MLP 学动作残差，通过冻结 decoder 的像素 loss 学 dynamics。任务阶段冻结世界模型，只学习奖励预测。Figure 4 多数任务使用 100-1000 有奖励交互步，而 MPO 为 1e6-1e8，但 COBRA 的预训练成本不在前一个数中。附录 E 有颜色对和目标/干扰物数量的组合留出；环境无物体碰撞，留出主要针对任务学习，不代表 dynamics 未见组合。

**B5. SCOFF: Object Files and Schemata / Factorizing Declarative and Procedural Knowledge in Structured, Dynamical Environments (ICLR 2021)** — [作者全文](https://arxiv.org/pdf/2006.16225)。Object files 维持时变状态，schemata 是共享 GRU 更新参数，每步通过 hard Gumbel 选择并通过 attention 通信。视频任务不需 action/goal，端到端解码。Table 1 IntPhys 消失异常相对分类错误 0.34，对 RMC 0.43；BabyAI 房间 6x6->10x10 的成功率 82%，同参数 GRU 74%。不同球数分别训练；有模式标记的切换实验不等于未见多事件程序。它直接覆盖“对象状态与可复用规则分离”，是必须讨论的概念先例。

**B6. HOWM: Toward Compositional Generalization in Object-Oriented World Modeling (ICML 2022)** — [论文](https://proceedings.mlr.press/v162/zhao22b/zhao22b.pdf)、[实现](https://github.com/linfeng-z/HOWM/blob/main/algorithms/homomorphic_wm.py)。Slot Attention、action-object binding 和共享 GNN，使用对齐对比目标；测试给初始图与离散 actions。每景 K=5，身份库 N=10/20/30，训练见过所有身份但留出组合。Table 1 N=20 五步 MRR：HOWM 75.1%、K-CSWM 32.2%、N-CSWM 99.9%；显存 3.7/1.5/8.1GB。结果展示性能-容量权衡，不是所有指标领先。论文联合训练表述与代码提供的解耦训练路径需在复现时区别。

### C. Latent Action 与条件生成

**C1. LAPA: Latent Action Pretraining From Videos (ICLR 2025)** — [论文](https://arxiv.org/html/2410.11758v2)、[代码](https://github.com/LatentActionPretraining/LAPA/blob/main/laq/laq_model/latent_action_quantization.py)。§3 与附录 A/F 的两帧 NSVQ 模型以 pixel L2 学转移；机器人时间窗口 0.6s、人类视频 2.4s，四个 tokens 共同编码一个帧间变化，不是四个事件。随后当前图与语言预测 latent labels；机器人适配会替换头并微调，不只是码到动作的轻量 decoder。Table 16 部分成功均分 50.09% 对 OpenVLA 43.87%，严格成功 35.19% 对 27.78%。§5.2 Figure 7 已有未做电机动作适配的语言闭环视频生成定性展示。因此语言->latent->原 decoder 路径不新，但未建立对象绑定时序组合。

**C2. LAPO: Learning to Act without Actions (ICLR 2024)** — [论文](https://arxiv.org/html/2312.10812v2)、[模型](https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py)。§4 IDM 读历史和下一帧，FDM 读历史与 VQ latent；下一帧 MSE 联训。后续 BC 预测量化前 continuous code。16 个 Procgen 环境分别训练，约 800万专家帧；Figure 3 的 400万交互步恢复专家水平，PPO 从零约 44%；Figure 4 少于 256 action 标签可达到该 PPO 水平。附录 A.2 无 VQ 重建更好而策略更差；§6.4 承认可见效果的延迟会导致 latent/action 时间错位。不能由 posterior 重建好推断 code 已可用于部署。

**C3. AdaWorld (ICML 2025)** — [论文](https://arxiv.org/html/2503.18938v4)、[latent 模型](https://github.com/Little-Podi/AdaWorld/blob/main/lam/lam/modules/lam.py)。32维 continuous beta-VAE 用帧对 pixel reconstruction+KL 学 action；固定编码器后另训 SVD 初始化的 video diffusion world model，最多六帧历史，不是同一粗 decoder 直接生成。约20亿帧预训练；Table 1 的配对迁移中 LIBERO FVD 767.0，对 discrete code 1504.5，人工成功率 70.5% 对 3.5%。输入是参考动作视频和目标初帧。Figure 5 已展示 latent 平均组合，20步序列迁移也存在，但不是仅历史+语言的新对象时序组合测试。

**C4. RepWAM (2026)** — [论文](https://arxiv.org/html/2606.13674v2)、[官方仓库](https://github.com/wdrink/RepWAM)。§3 先以 L1/perceptual/GAN/冻结视觉教师对齐训练 visual tokenizer，再固定它，以相邻潜帧推断四维 continuous transition，transport matrix+residual 预测状态并作正反向重建。没有 VQ/KL；视觉时间压缩四倍。语言模型 flow matching 联合生成 visual/action chunks。Table 4 两阶段训练 unseen-task gFVD 58.83，对无 latent action 72.91；PickFruit 50% 对 30%。其 transition 是全图级，不是每个对象。公开仓库内容有限，后续冻结范围不能仅凭 README 补全。可解码语义状态+latent transition+语言生成这一整体结构已经存在。

**C5. Genie: Generative Interactive Environments (ICML 2024)** — [论文](https://arxiv.org/html/2402.15391)。先训练 video VQ tokenizer，再训练读取下一帧的 latent action model 与 MaskGIT dynamics，动作码在 dynamics 路径 stop-gradient。8类码，逐帧、10Hz；约3万小时平台游戏视频。生成输入初图和用户选择的 codes，不是语言 prior。Table 2 pixel/token 两种输入的 delta-PSNR 为 1.91/1.33，而 FVD 为 40.1/38.8，说明控制响应与画质不同。论文报告约1FPS、16帧记忆，不能仅凭 latent 接口宣称高效。

**C6. CADDY / Playable Video Generation (CVPR 2021)** — [论文](https://arxiv.org/html/2101.12195)、[补充材料](https://willi-menapace.github.io/playable-video-generation-website/supplementary.pdf)、[代码](https://github.com/willi-menapace/PlayableVideoGeneration/blob/main/model/main_model/model.py)。联合 frame state、ConvLSTM dynamics、decoder；逐步 Gumbel action 和 continuous variability，重建/感知/feature/互信息/KL 目标。短重建 warm-up 后联合训练。Table 2 BAIR action-displacement error 54.8，对 SAVP+ 82.0，但 FVD 423 差于303。量化重建使用测试视频反演的 actions；交互才用人工 codes，variability 默认为零。已有 playable latent dynamics，不涉及多对象身份绑定或语言 prior。

### D. 时间抽象与语言

**D1. TextOCVP (2025 preprint / TMLR 2026)** — [论文 v3](https://arxiv.org/html/2502.11655v3)、[训练代码](https://github.com/angelvillar96/TextOCVP/blob/master/src/04_train_predictor.py)。对象模块先以 RGB/DINO 重建训练，再冻结，以 slot+RGB loss 学文本条件逐帧 predictor。测试一帧+文本，无真实未来。CATER-hard 30k、CLIPort 21k；Table 1 CATER 九帧 LPIPS 0.036 对 MAGE 0.108，Table 6 CLIPort 九帧 109.9ms 对198.8ms。CATER 两个动作同时开始，目标颜色留出不代表干扰物中从未出现该颜色。语言对象生成与一定效率已经有直接证据，但未见事件顺序留出不同。

**D2. Hierarchical Planning with Latent World Models (CoRL 2026; v3 2026-10-12)** — [论文](https://arxiv.org/html/2604.03208v3)。多时间尺度共用表征但有不同 Dynamics；真实 action chunks 编为宏动作。Franka/Push-T 随机路标、Maze 固定间隔，不是视频自发现事件边界。冻结视觉模型，latent 预测训练，RGB decoder 另训展示。测试当前与目标观测做动作搜索；Table 1 杯子搬运成功率 70% 对平层0%，Table 2 长时 Push-T 61% 对17%。时间抽象提高规划效率已有实测，但不是无 action、无未来的对象事件序列生成。

**D3. VTA: Variational Temporal Abstraction (NeurIPS 2019)** — [论文](https://arxiv.org/html/1910.00775)。HRSSM 学逐帧 state、片段 Gaussian latent 和 binary boundaries，重建+KL 联训 encoder/dynamics/decoder；有最大片段数/长度先验，去掉会退化成普通 RSSM。球视频无 action，迷宫实验有 action。Figure 3 展示碰撞分段；Figure 8 的5000回合导航以相同计算预算优于RSSM，但导航另使用目标附近3x3地图图像。已覆盖像素自发现时间结构，不覆盖对象绑定程序和组合留出。

**D4. LOVE: Learning Options via Compression (NeurIPS 2022)** — [论文](https://arxiv.org/html/2212.04590)、[实现](https://github.com/yidingjiang/love/blob/main/hssm_rl.py)。离散 skill 与可变边界，加序列描述长度目标。控制学习输入完整 observation-action trajectories，解码 actions，不是可渲染世界模型；还设置最短技能长度。二维从三目标示范到五目标，三维从两目标到四目标，但高层 RL 重新训练。Table 2 边界 F1 0.92，对 action-VTA 0.32、加最短长度0.36。压缩能帮助发现可用分段，但不是无 action 视频或零样本语言生成。

**D5. CompILE (ICML 2019)** — [论文](https://arxiv.org/html/1812.01483)。软分段、每段 latent 和 state-conditioned action policy，ELBO 联训。无边界标签但实验提供正确片段数；输入语义网格或机器人状态及 actions，不是 RGB。三任务训练，1024个新环境实例上测三/五任务；Table 1 五任务边界 F1 69.3%，LSTM surprisal 44.9%。模仿测试先看完整示范反演 skill，新任务 RL 另训高层。附录 E 指出 latent 倾向编码位置而非对象类型，不能据此证明主体绑定。

**D6. KeyIn: Keyframing the Future (L4DC 2020)** — [论文](https://arxiv.org/html/1904.05869)。预测 keyframes 与时间间隔，再插值视频；关键帧数预设。先训练插帧器再冻结，冻结模块保留上游梯度；posterior 读未来，生成从历史与 Gaussian sample 开始。20k grid/50k pushing videos；Table 1 推物事件 F1 0.43，对 Surprise 0.17；Table 2 规划64.2%，Jumpy58.8%，flat15%，但低层控制使用真实模拟器 Dynamics。时间位置学习已做过，不等于未知对象-事件程序的生成能力。

TextOCVP 的额外核对：附录 Table 7 的 CLIPort 19 帧 LPIPS 为 TextOCVP 0.078、MAGE 0.075，并非所有设置都领先；Table 11 只更换更强 renderer 后九帧 LPIPS 从 0.062 到 0.043。这支持将 Dynamics 与 renderer 的收益分开解释，而不是把更清晰的视频全部归功于动态表征。

### E. 表征与 Decoder

**E1. SAVi: Conditional Object-Centric Learning from Video (ICLR 2022)** — [论文](https://arxiv.org/pdf/2111.12594)。CNN+Slot Attention corrector 读取当前真帧，Transformer predictor 推进槽；共享 spatial-broadcast decoder。主设置监督 flow RGB，RGB 重建是另一设置；任务模块联合训练。条件测试给首帧框/点/分割提示，并持续读取后续图像。Table 1b MOVi++ 框条件 FG-ARI77.4/mIoU45.9，对传播基线37.8/33.3。长视频跟踪不是只看历史的未来预测；Figure5b 仍出现遮挡后交换。

**E2. STEVE (NeurIPS 2022)** — [论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf)、[实现](https://github.com/singhgautam/steve/blob/master/steve.py)。递归 slots 经 Transformer 预测同一帧的 dVAE tokens，dVAE 自身 RGB-MSE 与 slots 的 token-CE 分开；hard token target detach，RGB MSE 不直接沿完整 slot 路径回传。测试仍逐帧读真实图像。§5.3 MOVi-D/E FG-ARI47.67/52.15，对 flow-SAVi19.4/2.7。对象数/纹理外推是分解证据，不是 effects 或未来程序组合。

**E3. VideoSAUR (NeurIPS 2023)** — [论文](https://arxiv.org/html/2306.04829v2)。冻结 DINO，训练递归 slots、predictor 与 SlotMixer；当前 feature reconstruction 加跨帧 patch similarity CE，无 RGB loss。未来只用于制训练目标，测试仍逐帧读RGB。Table 3 YT-VIS：仅feature FG-ARI35.4，加下一帧feature37.9，加temporal similarity39.5。支持时间目标改善分组，不证明未来生成或精确物理实体识别。

**E4. DINOSAUR (ICLR 2023)** — [论文](https://arxiv.org/html/2209.14860v2)、[decoder](https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/decoding.py)。冻结 DINO，Slot Attention 与 MLP/AR decoder 重建 patch features，不重建RGB，不建模视频时间。Table 3 COCO MLP->Transformer：MSE0.31->0.16，FG-ARI40.5->34.1，instance mBO27.7->31.6。重建与分组不是同一个指标。AR feature decoder 读取右移真 target features；不能把它的 teacher-forced reconstruction 当纯 slots 的自由解码，MLP版本没有这条路径。

**E5. REPA (ICLR 2025)** — [论文](https://arxiv.org/html/2410.06940v3)。固定 KL-VAE 和视觉 teacher，训练 DiT/SiT 与 projection，把带噪 latent 的中间特征对齐干净图像特征，并优化 diffusion/velocity objective。无 object slots；这里 t 是噪声时间。Table 3 SiT-XL/2 400k updates 无CFG FID17.2->7.9，baseline7M为8.3。17.5倍是相近质量的更新数比较，不是机器人世界模型端到端加速；semantic alignment 促进生成不是新点。

**E6. RAE: Diffusion Transformers with Representation Autoencoders (2025 preprint / ICLR 2026)** — [论文](https://arxiv.org/html/2510.11690v1)、[Stage1代码](https://github.com/bytetriper/RAE/blob/main/src/train_stage1.py)、[Stage2代码](https://github.com/bytetriper/RAE/blob/main/src/train.py)。冻结 DINOv2/SigLIP2/MAE，训练 RGB decoder 的L1/LPIPS/GAN，再固定 autoencoder 训练 flow generation。Table1 DINOv2-B rFID0.49，对SD-VAE0.62；Table5 decoder加噪使gFID4.81->4.28而rFID0.49->0.57。冻结语义特征可以成像，但它保留256个高维patch tokens，不能直接推断压成少量对象后仍可成像。

### F. 表征用途、评测与数据

**F1. DynaMo (NeurIPS 2024)** — [论文](https://arxiv.org/abs/2409.12192)、[代码](https://github.com/jeffacce/dynamo_ssl)。无 action 的域内演示联合训练视觉 encoder、inverse/forward dynamics；不用 RGB decoder。冻结 encoder 后训练 imitation policy。Table 1 Push-T 覆盖率 0.66，Block Pushing 平均入框数 0.65/2，不是 65% 成功率。它主要证明控制用途，不是对象绑定、语言 future generation 或通用分类能力。

**F2. DINO-WM (ICLR 2025)** — [论文](https://arxiv.org/abs/2411.04983)、[代码](https://github.com/gaoyuezhou/dino_wm)。冻结 DINOv2 patch encoder，使用离线 RGB-action 轨迹学习未来 patch features，decoder 独立可选。推理给当前/目标图与候选 actions，用 CEM 规划。Table 3 未见配置 WallRandom 成功率 0.82、PushObj 0.34；Table 4 报告独立 decoder 的 LPIPS。它不是无 action 的新对象生成方法；适合 patch Dynamics 和下游规划参照。

**F3. V-JEPA 2 (2025)** — [论文](https://arxiv.org/abs/2506.09985)、[代码](https://github.com/facebookresearch/vjepa2)。基础模型做 masked-video feature learning；2-AC 冻结 encoder，在 Droid RGB、state、action 上训练约 300M predictor，latent teacher-forcing 与 rollout loss。推理是图像目标 CEM 规划。Table 3 规划时间 16 秒对 Cosmos 4 分钟，但候选数分别 800/80，不能当单候选同任务成本比较。附录 RGB decoder 是另训诊断工具，不与 Dynamics 联合训练。与我们的无 action object effects 不是相同输入问题。

**F4. TRAJAN: Direct Motion Models for Assessing Generated Videos (ICML 2025)** — [论文](https://arxiv.org/abs/2505.00209)、[项目](https://trajan-paper.github.io/)。从 BootsTAPIR 全视频 tracks 编码运动，由其他时空 query 重建轨迹和遮挡；不是从历史预测未来 RGB。Table 2 与部分人工运动评分的 Spearman 相关最高约 0.32，interaction 项约 0.09。它可辅助检测时空不一致，不是 motion GT，也不自动判断对象语义或条件遵循。

**F5. Physion (NeurIPS 2021 Datasets/Benchmarks)** — [论文](https://arxiv.org/abs/2106.08261)、[数据](https://github.com/cogtoolslab/physics-benchmarking-neurips2021)。指定对象对、观察前 1.5s，预测后续是否接触；包含 observed、observed+simulated 及全视频检测等不同信息条件。冻结表征后拟合场景级 readout，Figure 5 人类均值约 0.71。物理粒子输入模型与 RGB-only 模型不能混比。它测物理后果，不直接测 RGB 成像或语言条件变化。

**F6. Kubric / MOVi (CVPR 2022)** — [论文](https://arxiv.org/abs/2203.03570)、[数据](https://github.com/google-research/kubric)。模拟器生成 RGB、实例分割、flow、depth 与状态真值；原 Table 3 是 object decomposition 而非 world-model rollout。SAVi FG-ARI 在 MOVi-A/E 为 82.0/2.7，SIMONe 在 E 为 34.9；SAVi 加首帧框在 D/E 仍只有 9.9/7.5。可以用于独立结构/运动评价，但这些数字不是因果或组合生成结论。

## 11. 调研之后的最终研究取舍

### 11.1 不再作为贡献的部分

“objects + latent actions + shared dynamics + renderer + language”已经由多项工作共同或直接覆盖。仅把旧五秒 effect 改成逐帧/逐段 effects，属于修正项目，不足以构成新论文。联合 RGB 训练、冻语义特征再成像、可变时间抽象，也都不是空白。

### 11.2 候选核心机制：对象绑定的可变持续时间转移

比普通逐帧潜码更具体的研究对象，是：**哪个实体或实体关系在何时切换到哪种变化，并持续多久。** 对象对应与时间切换由视频证据一起学习，Dynamics 根据当前状态与交互决定后果。

- 主体绑定以当前对象状态为参照，不用固定 slot 编号作语义。
- 同一 effect 在一个时间段内持续使用，新的 evidence/condition 决定更新；对不同实体可以有不同更新时刻。
- effect 的切换不等同于物体停止：惯性、持续运动与被其他物体影响由状态及 Dynamics 继续表达。
- 顺序由 state rollout 保留。先推后挡与先挡后推不必相同；不把相加、反向或cycle自洽当作必然物理规律。

这不是宣称事件模型是首次提出。与 LPWM 的逐粒子逐帧 action、SCOFF 的共享参数规则、VTA/LOVE 的场景级时间抽象相比，需检验的增量是：**主体和时间一起组织，是否改善实际无未来输入的变化生成。** 文献调研支持把它作为候选问题，不证明贡献已经成立。

### 11.3 主干实现分成基础版本和论文变量

**基础版本**：可解码对象状态 + 逐步连续 effects + 交互 Dynamics + history/language prior。这部分优先复用成熟实现，不以其本身作为创新。

**论文变量**：把均匀逐步 effects 变成对象绑定的持续片段，学习更新边界/持续时间及选择内容。时间压缩有描述长度或更新代价，但不是让 latent 无条件保持不变；观测与 rollout 质量决定哪些变化必须保留。VTA/LOVE 为边界与压缩提供设计参考，不直接搬用其真实 action 输入。

原型接口可以写成：对象状态 $S_t=[s_t^1,\ldots,s_t^K]$；每个对象的二元更新变量 $b_t^k$ 决定是产生新 effect，还是沿用上一个 effect。训练期 posterior 用该段视频推断 $b,z$；生成期 prior 用历史状态与语言预测它们。共享 Dynamics 计算 $S_{t+1}=F(S_t,z_t,\Delta t)$，Renderer 输出 $\hat I_{t+1}=R(S_{t+1})$。这里 $z_t$ 持续不变不意味着 $S_t$ 不变。

代码张量以 `states[B,T,K,R,D]`、`update[B,T,K]`、`effects[B,T,K,Dz]` 表达，段序列是更新位置的紧凑存储；不先锁定 K/R/D/Dz。第一版让 Dynamics 按观测时间推进，压缩的是 effect 更新与 prior 预测次数。若 Dynamics 仍每帧运行，就不能把 effect 数减少直接写成整体等比例加速。

第一轮做三个相同主干的版本：

1. **全局逐步 effects**：没有对象绑定的变化组织。
2. **对象逐步 effects**：保留主体结构，但固定更新时间。
3. **对象片段 effects**：同时学习主体绑定与更新时间/持续时间。

这三个版本区分空间结构、时间结构的作用；外部 LPWM/TextOCVP 等提供已有方法比较。不要把三个消融当成全部主实验。

### 11.4 最小训练安排

1. **状态与变化共同训练**：用已匹配的预训练视觉特征/成像组件建立对象压缩，再同时优化观察重建、posterior rollout 和 effect 信息预算。若借用冻结 image decoder，新对象映射和 Dynamics 仍通过相应目标学习。
2. **加入自身预测状态的多步训练**：延长 sequence，比较逐步与对象片段更新，重建不仅在真实中间state上计算。
3. **条件 prior 训练**：使用同一表示学习从历史/语言选择主体、更新时刻与effect内容；先固定标签模型以明确目标，再研究短程联合微调，而不是一开始重训4B语言模型。
4. **正式共同任务训练**：对齐数据、条件输入和预测长度，训练/适配主baseline，在组合与真实数据上形成论文主表。

纯 RGB、feature+RGB、已匹配 RAE 可解码空间是成像方案选择，不同时作为三条新研究主线。依据可复用组件与最初完整训练结果选一个，其他仅在定位具体问题时比较。

### 11.5 主实验收敛为两部分

**受控主体-时序组合**：优先基于 CLIPort 式操作环境构建明确的对象角色、事件顺序与时间安排留出，使用环境生成视频及真实后果；CATER 作为低成本机制开发选项，不再作为另一条必做主实验。原 TextOCVP CATER 同时开始的两动作协议不能直接检验新先后顺序，扩展后的结果应明确标为新协议。训练见过局部变化，测试留出其组合，而不是由我们自己的 decoder/teacher 定义正确后果。

**真实机器人视频**：优先 LanguageTable 的语言条件生成，与 LPWM 对齐。Sketchy 作为条件重建的补充而非再扩一条主线。当前多源数据后续用于扩大场景覆盖；不要把它的规模当成机制成立的替代品。

必须能看到的输出很简单：目标物体是否正确、事件是否按正确顺序发生、运动是否准确、图像是否保持合理，以及同质量下花费多少计算。标准视频指标与实体/事件指标互补，不用更多指标掩盖核心失败。

### 11.6 Baseline 的角色

| 方法 | 在论文中的作用 |
| --- | --- |
| LPWM | 完整 object/particle latent-action world model 主对照 |
| TextOCVP | 相同初始视频/语言条件下的对象视频生成主对照 |
| PlaySlot InvDyn-M | per-object posterior effect 与条件重建参照 |
| RepWAM | 语义 visual/action tokenizer 与语言生成的重要结构相关工作；代码可用范围决定是否直接复现 |
| DiLA / WLA / DLAM | 分别检验跨场景迁移、变化算子组合、时间约束的已有证据与区别 |
| SCOFF / VTA / LOVE / HOWM | 机制及组合划分依据，不把需要action的系统强行当无action生成baseline |
| 同规模 non-object model | 空间结构的内部对照，匹配训练输入与目标 |

### 11.7 整体交付与依赖顺序

文献证据与论文命题 -> 数据/组合协议和模型接口 -> 基础可解码状态与Dynamics -> 对象片段effect机制 -> 历史/语言prior -> 正式外部比较与成本结果 -> 论文图表。

实现时保留视频读取、分布式运行、保存恢复与日志等已经能工作的基础设施。新模型单独实现，不把几十版旧损失、teacher审计或启动门禁一并迁入。当前不改变正在使用的模型文件。

## 12. 本轮交付状态

- [x] 给出模型、训练、数据、实验和实现的整体计划。
- [x] 启动五个 Sol 与一个 Luna，限定互不重复的调研范围。
- [x] 核对三个直接重合工作的方法与实验，并标记一个部分阅读工作。
- [x] 整合六组 36 篇逐篇证据，主线程另核对 3 篇直接先例及 1 篇部分正文。
- [x] 将论文命题收敛为已知基础与待检验的主体-时序组合能力，不宣称已确认全领域空白。
- [x] 给出主干实施顺序、训练工作包、实验与 baseline 角色。

本轮未修改模型代码、未提交训练任务、未声称已经证明创新或能力。

## 13. 实施启动：数据与平台锚定（2026-10-15）

本节是用户批准实施后的新记录，不将第12节的调研交付误称为模型完成。Codex active goal 已建立，目标覆盖数据锚定、主干实现、远端真实数据试验、账本维护与大规模运行交付。

### 13.1 运行位置

通过 SWXC 的「等待其他机器召唤」对话实际访问：thread `01a13dca-00cb-7c80-82ac-e61d655f9585`，主线程路由 host 为 `remote-control:env_e_6a2b6ece3898832eb9572cdf504d14cd`。该对话中的 `local` 指 SWXC 自身，不是本主线程的 Mac。

- 阿里：`ssh -p 22 root@10.67.16.10`，hostname `dsw-533922-6c5d8b7884-x7fnz`；1张空闲 H20，97871 MiB 显存；系统 Python3.12.3、torch2.7.0+cu126、CUDA可用。
- 百度补充：`ssh -p 8732 root@10.66.0.39`，已成功只读访问旧数据及语言产物。
- 阿里工作根：`/mnt/pfs/xuhaoming/wm-align`。`/mnt/pfs/xuhaoming` 是 rw 的 `fuse.aliyun-alinas-efc` 独立挂载，实测可用37597346791424 bytes，约35TiB。`/mnt/pfs` 本身只是overlay目录，不是该CPFS挂载。
- 源分片只读：`/mnt/data/baidu/sync_data/wm-align/object_video_v69_portable.parts`，为OSS挂载；不能用其512TiB的df显示代替PFS可用量。
- 两台Mac只负责源码维护和远程控制。数据、venv、模型权重、HF/torch/uv缓存、临时文件、训练与评测产物全部放阿里PFS。

容器根盘约281GiB可用只是一项环境事实。此前据它建议抽取子集的存储方案已取消，改为在CPFS完整展开现有portable包。

### 13.2 数据事实与能支持的实验

阿里目前有26个tar分片，合计864440913920 bytes，约805.07GiB；尚未展开。百度源export report为completed/missing=[]，去重媒体9173个，portable manifest运行路径均为相对路径。报告没有做全量解码验证。

| Source | Train | Historical held | Total |
| --- | ---: | ---: | ---: |
| AgiBot | 3600 | 400 | 4000 |
| Bridge | 1867 | 208 | 2075 |
| HY | 3600 | 400 | 4000 |
| RoboMIND | 3600 | 400 | 4000 |
| RoboTwin | 3367 | 374 | 3741 |
| Total | 16034 | 1782 | 17816 |

每条clip名义10秒，fps为5-30，raw_frames为51-301。整包约49.49小时clip时间；打包媒体可能还含未使用episode帧，所以805GiB不是10秒clip裁剪后的体积。

**可直接复用**：原始RGB、camera/frame_offset/真实时间、episode拆分，足以开始观察重建与posterior变化训练。无需重追踪；新RGB dataset不加载teacher.pt。

**需要补齐**：portable包没有独立instruction/provenance。百度已有 `data/language_object_effect_v70_step8750/language_manifest.json` 和 `outputs/v70_language_audits/stage2_language_full_metadata_trace_20261006.jsonl`，需作为数据sidecar迁入PFS，保留case/窗口/语言粒度。已有审计17096条可回连语言，不等于精确五秒事件标注。

**不能直接支持**：CoTracker3+relay轨迹不是独立实体/时间GT；原held已多次用于开发，不重新宣称最终未见测试集；现有集合也没有明确对象角色×事件顺序组合留出。因此T0/T1可先实施，组合主实验的数据要另行建立，不能以旧teacher一致性作替代。

### 13.3 首个实施包

- `code/igsw/object_event_wm/data.py`：RGB-only manifest、episode-uniform采样、原生尺寸padding、实际时间、可选精确语言窗口。继承train/held，旧held标记为development，无新test。
- `code/scripts/prepare_object_event_data_v71.py`：统计实际媒体/时间窗口可用性和语言覆盖，不hash、不生成新tracking。
- `code/scripts/prepare_object_event_storage_v71.sh`：把OSS的26个分片直接流式展开到PFS，不在Mac落盘、不拼出第二份805GiB tar，不重编码视频。
- 本地新分支 `codex/object-event-world-model-v71`。本包完整实现后经GitHub发布，由SWXC在阿里取代码并执行数据准备；模型与完整训练链路随后继续。

### 13.4 评测复用与新增

旧 `language_effect_evaluation_v70.py` 的逐轨迹绝对/运动归一化误差、静止漂移及1/3/5秒分布可作为辅助。必须补充同一renderer下的observed-state重建、posterior rollout和prior生成；主体与时间事件采用独立数据真值或明确标注，不由模型mask自行定义。先实现共同输出，再选择与LPWM/TextOCVP可比的正式协议。

### 13.5 当前进度

- [x] 建立active goal并读取历史数据/评测实现。
- [x] 经SWXC确认阿里GPU、CPFS和百度源metadata。
- [x] 完成首个RGB数据适配与PFS展开入口。
- [ ] GitHub发布并在阿里PFS完整展开数据。
- [ ] 补迁语言sidecar，运行真实数据读取和manifest准备链路。
- [ ] 配置PFS独立运行环境和已匹配的视觉/成像权重。
- [ ] 实现基础Object State、posterior、Dynamics、renderer并做真实数据端到端试验。
- [ ] 实现对象片段effects、条件prior和独立组合评测。
- [ ] 形成大规模阿里训练配置与前台运行交付。

## 14. 基础主干实施进度（2026-10-15）

### 14.1 数据准备的实际进展

首个数据包已推送：`2d5a91c664326f4390ed1f973acc02ea2a2bf4de`，分支 `codex/object-event-world-model-v71`。阿里通过GitHub直接HTTPS取到同一提交；百度旧代理在阿里超时，未修改系统网络配置，仅对下载命令取消失效代理。

语言sidecar已从百度8732流式迁入PFS：`metadata/language_manifest_v70.json`，367044123 bytes，68384个窗口；没有在Mac落盘。完整portable包在阿里PFS解包，既有storage PID27713、tar PID27717及SSH会话46234已确认实际运行。2026-10-15 12:54:21 CST读取约204.789GiB/25.437%，没有退出码或完成报告；这一数字是读取进度，不是数据质量完成度。

### 14.2 本轮实现范围

已编写 `code/igsw/object_event_wm/` 下的 codec/config/state/dynamics/model/training/evaluation，以及训练、评测、资产准备和一个真实资产端到端测试入口。尚未在GPU执行，不写成已通过。

- **共同视觉空间**：官方RAE的DINOv2-with-registers-base与ViT-XL decoder；RAE代码版本 `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`。512版使用官方 `ViTXL_n08_i512/model.pt` 和匹配统计量。输入从原始RGB等比例letterbox到512，DINO内部448，输出512；不声称这是原生像素分辨率预测。256版可选并使用其独立匹配权重。
- **对象状态**：竞争式对象分配与每候选局部tokens、跨帧更新；当前原型16个对象候选预算加1个额外成分，每成分4个局部tokens、width384。候选/alpha不自动等于物理实体或背景真值。
- **Effects**：`global_step`、`object_step`、`object_segment`。前两者每步更新；片段版以可学习边界选择更新或沿用effect。它们共享主要模块和数据，参数量由实际inventory报告，不预报效果。
- **联合目标**：观测/rollout的标准化视觉feature差、RGB L1、相邻feature变化误差、effect KL；片段版增加更新代价。RGB梯度穿过冻结decoder进入renderer/State/Dynamics。默认每次优化解码4个future时点，feature目标覆盖全部future；评测解码全部future。
- **训练运行**：single GPU或DDP，默认3s/16帧历史和5s/25帧未来，episode-uniform；数据游标、optimizer、scheduler、每rank RNG保存恢复；保留两个rolling recovery及编号snapshot。主训练前台运行，不访问GitHub或下载模型。
- **当前评测**：codec reference、observed State、posterior、zero effect、反向effect次序、State-copy与RGB-copy；输出逐帧L1/PSNR/SSIM和视频。`appearance_change_weighted_l1`仅表示RGB变化加权，不称物理运动GT。语言prior与独立对象/事件评测尚未实现。
- **一次完整测试**：三个effect模式各执行不间断两步与一步保存后恢复，再检查样本序列、模型差异、history prefix隔离，并在每源真实视频上输出重建。没有用小型假backbone代替正式资产。

### 14.3 未完成项与下一步

本轮只完成Python/Shell语法解析，不能代表GPU完整训练。阿里直连Hugging Face/Astral已确认，系统已有PyTorch、Transformers、PyAV、PyYAML、einops和HF Hub；uv、独立PFS环境及RAE权重未准备。新增依赖脚本被当前 `deps=ask` 权限拦截，已向用户请求具体依赖授权；没有绕过该限制安装。

环境准备获准后，在PFS下载配套资产并运行上述完整测试，使用真实反馈修正实现。之后继续条件prior、独立对象-次序数据与正式baseline比较；当前goal保持active，不因基础代码写完而完成。
