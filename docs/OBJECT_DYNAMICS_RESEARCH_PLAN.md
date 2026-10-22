# Object Dynamics: 研究定位、模型改造与实验总计划

日期：2026-10-15。本文是研究与实施计划，不是新模型的实验结果。

阅读顺序：先看第 11 节的调研后取舍，再看第 3-7 节的模型/训练/数据工作包；第 9-10 节是逐篇证据。初始候选计划与最终收窄方向保留在同一文档，避免把调研前的假设当成调研后的结论。

用户决策：重新审视主干，不要求继承旧版本的架构、冻结方式或 latent 形状；先给全局计划，再由六个 Sol/Luna agent 并行阅读论文的方法、实验与必要代码。调研后用户批准进入实施阶段；已完成主干端到端测试、100步真实视频pilot及正式受控数据准备，尚未启动正式八卡训练。最新数据结果见18.11，匹配128输入/输出的实测与命令见22.4-22.5；早期章节保留当时状态。

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

**2026-10-16实施状态**：当前已实现scene/global-step、object/global-step、object/object-step、object/object-segment四种内部配置，分别隔离状态组织、effect绑定与更新持续时间。global_step仍是逐时刻全局effect，不是整段视频共用一个z；E2的整段单一effect对照尚未实现，不能用现有global_step代替。正式对照训练尚未执行，目前只有工程短更新和100步真实视频pilot，不能宣称新版方法有效。科学证据需要先确认posterior真正还原变化且effect不可被忽略，再用相同初始历史的双指令真实后果和未见组合比较条件生成；主体区域由模拟器真值提供，不由模型自己的mask定义。分段update数量或较小Dynamics参数量也不自动等于系统更高效，成本要包含frontend、prior、Dynamics与renderer。

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

**当前平台优先级（用户2026-10-15更新）**：百度8732执行模型测试和后续实验。用户随后回复skip，阿里环境/权重准备已移出本阶段，不再重试其网络；已完成的阿里数据保留。代码经GitHub到百度，数据与环境不落Mac；百度复用旧运行根和独立benchmark环境。下述先前的阿里盘点和数据准备结果仍保留。

**百度统一目录**：源码仓库继续为 `/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source`，运行根继续为 `/mnt/pfs/public/xuhaoming/instruct_gs_world`，Python优先用该根下的 `.venv/bin/python`。V71发布代码放现有运行根的 `runtime/object_event_world_model_v71/releases/<revision>`，日志/结果仍放 `logs`/`outputs`。取消新增 `/mnt/pfs/public/xuhaoming/wm-align` 根目录的提议，不移动、复制或删除旧实验数据。

**Active goal的执行约束**：完整目标仍覆盖可解码状态、对象绑定时序effects、共同Dynamics、条件生成、独立评测和运行交付。当前工具中的原始goal描述仍写有“阿里优先”，平台顺序以本节用户最新指令为准；已有goal保持active，不建立重复goal，不把准备完成当作研究完成。当前待完成的是配套资产、真实模型训练/恢复/评测、完整条件prior及正式实验交付。

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

V71逐源评测采用固定种子、source内部episode-uniform无放回抽样，不按排序截取最前面的任务。各模型使用同一manifest和seed，逐例保留实际case_id与时间窗口；较少样例的完整运行测试仅检查工程执行，不用于比较总体效果。

`compare_object_event_evaluations_v71.py`仅配对case、完整帧索引、实际时间与输出尺寸相同的记录，先导出逐例CSV，再按episode汇总和source内bootstrap。报告绝对误差、逐例gain、分来源分位数与未配对情况；末帧缺失不拿更早帧替代。development上的区间描述这批配对样例，不替代独立test或对象语义评价。

### 13.5 当前进度

- [x] 建立active goal并读取历史数据/评测实现。
- [x] 经SWXC确认阿里GPU、CPFS和百度源metadata。
- [x] 完成首个RGB数据适配与PFS展开入口。
- [x] GitHub发布数据准备与基础主干代码，在阿里PFS建立独立checkout。
- [x] 在阿里PFS完整展开数据并取得真实manifest准备结果。
- [x] 补迁语言sidecar到阿里PFS。
- [x] 每源两个episode完成真实RGB读取，保存时间线和文件映射。
- [x] 运行完整模型训练/恢复/评测链路，取得真实结果。
- [x] 定位并修正初始化状态/后验量级异常，重新执行完整链路及逐帧trace。
- [x] 在修正后的完整模型上完成100步优化和固定40例held对照。
- [x] 百度旧环境与配套DINOv2/RAE资产已可执行；阿里准备按最新skip决定移出本阶段。
- [x] 实现基础Object State、posterior、Dynamics、renderer并做真实数据端到端试验。
- [x] 实现对象片段effects、条件prior和完整训练/恢复/生成链路。
- [x] 建立独立主体/顺序评分器，并在真实模拟序列及模型完整链路执行，见第18节。
- [x] 固定1000/100/200场景划分及原子关系覆盖计划。
- [x] 生成正式scene-disjoint集合及离线文本cache，实际统计见18.11。
- [ ] 在正式集合完成主模型与匹配baseline比较；外部LPWM仍待独立环境/权重准备与实际执行。
- [x] 形成百度八卡前台训练、显式恢复和单卡评测指令，见第16节；尚未启动正式长训。
- 阿里备用环境和配套权重：本阶段跳过，保留已准备的数据，不再重试网络。

## 14. 基础主干实施进度（2026-10-15）

### 14.1 数据准备的实际进展

首个数据包已推送：`2d5a91c664326f4390ed1f973acc02ea2a2bf4de`，分支 `codex/object-event-world-model-v71`。阿里通过GitHub直接HTTPS取到同一提交；百度旧代理在阿里超时，未修改系统网络配置，仅对下载命令取消失效代理。

语言sidecar已从百度8732流式迁入PFS：`metadata/language_manifest_v70.json`，367044123 bytes，68384个窗口；没有在Mac落盘。完整portable包在阿里PFS解包，既有storage PID27713、tar PID27717及SSH会话46234已确认实际运行。2026-10-15 13:16:30 CST读取718418075785 bytes，约669.079GiB/83.108%，没有退出码或完成报告；这一数字是读取进度，不是数据质量完成度。

基础主干提交为 `1048682e4c1ca68c3fe5c8ba289fa5b7eb665cdb`，已通过GitHub同步到阿里独立worktree `/mnt/pfs/xuhaoming/wm-align/releases/1048682`。运行中的数据准备仍使用 `repo` 下的 `2d5a91c`，没有在进程运行中切换源码。当前未启动模型训练。

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

本轮只完成Python/Shell语法解析，不能代表GPU完整训练。阿里直连Hugging Face/Astral已确认，系统已有PyTorch、Transformers、PyAV、PyYAML、einops和HF Hub；uv、独立PFS环境及RAE权重未准备。用户已批准 `deps=allow`，范围仅为阿里PFS中的uv、PyTorch2.7/torchvision、Transformers、HF Hub、PyAV、NumPy/Pillow、PyYAML/einops、LPIPS、SwanLab和官方RAE/DINO权重；不修改系统环境，不在Mac安装或下载。授权范围不包含新模拟器或其他模型依赖。

环境准备获准后，在PFS下载配套资产并运行上述完整测试，使用真实反馈修正实现。之后继续条件prior、独立对象-次序数据与正式baseline比较；当前goal保持active，不因基础代码写完而完成。

### 14.4 数据准备完成与实际语言覆盖

2026-10-15 13:23:46 CST解包结束，`EXTRACT_EXIT_CODE=0`；13:24:09生成 `/mnt/pfs/xuhaoming/wm-align/metadata/object_event_v71.json`，原执行会话最终退出0。报告保留全部17816条clip，`excluded=[]`。没有重追踪或重编码。

| 来源 | Clip/episode | 有语言窗口的clip | 语言窗口 | 原始尺寸与相机 |
| --- | ---: | ---: | ---: | --- |
| AgiBot | 4000 | 4000 | 16000 | 640x480，head |
| Bridge | 2075 | 2002 | 8008 | 256x256，image_0 |
| HY | 4000 | 3393 | 13572 | 424x240，cam_high |
| RoboMIND | 4000 | 3960 | 15840 | 640x480，front 2641条/top 1359条 |
| RoboTwin | 3741 | 3741 | 14964 | 518x518，cam_high |

共17096条clip有语言、720条无语言。54812个窗口是episode级目标，13572个HY窗口是history-anchor级任务；后一类不直接复制到任意重新采样的时间点。Bridge为5FPS，其余四源30FPS；都按真实时间生成3秒历史和5秒未来。此结果确认文件映射和metadata，不代表逐视频解码或模型测试完成。

新增 `inspect_object_event_data_v71.py` 从正式dataset读取每源视频，保存整段8秒预览、六时点timeline及原文件/帧索引/PTS映射。预览等比例缩放仅用于显示，不改portable媒体。该入口不读取teacher、不加载预训练模型，输出全部写PFS。

### 14.5 条件prior的代码接口

- **What changed**：新增 `ObjectEffectPrior`，复用历史对象候选的局部tokens、geometry和相对时间；对当前预测state生成Gaussian innovation及对象级更新边界。对象版只在对应候选的历史内读取局部证据，再做跨候选交互；全局对照读取全部历史候选。语言以带有效mask的冻结text tokens输入，不假定固定slot号就是主体。
- **Why**：posterior能解释真实未来，不等于生成期能够从历史选择正确变化。两个路径必须有独立输入边界与结果。
- **Impact**：prior训练目标为冻结posterior到prior的Gaussian KL，以及片段版的边界交叉熵；训练允许teacher-force此前的effect，但不把当前目标effect送入当前预测。生成只读取历史、请求时间、语言和随机数，然后递归调用共同Dynamics。当前新增的是prior模块/目标/生成接口；完整语言encoder、prior训练入口和生成比较尚未交付。完整工程测试追加一次history-only prior更新和自由rollout，未把它写成语言能力验证。

### 14.6 真实RGB读取结果

执行提交 `5b8cd9b023609f592cf0a312621624826d60f33e` 已通过GitHub拉到阿里 `/mnt/pfs/xuhaoming/wm-align/releases/5b8cd9b`。使用现有系统Python读取数据，没有安装包、加载模型或改写系统环境。

命令为 `inspect_object_event_data_v71.py --manifest /mnt/pfs/xuhaoming/wm-align/metadata/object_event_v71.json --out /mnt/pfs/xuhaoming/wm-align/outputs/v71_rgb_review_5b8cd9b --items-per-source 2`。实际退出0，五源各2条，共10条成功、0失败/排除。共410帧的frame_valid均为true；每条为16历史+25未来，相对时间-3至+5秒，约0.2秒间隔。前四源使用实际video PTS；RoboTwin使用RGB cache帧号/fps。Bridge原生5FPS，不伪称其被恢复为30FPS。

预览入口为 `/mnt/pfs/xuhaoming/wm-align/outputs/v71_rgb_review_5b8cd9b/index.html`，每例保留 `input_sequence.mp4`、`timeline.jpg`、`source.json`；日志为 `logs/v71_rgb_review_5b8cd9b.log`。RoboMIND这次两个样例均为front，没有由这两个样例推断全部top相机的画面语义。结果证明所选真实窗口可读和时间映射可检查，不代表全部数据质量或模型能力。

### 14.7 环境准备入口

`setup_object_event_env_v71.sh` 仅用于联网准备：在指定WORK_ROOT内通过uv创建独立环境，安装torch2.7.0/torchvision0.22.0及批准的软件包，取得官方RAE固定版本和512版配套DINO/decoder/statistics。uv、临时文件、HF/Torch缓存、Python环境和版本报告全部位于PFS。不修改系统Python或shell配置，不在训练脚本中调用下载。已有可用环境的平台直接运行资产准备和模型测试，无需重复安装PyTorch。

测试入口 `test_object_event_world_model_v71.sh` 与训练入口默认复用百度旧运行根；阿里通过显式WORK_ROOT切换到其已有PFS根。测试默认单卡、前台执行，日志落 `logs/<run>/test.log`；训练默认八卡、前台执行。两个运行入口均不访问GitHub或下载模型。

百度8732现有环境实测为Python3.11.13、torch2.8.0+cu126、torchvision0.23.0、Transformers5.10.2、PyAV17.0.1、SwanLab0.10.1；当前完整测试不调用未安装的LPIPS。用户确认继续使用DINOv2-B与配套RAE，不切换已有DINOv3。百度下载使用 `http_proxy=https_proxy=http://10.66.65.186:18000`。容器未安装Git，但代理访问GitHub/HF已成功；源码可从GitHub固定提交归档取得并放在旧runtime的版本目录，仍不允许从Mac直接传源码。资产清单通过显式 `--rae-revision` 记录所用官方源码版本，不依赖运行机器安装Git。

### 14.8 百度首次完整模型执行

2026-10-15 15:04:31-15:09:50 CST，提交 `a5a69301e5feaf4f0d9ced2ddcb2c7f23373c5de` 在百度单卡A100 80GB上完成512/BF16完整入口，退出0。源码/官方RAE归档通过旧Python标准库经指定代理下载；旧环境没有安装或升级包。资产和metadata准备均退出0。

三种effect模式都完成连续2步与1步保存后恢复到2步的对照，参数最大差异均为0，样本顺序相同；future feature swap后的history prefix差异0。history-only prior完成一次更新并生成25个state，effect形状 `[1,25,17,64]`。五源各一例产出八种条件的MP4。语言encoder及条件prior的生成质量没有在这次测试中评估。

**执行通过不代表优化正常**：global/object-step的初始effect KL约6045万、preclip gradient norm约2.91e14；object-segment的初始KL约1145万、gradient约1.24e11。Prior loss约1.866e11。与此同时观测feature MSE约0.907，观测RGB L1约0.090，说明只看重建标量和有限值断言会漏掉内部数值问题。下一项是逐帧记录state及posterior幅度，定位异常后改计算结构，不以降低KL权重或只增大clip阈值掩盖问题。

这次五例、两步模型的codec-reference RGB L1/PSNR/SSIM为0.0528/22.25/0.733；observed-state为0.2027/12.60/0.581；posterior为0.2110/12.35/0.569，zero-effect几乎相同。这仅是初始化附近的工程输出，不构成表示或effect效果结论；样例也采用当时的首项抽取，而非后续固定种子抽样。

实际结果根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_event_v71_e2e_a5a6930_20261015T070431Z`。其中 `test_report.json`、三种模式的 `uninterrupted/train.jsonl`/`recovered/train.jsonl`、`reconstruction/report.json`/`cases.jsonl`和视频均落盘。日志在同一旧运行根的 `logs/object_event_v71_e2e_a5a6930_20261015T070431Z`。原测试PID1238已正常结束，没有重启。

### 14.9 数值定位与状态更新修正

`73450f7` 在同一真实训练样例 `robotwin_ep26235_f20` 上执行初始化和旧step-2模型两次前向，均退出0。41帧DINO feature RMS处于0.915-0.953；初始化state RMS却从frame0的0.381增长到frame15的36.81、frame40的77414.31，最后posterior mean abs-max为135168。旧step-2模型最后state RMS进一步达到697298.38。异常已在history阶段发生，早于posterior，不是DINO输入feature整体变大。

- **What changed**：观测更新从无界自反馈残差，改为归一化状态输入、候选状态和门控凸组合；每帧memory输出再经过LayerNorm。新增首/末observed state RMS、末rollout state RMS和posterior mean最大幅度日志。
- **Why**：原renderer对输入做LayerNorm，使重建loss看起来普通，但posterior直接读取持续增大的state残差；两条路径对数值尺度的反应不同，单看重建loss会漏掉这个问题。
- **Impact**：不改DINO/RAE资产、不改KL权重；新模型从头执行完整测试，旧两步checkpoint只作诊断记录，不作为新结构resume起点。新增RMS字段仅用于数值健康，不用于证明物体语义或变化预测质量。

### 14.10 状态更新修正版的实际结果

执行提交 `93f5ae17fc9c78e13fd31276287e450b76e84162`；百度单A100、512/BF16、原17x4状态预算和KL权重0.001。2026-10-15 15:32:53 CST完整测试退出0，15:33:45两次trace也均退出0。没有安装包、重新下载权重或修改远端源码。

| 模式 | Step 1/2 KL | Step 1/2 preclip gradient norm | Resume参数差异 |
| --- | --- | --- | --- |
| global_step | 0.2420 / 0.2491 | 10.7337 / 6.3444 | 0 |
| object_step | 0.2560 / 0.2479 | 10.7216 / 5.2151 | 0 |
| object_segment | 0.1416 / 0.1235 | 12.0202 / 4.2640 | 0 |

三模式恢复后的样本顺序及同一步loss逐项相同。history prefix在future feature swap后差异0；codec保持冻结。history-only prior的loss为1.5362、gradient norm为63.2519，生成25个state；语言encoder仍未参与该工程测试。单样本峰值显存最高约21.89GiB，这不是更大batch的已测上限。

同一真实样例的初始化41帧state RMS范围为0.999962-0.999988；训练两步后为1.000113-1.000175，state abs-max最高5.77、posterior mean abs-max最高2.11。DINO输入与此前trace相同，支持修正解决了本次复现中的自反馈放大。归一化后的RMS近1是结构带来的数值性质，不能作为对象语义或动态信息有效性的证明。

结果根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_event_v71_e2e_93f5ae1_20261015T072730Z`。保留 `test_report.json`、三模式日志及checkpoint、`numerics/initial.json`、`numerics/trained_step2.json`、五源重建报告和40个视频。对应日志位于旧运行根下 `logs/object_event_v71_e2e_93f5ae1_20261015T072730Z`。会话27988最终退出0，测试PID13419已结束。

下一项是短程真实优化与固定episode对照，之后补全条件prior训练和独立组合实验。此处完成的是完整工程链路和一个具体数值故障的修正，不是长期训练或科学效果验收。

### 14.11 100步优化与40例配对结果

百度继续执行同一 `93f5ae1`：object_step从头训练100 optimizer steps，batch2、accum1、workers2、warmup10、rollout-ramp10、seed17，512/BF16及原模型维度不变。2026-10-15 15:55:54 CST训练退出0，随后指定工程step-2参考与本次step-100在同一40例development窗口上评测，两次均退出0，16:03:34结束；每源8例、无排除，case顺序/帧索引/时间相同。

| 条件 | Step-2参考 RGB L1 | Step-100 RGB L1 |
| --- | ---: | ---: |
| codec reference | 0.054364 | 0.054364 |
| observed state | 0.208791 | 0.187191 |
| posterior | 0.306446 | 0.192352 |
| zero effect | 0.315927 | 0.191795 |
| reversed effect order | 0.305534 | 0.192534 |
| source-state copy | 0.216521 | 0.187216 |
| RGB copy | 0.064772 | 0.064772 |

Posterior变化加权L1从0.301292降至0.229818，但zero effect为0.219842。可以判断短程优化改善重建，不能判断effect已有效；重建仍明显弱于codec reference，且posterior没有超过zero/copy。两步参考与100步pilot的batch/schedule不同，本项不是正式方法消融。

21个训练记录点中KL为0.1068-0.3872，gradient norm为0.6686-5.5426；observed state末帧RMS维持约1。Rollout末帧RMS最高5.7529，最后为2.2581，表明两条状态路径尺度不同，但不单凭该内部量判定动态信息质量。单步耗时中位数约3秒、吞吐0.667样例/秒，峰值allocated显存33.59GiB。

结果根为 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/object_event_v71_pilot100_93f5ae1_20261015T075006Z`，包含 `recovery_00000100.pt`、`reference_step2_eval` 和 `pilot_step100_eval` 的逐例JSON及视频；日志在旧运行根对应 `logs/object_event_v71_pilot100_93f5ae1_20261015T075006Z`。会话93801及训练/评测进程均已结束。

## 15. 完整条件prior链路实施

### 15.1 固定标签与语言

训练prior时，使用明确指定的world-model checkpoint。启动时将其硬链接为该run的 `teacher.pt`，随后所有恢复点引用这个固定依赖，不追随外部latest，不把冻结teacher重复写入每个checkpoint。State、posterior、Dynamics均冻结；teacher只生成状态与effect标签，不执行无用的RGB decoder。联合训练入口保持原有路径。

语言使用百度已存在的 `/mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct`。实际config为qwen3_vl，text hidden size2560；tokenizer与两个权重分片均存在。本轮先固定其文本encoder，仅编码instruction并缓存逐token特征，不输入视频、不生成描述、不重训4B。`prepare_object_event_text_v71.py`从已有language_windows去重，保留全文、不截断，按文件增量恢复；不新增语言、不复制RGB。每条训练样例仍使用自己被审计的语言窗口，缺语言片段不充当语言主实验。

### 15.2 训练和生成的边界

`train_object_event_prior_v71.py`/`.sh`复用现有DDP、episode-uniform sampler、保存恢复和训练标量链路。仅prior参与optimizer，学习冻结posterior到prior的Gaussian KL和片段更新边界；此前effect可以teacher-force，当前target effect不得作为当前输入。历史与语言cross-attention对所有未来query批处理，query之间只在同一时刻交互，不混合未来时刻。时间条件包含相邻步长和距历史终点的请求时间，不包含episode剩余时长。

10%样本丢弃语言，训练有效的无语言条件，供匹配对照使用；这是固定训练设计，不是本轮调参搜索。缓存文本不需要在prior训练进程加载Qwen，RGB decoder也不加载；只有标签所需的冻结DINO和world-model模块常驻。

`forecast_object_event_world_model_v71.py`从history_request读取明确列出的历史RGB帧，在线编码用户instruction，再由prior和同一Dynamics生成未来state/RGB。history_request不包含未来帧索引；输出记录 `future_rgb_read=false`、生成effects和实际时间。视频按实际PTS写入，不因历史/未来采样间距不同改变事件显示时间。

### 15.3 评测和完整运行测试

同一teacher/decoder下配对报告posterior、zero/reversed effect、copy，以及正确语言、无语言、打乱语言的生成样本。相同case的语言干预使用相同随机噪声；记录实际打乱文本，不将其一律称为已确认冲突指令。逐例、逐帧、分来源报告绝对RGB指标。多样本期望误差与按整条sequence选择的best-of-N分开，后者仅是oracle。缓存文本评测的耗时明确不含Qwen；独立forecast记录包含在线文本编码的完整预测耗时。

首次实测发现按文本排序相邻取值会选到近义任务描述，后续改为同一数据来源、不同task group且文本不同的固定种子抽样，并记录替代group/文本。不同group仍不自动等于语义冲突；准确的反事实后果需要独立任务真值，不能以原视频误差越大就断言“听懂了替代指令”。

现有单一完整测试入口加入真实文本cache、固定teacher、prior更新/保存恢复、配对生成和仅历史文件的forecast；还检查批处理未来query不会改变第一步输出及MP4的实际PTS。该链路已在15.4完成真实GPU执行；使用100步teacher不代表其effect已有效，正式科学实验仍需更充分训练与独立主体/顺序评价。单卡测试入口显式设置rank0/world1，不继承多机调度器的节点数作为测试进程数。

### 15.4 完整语言链路的实测结果

`a74a43c464b78a14b659c34d3eb3b0c1f30e4cfc` 已在百度旧目录/旧环境完整执行；2026-10-15 17:03:17 CST最终报告落盘，文本cache和完整测试均退出0。Qwen、DINO/RAE权重均复用现有文件，没有重新下载或安装包。

文本cache包含4564条unique instruction、宽度2560，覆盖五源68384个窗口与17096条clip，全文保留。三种joint模式以及文本prior连续/恢复参数差异均为0；history future-swap及后续teacher-forced state改变后的prior第一步差异也均为0。prior的固定teacher确认为真实step100、源revision93f5ae1；冻结teacher为27780272参数，prior为22810496可训练参数。

文本prior仅训练两步，loss为0.8821/1.3631，gradient norm为33.67/32.60。五例正确语言sample0的RGB L1为0.183676，无语言为0.184072，打乱语言为0.183665；没有语言有效性的正面结论，且该版打乱中有近义描述。两sample期望L1为0.183739，整段oracle best-of-2为0.183674，仅作为oracle覆盖记录。

独立forecast只读取16个历史帧，生成25个未来帧，记录future_rgb_read=false；MP4 PTS最大误差约0.477微秒，文本编码+历史编码+Dynamics+渲染耗时本例约1.862秒。该计时不包含首次模型加载，不能把它当作跨设备/批量的吞吐结论。

文本cache位于 `/mnt/pfs/public/xuhaoming/instruct_gs_world/metadata/object_event_text_v71`。完整结果根为 `outputs/object_event_v71_language_e2e_a74a43c_20261015T085418Z`（相对于百度旧运行根），包含 `test_report.json`、`language_prior`两分支checkpoint及固定teacher、`conditional_generation`逐例报告/视频、`history_only_forecast/forecast.mp4`及prediction.json；对应日志在旧运行根同名logs目录。会话31263最终退出0，PID187032已结束。

### 15.5 逐例比较、修正语言控制与batch实测

`d9bc431a50396548e9f2efec862c16c1e5a561df` 完成三项实际运行。40例配对结果中，step100相对step2的posterior RGB L1平均改善0.114093、中位改善0.114448，38/40例改善，source内episode bootstrap 95%区间为[0.095664,0.131243]。但step100 posterior相对zero-effect仅18/40例改善，平均gain=-0.000558，区间[-0.006238,0.005516]跨0。HY为7/8例改善，RoboTwin仅2/8，不能用总体均值掩盖来源差异，也不能宣布effect已有效。

逐例CSV/JSON保存在旧运行根 `outputs/object_event_v71_paired_d9bc431`。新语言控制在五例上完成，替换文本都能回连到同源不同task group，报告保存真实指令/替代指令及来源，不写成已验证语义反事实。结果在 `outputs/object_event_v71_language_controls_d9bc431_20261015T095529Z`。

单A100、512/BF16、batch4、accum1、原模型和损失的两步训练退出0。峰值allocated显存57.05GiB；第二步4.723秒、0.847样例/秒，梯度17.72、KL0.260。第一步包含3.62秒数据等待，不能用它或第二步单独推断八卡稳态吞吐。结果在 `outputs/object_event_v71_batch4_d9bc431_20261015T095529Z`。据此将八卡默认改为每卡batch4、累积8、global batch256；没有宣称已在八卡实测。

## 16. 百度执行指令

- **What changed**：固定可执行代码 `4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c`，将源码同步、单卡完整测试、八卡joint训练、恢复、prior训练和评测分开；每段命令独立设置路径。
- **Why**：百度容器没有Git/curl，但旧Python能经代理读取GitHub源码归档；训练必须离线使用服务器已有资产。
- **Impact**：保留旧源码目录与旧环境。版本源码只放在原运行根的 `runtime/object_event_world_model_v71/releases`。以下正式训练是待用户提交的实验，不表示已在八卡运行。

### 16.1 同步源码，只在准备机执行

这一段允许联网。它不安装依赖、不下载数据或模型，也不改共享源码checkout。服务器已经有DINOv2/RAE、RGB manifest和4564条文本cache，无需重做。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export ARCHIVE="${WORK_ROOT}/tmp/object_event_v71_${SOURCE_REVISION}.tar.gz"
mkdir -p "${WORK_ROOT}/tmp"
if [ ! -d "${ROOT}" ]; then
  "${PYTHON}" -c 'import os, urllib.request; urllib.request.urlretrieve("https://codeload.github.com/healenrens/Instruction-GS-World/tar.gz/" + os.environ["SOURCE_REVISION"], os.environ["ARCHIVE"])' &&
  mkdir -p "${ROOT}" &&
  tar -xzf "${ARCHIVE}" --strip-components=1 -C "${ROOT}"
fi
printf 'SOURCE_ROOT=%s\n' "${ROOT}"
```

### 16.2 单卡完整测试

使用已完成的真实step100作为prior工程测试teacher，不把它称为训练达标模型。测试同时运行三种joint模式、保存恢复、语言prior和仅历史生成；约束与实际结果见15.4。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/object_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME="object_event_v71_test_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
bash "${ROOT}/code/scripts/test_object_event_world_model_v71.sh" \
  --text-cache "${WORK_ROOT}/metadata/object_event_text_v71" \
  --prior-base-checkpoint "${WORK_ROOT}/outputs/object_event_v71_pilot100_93f5ae1_20261015T075006Z/recovery_00000100.pt"
TEST_RC=$?
printf 'TEST_RC=%s\nREPORT=%s/test_report.json\nLOG=%s/test.log\n' "${TEST_RC}" "${OUT}" "${LOG_ROOT}"
```

### 16.3 八卡joint训练，前台

本轮先提交 `object_step`。另两种匹配对照仅将 `EFFECT_MODE` 分别改为 `global_step`、`object_segment`；RUN_NAME随之变化，其余配置相同，不复用输出目录。每卡4、累积8、八卡global batch256。观察目标与posterior rollout联合学习，默认每步对4个未来时点计算RGB loss，feature目标覆盖25帧；不是先跑旧V69 State再跑旧Stage 2。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/object_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_PER_GPU=4 GRAD_ACCUM=8 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=4 DECODER_FRAME_BATCH=2
export EFFECT_MODE=object_step STEPS=10000
export RUN_NAME="object_event_v71_${EFFECT_MODE}_seed17_main"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
mkdir -p "${LOG_ROOT}"
bash "${ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  --seed 17 --precision bf16 --lr 0.0002 --warmup 500 --rollout-ramp 1000 \
  --save-every 500 --snapshot-every 2500 --log-every 10 \
  2>&1 | tee "${LOG_ROOT}/launcher.log"
TRAIN_RC=${PIPESTATUS[0]}
printf 'TRAIN_RC=%s\nOUT=%s\nLOG=%s/launcher.log\n' "${TRAIN_RC}" "${OUT}" "${LOG_ROOT}"
```

SwanLab使用服务器已有登录或任务注入的 `SWANLAB_API_KEY`，只上传训练标量。需要纯本地日志时，在运行前 `unset SWANLAB_PROJECT SWANLAB_WORKSPACE`；不在命令或仓库写入key。checkpoint为 `recovery_00000500.pt` 等，`latest.json`指向最近恢复点，`step_00002500.pt` 等编号快照保留；不生成 `latest.pt`。

### 16.4 同一实验八卡恢复

先把 `RUN_NAME` 设置为要恢复的已存在run。程序从checkpoint恢复原参数、optimizer、scheduler、每rank RNG和数据游标；本例仍使用相同八卡。不能用这一命令宣称跨卡数严格等价恢复。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/object_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_NAME=object_event_v71_object_step_seed17_main
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
export RESUME="${OUT}/latest.json"
unset INIT_FROM
mkdir -p "${LOG_ROOT}"
bash "${ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  2>&1 | tee -a "${LOG_ROOT}/launcher.log"
TRAIN_RC=${PIPESTATUS[0]}
printf 'TRAIN_RC=%s\n' "${TRAIN_RC}"
```

同一个入口也可恢复prior run：改为该prior的RUN_NAME即可，stage/固定teacher/text cache均从checkpoint恢复；不要重新创建teacher依赖。

### 16.5 单卡配对评测

例子使用joint的2500步编号快照。按实际完成快照改 `CHECKPOINT`，不以不存在的快照或最新文件名代替。每源64例，共最多320例；逐例指标保存全部，视频每源4例。历史held仍称development。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CHECKPOINT="${WORK_ROOT}/outputs/object_event_v71_object_step_seed17_main/step_00002500.pt"
export EVAL_OUT="${WORK_ROOT}/outputs/object_event_v71_eval_object_step_2500_$(date +%Y%m%d_%H%M%S)"
"${PYTHON}" "${ROOT}/code/scripts/evaluate_object_event_world_model_v71.py" \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${WORK_ROOT}/metadata/object_event_v71.json" \
  --assets "${WORK_ROOT}/metadata/rae_assets_512.json" \
  --partition development --items-per-source 64 --videos-per-source 4 \
  --precision bf16 --out "${EVAL_OUT}"
EVAL_RC=$?
printf 'EVAL_RC=%s\nREPORT=%s/report.json\nCASES=%s/cases.jsonl\nVIDEOS=%s/videos\n' \
  "${EVAL_RC}" "${EVAL_OUT}" "${EVAL_OUT}" "${EVAL_OUT}"
```

### 16.6 八卡语言prior训练

选择joint评测后的固定编号快照作为 `BASE_CHECKPOINT`，本例填写2500步仅说明路径。不要自动跟随joint的latest，也不要将100步工程teacher用于正式语言能力结论。prior只训练历史/语言到effect的生成模块，冻结joint模型；全文Qwen cache已完成，运行不加载Qwen或下载模型。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/object_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/object_event_text_v71"
export BASE_CHECKPOINT="${WORK_ROOT}/outputs/object_event_v71_object_step_seed17_main/step_00002500.pt"
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_PER_GPU=4 GRAD_ACCUM=8 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=4 DECODER_FRAME_BATCH=2
export STEPS=10000 LANGUAGE_DROPOUT=0.1
export RUN_NAME=object_event_prior_v71_object_step_teacher2500_seed17_main
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
mkdir -p "${LOG_ROOT}"
bash "${ROOT}/code/scripts/train_object_event_prior_v71.sh" \
  --seed 17 --precision bf16 --lr 0.0002 --warmup 500 \
  --save-every 500 --snapshot-every 2500 --log-every 10 \
  2>&1 | tee "${LOG_ROOT}/launcher.log"
TRAIN_RC=${PIPESTATUS[0]}
printf 'TRAIN_RC=%s\nOUT=%s\nLOG=%s/launcher.log\n' "${TRAIN_RC}" "${OUT}" "${LOG_ROOT}"
```

### 16.7 单卡语言生成评测

同一teacher与renderer比较正确语言、无语言、不同task-group语言、posterior和copy；固定单样本与4样本期望分栏，best-of-4只作为oracle。具体替代指令和逐帧结果保存本地，不上传评测视频到SwanLab。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=4bc9ddf2ddc747d7dbbb3d33d69a99246823df4c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PRIOR_CHECKPOINT="${WORK_ROOT}/outputs/object_event_prior_v71_object_step_teacher2500_seed17_main/step_00002500.pt"
export EVAL_OUT="${WORK_ROOT}/outputs/object_event_v71_eval_prior_2500_$(date +%Y%m%d_%H%M%S)"
"${PYTHON}" "${ROOT}/code/scripts/evaluate_object_event_world_model_v71.py" \
  --prior-checkpoint "${PRIOR_CHECKPOINT}" \
  --text-cache "${WORK_ROOT}/metadata/object_event_text_v71" \
  --manifest "${WORK_ROOT}/metadata/object_event_v71.json" \
  --assets "${WORK_ROOT}/metadata/rae_assets_512.json" \
  --partition development --items-per-source 64 --videos-per-source 4 --samples 4 \
  --precision bf16 --out "${EVAL_OUT}"
EVAL_RC=$?
printf 'EVAL_RC=%s\nREPORT=%s/report.json\nCASES=%s/cases.jsonl\nVIDEOS=%s/videos\n' \
  "${EVAL_RC}" "${EVAL_OUT}" "${EVAL_OUT}" "${EVAL_OUT}"
```

## 17. 独立benchmark准备

- **What changed**：新增作者公开TextOCVP/CLIPort视频归档的下载与实际结构清单入口 `prepare_textocvp_archive_v71.py`。先下载/读取ZIP目录及少量caption、图像，再决定adapter；默认不解压、不占GPU、不修改既有训练数据。
- **Why**：现有multirobot数据没有独立主体/事件顺序GT，不能用模型自己的mask或训练tracker来替代这项证据。
- **Impact**：公开archive只作为待核对的benchmark资产；21k、mask、时间信息等论文描述不直接当作已读取的文件事实。实际schema、可复用标签及最终组合协议待归档读取后记录。此次下载通过指定代理，在百度旧运行根 `data/benchmarks/textocvp_cliport` 进行；未改为新的训练分布。

**实际归档结果（2026-10-15）**：下载/目录解析于19:43:55 CST完成，退出0；文件20741534807 bytes，2269424个成员文件，未压缩体积97046404667 bytes。两套目录不能合并：`PutInBowl`本体含train20000、val500、test500；`PutInBowlSegm_new`另含val500、24588个segm文件和500个`object_info.pkl`。RGB样例360x320。新RGB manifest只取 `PutInBowl`；独立标注集单列，不能因为episode编号相同就认定RGB或标注帧对应。原本将val合计为1000的目录统计已在此分开。真实字段清单保存于百度benchmark目录 `archive_inventory.json` 与 `archive_schema_v71.json`。

### 17.1 公开RGB协议适配

- **What changed**：按[作者loader](https://github.com/angelvillar96/TextOCVP/blob/master/src/data/CLIPort.py)实现独立图像序列manifest，直接读取ZIP里的 `color/*_color.png` 与 `task_description.txt`，不展开或重编码RGB。保留作者train/val/test和四个明确排除的episode；默认首帧到后九帧，可另建十九帧预测manifest。沿用原joint/prior训练、episode-uniform采样、单卡完整测试与配对评测，真实机器人数据的时间契约不变。
- **Why**：公开loader按连续帧序号读取，没有提供物理FPS；不能把它改写为三秒历史/五秒未来。新数据以 `sampled_frame` 表示相邻发布图像的步数，历史请求仅列出实际历史图像，不列未来文件。预览采用5FPS并明确只是播放速度，逐帧指标使用frame offset而非seconds。
- **Impact**：这是公开语言条件成像benchmark的接口，不是独立实体/事件真值评价。原生图像尺寸由实际PIL读取，成像仍使用既有RAE512；现阶段结果必须标明该输出尺寸，不能直接与其他尺寸的论文数值混比。公开archive完成后执行真实数据端到端测试；在此之前仅为已实现接口。独立主体/顺序实验仍待模拟器GT或已核实的发布标注，不以模型mask替代。

### 17.2 图像序列准备与单卡测试命令

代码提交 `33378841aab3db4f4bfc008d2b91ded1d6830906`，明确只索引RGB本体。先按16.1的方法将SOURCE_REVISION替换为该完整提交，在准备机从GitHub取得源码。下面只读取已下载的ZIP及既有模型，不联网、不解压整份归档。执行过的步骤可直接使用其结果；无需重复跑完整测试来刷新报告日期。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=33378841aab3db4f4bfc008d2b91ded1d6830906
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MANIFEST="${WORK_ROOT}/metadata/textocvp_v71_1to9.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/textocvp_text_v71"
"${PYTHON}" "${ROOT}/code/scripts/prepare_textocvp_dataset_v71.py" \
  --archive "${WORK_ROOT}/data/benchmarks/textocvp_cliport/PutInBowl.zip" --out "${MANIFEST}" &&
"${PYTHON}" "${ROOT}/code/scripts/prepare_object_event_text_v71.py" \
  --manifest "${MANIFEST}" \
  --model /mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct --out "${TEXT_CACHE}" --batch 16
PREPARE_RC=$?
printf 'PREPARE_RC=%s\n' "${PREPARE_RC}"
```

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=33378841aab3db4f4bfc008d2b91ded1d6830906
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/textocvp_v71_1to9.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME="textocvp_v71_e2e_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
bash "${ROOT}/code/scripts/test_object_event_world_model_v71.sh" \
  --text-cache "${WORK_ROOT}/metadata/textocvp_text_v71"
TEST_RC=$?
printf 'TEST_RC=%s\nREPORT=%s/test_report.json\nVIDEOS=%s/conditional_generation/videos\n' \
  "${TEST_RC}" "${OUT}" "${OUT}"
```

正式benchmark训练复用16.3/16.6，但将manifest/text cache切换为本节路径，并使用独立的 `textocvp_v71_*` run名称。它是域内训练比较，不把多源100步模型的zero-shot结果当成与TextOCVP的公平主表。仍需对齐输出分辨率、预训练资产、训练预算和评测实现后再报告外部方法优劣。

### 17.3 独立模拟器环境

- **What changed**：用户已明确允许在百度原运行根建立独立benchmark uv环境，安装PyBullet、Gym及CLIPort必需依赖。新增 `setup_object_event_benchmark_v71.py`，环境位于 `environments/cliport-v71`，使用CPU版PyTorch2.7/torchvision0.22及固定包版本；官方CLIPort源码固定 `2be5c47b5bb9bb7040ad90693288b87b1e18e7ad`。uv二进制、缓存、临时包与源码全部在旧PFS根；经指定代理下载。
- **Why**：原训练 `.venv` 不承担模拟器依赖变更。CLIPort包入口会导入agent/model模块，因此隔离环境包含其导入依赖，但数据生成不加载策略权重、不训练动作policy。使用官方环境和oracle执行物理过程，后续采集实际body ID、位置和事件次序。
- **Impact**：环境准备本身不算模拟器执行成功。独立生成与真实渲染仍需完整运行后记录。已读公开分割集只有前后两个action级info快照，不是逐视频帧pose；同编号 `episode00001` 的主RGB/分割集caption相同，但51/50帧且首帧像素不同，不能交叉挂载mask。segm灰度为0、25、51等11个值，尚未确认到body ID的编码，不据此生成目标物体评分。

### 17.4 主体与顺序的真实生成入口

- **What changed**：`generate_cliport_event_sequences_v71.py` 使用官方CLIPort Environment、Task.oracle和UR5抓放primitive，在同一随机场景执行A后B、B后A；三组彩色block/bowl中两组参与事件、一组保留作背景干扰。默认两场景四条完整轨迹，先做真实物理生成与记录检查，不预先称为正式训练/测试划分。
- **Why**：公开文件中的同编号序列和归一化灰度mask不足以直接恢复严格的物体/时间真值。这里每帧直接保存实际body ID、六个实体的位置/姿态、可见像素数、接触对象对及模拟器物理tick。采样5Hz，physics240Hz；单独数真实stepSimulation调用，不使用上游movej重复递增的step_counter。事件记录区分oracle控制开始/结束和目标reward，不把控制开始误称物体运动onset。
- **Impact**：每条保留原生camera RGB视频、timeline、truth.npz、episode.json，集合给出同场景两种指令的history像素是否相同及每条真实reward。模型未来输入仅为历史RGB和指令；机器人action、mask和pose只服务数据生成或独立评测。场景渲染/抓放/记录需在新独立环境执行后才算通过；正式组合留出、评测实现和规模化生成仍待后续完成。

### 17.5 公开数据与独立环境的实际执行状态

`3337884` 在百度完成仅RGB本体的manifest重建及四例development预览，均退出0。随后发现GPU0有其他compute进程、约40GiB实际占用，因此本轮没有启动文本cache与完整GPU训练测试，没有终止这些进程。该项不能沿用先前multirobot测试结论宣称新benchmark链路已通过。

独立环境准备退出0，`environments/cliport-v71` 实际安装torch2.7.0+cpu/torchvision0.22.0+cpu，官方CLIPort固定源码已到 `third_party/cliport-2be5c47`；运行环境清单位于 `metadata/object_event_benchmark_environment_v71.json`。旧训练环境未改动。

首次完整模拟入口 `9c2db59` 在导入官方CLIport utils时因缺少matplotlib退出1，没有生成轨迹。源码修订 `cb53f5d26beed6f39833bbad25fa42264035a978` 只补充matplotlib3.10.3，已推送并交由SWXC从GitHub获取，用同一场景、seeds17/18重跑。此次失败日志在旧运行根 `logs/cliport_event_sequences_v71_9c2db59_20261015T121641Z/generate.log`；修订后的实际结果待返回。不能把包安装完成写成物理生成、事件标签或模型能力验证完成。

### 17.6 四条真实物理轨迹已完成

`cb53f5d` 经GitHub在百度取得，独立环境更新退出0，完整模拟生成也退出0。两组seed17/18各执行正反顺序，共四条，未更换seed或丢弃失败样例；四条官方目标reward均为1.0，每次放置事件各得到0.5。相同seed两条序列的3秒、16帧历史RGB逐像素一致，后续指令和实际执行次序不同。

| 场景/顺序 | 总帧数 | 历史终点后的记录长度 | 完成的放置事件 |
| --- | ---: | ---: | ---: |
| seed17 / 01 | 48 | 6.3083秒 | 2 |
| seed17 / 10 | 52 | 7.0792秒 | 2 |
| seed18 / 01 | 58 | 8.2917秒 | 2 |
| seed18 / 10 | 53 | 7.3292秒 | 2 |

RGB为640x480，240Hz物理引擎每48步采样一次，并保留不满0.2秒的最终时点。每例实际落盘 `video.mp4`、`timeline.jpg`、`episode.json`、`truth.npz`；后者含 `[T,480,640]` uint8直接分割ID、`[T,6,7]` XYZ/四元数、`[T,6]`可见像素数、真实tick与秒。六个实体ID为5--10，机器人body ID为2；标签来自实际模拟器，不来自模型mask或tracker。

结果目录：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/cliport_event_sequences_v71_cb53f5d_20261015T122104Z`，入口 `index.html`；日志在同名 `logs` 目录。举例seed17的指令是先把green block放入cyan bowl，再把brown block放入yellow bowl；反序样例交换这两个事件，场景初始内容不变。事件tick记录oracle控制开始/结束，不以此冒充物体实际motion onset。上述reward只确认数据生成任务完成，不是所训练模型的预测成功率。

下一项使用这些真实标签建立独立主体/顺序/轨迹评测，再冻结按scene与组合划分的规模化生成协议；四条工程样例本身不构成正式benchmark结果。常规TextOCVP GPU测试仍待空闲GPU，不能以CPU模拟成功替代。

**训练源修订**：上述初次运行保存了有损H.264展示视频，history相同结论来自编码前原图，不能直接扩展到解码MP4的像素。生成器现同时保存 `rgb.npz` 的无损uint8原始序列及时间，以它作为训练/评测输入；`video.mp4`仅作preview。两种指令必须在真正交付给模型的history原图上相同，不能让视频压缩差异成为未来条件线索。需要按原seed重新生成无损版本，旧结果保留作历史执行证据。

**无损版实测完成**：`c37762c` 在20:33:27 CST正常结束，四条reward/事件/帧数与原运行相同，生成进程376331已退出。从实际 `rgb.npz` 读回的shape依次为 `[48,480,640,3]`、`[52,480,640,3]`、`[58,480,640,3]`、`[53,480,640,3]`，均为uint8。两组16帧history的最大/平均像素差均为0；RGB/episode/truth的帧数、时间及物理tick完全对应，最大时间差0。四个preview视频均可解码且帧数匹配。单例无损RGB文件约1.20--1.43MB。

当前应查看和使用的结果根是 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/cliport_event_sequences_v71_c37762c_20261015T123110Z`，有 `index.html`、集合manifest及 `lossless_rgb_verification.json`。它替代旧cb53f5d目录作为后续模型输入来源；旧目录不删除。此结果完成了可追溯的受控数据生成链路，独立模型评测和正式组合划分仍未完成。

已准备服务器上可独立重跑以下CPU生成命令，不联网：

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=c37762c476c2df2c192f15d76600f8e7f8b3492c
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/environments/cliport-v71/bin/python"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="${WORK_ROOT}/tmp/benchmark-v71"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/benchmark"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/benchmark-pycache"
export OUT="${WORK_ROOT}/outputs/cliport_event_sequences_v71_$(date +%Y%m%d_%H%M%S)"
"${PYTHON}" "${ROOT}/code/scripts/generate_cliport_event_sequences_v71.py" \
  --environment-report "${WORK_ROOT}/metadata/object_event_benchmark_environment_v71.json" \
  --out "${OUT}" --seeds 17,18 --source-revision "${SOURCE_REVISION}"
GENERATE_RC=$?
printf 'GENERATE_RC=%s\nREVIEW=%s/index.html\n' "${GENERATE_RC}" "${OUT}"
```

## 18. 独立主体与顺序测量

### 18.1 固定查询，不泄露真实结束时刻

- **What changed**：受控生成默认保留3秒历史，并继续真实物理模拟至至少未来10秒；模型统一请求16帧历史、50帧未来，而不使用每个oracle任务的实际结束时间选择query长度。额外保存仅含历史的 `history.npz`，部署请求不再指向包含真实未来的文件。旧机器人数据与公开TextOCVP帧步协议不变。
- **Why**：正反顺序实际执行耗时不同；把视频末尾时间作为模型请求条件，会把本应预测的未来信息带进输入。
- **Impact**：`simulation_data.py`接入共同训练与评测，输入只有RGB、请求时间和指令，不加载pose/mask truth。真实标签仅由evaluator另读。无匹配真帧的时间点标记无效，不把更早末帧充作终点。未完成oracle程序保留RGB数据，但不作为已实现语言结果训练prior；实际任务完成状态始终记录。

### 18.2 测量定义和不能支持的结论

1. **初始绑定与后续归属**：只在最后一帧已观察图像上，用Hungarian匹配候选标签和模拟器实体ID，随后不再用未来GT重排。逐实体、逐帧报告IoU与未预测到的帧数。V71提供的是上采样后的feature-field ownership，不是独立RGB segmentation head；这一指标测空间归属，不独自证明RGB decoder的删除局部性。
2. **Visible-centroid误差**：比较预测区域与GT可见区域的质心，单位为原生像素；另比较各自相对历史终点的位移。归一化分母是该实体GT可见质心的RMS位移，低于1原生像素时不计算ratio，而报告漂移。无预测区域记为图像对角线误差并保留missing计数，不丢弃失败帧。遮挡会改变可见质心，因此不将它称为3D物理中心误差；另列模拟器真实world-center位移/路径长度供解释。
3. **主体区域RGB误差**：区域来自真实actor在当前帧、历史终点和配对另一条真实未来的mask并集；剔除两条真值中的机器人/工具像素。计算原生坐标下的RGB L1，按主体和时间逐项保留。它不根据模型自己声称的区域选择评分位置。
4. **顺序结果偏好**：在完全相同场景和相同请求时间，分别计算预测到实际A后B和实际B后A结果的主体区域误差；差值表示更接近哪一条真实结果。这是配对结果偏好，不是通用任务成功率或motion-onset分类。更换为反序语言后，评分参考也切换到反序真值，不能用原序GT惩罚正确服从的新结果；无语言条件的合法结果不唯一，单独标记。

模型入口 `evaluate_cliport_events_v71.py` 将posterior、zero effect、state-copy和可选的正确/无/反序语言生成全部接到上述同一评分器。输出逐帧JSON、原生分辨率预测NPZ/视频和仅历史的推理请求；posterior明确使用真实未来，prior不读取未来条件。

### 18.3 先用真实控制结果验证评分器

`check_cliport_event_metrics_v71.py`对同一实测场景比较真实正确结果、真实反序结果、当前图复制、两结果的RGB均值，以及只交换future实体标签但保持RGB正确的控制。检查正确结果RGB误差0/IoU1、反序偏好符号相反、均值无顺序偏好、身份交换只改变跟踪指标而不改变RGB误差。这不是训练模型成绩。每一条原始frame/entity记录先落盘，再给逐例摘要；当前代码已实现，真实CPU执行结果待返回。

### 18.4 正式划分计划

`make_cliport_event_plan_v71.py`在模拟前固定scene seed、颜色和允许的指令，不按未来运动或执行成功挑选场景。默认1000/100/200个train/development/test场景，同一场景的全部程序保持在一个集合内。六种颜色中的有向相邻actor-color对作为未见顺序组合，训练排除这些方向，但可学习反方向和各自原子放置。development/test同时包含已见和留出方向，并报告测试原子actor-color/target-color关系是否已在train出现。

这项留出检验的是“已见原子放置的新主体先后组合”，不称新物理技能，也不把test新颜色当主要变量。正式计划和生成后的成功/失败覆盖需另列；四条工程样例不被重新包装为正式test。训练按scene均匀访问，再选其中一个可用指令，避免双指令场景自动获得双倍权重。

### 18.5 百度实际端到端结果

- **What changed**：`92db4e782fcc71bc93b2783ae8235a314b2d74de` 在百度完成固定10秒模拟数据读取、真实结果评分校准、完整world/prior更新与保存恢复、history-only生成，以及同一模型的独立逐例评价，所有进程正常结束。新数据根为 `outputs/cliport_event_sequences_v71_c682ec9_20261015T133041Z`，四例均66帧、reward1、16帧history相同；模型manifest为 `metadata/cliport_event_v71_engineering_c682ec9.json`。world/prior各两步，resume参数差异0，causal差异0；history-only请求只读取 `history.npz`，产生50帧/10秒结果。
- **Why**：需要分别确认评分器能区分已知正确/错误结果、模型链路能实际运行，以及训练模型究竟表现如何，不能把三者合并成一个“通过”。
- **Impact**：这仍是四例工程测试，不是正式留出结果。正式plan已落 `metadata/cliport_event_v71_plan_c682ec9.json`：1000/100/200场景、1600/200/400程序，scene不重叠，train覆盖30种原子颜色关系，dev/test没有未见原子关系。1300个场景尚未生成，也未开始八卡长训。

评分器的真实控制结果：四例的11项检查均通过。正确结果RGB误差0、IoU1、centroid误差0；seed17/18的正确顺序margin为0.134194289/0.175817626，反序为对应负值，两真实结果RGB均值的margin约0。只交换实体标签时RGB误差仍0，但主体IoU降为0，说明成像与身份不能相互替代。证据在 `outputs/cliport_event_metric_check_92db4e7_20261015T134435Z/{report.json,cases.jsonl}`。

两步模型的主体区域RGB L1：posterior 0.554148、zero 0.550133、state-copy 0.402913、正确语言0.547555、无语言0.547535、原图copy 0.169803。四例共8个actor没有匹配到候选field，400/400个可见actor帧缺少预测，IoU为0；800px数值是显式missing惩罚，不是测得的实际轨迹偏差。它说明初始化两步模型尚未建立有效表示，不能由此判断长训后方法是否成立。逐例与视频在 `outputs/cliport_event_eval_92db4e7_20261015T134435Z`；恢复证据在 `outputs/cliport_event_e2e_92db4e7_20261015T134435Z/test_report.json`。

同一新协议的单A100 B2 profile完成，未改变模型或loss：

| 实测项 | Step 1 | Step 2 |
| --- | ---: | ---: |
| Loss | 2.7984633446 | 2.7513113022 |
| Gradient norm | 12.6962766647 | 3.2945632935 |
| 单步秒数 | 7.124901 | 4.366188 |
| 样例/秒 | 0.280706 | 0.458066 |
| Peak allocated GiB | 51.331513 | 51.542677 |

实际profile根：`outputs/cliport_event_batch2_92db4e7_20261015T141409Z`，日志在同名 `logs` 目录，退出0。后续受控16+50帧协议八卡采用每卡B2、累积16、global batch256；原3+5秒41帧的B4实测不直接迁移到这个更长窗口。上述吞吐只是单卡两步，不是八卡稳态测量。

公开TextOCVP本体的 `3337884` 单卡完整测试也已完成：三种effect模式及prior恢复参数差异均0，78条完整指令cache可用，1+9帧单位为sampled_frame，峰值显存13.408241GiB。它同样只确认运行链路，没有正式benchmark训练成绩。

### 18.6 消除固定输出的顺序偏好

- **What changed**：新增 `summarize_cliport_event_pairs_v71.py`，直接读取现有 `cases.jsonl`，不加载模型或重跑视频。每个scene先保留两种指令逐帧的“另一真实结果误差减指定结果误差”，再在共同时间取两者平均，最后按scene汇总。evaluator与评分控制入口使用同一实现。绝对主体区域RGB误差继续并列报告。
- **Why**：同一个不随语言改变的预测，可能偶然更像其中一种顺序；只看单例margin或“2/4正值”不能证明语言有效。两种指令使用相同history、噪声和对称评分区域时，固定输出的成对margin应为0。
- **Impact**：成对分数只测真实配对结果偏好，不宣称任务成功或物理事件onset。posterior/真实oracle读取未来，只作为上限；仅history/语言条件承担部署结论。真实控制应表现为正确结果正值、反序负值、copy和两结果均值约0；已保存四例结果的CPU配对汇总待执行。

**实际配对汇总完成**：百度两个CPU入口均退出0，未重算模型/视频，使用 `280021f`。两场景各50个共同未来帧，无未配对场景。正确真实结果的scene平均margin为0.15500595772638917，反序为对应负数；copy和两结果RGB均值严格为0。两步模型正确语言margin仅0.0000048857927322387695，无语言为0，posterior为0.00000184670090675354；正确语言主体RGB L1为0.5475552627，copy为0.1698028974。前述单例正负偏好不再被解释成语言遵循。原始结果目录下均新增 `paired_summary_280021f.json`；identity-swap的RGB完全未变，因此仍得到oracle的RGB分数，其身份错误由独立ownership测量识别。

**组合切片接入**：正式evaluator保留每个program的 `combination`，反序语言同时切换 `reference_combination`。配对汇总分别报告contains-held-order与seen-only场景；单方向的绝对actor RGB误差按实际参考指令的seen/held类别报告，并保留逐例值。这里不把某一方向的单例margin当语言收益，也不将工程样例归入正式seen/held。旧报告若缺标签明确列为unspecified；不根据case编号反推标签。标签只进入评价记录，不成为模型输入或训练监督。

正式生成已出现oracle未完成的实际样本，因此摘要同时保留各program的oracle状态；某条件只有一侧结果、或没有共同可评分真帧时，配对结果明确列出 `unscored_conditions` 和原因。逐条件scene分母只含实际评分的配对，不把失败样例写成零误差，也不只在原始大JSON中隐藏这些缺失。

### 18.7 平台范围更新

- **What changed**：用户本轮明确回复skip，跳过阿里备用阶段的网络、环境和权重准备；不再查询或重试其代理访问。已完成的阿里数据和只读记录保留。
- **Why**：百度已经完成真实数据、模拟器和GPU链路，后续实验以百度旧目录为主。
- **Impact**：安装、数据、模型与结果仍全部放服务器PFS，不放Mac；新代码继续本地提交GitHub、SWXC从GitHub取到百度执行。阿里没有被记为GPU实验通过。

### 18.8 已有结果的独立CPU配对汇总

`280021f08636f525e62c88dfb032f571a478ec98` 已推送。准备机按16.1取得该release后执行以下命令；不联网、不读GPU、不重算已有视频，原始report/cases保持不变。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=280021f08636f525e62c88dfb032f571a478ec98
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/environments/cliport-v71/bin/python"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for RESULT in cliport_event_metric_check_92db4e7_20261015T134435Z cliport_event_eval_92db4e7_20261015T134435Z; do
  "${PYTHON}" "${ROOT}/code/scripts/summarize_cliport_event_pairs_v71.py" \
    --cases "${WORK_ROOT}/outputs/${RESULT}/cases.jsonl" \
    --out "${WORK_ROOT}/outputs/${RESULT}/paired_summary_280021f.json"
  printf 'SUMMARY_RC=%s RESULT=%s\n' "$?" "${RESULT}"
done
```

后续完整 `evaluate_cliport_events_v71.py` 自动将同场景两条程序一起选入 `--items` 的预算，并写入 `report.json/paired_outcome_preference`。`--items` 是case数而非scene数，使用偶数；未配对场景明确列出，不能在均值中隐去缺失。

### 18.9 正式集合的并行与续跑

- **What changed**：生成入口新增 `--workers`，每个spawn子进程独立创建PyBullet环境并处理一整个scene的程序；子进程只写自己的scene目录，主进程统一写集合manifest和review。`progress.json`区分已落盘程序、oracle成功程序和本轮复用程序。显式 `--resume` 在同一plan/时间协议下复用已落盘 `episode.json`，包括oracle未完成的样本；不替换seed、不为成功而重试。
- **Why**：正式计划有1300场景、2200程序，串行生成与中断后全部重做会重复消耗CPU时间。续跑应恢复生成进度，而不是选择更容易成功的数据。
- **Impact**：每条RGB、truth、history及preview写完后才原子发布episode元数据。未发布的程序可重新生成；已发布程序不改写。默认workers=1保留原入口，完整真实测试采用2个CPU进程生成四例并再次续跑，再做已有独立评分；不新增训练环境、GPU依赖或大量单元门禁。正式集合尚未因实现此入口而被标记已完成。

**真实并行/续跑完成**：`d1301eb54293bf25227a97121bf15192cec3267f`在百度独立CPU环境执行四例，首次两进程生成86.553秒，resume18.143秒，评分30.948秒，manifest准备1.676秒，全部退出0。四例66帧、reward1；与原串行c682ec9对应RGB的最大差均0。resume为4复用/0新生成，16个episode/NPZ文件的mtime与大小完全未变。两场景16帧history相同，paired控制检查全部通过。结果根为 `outputs/cliport_event_parallel_d1301eb_20261015T151251Z`，实际回执为其中 `parallel_verification_receipt.json`。容器CPU配额11核、内存40GiB；正式生成选4进程，不能把宿主可见116核当可独占额度。

统一CPU入口 `prepare_cliport_ordered_benchmark_v71.sh` 只顺序执行真实生成与manifest准备，不联网、不使用GPU、不启动训练。默认沿用已冻结的 `metadata/cliport_event_v71_plan_c682ec9.json`，数据写入旧运行根 `data/benchmarks/cliport_ordered_v71`，完成后写 `metadata/cliport_event_v71.json`。`GENERATE_WORKERS`默认4，续跑显式设 `RESUME_DATA=1`；中断后不改plan或时间协议。每次启动有独立日志，进度文件区分生成落盘和oracle成功，不能把两者混为一谈。

**正式生成当前状态**：已向授权SWXC对话发送048a3fc的4进程启动请求，但远控返回 `Timed out waiting for MCP response to fs/createDirectory`，随后读取同一thread暂不可达。尚无本次正式生成的session/PID/progress证据，因此不记为running或failed，也不因观察超时重复启动。恢复后先读取该对话与同一集合的实际进程状态，再决定继续观察或启动；四例d1301eb已完成的结果不受此项影响。

**随后实际核对（2026-10-15 23:45 CST）**：SWXC恢复后在百度查询到对应生成进程为空、collection目录不存在，确认此前请求没有启动正式生成。新提交0fd4346随后经GitHub取得，正式数据参数和计划保持不变；已让同一授权对话直接运行已有prepare入口，不新增封装。之后远控再次暂不可达，尚未取得这次新请求的进程回执，不把“源码到达”记成“数据生成运行中”。

### 18.10 正式数据与首轮八卡命令

准备机先按16.1取得 `048a3fcd95b8041577929be8a29d7381b526a529` release。下面各段可独立执行，训练不访问GitHub或下载模型；数据生成完毕后再使用其manifest。测试数据仍保留，不依据测试结果选择模型。

**CPU数据生成/续跑**：只运行这段不会训练，也不占用GPU。`RESUME_DATA=1`明确表示复用该集合已落盘程序。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=048a3fcd95b8041577929be8a29d7381b526a529
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export GENERATE_WORKERS=4 RESUME_DATA=1
export PLAN="${WORK_ROOT}/metadata/cliport_event_v71_plan_c682ec9.json"
export COLLECTION="${WORK_ROOT}/data/benchmarks/cliport_ordered_v71"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
bash "${ROOT}/code/scripts/prepare_cliport_ordered_benchmark_v71.sh"
DATA_RC=$?
printf 'DATA_RC=%s\nPROGRESS=%s/progress.json\nREVIEW=%s/index.html\n' \
  "${DATA_RC}" "${COLLECTION}" "${COLLECTION}"
```

**单卡指令特征准备**：读取已经存在的Qwen，只缓存完整文本，不重新编码视频。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=048a3fcd95b8041577929be8a29d7381b526a529
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
"${WORK_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/prepare_object_event_text_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" \
  --model /mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct \
  --out "${WORK_ROOT}/metadata/cliport_ordered_text_v71" --batch 16
```

**八卡joint首轮**：固定1000 optimizer steps、global batch256，warmup50、rollout ramp100；这是新受控集合上的首轮测量预算，不是已证明收敛的训练步数。三种effect模式使用同一预算/seed/split，先跑object_step；改变 `EFFECT_MODE` 为global_step或object_segment会产生独立输出目录。单卡B2的实测只作为显存配置依据，不承诺八卡吞吐。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=048a3fcd95b8041577929be8a29d7381b526a529
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export EFFECT_MODE=object_step
export RUN_NAME="cliport_ordered_v71_${EFFECT_MODE}_seed17_048a3fc_1000"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export BATCH_PER_GPU=2 GRAD_ACCUM=16 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=4 DECODER_FRAME_BATCH=2
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  --warmup 50 --rollout-ramp 100 --save-every 100 --snapshot-every 1000
TRAIN_RC=$?
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "${TRAIN_RC}" "${OUT}"
```

恢复使用完整同段命令，但把 `unset RESUME INIT_FROM` 改成 `unset INIT_FROM`，并加 `export RESUME="${OUT}/latest.json"`。恢复读取保存的优化器、schedule、RNG和data cursor；不改卡数/批量继续同一个run。准备数据的 `RESUME_DATA` 与训练 `RESUME` 是两件事。

**八卡prior首轮**：只在对应world快照完成后运行。world/renderer固定，训练history/完整指令到effect；不是读真实未来的posterior生成。每种world模式对应独立prior，不混用teacher。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=048a3fcd95b8041577929be8a29d7381b526a529
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/cliport_ordered_text_v71"
export EFFECT_MODE=object_step
export BASE_CHECKPOINT="${WORK_ROOT}/outputs/cliport_ordered_v71_${EFFECT_MODE}_seed17_048a3fc_1000/step_00001000.pt"
export RUN_NAME="cliport_ordered_prior_v71_${EFFECT_MODE}_seed17_048a3fc_1000"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export BATCH_PER_GPU=32 GRAD_ACCUM=1 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=64 DECODER_FRAME_BATCH=2 LANGUAGE_DROPOUT=0.1
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_prior_v71.sh" \
  --warmup 50 --save-every 100 --snapshot-every 1000
TRAIN_RC=$?
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "${TRAIN_RC}" "${OUT}"
```

**单卡development评测**：200个case对应100个scene的两种程序；每例保存原生RGB、独立主体评分和history-only请求，report自动给出成对顺序分数。正式test不参与首轮选择。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=048a3fcd95b8041577929be8a29d7381b526a529
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export EFFECT_MODE=object_step
export OUT="${WORK_ROOT}/outputs/cliport_ordered_eval_v71_${EFFECT_MODE}_1000_development"
"${WORK_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/evaluate_cliport_events_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" \
  --assets "${WORK_ROOT}/metadata/rae_assets_512.json" \
  --prior-checkpoint "${WORK_ROOT}/outputs/cliport_ordered_prior_v71_${EFFECT_MODE}_seed17_048a3fc_1000/step_00001000.pt" \
  --text-cache "${WORK_ROOT}/metadata/cliport_ordered_text_v71" \
  --partition development --items 200 --out "${OUT}"
printf 'EVAL_RC=%s\nREPORT=%s/report.json\n' "$?" "${OUT}"
```

### 18.11 正式集合与文本cache完成（2026-10-16）

- **What changed**：原session32782正常退出0，完成1300个场景、2200条program视频，并自动写出训练manifest；没有重启生成。随后现有Qwen离线单卡文本cache退出0，保存343条唯一完整指令，width2560、input=instruction_only、truncation=false。
- **Why**：正式训练现在有了实际已生成的scene-disjoint集合，不再只有四例工程数据或尚未兑现的采样计划。
- **Impact**：joint保留全部1600条训练RGB，语言prior只使用1597条oracle完成的训练program；按scene均匀访问，不按program条数增加scene权重。未完成样本原样保留，未为成功率换seed或重试。正式GPU长训和模型间比较尚未开始。

统计来自实际读取2200个episode.json及两个manifest，case ID集合一致、partition/combination/status无冲突：

| Partition | Scenes | Programs | Oracle完成 | Oracle未完成 | Seen actor order | Held actor order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1000 | 1600 | 1597 | 3 | 1600 | 0 |
| development | 100 | 200 | 200 | 0 | 160 | 40 |
| test | 200 | 400 | 399 | 1 | 320 | 80 |
| 合计 | 1300 | 2200 | 2196 | 4 | 2080 | 120 |

三个partition的scene交集均为0。各集合实际覆盖30种原子actor-color/target-color关系，仅统计oracle完成program也均为30；development/test相对train缺失原子关系均为空。有序actor-color pair数分别为24、30、30，测试的是已见原子关系的新先后组合，不称为新物理技能。

集合中900个scene有两种program，生成器记录的900组历史RGB逐像素比较均相同，false/缺失均为0；另400个scene只有一种允许的训练program。语言prior实际采样分母为998个scene，而不是1000或1597：scene1129、1990的唯一program未完成，因此没有语言训练样本；scene1008仍有另一条已完成program。joint分母仍为1000个scene。

保留的oracle未完成记录：train的 `scene_001008_order_10`、`scene_001129_order_10`、`scene_001990_order_10`，reward均0.5；test的 `scene_002170_order_01`，reward0。最终progress为completed，planned/finished一致，reused=0。它们的真实RGB仍可用于重建；没有把未执行完成的指令当作成功后果监督。独立条件评测必须保留这些状态与实际分母。

正式产物均在百度旧运行根 `/mnt/pfs/public/xuhaoming/instruct_gs_world`：

- 数据与人工查看：`data/benchmarks/cliport_ordered_v71/`、其中 `index.html`。
- 原集合及物理真值映射：`data/benchmarks/cliport_ordered_v71/manifest.json`、各case的 `episode.json` 与 `truth.npz`。
- 训练统一入口：`metadata/cliport_event_v71.json`，3秒/16帧历史、固定10秒/50帧未来。
- 文本cache：`metadata/cliport_ordered_text_v71/`，343个 `.pt` 与 `index.json`、`completion.json`。
- 原数据生成日志：`logs/cliport_ordered_v71_data/20261015T155452Z.log`。

数据生成源码为 `0fd4346ffd3b5437ff63291e45777ad8c2fbf2fa`；文本准备与当前完整模型测试源码为 `aab89134da946b4ea2286929059cac503745dc22`。只更新文档不要求重造数据或重复测试。18.10/22.5的正式manifest和cache路径现在对应已存在的产物。

## 19. 外部对照的实际接口准备

- **What changed**：再次读取[LPWM官方LanguageTable准备脚本](https://github.com/taldatech/lpwm/blob/main/datasets/langtable_preparation.py)、[loader](https://github.com/taldatech/lpwm/blob/main/datasets/langtable_ds.py)与[运行说明](https://github.com/taldatech/lpwm)。官方路径从RLDS读取，选择15至50帧episode，用seed42划分80/10/10，保存逐帧PNG和T5-large文本；loader默认128尺寸、存在末帧padding及有效标记，valid默认只取100个episode。此处记录的是代码事实，不声称已经复现论文表格或取得其精确划分清单。
- **Why**：不能用已有multirobot或四条模拟工程样例替代已规划的LanguageTable主实验，也不能把三种V71内部effect变体当全部外部baseline。
- **Impact**：下一步先取得官方原始数据/episode身份与真实准备结果，再冻结共同窗口与评价清单；padding帧不当作真实未来，T5默认32-token截断与本项目完整指令cache的条件差异需分开报告。后续LPWM/T5/VGG/LPIPS准备及单卡测试已获授权并完成，见21.5；TensorFlow/TFDS与LanguageTable全量数据不在本轮执行范围。正式模型比较仍未完成。

**已经核对的LPWM调用边界**：[官方 `DLP.sample_from_x`](https://github.com/taldatech/lpwm/blob/main/models.py) 支持直接传history RGB。部署生成必须使用 `use_all_ctx=False`、`n_pred_eq_gt=False`，明确给 `cond_steps`、`num_steps` 和文本embedding；后一个参数关闭“输入GT长度必须等于输出长度”的假设，不需要把真实未来或dummy future拼进模型输入。`use_all_ctx=True` 会从完整视频提取context，只用于标明读未来的posterior reconstruction。构造可复用[官方配置loader](https://github.com/taldatech/lpwm/blob/main/generate_lpwm_video_prediction.py)的无checkpoint路径，再直接 `load_state_dict` 加载指定快照；不沿用其捕获所有加载错误后继续生成的分支。尚未据此声称baseline已运行。

### 19.1 官方原始数据字段已确认

- **What changed**：2026-10-16直接读取官方GCS的 [dataset_info.json](https://storage.googleapis.com/gresearch/robotics/language_table/0.0.1/dataset_info.json) 和 [features.json](https://storage.googleapis.com/gresearch/robotics/language_table/0.0.1/features.json)，而非从论文摘要猜schema。版本0.0.1只有名为train的原始split，登记numBytes为428,665,067,806；顶层为字符串episode_id与可变长steps。每步RGB为360x640、uint8/JPEG，instruction为512维int32的UTF-8字节序列。其余是二维action/effector坐标、reward与first/last/terminal标记，没有显式帧时间戳、对象mask或对象位姿。
- **Why**：LPWM准备脚本再从原始集合划80/10/10，不能将其二次划分称为原始官方test；[官方README](https://github.com/google-research/language-table)同时明确原始real集合经过relabel。仅有episode_id字段尚不能证明不同ID对应完全独立的录制，真实ID格式与重标注来源映射仍需读取样本后确认。
- **Impact**：首个适配以frame-step记录时序，除非获得实际采集时间证据，否则不填写推测FPS/秒数。只将RGB和原始指令接入预测模型，机器人action不成为隐藏条件；它用于真实视频重建/条件生成，不提供本项目独立主体或事件顺序GT。先形成可追溯的episode/窗口划分再比较；不以随机拆重标注ID自动证明录制级留出。当前仅远程读取少量公开JSON元数据，未下载视频shards、未安装TensorFlow/TFDS，也未生成新的训练集合。

## 20. 相同token预算的非对象对照

- **What changed**：新增 `scene_state.py` 与 `code/configs/object_event_scene_control_v71.json`。保留相同DINOv2/RAE、68个384维state tokens、17个64维global effect tokens、Dynamics层数、目标和采样；每个scene token独立读取全图，取消对象之间的竞争分配，memory在68个tokens之间全局attention。renderer将所有tokens汇成一个场景field，不再为每个候选对象独立成像和mask合成。使用 `global_step` effect路径，历史和effect cross-attention都是全局的。
- **Why**：只有global-step/object-step/object-segment三个对象模型，不能隔离对象状态组织本身的价值。此对照与object-layout/global-step先比较状态与renderer组织；后者再与object-step比较effect绑定，object-step与object-segment比较持续时间。它是受控内部对照，不替代LPWM/TextOCVP外部主比较。
- **Impact**：场景对照的 `[17,4]` 只是共享存储接口，不代表17个物体。参数形状除renderer去掉一个输出mask logit外保持一致，参数差为384+1=385；run/test报告记录实际模块参数、state tokens及effect维度，不能把近同参数写成同FLOPs。场景renderer不输出实体分割，因此独立evaluator将ownership列为不可用，而非把整图label当物体真值；外部actor区域RGB与配对顺序分数仍使用同一GT。旧checkpoint默认object-layout，新scene模型从头训练，不重写已有实验。完整单卡训练/恢复/prior/生成测试仍待服务器执行，尚未宣称对照有效。

### 20.1 单卡完整测试

当前数值测试修订为 `0fcca583f4b475fae6cc1f05ff8e65b20f4eea7b`，默认启用明确的确定性计算设置，保留原比较阈值；此前0fd4346失败证据见20.4。准备机取得该release后运行；这段不联网、不改已有模型或数据。文本cache路径从已完成的四例prior运行记录读取，不猜测其目录名。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=0fcca583f4b475fae6cc1f05ff8e65b20f4eea7b
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71_engineering_c682ec9.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME="cliport_scene_control_e2e_0fcca58_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
export TEXT_CACHE="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["args"]["text_cache"])' \
  "${WORK_ROOT}/outputs/cliport_event_e2e_92db4e7_20261015T134435Z/language_prior/recovered/run.json")"
bash "${ROOT}/code/scripts/test_object_event_world_model_v71.sh" \
  --model-config "${ROOT}/code/configs/object_event_scene_control_v71.json" \
  --modes global_step --train-partition engineering --evaluation-partition engineering \
  --text-cache "${TEXT_CACHE}"
printf 'TEST_RC=%s\nREPORT=%s/test_report.json\n' "$?" "${OUT}"
```

### 20.2 八卡scene对照

以下joint/prior分别前台运行；正式数据与文本cache完成后使用。预算与18.10的object对照相同，模型从头训练。内部主比较是scene/global与object/global，不把scene/global和object/segment之间的全部差异归因于单一机制。此处保留高吞吐算法默认设置，恢复完整训练状态但不宣称CUDA浮点逐位一致；20.1的数值一致性测试单独使用确定性设置。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=0fcca583f4b475fae6cc1f05ff8e65b20f4eea7b
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export EFFECT_MODE=global_step
export RUN_NAME=cliport_scene_global_v71_seed17_0fcca58_1000
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export BATCH_PER_GPU=2 GRAD_ACCUM=16 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=4 DECODER_FRAME_BATCH=2
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  --model-config "${ROOT}/code/configs/object_event_scene_control_v71.json" \
  --warmup 50 --rollout-ramp 100 --save-every 100 --snapshot-every 1000
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "$?" "${OUT}"
```

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=0fcca583f4b475fae6cc1f05ff8e65b20f4eea7b
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/cliport_ordered_text_v71"
export BASE_CHECKPOINT="${WORK_ROOT}/outputs/cliport_scene_global_v71_seed17_0fcca58_1000/step_00001000.pt"
export RUN_NAME=cliport_scene_global_prior_v71_seed17_0fcca58_1000
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export BATCH_PER_GPU=32 GRAD_ACCUM=1 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=64 DECODER_FRAME_BATCH=2 LANGUAGE_DROPOUT=0.1
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_prior_v71.sh" \
  --warmup 50 --save-every 100 --snapshot-every 1000
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "$?" "${OUT}"
```

恢复时保留完整环境，替换 `unset RESUME INIT_FROM` 为 `unset INIT_FROM`，增加 `export RESUME="${OUT}/latest.json"`；实际scene配置从checkpoint恢复。正式评价仍使用18.10的同一evaluator和development清单，只将prior-checkpoint与OUT切换为对应scene run，不为scene单独挑case或改变actor评分区域。

### 20.3 当前执行事实

`0fd4346` 已在百度release中取得，实际文本cache路径从旧prior记录读出为 `metadata/cliport_event_text_v71_c682ec9`。单卡完整测试及四例独立评价已交由同一SWXC对话执行，但截至本轮尚无新训练进程或test_report回执，远控读取暂不可达。模型/模块参数、恢复数值和运行显存仍需实际结果，不以代码差值或旧object-layout测试替代新scene对照的运行证据。正式八卡命令已交付，GPU长训尚未启动。

### 20.4 实测与数值恢复问题

- **What changed**：实际测试 `outputs/cliport_scene_control_e2e_0fd4346_20261015T160129Z` 已执行world训练/恢复及prior训练，但在prior最终数值对比处退出1，未运行独立eval。scene模型实测27,779,887参数（encoder3,287,040、renderer599,461、posterior2,270,344、Dynamics21,623,042），prior22,810,496参数；预算68 tokens/1088 effect标量。world两步loss2.813741/2.721269，梯度有限，峰值13.8307GiB。
- **Why**：CPU读取既有checkpoint发现两次独立启动在第1步就不同：参数最大差0.00039067585，一阶矩最大差2.59746e-5；第2步差0.00058930879。两步的四类RNG、case顺序、cursor、scheduler与学习率相同。因此这不是已经定位的resume丢状态问题，具体CUDA数值来源仍待进一步实测。
- **Impact**：新增训练 `--deterministic`，在CUDA初始化前明确设置cuBLAS workspace、确定性算法和cuDNN选项，跟随args进入checkpoint；默认高吞吐训练不变。完整数值恢复测试默认使用该设置，保留原1e-5阈值，并先报告恢复前第1步的差异。每段测试在断言前写 `resume_report.json`，失败也保留实际数值，不吞异常、不改loss。依据为[PyTorch2.8重现性说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)及[SDPA实现说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)；尚未把该设置写成已经修复。高吞吐模式完整状态恢复与逐位数值等价分开描述。

正式CPU集合已实际启动，session32782、generator PID440957，沿用0fd4346和四进程原计划。最近读取为100/1300场景、160/2200程序，其中159个oracle完成；未完成样本保留，未重复启动。进度文件为 `data/benchmarks/cliport_ordered_v71/progress.json`，日志 `logs/cliport_ordered_v71_data/20261015T155452Z.log`。这是运行中快照，不是数据完成报告。

### 20.5 确定性完整链路实测完成

- **What changed**：`0fcca583f4b475fae6cc1f05ff8e65b20f4eea7b` 在百度单A100完成scene完整测试与四例独立评价，两个入口退出0。world与语言prior在第1步和最终恢复后的参数差异均0，sample顺序相同、梯度有限；原阈值1e-5没有放宽。future-feature swap前缀差异与later-state swap首次预测差异均0，history-only forecast记录future_rgb_read=false、生成50个states。
- **Why**：同一数据/RNG不能自动保证CUDA运算逐次一致。此次明确计算设置后，恢复前的独立启动与恢复后的路径均对齐，解决了该完整测试中混入的数值歧义；这不等于已定位某一个具体kernel，也不把两步优化视为方法收益。
- **Impact**：world最高peak allocated13.87884GiB，prior1.44398GiB；独立eval没有单独采集峰值，不沿用训练数值冒充。四例六种模型条件共24/24正确标明ownership不可用，scene不伪造对象分割；actor RGB/order仍按相同模拟器真值计算。正式CPU进程随后确认仍活着，141场景/225程序、223个oracle完成，另2条保留；不重启它。

本次输出根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/cliport_scene_control_e2e_0fcca58_20261015T162702Z`。主要证据为 `test_report.json`、`global_step/resume_report.json`、`language_prior/resume_report.json`、`independent_cliport_events/report.json` 和 `independent_cliport_events/cases.jsonl`。新增组合标签统计的 `7c3612d` 只安排既有四例的CPU评分检查，不重复模型训练或GPU生成。

**评分元数据CPU实测完成**：`7c3612d21f0d699a46e3e3b0822a5b3d13e80d70` 的既有四例检查退出0、status=passed；两scene的全部paired checks通过。组合分组明确为engineering，5个条件各含2个scene pair；参考类别的绝对误差分组同为engineering，各条件含4个case。unscored与unpaired列表均空，没有把工程样例标为正式seen/held。报告为 `outputs/cliport_metric_composition_7c3612d_20261015T164619Z/report.json`，model_evaluated=false。真实held/失败program的正式统计仍等待完整集合与后续评价，不能由这四例代替。

### 20.6 Prior批量配置实测

- **What changed**：用相同scene teacher、完整16+50帧输入测量prior的B32、累积1、encoder frame-batch64，默认高吞吐算法，单卡退出0。prior入口默认设置更新为B32/acc1/frame64；显式环境值仍优先。18.10与20.2的八卡prior命令同步，global batch仍为256，joint配置不改。
- **Why**：prior不加载RAE decoder，也不对frozen teacher回传，其B1测试峰值只有1.44GiB。将joint的B2设置直接用于prior会留下很大的批量空间。
- **Impact**：实际profile仅两步、effective batch32，不是八卡性能测量。第一步21.1444秒（data13.3720秒）、1.5134样例/秒、22.5032GiB；第二步4.80839秒（data0.000577秒）、6.65504样例/秒、22.6823GiB。loss2.013954/1.619500，梯度60.00037/59.24242，均有限；预取让第二步几乎不等数据，不能把该点写成长期吞吐。结果在 `outputs/cliport_prior_batch32_0fcca58_20261015T165942Z`，未启动八卡长训，也未为profile重启正式数据生成。

## 21. LPWM原生外部对照接入

- **What changed**：固定官方源码 `4cf53c403433e64c01652ac2adbec66231a46dea`，独立 `object_event_baselines/lpwm.py` 调用官方DLP、ELBO和sampling方法，不替换为V71网络或loss。用户已授权百度独立环境与单卡完整测试；环境、权重和engineering文本cache已完成，`20f5e23`完整测试退出0，实际范围与结果见21.5。不是正式训练后的benchmark成绩。
- **Why**：内部scene/global/object/segment控制不能代替外部已有方法。接入要先核对真实调用与原生训练目标，再用同一视频/语言来源执行，不能仅凭论文摘要仿写一个“LPWM-like”模型。
- **Impact**：先接当前受控数据和公开图像序列接口。LanguageTable原始RLDS读取、LPWM正式主体/顺序评分与FVD对齐仍未完成；本轮不下载LanguageTable全量数据。环境/资产全部在百度旧PFS根，不在Mac安装或下载，不改原训练环境。

### 21.1 已核对的模型与loss

依据[固定版本DLP实现](https://github.com/taldatech/lpwm/blob/4cf53c403433e64c01652ac2adbec66231a46dea/models.py)与[原生训练代码](https://github.com/taldatech/lpwm/blob/4cf53c403433e64c01652ac2adbec66231a46dea/train_lpwm.py)：

1. 动态模型将encoder粒子数设为prior粒子数；LanguageTable配置因此编码256个particles，再为decoder选24个。新inventory读取实际对象属性，不把配置的24直接当总计算量，也不把粒子数当已验证物体数。
2. 保留原生Adam参数、ELBO系数、VGG/LPIPS重建及FP32计算。当前原生配置无LR scheduler；共享checkpoint保存支持scheduler=None，V71已有非空scheduler行为不变。warmup仍以数据epoch表示，scene均匀采样的epoch按实际访问数计算。
3. 原生loss可自行下载VGG/LPIPS。接入只改权重构造方式：从PFS显式加载TorchVision VGG与官方LPIPS文件，forward直接继承官方实现。训练进程不调用下载或hash校验，不改成MSE替代方法。criterion不进入model optimizer，模型checkpoint不重复保存该冻结VGG。
4. 未来posterior路径调用 `sample_from_x(... use_all_ctx=True)`；部署路径只传history，使用 `use_all_ctx=False, n_pred_eq_gt=False`。当前接口不读取机器人action或image goal。两类结果明确分栏。
5. 使用PyTorch原生non-reentrant activation checkpointing包装29个原生Transformer blocks，保留原forward、dropout RNG和state_dict键；不切换官方torch_attn分支的不同mask语义。FP32与完整16+50帧不变。所有成本比较需计入重算开销。

### 21.2 数据与条件对齐的边界

受控协议固定16历史/50未来，因此模型horizon由原生20调整为65，并记录原值/实际值；这是新受控任务适配，不宣称复现原论文原始训练配置。保留原生128像素模型，但改用全视野letterbox，避免中心裁剪丢掉边缘操作对象。当前128像素RGB分数不能直接与V71的512像素分数混表，正式共同尺度评测仍需完成。

语言采用本地T5-large。保留原生不传padding attention mask的编码方式；不截断原指令，cache长度取max(32,实际最长token数)，并在模型配置记录必要的language_max_len调整。若所有指令都不超过32，长度与原生配置一致；超过32时明确是完整指令适配，不是原样的32-token条件。T5只读语言，cache只保存小型文本张量，不保存视频特征。

无语言对照使用T5编码的空字符串，保留32个语言位置，不将lang_embed设为None。固定官方实现会在None输入时跳过语言token拼接，但仍删除末尾32个tokens，误删物理particles并造成维度错误。空字符串移除的是指令信息，不是拆掉网络分支；metadata记录这一对照定义，训练任务指令保持不变。

LPWM训练联合使用语言条件，因此使用与V71 prior相同的有效语言窗口集合；受控oracle未完成的视频仍保留在数据中，但不作为成功实现指令的LPWM训练标签。V71 joint可使用这些无标签RGB，两者的实际episode数分别记录，不能声称视觉样本暴露次数完全相同。

原生batch3/epoch150是外部方法的初始recipe，不等于V71的global batch256/1000步预算。正式比较前需报告实际访问数、更新数、两阶段总成本和预训练资产；不把不同loss的数值直接比较。完整长序列已测单卡micro-batch1；下列八卡模板用B1/累积3保持global batch24，不把它说成原生B3逐位等价或已测八卡吞吐。

### 21.3 工程入口与测试定义

| 文件 | 作用 |
| --- | --- |
| `setup_lpwm_environment_v71.py` | 用已有PFS uv创建独立CUDA环境，不改旧训练或CLIPort环境；不安装TensorFlow。 |
| `prepare_lpwm_assets_v71.py` | 准备固定LPWM源码、T5、VGG/LPIPS；只有此准备阶段联网。 |
| `prepare_lpwm_instruction_cache_v71.py` | 本地T5完整指令cache，记录token长度与条件变化。 |
| `train_lpwm_object_events_v71.py/.sh` | 原生FP32目标、scene均匀采样、DDP、完整checkpoint、仅训练标量SwanLab。 |
| `evaluate_lpwm_rgb_v71.py` | posterior、正确语言单样本、无语言及当前图copy，逐帧RGB/PSNR/SSIM/LPIPS和本地视频。 |
| `test_lpwm_object_events_v71.py/.sh` | 一次真实短更新、保存恢复、history-only生成、future swap及视频评价。 |

LPWM的CUDA STN含grid_sample，其backward可能非确定，见[PyTorch2.8说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.grid_sample.html)。测试验证模型/优化器/RNG的精确序列化往返、恢复后的step与采样顺序，并报告独立轨迹的参数差异，不把浮点逐位一致当已保证能力。测试只有在后续RGB评价也完成后才写passed；21.5记录本轮实际通过的范围。

### 21.4 独立执行命令

以下命令在百度准备机/训练任务中执行，不在Mac运行。环境、权重与单卡完整测试已执行；下列为可独立复现入口，不要求因文档更新重做准备。正式八卡训练尚未启动。

**源码同步，仅此段和资产准备段联网**：

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
"${WORK_ROOT}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlretrieve
root = Path(os.environ["WORK_ROOT"])
revision = os.environ["SOURCE_REVISION"]
destination = root/"runtime/object_event_world_model_v71/releases"/revision
temporary = root/"tmp/lpwm-v71"
temporary.mkdir(parents=True, exist_ok=True)
if not destination.is_dir():
    archive = temporary/f"source-{revision}.tar.gz"
    urlretrieve(f"https://codeload.github.com/healenrens/Instruction-GS-World/tar.gz/{revision}", archive)
    stage = Path(tempfile.mkdtemp(prefix="release-", dir=temporary))
    subprocess.run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(stage)], check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(destination)
print("ROOT="+str(destination))
PY
```

**独立环境和资产准备**：使用已安装的PFS uv，不动旧训练 `.venv`。不包含LanguageTable全量下载或TensorFlow安装；已完成的环境和权重可直接复用。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${WORK_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/setup_lpwm_environment_v71.py" \
  --work-root "${WORK_ROOT}" --python "${WORK_ROOT}/.venv/bin/python" &&
"${WORK_ROOT}/environments/lpwm-v71/bin/python" "${ROOT}/code/scripts/prepare_lpwm_assets_v71.py" \
  --work-root "${WORK_ROOT}"
printf 'PREPARE_RC=%s\nASSETS=%s/metadata/lpwm_assets_v71.json\n' "$?" "${WORK_ROOT}"
```

**四例已有视频的T5 cache与单卡完整测试**：复用已有RGB，不生成新视频数据，不下载任何模型。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/environments/lpwm-v71/bin/python"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71_engineering_c682ec9.json"
export ASSETS="${WORK_ROOT}/metadata/lpwm_assets_v71.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/lpwm_cliport_engineering_text_v71_empty"
export RUN_NAME="lpwm_v71_e2e_20f5e23_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
"${PYTHON}" "${ROOT}/code/scripts/prepare_lpwm_instruction_cache_v71.py" \
  --manifest "${MANIFEST}" --model "${WORK_ROOT}/models/lpwm-v71/t5-large" \
  --out "${TEXT_CACHE}" --batch 16 &&
bash "${ROOT}/code/scripts/test_lpwm_object_events_v71.sh"
printf 'TEST_RC=%s\nREPORT=%s/test_report.json\n' "$?" "${OUT}"
```

**正式集合完成后的文本cache**：

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${WORK_ROOT}/environments/lpwm-v71/bin/python" "${ROOT}/code/scripts/prepare_lpwm_instruction_cache_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" \
  --model "${WORK_ROOT}/models/lpwm-v71/t5-large" \
  --out "${WORK_ROOT}/metadata/lpwm_cliport_text_v71" --batch 16
```

**八卡前台模板**：B1来自完整长序列实测，累积3保持global batch24；八卡硬件执行尚未测量。默认150个scene访问epoch，实际步数由998个可采样训练scene/global batch计算，日志记录真实访问数。没有Git、模型下载或后台启动。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/environments/lpwm-v71/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/lpwm_assets_v71.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/lpwm_cliport_text_v71"
export RUN_NAME=lpwm_cliport_v71_native_seed17_20f5e23
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export BATCH_PER_GPU=1 GRAD_ACCUM=3 EPOCHS=150 WORKERS_PER_RANK=2
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset STEPS RESUME
bash "${ROOT}/code/scripts/train_lpwm_object_events_v71.sh"
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "$?" "${OUT}"
```

恢复使用同一完整环境，将 `unset STEPS RESUME` 改为 `unset STEPS`，并设置 `export RESUME="${OUT}/latest.json"`。单卡RGB评价可用：

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=20f5e232fddfc46ab1f6cf97054bdf50746b94d6
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
"${WORK_ROOT}/environments/lpwm-v71/bin/python" "${ROOT}/code/scripts/evaluate_lpwm_rgb_v71.py" \
  --checkpoint "${WORK_ROOT}/outputs/lpwm_cliport_v71_native_seed17_20f5e23/latest.json" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" --partition development \
  --items-per-source 200 --samples 1 \
  --out "${WORK_ROOT}/outputs/lpwm_cliport_v71_native_seed17_20f5e23_development"
printf 'EVAL_RC=%s\n' "$?"
```

上述评价是当前逐帧RGB接口，不自动包含共同分辨率主表、FVD或独立主体/事件成功率；这些属于仍需完成的外部对照工作，不沿用内部V71结果冒充。

### 21.5 LPWM真实单卡链路完成（2026-10-16）

- **What changed**：用户授权后，独立 `environments/lpwm-v71`、固定LPWM源码、T5-large、VGG和LPIPS均在百度旧PFS根准备完成。`20f5e232fddfc46ab1f6cf97054bdf50746b94d6`完整单卡测试退出0，test_report.status=passed，覆盖两步更新、模型/optimizer/RNG精确存取、恢复后的step与采样顺序、history-only 50帧生成、future swap和真实RGB评价。
- **Why**：外部对照需要实际可执行，不能只用内部消融或静态代码替代。完整测试沿用原生FP32/ELBO、128输入及16历史/50未来，不缩短序列或改loss获得通过。
- **Impact**：模型实际146,507,261参数，encoder256/decoder24 particles、context dim7。启用29个Transformer blocks的activation checkpointing后，单卡B1峰值31.64013GiB。保存往返model/optimizer/RNG均相同，恢复到step2、optimizer steps=[2]、采样顺序相同；两条独立CUDA训练轨迹最终参数差0.0001783436，并非逐位等价。history生成case为scene_000017_order_01，输出[1,50,3,128,128]，future swap差0。

两步loss为133.57617和128.66869，首步grad norm约7.74e8、第二步约2189.9。它们有限，但初始梯度很大；测试为覆盖完整目标关闭warmup，不能据两步有限数值宣称长训稳定。正式模板保留原生1 epoch warmup。RGB评价实际只覆盖scene_000018_order_01：posterior L1=0.370392、正确语言=0.370777、空指令=0.370777，copy=0.061469。此为两步工程输出，未显示预测或语言能力；正式组合benchmark还未运行。

环境记录为 `metadata/lpwm_environment_v71.json`，资产记录为 `metadata/lpwm_assets_v71.json`。已安装torch2.8.0+cu126、torchvision0.23.0+cu126、Transformers4.57.1、NumPy1.26.4、SwanLab0.10.1、protobuf6.31.1，原训练环境不变。T5 revision为 `150ebc2c4b72291e770f58e6057481c8d2ed331a`；LPIPS使用作者官方 `082bb24f84c091ea94de2867d34c4544f68e0963` 的v0.1 VGG权重，旧上游mirror与实际来源均记录，不替换原loss forward。

完整证据根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/lpwm_cliport_e2e_20f5e23_20261016T012137Z`；含test_report.json、checkpoint_roundtrip.json、uninterrupted/recovered训练日志及evaluation/report.json、cases.jsonl、视频。工程T5 cache为 `metadata/lpwm_cliport_engineering_text_v71_empty`，4条任务指令加空串共5条、32 tokens、width1024。正式训练、四例独立配对导出与共同评分尚未执行。

正式T5准备随后也退出0：`metadata/lpwm_cliport_text_v71` 含344个 `.pt`，为343条既有任务指令加空字符串。实际token长度1-20，统一32、width1024、FP32，无截断；index与completion及文件数一致。此缓存只编码文本，没有生成新视频、安装新包或启动正式训练。

## 22. 共同真值与统一成像评价

- **What changed**：新增 `score_cliport_predictions_common_v71.py`，直接读取既有无损预测NPZ，使用同一manifest中的原始RGB和模拟器mask重新评分，不运行模型。操作物区域与顺序分数仍在原生坐标计算；RGB/PSNR/SSIM统一从原生视图缩放到最长边128，保留比例、采用bilinear antialias。GT不再来自各模型已经缩放过的训练target。
- **Why**：LPWM的128像素结果与V71的512像素结果不能直接混合原有分数，模型resize链路也不能各自改变参照GT。统一评价尺寸只解决测量尺度，不消除输入分辨率、预训练资产或训练预算的差别。
- **Impact**：旧预测与旧report不覆盖，新分数写独立目录。逐例、逐帧、seen/held标签、反序参考指令、时间匹配与缺失原因均保留；原生空间field ownership仍属于V71既有单独指标，不借RGB分数替代。新增LPWM受控预测导出入口，读取与V71相同的scene pairs，提供posterior、正确/无/反序语言以及copy；反序语言按真实反序后果评分。LPWM环境与单卡RGB入口已完成，受控paired exporter仍待实际执行。

### 22.1 评分口径

预测文件存储最终原生视图RGB。所有方法的这些视图和同一原始GT通过完全相同的函数得到共同尺寸，因此同一case的GT相同。模型本身的输入尺寸单独记录；512模型可能仍保留更多信息，不能把共同尺寸分数的全部差异归于object结构。

如果另一条真实顺序缺少某时点，只取消该时点的配对偏好，当前参考视频的绝对误差仍可计算；不把另一条更早的末帧拿来充数。模型文件的时间网格若与固定请求不符，记为未评分条件及原因，不静默重定时。结尾真帧是否存在单独报告。无语言本来可能对应多种未来，明确标记为ambiguous。

采样结果保留每个sample的分数，并逐case报告sample 0、固定采样数下的平均误差，以及单列的oracle best-of-N；后者不是部署精度。LPWM paired exporter以scene和sample编号确定噪声种子，使同场景两种指令及语言干预复用同一噪声。已有V71单样本结果可直接重评，无需重跑GPU。

### 22.2 真实控制校准

既有CPU控制入口增加 `--save-predictions`，输出真实oracle、反序、copy、两结果均值和identity-swap的RGB文件，供共同评分器完整读取。另输出128/512分辨率的纯resize参考：它们使用真实未来、不含模型，只测缩放与还原本身造成的误差，不叫生成能力或完整codec上限。控制还遮去一个alternative终点的可用性，确认不回填更早时刻。

下一步按同一流程重评这些真实控制、已有object预测和已有scene预测，核对native actor/order旧值是否保持，以及共同128尺度的实际数值。该CPU链路无需新包或权重；尚未执行前不写成通过。

### 22.3 配对比较与执行

`compare_common_cliport_scores_v71.py`只比较同一manifest、请求序列、resampling协议与共同case。逐case保留两边绝对值及candidate-reference差值，缺条件、缺case或参考帧不同均列出；协议不同则不生成跨方法差值。单例顺序margin不拿来判断语言能力，顺序比较继续使用同场景两条指令的成对汇总。当前不计算显著性，也不把同输出分辨率称为同输入或同训练预算。

共同评分入口代码为 `6a120b9b64eb25f0e40f655c4a8f22ea8b468273`；准备机经GitHub取得该release后，下面可直接重评已保存的scene结果，不联网、不占GPU、不重新生成视频：

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=6a120b9b64eb25f0e40f655c4a8f22ea8b468273
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export INPUT="${WORK_ROOT}/outputs/cliport_scene_control_e2e_0fcca58_20261015T162702Z/independent_cliport_events"
export OUTPUT="${WORK_ROOT}/outputs/cliport_scene_common128_v71_$(date +%Y%m%d_%H%M%S)"
"${WORK_ROOT}/environments/cliport-v71/bin/python" "${ROOT}/code/scripts/score_cliport_predictions_common_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71_engineering_c682ec9.json" \
  --predictions "${INPUT}" --out "${OUTPUT}" --max-edge 128 --model-image-size 512 --cpu-threads 2
printf 'SCORE_RC=%s\nREPORT=%s/report.json\nCASES=%s/cases.jsonl\n' "$?" "${OUTPUT}" "${OUTPUT}"
```

LPWM完成自身测试和训练后，先用 `export_lpwm_cliport_predictions_v71.py` 输出原生视图NPZ，再用同一个共同评分入口读取；不能将只含有损MP4的通用RGB evaluator目录当作无损预测目录。LPWM exporter和实际跨模型比较仍待对应资产/实验，不用当前工程模型重评分数宣布方法优劣。

### 22.4 实际校准与分辨率控制

`6a120b9`在百度完成四个CPU入口，全部退出0：真实控制导出、共同校准评分、已有object重评、已有scene重评。四例各12项控制检查通过；共同128下真实oracle的RGB L1=0、SSIM=1，copy与两结果均值的paired margin均为0。旧object/scene native actor/order分数只出现约1e-8的数值差异，未发生实质变化；三组未评分条件均为0。

| 纯resize参考，不运行模型 | Native actor L1 | 共同128 RGB L1 | 共同128 SSIM |
| --- | ---: | ---: | ---: |
| 真实未来先经过128再还原 | 0.02735374 | 0.006563556 | 0.98982698 |
| 真实未来先经过512再还原 | 0.00642120 | 0.000366225 | 0.99997446 |

这些差异在模型之外就存在。因此只统一评分分辨率仍不足以将提升归因于object机制。新加 `object_event_view128_v71.json` 和 `object_event_scene_view128_v71.json`，明确把V71观察输入与RGB训练目标都先变为同一128全视野图像；预测也以128输出。DINO内部仍按448计算、RAE仍为原512 decoder，其输出经过固定、不重叠的area平均到128，未新增或声称有128原生RAE权重。它匹配信息输入/成像输出的尺寸，不匹配预训练资产或内部计算成本。

默认view_size=0保留原512主版本，旧checkpoint默认行为不变。匹配128配置贯穿joint、冻结teacher标签、prior、history-only forecast和所有decode控制，不仅仅在evaluator最后缩图。`aab8913`完整单卡链路已退出0：实际input/target/prediction均为128x128，内部encoder/decoder仍为448/512。world和prior第1步及最终恢复参数差均0，两项因果差均0，梯度有限；world最高allocated为30.94620GiB、prior为1.28516GiB。后续四例独立评测与共同评分均退出0，共同评分覆盖四例、每例七条件，unscored为空。

CPU证据根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/cliport_common_score_6a120b9_20261015T193640Z`。包含 `calibration/report.json`、`common_calibration/report.json`、`common_object/report.json`、`common_scene/report.json` 和 `native_score_difference.json`；原预测与原报告未修改，没有重跑GPU。

新单卡测试根：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/cliport_view128_e2e_aab8913_20261015T200205Z`，完整过程日志位于运行根 `logs/cliport_view128_e2e_aab8913_20261015T200205Z/test.log`。另一个CPU比较入口已确认旧object与scene的四例全部配对、协议差异和排除清单均为空；这些是两步且effect模式不同的工程模型，不用其误差差值证明object结构优劣。

**view128四例实际结果**：逐例预测/分数在 `independent_cliport_events/`，共同评分在 `common_view/`。正确语言的共同RGB L1为0.34515754、SSIM为0.50522388，native actor L1为0.45846309；无语言actor L1为0.45878868，当前图像复制为0.16980290。正确语言paired margin仅7.98e-7，无语言和copy均为0；posterior actor L1为0.47817518，zero effect为0.47423290。这只证明完整测量链路可执行，两步模型没有展示有效运动还原或顺序遵循，不据此宣称128优于512或对象结构优于scene控制。后续使用正式训练及同输入/预算的对照判断机制。

### 22.5 匹配128实验的独立命令

这些命令使用已发布并实测的 `aab89134da946b4ea2286929059cac503745dc22`。准备机按16.1的GitHub同步方式取该release；下列测试、训练和评价均不联网。它是128信息输入/输出的匹配实验，不覆盖原512版本。正式集合和文本cache由18.10的准备入口产生，不能用四例工程集合替代。正式训练尚未启动，八卡吞吐尚未实测。

**单卡完整测试**：已有本节实测结果，不必仅因文档更新再跑；下面用于独立复现。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=aab89134da946b4ea2286929059cac503745dc22
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71_engineering_c682ec9.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_NAME="cliport_view128_e2e_aab8913_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}" LOG_ROOT="${WORK_ROOT}/logs/${RUN_NAME}"
bash "${ROOT}/code/scripts/test_object_event_world_model_v71.sh" \
  --model-config "${ROOT}/code/configs/object_event_view128_v71.json" \
  --modes object_step --train-partition engineering --evaluation-partition engineering \
  --text-cache "${WORK_ROOT}/metadata/cliport_event_text_v71_c682ec9"
printf 'TEST_RC=%s\nREPORT=%s/test_report.json\n' "$?" "${OUT}"
```

**八卡joint首轮**：object_step、1000步、global batch256。每卡2的容量依据来自相同内部DINO/RAE路径的实际profile；不因为最终输出变成128就宣称计算也减到1/16。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=aab89134da946b4ea2286929059cac503745dc22
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export EFFECT_MODE=object_step
export RUN_NAME="cliport_view128_v71_${EFFECT_MODE}_seed17_aab8913_1000"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export BATCH_PER_GPU=2 GRAD_ACCUM=16 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=4 DECODER_FRAME_BATCH=2
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_world_model_v71.sh" \
  --model-config "${ROOT}/code/configs/object_event_view128_v71.json" \
  --warmup 50 --rollout-ramp 100 --save-every 100 --snapshot-every 1000
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "$?" "${OUT}"
```

同一object配置下，仅将 `EFFECT_MODE` 改为 `global_step` 或 `object_segment` 会进入对应独立目录。scene对照使用 `object_event_scene_view128_v71.json`、`EFFECT_MODE=global_step` 和独立 `RUN_NAME=cliport_scene_view128_v71_global_step_seed17_aab8913_1000`。首先用scene/global与object/global比较状态组织，再用object/global与object/step比较绑定；不能直接用scene/global对object/step的差值归因于单个因素。此处不自动提交四个长训。

**八卡prior**：明确读取同一模式的1000步编号快照，配置自动继承其中的view_size；不误用512 teacher。八卡32x1=256保持原批量协议。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=aab89134da946b4ea2286929059cac503745dc22
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export TEXT_CACHE="${WORK_ROOT}/metadata/cliport_ordered_text_v71"
export EFFECT_MODE=object_step
export BASE_CHECKPOINT="${WORK_ROOT}/outputs/cliport_view128_v71_${EFFECT_MODE}_seed17_aab8913_1000/step_00001000.pt"
export RUN_NAME="cliport_view128_prior_v71_${EFFECT_MODE}_seed17_aab8913_1000"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
export BATCH_PER_GPU=32 GRAD_ACCUM=1 STEPS=1000 WORKERS_PER_RANK=2
export ENCODER_FRAME_BATCH=64 DECODER_FRAME_BATCH=2 LANGUAGE_DROPOUT=0.1
export SWANLAB_PROJECT=instruct-gs-world SWANLAB_WORKSPACE=healenrens
unset RESUME INIT_FROM
bash "${ROOT}/code/scripts/train_object_event_prior_v71.sh" \
  --warmup 50 --save-every 100 --snapshot-every 1000
printf 'TRAIN_RC=%s\nTRAIN_LOG=%s/train.jsonl\n' "$?" "${OUT}"
```

joint/prior恢复分别使用原完整命令，将 `unset RESUME INIT_FROM` 改为 `unset INIT_FROM`，并在启动前设置 `export RESUME="${OUT}/latest.json"`。不改变原run的卡数、batch、accum和模型配置，不自动恢复其他目录。

**单卡development与共同评分**：预测只运行一次，随后CPU使用保存的无损NPZ评分；不以有损MP4重新计算指标。固定100个development场景，不访问最终test做选型。

```bash
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=aab89134da946b4ea2286929059cac503745dc22
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export EFFECT_MODE=object_step
export OUT="${WORK_ROOT}/outputs/cliport_view128_eval_v71_${EFFECT_MODE}_aab8913_1000_development"
"${WORK_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/evaluate_cliport_events_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" \
  --assets "${WORK_ROOT}/metadata/rae_assets_512.json" \
  --prior-checkpoint "${WORK_ROOT}/outputs/cliport_view128_prior_v71_${EFFECT_MODE}_seed17_aab8913_1000/step_00001000.pt" \
  --text-cache "${WORK_ROOT}/metadata/cliport_ordered_text_v71" \
  --partition development --items 200 --out "${OUT}/predictions" &&
"${WORK_ROOT}/environments/cliport-v71/bin/python" "${ROOT}/code/scripts/score_cliport_predictions_common_v71.py" \
  --manifest "${WORK_ROOT}/metadata/cliport_event_v71.json" \
  --predictions "${OUT}/predictions" --out "${OUT}/common128" \
  --max-edge 128 --model-image-size 128 --cpu-threads 2
printf 'EVAL_RC=%s\nREPORT=%s/common128/report.json\n' "$?" "${OUT}"
```

## 23. Stage 1：500步held变化诊断

**What changed**：针对正式run `v71_joint_object_step_view128_seed17_20f5e23_20261020_021715` 新增单卡、无需Prior的评测入口。不改变训练，不将最近训练loss下降写成动态能力。固定500步编号checkpoint，使用全部development 100 scenes/200 programs，不访问最终test集。

**Why**：SwanLab step690之前的重建误差有改善，但feature-change几乎不变。需要区分“观测状态可还原”“Dynamics使用了正确effect”和“静态画面捷径”，不能用同一个平均重建数解释所有问题。

### 23.1 条件与指标

| 条件 | 做什么 | 是否读取真实未来 |
| --- | --- | --- |
| posterior | 正确后验mean effects，连续rollout，共用renderer | 是 |
| zero_effect | 同一历史末态、时间和Dynamics，所有effects置零 | 否 |
| reversed_effect_order | 反转后验effect时间顺序，其他保持不变 | 是 |
| source_state_copy | 解码历史末态，复制到每个未来时刻 | 否 |
| rgb_copy | 直接复制原生相机的最后观测帧，不经过模型 | 否 |
| observed_state | 逐帧观测真实未来后重建，检查State/renderer | 是 |
| codec_reference | 真实未来DINO特征经冻结RAE重建，作为视觉codec参考 | 是 |
| resize_reference | 真实未来仅走当前模型的缩放/还原视图链路 | 是 |

所有模型RGB先限制到 `[0,1]` 再还原到原生视图，与原始无损RGB比较，因此不是训练中未clamp的RGB loss。原图copy数值不经过缩放；视频仅为展示会缩放。codec/resize是参考链路，不声称其误差是所有模型的理论下界。

- **Actor RGB L1**：在真实actor初始位置与当前可见位置的mask并集上比较RGB，排除当前机械臂/tool像素。区域不由模型mask决定。保留每个actor、每帧和逐case值。
- **Actor anchor-change L1**：各预测路径先减去该路径自己的历史末帧重建，再与真实RGB相对真实历史末帧的变化比较；排除初始与当前机械臂像素。这分离静态重建偏差与变化误差，但仍是RGB变化，不是物理运动。
- **Actor step-change L1**：比较相邻预测帧的RGB差和相邻真实帧的RGB差；第一步使用各路径自身anchor。加入上一帧actor位置，排除相关机械臂像素，只评分真实相邻的有效帧，不跨缺帧计算。
- **Change-normalized error**：在每个actor内部，用anchor-change误差除以该actor真实RGB变化幅度。只有分母至少为 `1/255` 才计算；这是uint8颜色分辨率，不是运动阈值或teacher筛点参数。copy对有变化actor的比值为1；静止actor不强行计算ratio。
- **Feature MSE / feature-change L1**：使用与训练相同的normalized DINO、有效patch与相邻时间差，覆盖全部未来帧。它们用于解释训练目标，不等同于独立object质量。
- **外部事实与辅助诊断**：保留GT body的3D位移/路径长度，以及初始帧固定匹配的field ownership、visible-centroid误差。field不是RGB实例分割，visible centroid不是3D body center。末帧缺失或actor被遮挡时，不用更早时刻冒充可见终点。

统计顺序：逐帧/actor → case → 同scene两个program平均 → scene分布。报告均值、median、p10/p90，以及posterior对zero/reverse/copy逐case和逐scene的差值、胜/平数量。0-2s、2-5s、5-10s分别报告，不只给整段平均；seen/held combination保留独立切片。没有新增自动“通过”门槛。

### 23.2 同步与单卡执行

准备机同步允许联网；运行命令无Git、模型下载、后台训练行为。下面同步只fetch/export评测代码，不切换或覆盖八卡任务正在使用的release。

```bash
(
set -euo pipefail
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git -C "${SOURCE_ROOT}" fetch origin codex/object-event-world-model-v71
export SOURCE_REVISION=3b3f097510e38bf60edcf7b58d5474b2430a9b4d
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
mkdir -p "${ROOT}"
git -C "${SOURCE_ROOT}" archive "${SOURCE_REVISION}" | tar -xf - -C "${ROOT}"
printf 'SOURCE_REVISION=%s\nROOT=%s\n' "${SOURCE_REVISION}" "${ROOT}"
)
```

评测代码提交为 `3b3f097510e38bf60edcf7b58d5474b2430a9b4d`。同步时可将 `SOURCE_REVISION` 固定为该提交；不使用旧训练release来寻找新增评测脚本。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=3b3f097510e38bf60edcf7b58d5474b2430a9b4d
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CHECKPOINT="${WORK_ROOT}/outputs/v71_joint_object_step_view128_seed17_20f5e23_20261020_021715/step_00000500.pt"
export OUT="${WORK_ROOT}/outputs/v71_stage1_held_step500_${SOURCE_REVISION:0:7}"
export CUDA_VISIBLE_DEVICES=0
export PARTITION=development ITEMS=200 VIDEO_CASES=12
export FRAME_BATCH=4 DECODE_BATCH=2 PRECISION=bf16 RESUME_EVAL=1
mkdir -p "${OUT}"
bash "${ROOT}/code/scripts/evaluate_object_event_stage1_v71.sh" 2>&1 | tee -a "${OUT}/eval.log"
EVAL_RC=$?
printf 'EVAL_RC=%s\nREVIEW=%s/index.html\nREPORT=%s/report.json\n' "${EVAL_RC}" "${OUT}" "${OUT}"
```

`RESUME_EVAL=1` 只复用同一输出目录的已完成评测case，不恢复或修改训练。比较其他checkpoint必须更换 `CHECKPOINT` 和 `OUT`。默认12个视频取固定scene排列的前12例，不按效果选择；全部200例仍完整评分。结果为 `index.html`、`report.json`、`cases.jsonl`、逐case `metrics.json` 和所选case的 `comparison.mp4/terminal.jpg`。视频前3秒是相同的真实历史，后10秒才展示各条件；不上传SwanLab评测数据或媒体。

### 23.3 已完成的本地执行

`test_object_event_stage1_metrics_v71.py --pipeline` 在已有CPU环境运行退出0：精确预测actor误差0、copy变化比值1；机械臂不计入actor变化，静止比值为空，缺帧不跨帧做差，遮挡/缺失终点不代换。随后实际小尺寸ObjectEventWorldModel以BF16跑完8条件、JSON、5帧视频和HTML，resume复用case且不重写结果。codec使用固定测试替身，非正式DINO/RAE权重；真实500步单卡held评测尚待执行，不把本地管线测试当作实验成绩。

## 24. 固定16帧：隔离State与renderer的重建诊断

**What changed**：step500的200-case held及用户视频反馈已确认learned路径丢失物体细节；不再把它仅称为长时Dynamics失败。新增独立单卡入口，只优化现有encoder/renderer，保持DINO、RAE、State尺寸、renderer结构和原有重建loss不变；不覆盖原八卡模型。

**Why**：直接DINO/RAE参考仍能还原物体，失败发生在加入学习的State/renderer之后。先把任务简化成固定少量图像的独立重建，区分基础重建是否可优化与联合/时序训练的问题，而非同时换encoder、扩容量、改loss。

### 24.1 数据、训练与判断

- 从现有train split选4个不同scene，每个scene选一个已有program，取请求时间0/2/5/10秒。seed17、按scene均匀、不看未来运动挑帧。记录真实case/frame/time；缺失请求帧列入skipped，不冒充准确时刻。正式数据预计16帧，实际数量写入输出。
- 当前帧 `[B,1024,768]` DINO特征进入原encoder，每帧独立State初值，得到 `[B,17,4,384]` state tokens与centers；原renderer恢复feature，原冻结RAE解码并输出view128。没有前序图片、帧ID或任务ID条件。它与原16帧历史的时序任务不同，不能把初始分数混为同一协议。
- 默认加载500步encoder/renderer，重建诊断使用新AdamW。只更新这两个模块；posterior、Dynamics不调用，DINO/RAE不更新，但RAE保留对输入feature的梯度。
- 目标仍是原权重的feature MSE＋RGB L1。模拟器actor masks仅用于输出评分，不进入模型或loss；不新增trajectory、KL或contrast目标。
- micro-batch4，完整固定集合参与每次更新，默认1000 optimizer steps，lr2e-4、weight_decay0.01、clip1。不将其当作正式长训或held成绩。
- 先看同帧Target/codec/before/after是否恢复物体，再看逐帧actor RGB误差和原目标是否改善。固定集合能拟合，只说明当前链路在该任务上可优化；不能证明泛化、object grouping或Dynamics。不能拟合也不唯一证明“容量不足”，下一步需区分初始化与模块责任。
- 不产生海量逐时刻JSON。`report.json`只含before/after/codec均值和16帧摘要；`frames/frame_00.png`至`frame_15.png`用于逐例查看，`index.html`汇总全部图片。`selection.json`记录来源；`fixed_frames.pt`仅缓存这16帧及目标以保证恢复；`latest.pt`只保存本实验encoder/renderer、optimizer、RNG与SwanLab ID，不是原joint训练恢复点。

### 24.2 独立执行

同步和运行分开。准备机可联网fetch/export；实际GPU运行不访问GitHub或下载权重。

```bash
(
set -euo pipefail
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git -C "${SOURCE_ROOT}" fetch origin codex/object-event-world-model-v71
export SOURCE_REVISION=8388079b73a22d1915114d903be76f84dcb5cae7
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
mkdir -p "${ROOT}"
git -C "${SOURCE_ROOT}" archive "${SOURCE_REVISION}" | tar -xf - -C "${ROOT}"
printf 'ROOT=%s\n' "${ROOT}"
)
```

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=8388079b73a22d1915114d903be76f84dcb5cae7
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CHECKPOINT="${WORK_ROOT}/outputs/v71_joint_object_step_view128_seed17_20f5e23_20261020_021715/step_00000500.pt"
export RUN_NAME="v71_reconstruction_probe_step500_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0 INITIALIZATION=checkpoint
export SCENES=4 FRAME_SECONDS=0,2,5,10 STEPS=1000 BATCH=4
export FRAME_BATCH=4 DECODE_BATCH=2 PRECISION=bf16 LOG_EVERY=10 SAVE_EVERY=100
export PROBE_SWANLAB_PROJECT=instruct-gs-world PROBE_SWANLAB_WORKSPACE=healenrens
unset RESUME
mkdir -p "${OUT}"
bash "${ROOT}/code/scripts/fit_object_state_reconstruction_v71.sh" 2>&1 | tee -a "${OUT}/launcher.log"
RC=$?
printf 'PROBE_RC=%s\nREVIEW=%s/index.html\nREPORT=%s/report.json\n' "${RC}" "${OUT}" "${OUT}"
```

运行机器需要设置 `SWANLAB_API_KEY`。该值仅留在运行环境，不写入源码或本文档。项目和workspace使用 `PROBE_SWANLAB_PROJECT/PROBE_SWANLAB_WORKSPACE`，传给SDK的显式参数；入口继续接受旧launcher变量，但在SDK初始化前移除冲突的 `SWANLAB_PROJECT/SWANLAB_WORKSPACE`。SwanLab新版将project环境字段解释为结构化JSON，不能把普通项目名字符串直接留给其Settings解析。

恢复时重用原OUT，设置 `RESUME="${OUT}/latest.pt"`，并运行同一入口；不要使用新的时间戳目录。新初始化对照使用 `INITIALIZATION=fresh` 和独立OUT；相同seed/数据配置选同一组帧，原500步checkpoint仅提供模型配置。首轮默认仅跑checkpoint初始化。

### 24.3 本地结果

在已有CPU环境运行完整BF16小模型流程：准备4帧、训练4步、写PNG/HTML/compact JSON、只更新encoder/renderer、保存和2+2步恢复，退出0；与4步连续训练参数差0。复用了现有单个端到端测试入口，没有安装或下载本地权重。正式DINO/RAE和实际16帧GPU重建结果仍待运行。

## 25. 绕过encoder：逐图Free State与同一renderer

**What changed**：固定16帧入口新增 `--state-source free` 和 `--reference-run`，不另建world model。直接读取上一轮完整固定帧bank，保持case、原帧、目标和评分区域相同。每帧tokens与centers从原joint step500 encoder的独立帧输出初始化；冻结encoder，优化逐图State和同一原step500 renderer，1000步、AdamW lr2e-4、weight_decay0.01、clip1、micro4/effective16。原数据、旧probe checkpoint及正式训练均保持原样。

**Why**：用户提供的after图仍丢失小物体。若允许逐图直接拟合State后能够恢复物体，优先定位encoder聚合/优化；若不能，优先定位renderer、feature解码接口或共同目标优化。逐图State允许记忆，是诊断而非部署模型，也不是同参数量的泛化对比。

**Impact**：同帧五列PNG展示Target、codec_reference、before、旧probe终点encoder_after、free_state_after。旧probe终点只用于一次参考计算，计算后恢复原step500参数，新优化不从已训1000步renderer继续。每100步输出compact report与图片，SwanLab补齐重建actor误差、codec/旧encoder参考与逐帧逐actor标量，不上传媒体。报告、free State、optimizer及初始参考写入新OUT，显式resume恢复；旧encoder模式继续兼容。

### 25.1 同步代码

```bash
(
set -euo pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
export SOURCE_REVISION=12443559a7f01a1421bca2573e1036f8b721e2d4
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
git -C "${SOURCE_ROOT}" fetch git@github.com:healenrens/Instruction-GS-World.git codex/object-event-world-model-v71
mkdir -p "${ROOT}"
git -C "${SOURCE_ROOT}" archive "${SOURCE_REVISION}" | tar -xf - -C "${ROOT}"
printf 'ROOT=%s\n' "${ROOT}"
)
```

### 25.2 单卡前台运行

运行环境设置SWANLAB_API_KEY，值不写入源码或文档。此命令不访问GitHub或下载模型。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=12443559a7f01a1421bca2573e1036f8b721e2d4
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CHECKPOINT="${WORK_ROOT}/outputs/v71_joint_object_step_view128_seed17_20f5e23_20261020_021715/step_00000500.pt"
export REFERENCE_RUN="${WORK_ROOT}/outputs/v71_reconstruction_probe_step500_20261021_030844"
export RUN_NAME="v71_free_state_probe_step500_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0 STATE_SOURCE=free INITIALIZATION=checkpoint
export STEPS=1000 BATCH=4 FRAME_BATCH=4 DECODE_BATCH=2 PRECISION=bf16
export LOG_EVERY=10 SAVE_EVERY=100
export PROBE_SWANLAB_PROJECT=instruct-gs-world PROBE_SWANLAB_WORKSPACE=healenrens
unset SWANLAB_PROJECT SWANLAB_WORKSPACE RESUME
mkdir -p "${OUT}"
bash "${ROOT}/code/scripts/fit_object_state_reconstruction_v71.sh" 2>&1 | tee -a "${OUT}/launcher.log"
RC=$?
printf 'PROBE_RC=%s\nREVIEW=%s/index.html\nREPORT=%s/report.json\n' "${RC}" "${OUT}" "${OUT}"
```

显式恢复时沿用原OUT，增加 `RESUME="${OUT}/latest.pt"`，原frames/初始参考/free State/optimizer/RNG会恢复。CPU BF16小尺寸完整流程已通过：训练、encoder冻结、五列PNG、逐actor评分、4步与2+2步恢复参数一致；真实DINO/RAE的16帧结果尚未执行，不据工程通过宣布方法有效。

## 26. 同一Free State实验的八卡执行

用户于2026-10-23要求八卡。新增单机torchrun/DDP，固定16帧、每卡2帧；renderer与每图State在8rank间同步，保持1000步、原loss和AdamW recipe。评分/图片/日志仅rank0输出，完整16帧重建参考保持一致；显式恢复保存各rank RNG。该配置用于并行当前诊断，不是增加八份数据或训练八个独立renderer。单卡仍用NPROC_PER_NODE=1。

准备机按第25.1节同步，将SOURCE_REVISION替换为 `cd7c375ced82aea27cc1d10aa81c3c6a63361712`；实际运行命令独立且不访问GitHub或下载模型。运行环境设置SWANLAB_API_KEY。用户追加要求多跑steps，本轮改为10000步、每1000步保存；第1000步用于同预算对照，其后仅观察进一步拟合，不能将10000步对1000步差值全部归因于结构。

```bash
set -o pipefail
export WORK_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export SOURCE_REVISION=cd7c375ced82aea27cc1d10aa81c3c6a63361712
export ROOT="${WORK_ROOT}/runtime/object_event_world_model_v71/releases/${SOURCE_REVISION}"
export PYTHON="${WORK_ROOT}/.venv/bin/python"
export MANIFEST="${WORK_ROOT}/metadata/cliport_event_v71.json"
export ASSETS="${WORK_ROOT}/metadata/rae_assets_512.json"
export CHECKPOINT="${WORK_ROOT}/outputs/v71_joint_object_step_view128_seed17_20f5e23_20261020_021715/step_00000500.pt"
export REFERENCE_RUN="${WORK_ROOT}/outputs/v71_reconstruction_probe_step500_20261021_030844"
export RUN_NAME="v71_free_state_probe_8gpu_$(date +%Y%m%d_%H%M%S)"
export OUT="${WORK_ROOT}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8
export STATE_SOURCE=free INITIALIZATION=checkpoint STEPS=10000 BATCH=2
export FRAME_BATCH=4 DECODE_BATCH=2 PRECISION=bf16 LOG_EVERY=20 SAVE_EVERY=1000
export PROBE_SWANLAB_PROJECT=instruct-gs-world PROBE_SWANLAB_WORKSPACE=healenrens
unset SWANLAB_PROJECT SWANLAB_WORKSPACE RESUME
mkdir -p "${OUT}"
bash "${ROOT}/code/scripts/fit_object_state_reconstruction_v71.sh" 2>&1 | tee -a "${OUT}/launcher.log"
RC=$?
printf 'PROBE_RC=%s\nREVIEW=%s/index.html\nREPORT=%s/report.json\n' "${RC}" "${OUT}" "${OUT}"
```

恢复仍保留八卡与BATCH=2，重用原OUT并设RESUME为该目录latest.pt。两rank CPU/Gloo完整测试脚本为 `code/scripts/test_reconstruction_probe_distributed_v71.py`；目标为与单进程固定全集优化近似一致、DDP连续与恢复一致，并验证单份结果及每rank RNG。当前静态检查完成，远端运行待回执，不声称八张A100的实际吞吐或显存已经测得。
