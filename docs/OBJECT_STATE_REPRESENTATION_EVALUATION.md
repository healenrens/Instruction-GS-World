# Object State Representation Sufficiency Evaluation Protocol

> 生效日期：2026-08-30
> 适用主线：V61 及后续 Object State / latent-effect world model
> 绑定 Gate：G2 Persistent State、G3 Independent Object Validity
> 永久账本：`/Users/hela/Instruct-GS-World-recovered-20260725/docs/OBJECT_WORLD_MODEL_PROGRESS_V57.md`

## 1. 核心判断

当前模型在功能上承担一个 representation autoencoder 的角色：它把 RGB video 压缩成
object-level latent state，再要求该 state 保留 identity、dynamic、geometry 和 lifecycle 信息。
因此，训练 loss、teacher alignment 或 observation reconstruction 下降，只能证明目标可以被优化，
不能单独证明 latent 是好的 object representation。

数学上需要区分两种情况：

1. 当前 V61 Object State 是 deterministic continuous representation，没有显式
   $q_\phi(z\mid x)=\mathcal N(\mu,\sigma^2)$、prior 或 KL，因此不能伪造 VAE 的 `Rate`。
2. 后续 latent-effect posterior 如果输出 $\mu$、$\log\sigma^2$ 并相对 prior 正则化，才使用
   KL、posterior sampling stability 和 rate-distortion 曲线。

本项目统一使用以下三维评价地图：

$$
\text{Distortion} + \text{Rate or Capacity} + \text{Utility}.
$$

- **Distortion**：latent 对真实 object observation/state 丢失了多少信息。
- **Rate or Capacity**：latent 实际使用了多少独立方向、carrier 和 owner；是否坍缩或冗余。
- **Utility**：任务相关信息能否被低复杂度模型稳定读取，并用于 temporal prediction。

任何单一指标都不能替代这三类证据。

## 2. 被评价的 representation

V61 state 不能只压平成一个不区分语义的向量。需要分别评价：

```text
identity state     z_id   : 跨时间和遮挡保持对象身份
dynamic state      z_dyn  : 表示对象当前可变化状态
geometry state     z_geo  : center、support、relative scale 等相对几何
lifecycle state    z_life : visibility、presence/unknown
joint object state z_obj  : 上述状态的联合表示
```

对 set-valued carriers/roots，必须先用 external query、track correspondence 或 permutation matching
确定比较对象；禁止直接按 carrier/root index 展平后计算 rank、retrieval 或 temporal consistency。

若 encoder 是 deterministic，probe 使用其直接输出。若未来使用 stochastic posterior，则主要特征为
posterior mean：

$$
f(x)=\mu_\phi(x),
$$

并额外报告多次采样

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
$$

对同一 probe 预测的方差。采样稳定性不能代替 posterior mean 的任务可用性。

## 3. 第一层：表示是否存在且没有坍缩

### 3.1 Active Units

对 track/query matched latent 的第 $j$ 维，定义：

$$
\operatorname{AU}_j=
\mathbb I\left[\operatorname{Var}_x(f_j(x))>\epsilon\right].
$$

报告 active-unit 数和比例，但不机械要求全部维度活跃。必须分别统计 `identity`、`dynamic` 和
`joint`，避免 identity 的稳定性掩盖 dynamic 全零。

### 3.2 Effective Rank

在 held samples 上计算协方差

$$
C=\operatorname{Cov}_x[f(x)],
$$

并以归一化特征值 $p_i$ 定义

$$
\operatorname{erank}(C)=\exp\left(-\sum_i p_i\log p_i\right).
$$

同时报告前 1、4、8、16 个主成分解释的方差比例。active units 很多但 effective rank 很低，仍然
属于低维坍缩或强冗余。

### 3.3 Set-state 容量使用

当前模型还必须报告：

- active/effective carriers；
- owner entropy 和折算后的 effective owner count；
- scene owner fraction；
- carrier/root feature 的平均 pairwise cosine；
- identity 与 dynamic 的方差比；
- owner、carrier 或 presence 是否长期卡在最小值或最大值。

`effective_object_roots` 只反映 presence 时，不能作为 owner 使用率。owner 使用率必须由 owner
distribution 的 entropy 单独计算。

### 3.4 Stochastic posterior 专用诊断

只有模型实际存在 Gaussian posterior 和 prior 时，才报告：

$$
R=\mathbb E_x D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)),
$$

以及 per-dimension $R_j$、active KL dimensions、posterior variance 和 sampling stability。
KL 接近零且 reconstruction 良好是 posterior collapse 风险；KL 大也不等于任务信息充分。

## 4. 第二层：任务信息是否容易读取

所有 probe 必须冻结 Object State encoder，并使用相同 held split。

### 4.1 无训练 retrieval

至少报告：

- same-object temporal Recall@1/5/10；
- occlusion-to-reappearance retrieval；
- same-task-phase retrieval；
- different-object rejection；
- cross-source、cross-background 和 cross-camera retrieval。

当前 `heldout_track_appearance_temporal_error` 只检查同一 track 前后是否相似。它必须与
retrieval accuracy 联合解释：所有 identity 都相似时，temporal error 可以很低，但 retrieval 会失败。

### 4.2 Linear Probe

冻结 encoder，只训练线性或 logistic head：

$$
\hat y=Wf(x)+b.
$$

根据可用 truth，分别 probe：

- object identity / object relation；
- relative center、scale、support 和 motion；
- visibility、occlusion、reappearance；
- task phase、progress、success；
- gripper state 和 contact，仅作为 state sufficiency probe，不把显式 action 输入 world model。

### 4.3 有限容量 MLP Probe

再使用固定宽度的两层 MLP。解释规则为：

| Linear | MLP | 结论 |
|---|---|---|
| 高 | 高 | 信息存在且组织清晰 |
| 低 | 高 | 信息存在但纠缠，latent geometry 较差 |
| 低 | 低 | 信息缺失或严重坍缩 |
| 仅完全微调后高 | 高 | 下游网络重新学习任务，不能归功于原 representation |

### 4.4 低数据量曲线

probe 使用 $1\%,5\%,10\%,25\%,100\%$ truth。Object State 的价值必须体现在样本效率，而不是
仅在 100% truth 和高容量 MLP 下被补救。

## 5. 第三层：是否编码了正确的信息

### 5.1 Task-relevant 与 nuisance probe

除任务变量外，同时尝试从 latent 预测：

- camera ID / camera view；
- dataset source；
- background appearance；
- illumination、颜色和压缩编码；
- episode ID 或文件 shard。

目标不是盲目删除全部这些信息，而是验证：object identity、relative geometry 和 dynamic state 的
可读性不能主要依赖 source、camera 或 background shortcut。哪些变量是 nuisance 必须按下游任务声明。

### 5.2 不变性与敏感性

对不应改变 object semantics 的变换 $T_n$，报告归一化不变性：

$$
S_{\mathrm{inv}}(T_n)=
\frac{\mathbb E\|f(x)-f(T_n(x))\|_2}
{\mathbb E\|f(x)-f(x')\|_2}.
$$

同时对 object movement、visibility change、relation change 等任务相关变换 $T_r$ 报告 sensitivity。
合格表示必须对 nuisance 稳定、对相关变化敏感；“所有情况下都不变”同样是坍缩。

## 6. 第四层：绝对 Distortion 与 compositional validity

评测必须报告 prediction/reconstruction 相对真实 held observation 的绝对误差，而不能只报告
“相对 persistence 改善多少”。至少包含：

- object-local appearance/semantic observation error；
- relative geometry absolute error；
- visibility/lifecycle calibration error；
- track-covered region 的 deletion locality；
- cross-object leakage；
- scene/background reconstruction 或 residual error。

RGB reconstruction 可以作为辅助 Distortion，不作为 object semantics 的唯一监督。DINO/SigLIP feature
reconstruction 也只能说明 perception information 可解码，不能单独证明 component 是 object。

若存在 decoder，slot/carrier deletion 必须只影响 external truth 对应的 object region。全局 decoder
通过其他 component 补回删除区域时，reconstruction 不能作为 compositional success。

## 7. 第五层：时序状态充分性

### 7.1 Markov Sufficiency

冻结 state encoder，比较容量相同的两个 predictor：

$$
p(o_{t+1}\mid O_t,z_t)
$$

与

$$
p(o_{t+1}\mid O_t,O_{t-1},\ldots,z_t).
$$

若加入原始历史 state 后仍显著改善，说明 $O_t$ 丢失了速度、遮挡对象、接触或任务进度，不是充分
state。该测试必须分别在 motion-active、occlusion 和 long-horizon subset 上执行。

### 7.2 Dynamics Predictability

只有 G2/G3 通过后，才训练低容量 Dynamics。首先评价 correct posterior effect 是否比 zero、
shuffled 和 persistence 更好，再评价多步 rollout 的误差增长。单步 prediction 下降不能覆盖 identity、
lifecycle 或 compositional failure。

### 7.3 下游数据效率

最终冻结 encoder，只训练小型 effect selector、action head 或 policy，比较相同 demonstration 数量下
的 task A success、跨背景/相机/对象泛化，以及冻结与微调差距。该层属于 G6，不得提前替代 G2/G3。

## 8. 基线与报告矩阵

每个正式版本至少比较：

- random encoder；
- frozen DINO frame/point feature；
- frozen SigLIP frame/object feature；
- 同容量 deterministic encoder/AE；
- 最近 accepted Object State checkpoint；
- 当前完整方法。

统一报告矩阵：

| 维度 | 核心问题 | 主要指标 |
|---|---|---|
| Distortion | 信息丢失多少？ | absolute object observation、geometry、lifecycle error |
| Rate/Capacity | latent 是否坍缩或冗余？ | KL（仅 stochastic）、AU、effective rank、owner/carrier usage |
| Utility | 信息是否低复杂度可读？ | kNN、Recall@K、linear/MLP probe、low-data curve |
| Correctness | 学到 object 还是 shortcut？ | nuisance probe、cross-source/view、independent truth |
| Temporal | 当前 state 是否充分？ | reappearance、Markov test、multi-horizon rollout |

正式选择 checkpoint 时，寻找 $(D,R,U)$ 或 $(D,C,U)$ 的 Pareto improvement；不得只选择 training
loss 最低、reconstruction 最好或 latent dimension 最大的 checkpoint。

## 9. G2/G3 晋级规则

Object State 只有同时满足以下条件才可晋级：

1. 没有 identity、dynamic、owner 或 carrier collapse；AU、effective rank 和 capacity usage 全部报告。
2. held independent identity retrieval 显著高于 chance 和 random encoder；不能只依赖 temporal error。
3. frozen linear/MLP probe 能读取 geometry、motion 和 lifecycle，且低数据量曲线优于 frame baseline。
4. nuisance probe 与 cross-source/view 测试排除 background、camera 和 dataset shortcut。
5. absolute object observation error、deletion locality 和 cross-object leakage 在 independent truth 上成立。
6. Markov sufficiency 测试没有显示当前 state 遗漏决定未来所需的大量历史信息。
7. teacher-held 与 independent evaluation 分开报告；训练 CoTracker 不能同时定义最终 truth。

任一条失败，判决为 `iterate at G2/G3`。不得用更多训练 steps、较低 reconstruction loss 或后续
Dynamics 指标覆盖前置失败。

## 10. 当前 V61 证据与下一步

截至 2026-08-30，四个同预算 probe 均已完成 3,000 steps：`dino/fcak90ya`、
`siglip/9ttfq8h4`、`siglip_dino/60mczuva`、`siglip_dino_object/vaos6ix0`。四组使用相同提交、
seed、batch、held split 和训练步数，均无 non-finite。

最后 480 steps 的 held identity retrieval 分别为 `0.378% / 0.345% / 7.173% / 9.927%`。
因此当前证据支持三项判断：

1. 单独把 DINO Student 换成 SigLIP Student 没有效果，二者都发生 identity homogenization。
2. frozen DINO local alignment 提供了有效的外部 identity anchor，是 retrieval 持续增长的主要来源。
3. object-semantic target 继续提高 retrieval，但完整 variant 的 scene owner fraction 降到 `0.0168%`，
   effective owner categories 只有 `2.87 / 17`，coordinate、visibility 和 presence 也明显退化；它通过
   集中 owner assignment 获得部分语义收益，尚不是合格的 compositional Object State。

四组 motion error 都约为 `0.01393`，128 个 carriers 都有约 127 个持续 active。因此当前只验证了
semantic identity signal，尚未验证 dynamic state、adaptive capacity、independent object validity 或
Markov sufficiency。

`siglip_dino` 是下一轮统一表征复评的主要 Pareto candidate，`siglip_dino_object` 是检验 semantic
gain 与 owner collapse 关系的必要对照。下一步必须在两个 checkpoint 上执行本文档第 3 至第 7 节的
完整 evaluator；在 AU/effective rank、Recall@K、frozen probe、nuisance、occlusion/reappearance、
deletion locality、scene leakage 与 Markov tests 返回前，仍禁止晋级 latent-effect Dynamics。

## 11. V61 统一复评实现

统一复评入口为：

- `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/evaluate_representation_sufficiency_v61.sh`
- `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/evaluate_representation_sufficiency_v61.py`
- `/Users/hela/Instruct-GS-World-recovered-20260725/code/igsw/adaptive_gaussian_wm/representation_sufficiency_v61.py`

一次前台运行同时加载 `siglip_dino` 与 `siglip_dino_object`。每个 held RGB clip 只运行一次 frozen
DINO、CoTracker 和 object-component teacher，再分别通过两个 Student checkpoint，避免 teacher
计算差异污染比较。最终只有一个 W&B comparison run，所有标量分别写入
`sufficiency/siglip_dino/*`、`sufficiency/siglip_dino_object/*` 与 `comparison/*`。

该 evaluator 输出：

1. identity/dynamic active units、participation-ratio effective rank 和跨样本 cosine collapse；
2. Student 与 frozen DINO point teacher 各自的 first-to-last、occlusion/reappearance
   Recall@1/5/10，并同时报告 chance；
3. carrier/root/owner capacity、scene fraction、object-to-scene 与 scene-to-object leakage；
4. external-track same/different identity margin，以及删除一个 root assignment 后对 same/different
   tracks 的影响比例；
5. 按 task group 隔离 train/test 的 frozen ridge 与小型 MLP probes：coordinate、motion、visibility；
6. identity 对 dataset source 的 nuisance probe；
7. current dynamic state 与 current+previous dynamic state 对 future motion 的 probe gain 差，作为
   Markov sufficiency 诊断；
8. 原 held objective 中 coordinate、motion、lifecycle、relation 和 teacher alignment 的 absolute
   distortion。

所有无需拟合 probe 的指标同时按六个 source 与 clip length 分解；数据读取的 decode replacement
rate 单独记录，不能把替换样本比例差异解释为模型收益。motion linear probe 额外报告
10%/25%/50%/100% train-sample curve，用于判断信息是否能被低数据量读出。

V61 没有 compositional image decoder，因此第 4 项只能称为 `external-track root-assignment deletion
locality`，不能写成独立图像重建 locality。整个 evaluator 的 truth scope 固定为
`held_training_teacher_not_independent_object_truth`：它用于 G2 checkpoint 选择，不构成 G3 通过。

## 12. V61 统一复评结果与 Gate 判决

W&B run `t33taede` 已完成，evaluator revision 为
`0cd9a40a23bde8d8b07c1756c66e4d022cb822c0`。它在 144 个 held conditions 上比较两个 seed 17、
step 3,000 checkpoint，输入数据、teacher 输出和 probe split 相同。

### 12.1 Capacity 与 retrieval

`siglip_dino` 的 identity active units/effective rank 为 `29 / 5.04`；`siglip_dino_object` 为
`88 / 2.67`。完整 variant 虽然让更多维度越过方差阈值，独立信息维度反而更少。两者 identity
off-diagonal cosine 为 `0.9949/0.9812`，仍高度同质化。

first-to-last Recall@1 为 `40.29%/34.09%`，chance 为 `2.38%`，frozen DINO teacher 为 `49.27%`；
reappearance Recall@1 为 `32.95%/23.80%`，chance 为 `1.05%`，teacher 为 `34.32%`。因此
`siglip_dino` 保留了可用于 correspondence 的信息，而完整 semantic objective 使正式 held
persistent-identity 指标退化。

### 12.2 Shortcut 与 compositionality

identity 对 source 的 linear balanced accuracy 为 `97.39%/99.38%`，远高于六分类 chance
`16.67%`；coordinate-from-identity linear gain 为 `0.941/0.862`。identity 因此高度携带 dataset
domain 与绝对空间信息，而非纯 object identity。

`siglip_dino_object` 的 scene owner fraction 为 `0.0022%`，effective owner categories 为
`2.90 / 17`；`siglip_dino` 为 `5.78%` 和 `3.16 / 17`。external-track root-assignment deletion
locality 从 `6.17` 降到 `4.27`。完整 variant 的 same-different margin 虽从 `0.0123` 增至
`0.0284`，但这是以 owner/scene collapse、较低 effective rank 和较差 locality 为代价。

### 12.3 Dynamic utility 与 Markov 诊断

两者 dynamic active units 都是 `128 / 128`，effective rank 却只有 `2.08/2.75`。motion linear gain
为 `-0.445/-0.503`，小型 MLP gain 为 `-0.037/-0.059`；负值表示 probe 比训练集 motion 均值基线更差。
visibility MLP balanced accuracy 为 `50.0%/53.1%`，也接近 chance。

`siglip_dino` 的 current-only motion gain 为 `-0.401`，加入 previous state 后为 `-0.102`，history
incremental gain 为 `+0.299`。历史含有当前 state 丢失的信息，但两条路径都未超过常数基线。
`siglip_dino_object` 的 current/history gain 为 `-0.471/-0.518`。因此不能将较小的 history 增益解释为
Markov sufficiency；当前首先失败的是 dynamic information 的可读性。

### 12.4 最终判决

本次 evaluation 状态为 `completed`，但 promotion decision 为
`not_automatic_requires_g2_and_independent_g3_review`。按第 9 节规则：

- 两个 checkpoint 均为 `iterate at G2/G3`；
- `siglip_dino` 仅保留为诊断 baseline，`siglip_dino_object` 不晋级；
- 当前禁止训练 latent-effect Dynamics；
- 下一版只修正 representation objective，不增加 backbone variant，也不依靠延长训练；
- 必须提高 identity/dynamic effective rank，降低 source/coordinate shortcut，让 motion/visibility
  可由 frozen probe 读出，并恢复 scene/object compositional decomposition，再复用本 evaluator 和
  independent G3 truth 复评。
