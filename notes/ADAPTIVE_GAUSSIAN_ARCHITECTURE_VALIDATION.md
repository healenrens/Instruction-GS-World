# Adaptive Gaussian Object-JEPA Architecture Validation

日期：2026-07-18

状态：执行中的架构级验证协议。

## 1. 验证对象

本轮不把对象化、自适应密度和多模态先验视为三个彼此独立的技巧。
需要验证的完整因果链是：

`current/history features`
`-> adaptive GPSTokens`
`-> competitive object slots`
`-> latent Action Posterior / history-only Flow Prior`
`-> joint Object-JEPA Dynamics`
`-> future latent and auxiliary Gaussian feature readout`

只有完整架构相对机制退化版本产生一致、可重复、可解释的联合增益，
才能认为方案成立。

未来特征、未来轨迹、对象标签和 mode 标签只允许作为训练目标或评估标注，
不得进入 current encoder、GPSToken placement、prior context 或部署时样本选择。

### 1.1 GPSToken 没有被废弃

保留两级表征：

1. GPSToken 把当前稠密特征压缩为带中心、协方差、feature、opacity 和
   activation 的自适应 micro-Gaussian tokens。
2. competitive object slots 再把这些 GPSTokens 聚合为跨时刻稳定的
   object latent，供 JEPA Dynamics 使用。

已验证过三种可选写回：

`O_k <- O_k + 0.5 G([c_k, f_k])`

`O_k <- O_k + 0.5 W_c c_k`

`O_k <- O_k + 0.5 W_f f_k`

其中：

- `O_k` 是对象 `k` 的 slot latent。
- `c_k` 是由 GPSToken assignment 加权得到的二维中心。
- `f_k` 是同一 assignment 加权得到的对象 feature。
- `G` 是 LayerNorm 和两层 MLP 组成的可学习映射。
- `W_c` 是仅映射二维中心的线性层。
- `W_f` 是仅映射对象 feature 的线性层。

full geometry fusion 和 center-only fusion 都在受控诊断中降低了 center
observability，因此只保留为兼容消融，不能作为正式候选。无写回是当前
几何基线；feature-only fusion 只用于检验 object latent 是否保留可预测的
语义特征，必须先证明它不破坏纯几何 observability 才能进入正式候选。
Dynamics 始终预测 `O_k`，而不是直接递归预测 Gaussian 参数。

被废弃的是“仅按 assignment entropy 做 hard token 分割”以及使用未来轨迹
决定 placement 的 oracle 规则。前者没有证明能分配有效预算，后者违反
严格因果边界。新版 GPSToken placement 和 activation 只读取当前帧特征与
二维归一化网格坐标；未来只用于 target、posterior 和评估。

## 2. 旧证据为什么不足

### 2.1 对象化对照不匹配

旧 synthetic tiny Object-JEPA 约有 `314,010` 个可训练参数，
旧 `FlatFeaturePredictor` 只有约 `14,320` 个参数。

参数量差异本身不会使旧 flat 结果无效，但它不能隔离“竞争式对象聚合”
是否产生增益。新对照复用完全相同的参数，只改变聚合归一化方式。

### 2.2 density 不可识别

旧重建使用：

`F_hat_n = sum_m(alpha_m a_mn f_m) / sum_m(alpha_m a_mn)`

其中：

- `n` 是稠密网格位置。
- `m` 是候选 GPSToken。
- `a_mn` 是位置 `n` 对 token `m` 的软分配。
- `alpha_m` 是 token activation。
- `f_m` 是 token 解码特征。
- `F_hat_n` 是位置 `n` 的重建特征。

若所有 `alpha_m` 同时乘以同一个正数，分子和分母会抵消。
因此旧损失不能可靠识别样本应该使用多少 token。

### 2.3 旧 prior 不是联合相关先验

旧速度场分别处理每个 action token。
不同 action token 只共享 context，不能在一次速度场计算中相互注意，
因此不能表达 token 间及不同未来时刻间的联合相关结构。

新版 Flow Prior 对所有 `Q x A` 个 action token 联合执行 self-attention。

其中：

- `Q` 是联合预测的未来时刻数。
- `A` 是每个未来时刻的 latent action token 数。

最终 prior 消融使用完全相同的 Transformer 参数。`joint` 将
`Q x A` 个 token 组成一个 attention sequence；`factorized` 将每个 token
改为长度 `1` 的独立 sequence。两者参数量严格相同。

## 3. 统一对照矩阵

| 变体 | GPSToken density | latent 聚合 | prior |
|---|---|---|---|
| `full` | adaptive | competitive | joint flow |
| `no_object` | adaptive | global shared | joint flow |
| `independent_slots` | adaptive | non-competitive | joint flow |
| `fixed_density` | fixed expected budget | competitive | joint flow |
| `independent_prior` | adaptive | competitive | per-token flow |
| `all_degraded` | fixed expected budget | global shared | per-token flow |

所有变体保持：

- 相同 current-only 输入。
- 相同 target encoder 规则。
- 相同 Dynamics 深度和宽度。
- 相同训练 batch、step、优化器和 future horizon。
- 对象化对照使用同一组聚合参数，参数量差为零。
- density 对照的平均 activation budget 差异不超过 `5%`。

## 4. 对象化假设

### 4.1 假设

竞争式 slot 应把具有共同外观和运动的 GPSToken 聚合到稳定对象 latent，
并在独立对象运动、遮挡或跨时刻预测时优于不竞争的全局 latent。

### 4.2 Synthetic 指标

定义软 contingency matrix：

`C_kc = sum_n p_nk y_nc`

其中：

- `p_nk` 是网格位置 `n` 属于 slot `k` 的软概率。
- `y_nc` 是位置 `n` 属于合成对象 `c` 的 one-hot 评估标签。
- `C_kc` 是 slot `k` 与对象 `c` 的软重合量。

Purity：

`P = sum_k max_c C_kc / sum_kc C_kc`

Completeness：

`R = sum_c max_k C_kc / sum_kc C_kc`

Object F1：

`F_obj = 2 P R / (P + R)`

Purity 防止一个 slot 混合多个对象，completeness 防止一个对象被任意切碎。

### 4.3 Real evaluation-only 指标

真实 strict-causal pair 中的未来 tracker 位移只用于评估：

`d_i = traj_i[K] - traj_i[0]`

其中：

- `i` 是当前帧固定采样点。
- `traj_i[0]` 是点 `i` 的当前三维位置。
- `traj_i[K]` 是评估终点的三维位置。
- `d_i` 是未来位移，不进入模型输入。

用当前 slot assignment 对 `d_i` 做加权分组并计算：

`R2_motion = 1 - SSE_slot / SSE_global`

其中：

- `SSE_slot` 是用 slot 内平均位移预测每个点后的平方误差。
- `SSE_global` 是用全局平均位移预测后的平方误差。

### 4.4 Gate

对象化成立需要同时满足：

1. `full` 的 synthetic `F_obj` 在至少 4/5 个 seed 高于 `no_object`。
2. `full` 的 held feature/latent error 均值不高于容量匹配对照。
3. real `R2_motion` 高于 global 和 assignment-shuffle 对照。
4. 在至少 4/5 个 seed 中，打乱跨帧 slot identity 使三帧 latent MSE
   至少增加 `5%`，证明模型实际使用对象身份。

## 5. 自适应 density 假设

### 5.1 可识别重建

新版重建为：

`F_hat_n = sum_m(alpha_m a_mn f_m) + (1-c_n) F_bar`

`c_n = sum_m(alpha_m a_mn)`

其中：

- `F_bar` 是当前样本有效网格特征的均值。
- `c_n` 是位置 `n` 被活跃 token 覆盖的程度。

activation 降低时，局部特征会退回全局均值，因此 token 数会影响失真。

### 5.2 预算

每个样本的有效 token 数为：

`M_eff = sum_m alpha_m`

只约束 batch 平均 activation fraction 接近 `r`：

`L_budget = (mean(M_eff / M) - r)^2`

其中：

- `M` 是最大候选 token 数。
- `r` 是固定预算比例，本轮为 `0.5`。

`full` 可在样本间重新分配预算，`fixed_density` 每个样本严格使用 `rM`
的连续 activation budget。

为避免平均 MSE 把预算优先分配给容易样本，定义 current-only
可压缩性幅度：

`u_b = sqrt(mean_n ||F_bn - F_bar_b||^2)`

其中：

- `b` 是 batch 中的样本索引。
- `n` 是当前帧有效网格位置。
- `F_bn` 是样本 `b` 在位置 `n` 的当前视觉特征。
- `F_bar_b` 是样本 `b` 的当前视觉特征均值。
- `u_b` 是当前特征相对全局均值的失真幅度。

连续预算排序目标为：

`r_b = clip(r sqrt(u_b / mean_b u_b), r_min, r_max)`

其中：

- `r_b` 是样本 `b` 的连续 activation fraction 目标。
- `r` 是整个 batch 的平均预算比例。
- `r_min`、`r_max` 是防止全关或全开的边界。

`r_b` 只提供当前特征可压缩性的排序，不提供对象数、未来运动或区域边界。
最终价值仍由同平均预算下相对 `fixed_density` 的预测误差证明。

### 5.3 Gate

自适应 density 成立需要：

1. `full` 与 `fixed_density` 平均预算差异不超过 `5%`。
2. complexity 与 `M_eff` 的相关性在至少 4/5 个 seed 为正且大于 `0.3`。
3. 高复杂度样本上 `full` 的预测误差低于固定预算。
4. 低复杂度样本不能以超过 `2%` 的代价换取高复杂度收益。
5. 在至少 4/5 个 seed 中，按有效 token 数做 rank-reverse activation
   交换后，当前特征重建 MSE 至少增加 `2%`。

complexity 只在合成评估中使用真实对象数和背景空间频率计算，
不作为模型训练输入或 density target。

## 6. 多模态先验假设

### 6.1 数据条件

同一个 current/history 被复制为一个 future group。
ambiguous group 有三个真实 future mode：

`mode in {-1, 0, +1}`

deterministic group 的三个副本拥有同一个 future。

current history 中只包含“该条件是否允许分支”的可见 cue，
不包含最终选择了哪个 future mode。

正式 synthetic benchmark 使用至少 `12 x 12` 的 feature grid；当前
几何 coverage 预检使用 `16 x 16`。预检要求：
在 deterministic group 上，用最后两帧 target slot center 做恒速外推，
其 future center MSE 必须低于 current-center copy。该预检只判断观测运动
是否高于网格/slot 噪声，不向模型提供 future。grid 8 的预检失败，因此其
多模态 coverage 结果只作诊断，不作为架构证据。

三条 future 分支共享的运动轴必须由当前可见对象位置和外观确定：

`v_branch,k = 0.30 normalize(p_k + lambda f_k[0:2])`

其中：

- `k` 是对象索引。
- `p_k` 是对象当前二维位置。
- `f_k[0:2]` 是对象当前外观特征的前两个可见分量。
- `lambda` 是固定混合系数。
- `v_branch,k` 是对象 `k` 的可见条件分支速度；`0.30` 用于确保三个
  synthetic mode 的间隔高于 target encoder 数值噪声。

纯几何子基准中的未来 mode 只选择 `-v_branch,k`、`0` 或
`+v_branch,k`。
若分支轴由未来随机变量生成，history-only prior 不可能知道真实 mode
位于哪条轴上，该数据不能用于验证条件多模态。

语义子基准保持相同 current/history，并加入由当前对象外观唯一确定的
正交变化方向：

`u_k = normalize(roll(f_k) - <roll(f_k), f_k> f_k)`

`f^+_qk = normalize(f_k + beta h_q mode u_k)`

其中：

- `roll(f_k)` 是对当前对象 feature 通道做固定循环移位。
- `<.,.>` 是内积；减去投影后，`u_k` 与当前 feature `f_k` 正交。
- `beta` 是语义分支强度，只用于受控 benchmark。
- `h_q` 是归一化 future horizon，最后一个预测时刻为 `1`。
- `mode` 仍为 `-1`、`0` 或 `+1`，不进入模型输入。
- `f^+_qk` 是语义子基准的 future 对象 feature。

因此分支轴完全由 current feature 决定，future 隐变量只选择沿该轴的
方向。纯几何子基准验证 center coverage；语义子基准验证 JEPA
feature/latent coverage，二者分别报告，不用一个指标替代另一个。

### 6.2 Stateful Object-delta Posterior

对齐后的当前和未来对象 slot 构造：

`h_qk = W_d(O_qk - O_0k) + W_0 O_0k + W_+ O_qk + W_a a_qk`

其中：

- `O_0k` 是对象 `k` 的当前 slot。
- `O_qk` 是未来时刻 `q` 的目标 slot。
- `a_qk` 是 future target slot 的 activity。
- `W_d`、`W_0`、`W_+`、`W_a` 是可学习线性映射。
- `h_qk` 是对象级变化 token。

`A` 个 latent action query 对全部 `K` 个 `h_qk` 做 cross-attention，
避免不同对象的相反运动在均值池化中抵消。delta-only posterior 曾作为
动作坐标系稳定化假设接受测试，但在同一 grid 和训练预算下不如 stateful
posterior；因此当前候选保留 absolute current/future 和已观测 history
摘要。future absolute slot 只进入训练期 posterior，不进入 prior context。

一对象一 action token 的 object-aligned 版本也做过相同容量对照，但
同时降低 latent 和 center posterior-oracle recall，故只保留为消融；
当前候选继续使用少量 free latent action queries。

为防止 posterior action 坐标系随场景任意旋转，新增二维中心变化锚点：

`Delta c_qk = c^+_qk - c^0_k`

`g_qk = [Delta c_qk, ||Delta c_qk||_2]`

`h_qk <- h_qk + W_g g_qk`

其中：

- `c^0_k` 是 history 当前时刻对象 `k` 的二维 slot 中心。
- `c^+_qk` 是 future 时刻 `q` 的 target slot 中心。
- `Delta c_qk` 是训练期可见的对象中心变化，只进入 posterior。
- `g_qk` 拼接二维变化和变化幅度。
- `W_g` 是把三维几何描述映射到 posterior hidden space 的可学习层。

另用 action 解码全部对象中心变化：

`Delta C_hat_q = D_c(vec(A_q))`

`L_action-center = mean_qk SmoothL1(Delta C_hat_qk, Delta c_qk)`

其中：

- `A_q` 是时刻 `q` 的全部 latent action tokens。
- `vec` 表示拼接 action token。
- `D_c` 是训练期辅助解码器。
- `Delta C_hat_qk` 是 action 对对象 `k` 中心变化的预测。

部署时 prior context 只把 history slot 中心经 `W_context` 加到 history
slot token；它不读取 `c^+`、`Delta c` 或任何 future 字段。因此中心锚定
约束 action 语义，但不改变严格因果边界。

history-only prior 还必须保留时间顺序：

`H_tk = W_s O_tk + W_c c_tk + W_tau s(Delta t_t)`

其中：

- `t` 是已观测 history 帧索引。
- `O_tk`、`c_tk` 是 history 帧 `t` 的对象 slot 和二维中心。
- `Delta t_t` 是该 history 帧相对当前帧的已知时间差。
- `s(Delta t) = sign(Delta t) log(1 + |Delta t| / tau_ref)`。
- `W_s`、`W_c`、`W_tau` 是可学习映射。

若省略 `W_tau s(Delta t_t)`，prior 对 history 帧排列近似不变，无法从多个
已观测中心判断速度方向。这里的时间差来自已观测 history，不是未来状态。

### 6.3 覆盖指标

对每个 current context 采样 `S` 个 prior future。
每个样本到真实 mode `j` 的 latent 距离为 `e_sj`。

令 `d_jl` 为同一 context 下 mode `j` 与 mode `l` 的 RMS 距离。
命中半径定义为：

`rho = 0.5 min_{j != l} d_jl`

样本命中 mode `j` 当且仅当 `e_sj <= rho`。代码保留 MSE 形式计算，
因此命中阈值等价于最小 mode 间 MSE 的 `0.25`，而不是 `0.5`。
这样不同 mode 的命中球不重叠。Best-of-N 仍报告 MSE，便于直接比较误差。

Mode Recall：

`Recall_mode = hit modes / true modes`

Sample Precision：

`Precision_sample = valid samples / S`

其中 valid sample 至少落入一个真实 mode 的命中半径。

Best-of-N：

`E_N = mean_i min_{s<=N} error(pred_is, target_i)`

其中：

- `i` 是评估样本。
- `s` 是 prior sample。
- `N` 是允许的采样数。

### 6.4 Gate

多模态先验成立需要：

1. prior context 对 future swap 的差异严格为 `0`。
2. posterior action 的不同 mode 距离显著大于相同 mode 数值噪声。
3. 使用每个真实 future 对应的 posterior action 时，纯几何子基准的 center
   和语义子基准的 JEPA latent/feature `Recall_mode >= 0.8`；否则对应的
   prior coverage gate 在下游不可达。
4. ambiguous group 的 `Recall_mode@16 >= 0.8`。
5. ambiguous group 的 `Precision_sample@16 >= 0.7`。
6. `E_16` 相对 deterministic prediction 至少改善 `15%`。
7. deterministic group 的 sample diversity 明显低于 ambiguous group。

仅有 sample 方差不算多模态证明。

## 7. 整体架构 Gate

完整架构通过需要：

1. 原有严格因果、梯度、单/三帧、DDP 和 checkpoint gate 全部保持。
2. 第 4、5、6 节的机制 gate 分别通过。
3. `full` 在统一 integrated benchmark 上优于每个单项退化和
   `all_degraded`，而不是只在某个局部指标获胜。
4. synthetic 结论至少覆盖 5 个训练 seed。
5. real held-seed 和 held-task 至少覆盖 3 个训练 seed。
6. 报告均值、标准差和逐 seed 原始结果，不用单次最好结果替代统计。

若任一机制 gate 未通过，结论必须是“架构尚未证明”，并定位失败路径；
不得用其他指标通过来覆盖该失败。

Integrated 对照按机制使用对应端点：`independent_slots` 比较
feature/latent MSE，`independent_prior` 比较 action Recall 和 Best-of-16，
`all_degraded` 比较 feature/latent MSE。每项至少在 3/5 个配对 seed
上由 `full` 获胜。六个变体还必须报告完全相同的可训练参数量。

## 8. 权威路径

本地代码根：

`/Users/hela/Instruct-GS-World/`

远端运行根：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/`

真实 strict-causal pair：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_pairs_e2e_20260717_v1/`

真实 DINO sidecar：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/rt2_causal_pairs_e2e_20260717_v1_dino32/`

所有测试和实验只在远端既有环境执行：

`/mnt/pfs/public/xuhaoming/instruct_gs_world/.venv/`
