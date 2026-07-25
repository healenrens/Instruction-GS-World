# Adaptive GPSToken Object-JEPA World Model

日期：2026-07-18
状态：实施规格；用于约束代码、测试和小规模可行性实验。

## 1. 结论先行

GPSToken 不废弃。废弃的是旧 GPSToken 的三个具体机制：
1. 用手工图像熵递归分割决定 token 位置。
2. 用固定 `active_count` 强制所有样本采用相同密度。
3. 用未来轨迹位移 `traj[K] - traj[0]` 选择 mover token。

新版 GPSToken 的职责是：
- 只读取当前或历史可见帧的稠密视觉特征。
- 学习把有限的微高斯 token 分配到信息密度不同的图像区域。
- 输出连续 activation，使不同样本拥有不同的有效 token 数。
- 为对象级 slot 提供细粒度、可渲染的 2D Gaussian 表征。

对象 slot 和 GPSToken 不是二选一：
- GPSToken 解决局部空间覆盖与自适应密度。
- Object Slot 解决对象聚合、身份和交互。
- Dynamics 预测 Object Slot 的未来 latent。
- Gaussian Decoder 将预测 latent 读出为微高斯特征和辅助属性。

显式 Gaussian 属性不是 Dynamics 的主状态，也不递归反馈给 Dynamics。

## 2. 研究目标

只依赖视频，学习：

1. 从一帧或三帧历史中提取对象级状态。
2. 从未来视频提取训练目标。
3. 用 future-conditioned Action Posterior 表示实际发生的潜在变化。
4. 用 current/history-only Conditional Flow Prior 在推理时采样潜在变化。
5. 联合预测多个未来时刻的对象 latent，而非逐步回归显式 Gaussian 属性。

核心目标是 JEPA 式特征预测对齐。2D Gaussian feature rendering、RGB、相对几何均为辅助约束。

## 3. 因果边界

### 3.1 可进入部署路径的信息

- 当前帧或历史帧 RGB。
- 从这些帧独立提取的冻结视觉特征。
- 这些帧的相对物理时间间隔。
- 可选语言条件；视频模型成立不依赖语言。
- current/history-only Flow Prior 的随机样本。

### 3.2 只允许作为训练监督的信息

- 未来帧 RGB 和未来帧视觉特征。
- 未来 target encoder 的对象 latent。
- Action Posterior 读取的未来 latent。
- 可选 tracker、相对深度或渲染监督。

### 3.3 禁止路径

- 未来可见性改变当前 proposal 或 token 数。
- 未来位移改变当前 token 位置。
- 未来特征进入当前 encoder、Flow Prior context 或历史 mask 分支。
- 用 Action Posterior 帮助重建被遮盖的历史对象；这会直接泄露答案。
- 将 best-of-N 的 oracle 样本选择当作可部署预测。

## 4. 输入、符号与张量

| 符号 | 形状 | 含义 |
|---|---|---|
| `B` | 标量 | batch size |
| `T_h` | `1` 或 `3` | 可见历史帧数 |
| `Q` | 标量 | 联合预测的未来时刻数 |
| `N` | 标量 | 每帧稠密视觉网格位置数 |
| `C_f` | 标量 | 冻结视觉特征维度 |
| `M` | 标量 | 每帧允许的最大 GPSToken 数 |
| `K` | 标量 | 每个场景的最大对象 slot 数 |
| `D_g` | 标量 | GPSToken latent 维度 |
| `D_o` | 标量 | Object Slot latent 维度 |
| `A` | 标量 | 每个未来间隔的 latent action token 数 |
| `D_a` | 标量 | latent action token 维度 |
| `F` | `[B,T_h,N,C_f]` | 历史稠密视觉特征 |
| `P` | `[B,T_h,N,2]` | 归一化 2D 网格坐标，范围 `[-1,1]` |
| `V` | `[B,T_h,N]` | 图像 padding 有效位；不是对象或可见性标注 |
| `Delta` | `[B,T_h+Q]` | 相对当前时刻的物理时间间隔 |

历史的最后一帧定义为当前时刻，因而它的 `Delta=0`。更早历史的 `Delta<0`，未来目标的 `Delta>0`。

相机内参、深度、对象 mask、机械臂 mask、显式机器人 action 均不是必需输入。

## 5. 模块一：Learnable GPSToken Allocator

### 5.1 目的

把每帧 `N` 个稠密特征位置压缩为最多 `M` 个微高斯 token，并让有效 token 密度随场景自适应变化。

### 5.2 计算

先将视觉特征和坐标对齐到公共维度：

`x_n = MLP_f(F_n) + MLP_p(P_n)`

其中：

- `n` 是稠密网格位置索引。
- `F_n` 是该位置的冻结视觉特征。
- `P_n` 是该位置的 2D 坐标。
- `x_n` 是对齐后的网格 latent。

每个候选 GPSToken 有一个可学习查询 `q_m`。查询与当前帧的全局摘要共同生成分配 logits：

`ell_mn = <W_q(q_m + c), W_k x_n> / sqrt(D_g)`

其中：

- `m` 是 GPSToken 索引。
- `c` 是当前帧所有有效 `x_n` 的池化摘要。
- `W_q`、`W_k` 是可学习线性映射。
- `ell_mn` 是位置 `n` 分配给 token `m` 的未归一化分数。

对每个网格位置在 token 轴上归一化：

`a_mn = softmax_m(ell_mn)`

`a_mn` 表示位置 `n` 对 token `m` 的软归属。token 占用率为：

`rho_m = sum_n(V_n a_mn) / sum_n(V_n)`

token latent 和中心为：

`g_m = sum_n(V_n a_mn x_n) / sum_n(V_n a_mn)`

`mu_m = sum_n(V_n a_mn P_n) / sum_n(V_n a_mn)`

协方差由加权二阶矩得到，并加正下界：

`Sigma_m = sum_n(w_mn (P_n-mu_m)(P_n-mu_m)^T) + epsilon I`

其中：

- `w_mn` 是归一化后的 `V_n a_mn`。
- `epsilon` 是防止协方差退化的正数。
- `I` 是 `2x2` 单位矩阵。

activation 为：

`alpha_m = sigmoid(MLP_a([g_m,rho_m]))`

不同样本的 `sum_m alpha_m` 可以不同，因此 `M` 只是容量上限，不是固定有效 token 数。

### 5.3 输出

`LearnableGPSTokenAllocator.forward(features, coordinates, valid_mask)` 返回：
- `latent`: `[B,M,D_g]`
- `center`: `[B,M,2]`
- `covariance`: `[B,M,2,2]`
- `depth_order`: `[B,M,1]`
- `opacity`: `[B,M,1]`
- `activation`: `[B,M,1]`
- `assignment`: `[B,M,N]`
- `occupancy`: `[B,M,1]`

`depth_order` 只表示无标定 2D feature splatting 的相对前后顺序，不声称是公制深度。

## 6. 模块二：Object Slot Aggregator

### 6.1 目的

将同一对象上的多个 GPSToken 聚合成一个对象 latent，同时保留微高斯用于局部解码。

### 6.2 身份锚点

最早可见历史帧使用 `K` 个可学习 slot 查询执行竞争式 Slot Attention，得到身份锚点 `O_anchor`。

后续历史帧不重新随机初始化 slot，而是以 `O_anchor` 为查询读取该帧 GPSToken。这样 slot 索引由最早帧锚定，不需要外部对象 mask、tracker identity 或 Hungarian 后处理。

### 6.3 计算

竞争分配：

`pi_mk = softmax_k(<W_g g_m, W_o o_k> / sqrt(D_o))`

其中：

- `g_m` 是 GPSToken latent。
- `o_k` 是对象 slot。
- `pi_mk` 是微 token `m` 属于对象 `k` 的软概率。

对象更新：

`u_k = sum_m(alpha_m pi_mk W_v g_m) / sum_m(alpha_m pi_mk)`

`o_k <- GRU(u_k, o_k) + MLP(LN(o_k))`

其中：

- `alpha_m` 是 GPSToken activation。
- `W_v` 是 value 投影。
- `GRU` 是门控循环更新，只用于 Slot Attention 内部迭代，不表示时间递归。
- `LN` 是 LayerNorm。

### 6.4 输出

`ObjectSlotAggregator.forward(tokens, anchor_slots=None)` 返回：

- `slots`: `[B,K,D_o]`
- `assignment`: `[B,M,K]`
- `activity`: `[B,K]`

## 7. 模块三：Signed Gap Scale Modulation

模型不使用离散 frame id、显式 horizon token 或 learned time embedding。

物理时间间隔映射为：

`s(Delta t) = sign(Delta t) log(1 + |Delta t| / tau_ref)`

其中：

- `Delta t` 是该帧相对当前时刻的时间差。
- `tau_ref` 是数据集采样间隔的参考尺度。
- `s` 是无量纲 signed gap scale。

`s` 只通过 AdaLN/FiLM 调制 Transformer：

`AdaLN(h,s) = (1 + gamma(s)) LN(h) + beta(s)`

其中 `gamma` 和 `beta` 是由小型 MLP 从 `s` 生成的缩放与平移。模型因此知道变化尺度，但不会把时间作为一个可被直接复制的显式 token。

## 8. 模块四：Joint Object Latent Dynamics

### 8.1 输入

- 可见历史对象 slot：`[B,T_h,K,D_o]`
- 历史对象级随机 mask。
- `Q` 组未来 mask query。
- current-only 或 posterior latent action tokens。
- 每组 token 对应的 signed gap scale。

### 8.2 对象级历史 masking

最早身份锚点不全部遮盖。其余历史对象以对象为单位遮盖，同一对象的整个 slot latent 被 mask token 替换，而不是随机遮盖 latent 维度。

历史重建分支只允许使用零 action 或 current-only prior action，禁止使用 future-conditioned posterior action。

### 8.3 联合预测

所有可见历史 slot、被遮盖历史 query 和未来 query 一次进入双向 Transformer。未来 `Q*K` 个 slot 同时预测：

`O_hat_future = Transformer(O_visible, O_masked_query, O_future_query, A, s)`

其中：

- `O_visible` 是未遮盖历史对象 latent。
- `O_masked_query` 是历史 mask query。
- `O_future_query` 是未来对象 query。
- `A` 是 latent action tokens。
- `s` 是每个 token 对应的 signed gap scale。
- `O_hat_future` 是预测的未来对象 latent。

模型不自回归地把显式 Gaussian 属性送回下一步，因此不会累积中心、协方差或 opacity 的数值误差。

### 8.4 规模

完整目标配置：

- `D_o=1536`
- 28 个 Transformer blocks
- 16 个 attention heads
- `K=16`
- `M=256`

小规模验证配置可以降低宽度、层数、`K` 和 `M`，但接口和因果边界必须完全一致。

## 9. 模块五：Latent Action Posterior 与 Flow Prior

### 9.1 Posterior

`ActionPosterior(history_slots, stopgrad(future_slots), gap_scale)` 输出：

`A_q`，形状 `[B,Q,A,D_a]`。

Posterior 可以读取未来，因为它只在训练期用于提取“实际发生了什么”。未来 target 必须 stop-gradient，避免 posterior 反向改变 target encoder。

### 9.2 Prior

`ConditionalFlowPrior(history_slots, gap_scale, noise)` 输出同形状的 `A_p`。

Prior context 只由历史对象 slot 构成。训练使用 conditional flow matching：

`z_tau = (1-tau) z_0 + tau A_q`

`v_target = A_q - z_0`

`L_flow = ||v_theta(z_tau,tau,context) - v_target||^2`

其中：

- `z_0` 是标准高斯噪声。
- `tau` 是 flow 插值时间，均匀采样自 `[0,1]`。
- `z_tau` 是噪声与 posterior action 之间的线性插值。
- `v_target` 是目标速度。
- `v_theta` 是可学习条件速度场。
- `context` 只含历史信息。

这里的 `tau` 是生成流的积分变量，不是视频时间；视频时间只通过 signed gap scale 表达。

## 10. 模块六：Auxiliary Gaussian Decoder

`GaussianReadout(predicted_object_slots, current_micro_tokens)` 输出每个微 token 的：

- feature residual
- center residual
- covariance residual
- depth-order residual
- opacity residual
- activation residual

微 token 对对象 slot 的软归属用于广播对象预测。显式属性只参与辅助监督和可视化，不进入主 Dynamics 的下一次预测。

无相机内参时，decoder 只进行归一化图像平面的 2D Gaussian feature splatting。不存在伪造相机射线。

若数据提供可信相对深度与 confidence，可以启用相对 depth-order loss；否则该 loss 权重严格为零。

## 11. Loss

### 11.1 主未来特征目标

`L_future = mean(m_qk ||Norm(P(O_hat_qk)) - stopgrad(Norm(O_target_qk))||^2)`

其中：

- `q` 是未来时刻索引。
- `k` 是对象 slot 索引。
- `P` 是 predictor projector。
- `Norm` 是无可学习参数的向量归一化。
- `m_qk` 是由 target slot 自身 activity 产生的软权重，不是外部 mask。

### 11.2 历史遮盖目标

`L_history` 使用同一形式，对被遮盖的历史对象 latent 做预测，仅在 `T_h=3` 时启用。

### 11.3 辅助目标

- `L_flow`：Conditional Flow Matching。
- `L_feature`：2D Gaussian splatting 后的稠密视觉特征重建。
- `L_rgb`：可选 RGB 重建。
- `L_geometry`：可选 confidence-weighted 相对 depth-order。
- `L_alloc`：覆盖、token 多样性、activation 稀疏和预算正则。
- `L_slot`：slot 使用均衡与 assignment 低熵正则。

总损失：

`L = L_future + lambda_h L_history + lambda_f L_flow + lambda_feat L_feature + lambda_rgb L_rgb + lambda_geo L_geometry + lambda_a L_alloc + lambda_s L_slot`

所有 `lambda` 都是非负权重。默认训练顺序确保 `L_future` 是主目标，辅助项不能反客为主。

## 12. 梯度路由

1. Target encoder 输出 stop-gradient。
2. `L_future` 更新 current encoder、GPSToken、Object Aggregator 和 Dynamics。
3. `L_history` 不更新 Action Posterior。
4. `L_flow` 只更新 Flow Prior；posterior action 在该 loss 中 detach。
5. 辅助 Gaussian loss 更新 allocator/readout；进入主 Dynamics 的梯度使用小权重。
6. 冻结视觉 backbone 默认不更新。

## 13. 训练阶段

### 阶段 A：表征预训练

- Learnable GPSToken + Object Slot。
- 稠密 feature reconstruction、跨帧 slot consistency、allocation regularization。
- 不训练未来 Dynamics。

### 阶段 B：JEPA Dynamics

- 混合 `25%` 单帧历史和 `75%` 三帧历史。
- 对象级历史 masking。
- 联合预测多个未来 latent。
- 先使用 posterior action，验证模型能解释已知未来。

### 阶段 C：Flow Prior

- 冻结或低学习率更新 posterior/dynamics。
- 训练 current-only Conditional Flow Prior 拟合 posterior action。
- 检查采样多样性、coverage 和 calibration。

### 阶段 D：辅助解码

- 启用 2D Gaussian feature splatting。
- 有可信监督时再启用 RGB 或相对几何。
- 不允许辅助属性成为 Dynamics 主状态。

### 阶段 E：大规模多机训练

只有结构、因果、梯度、合成任务和真实小样本 gate 全部通过后，才进入两台八卡训练。

## 14. 实现文件和接口

新增隔离目录：

`/Users/hela/Instruct-GS-World/code/igsw/adaptive_gaussian_wm/`

计划文件：

- `config.py`：完整与 tiny 配置、参数合法性检查。
- `scale.py`：signed gap scale 和 scale-conditioned Transformer block。
- `gpstoken.py`：可学习 GPSToken 分配器。
- `object_slots.py`：身份锚定的 Object Slot 聚合。
- `latent_action.py`：Posterior 与结构化 Conditional Flow Prior。
- `dynamics.py`：对象级 mask 和联合未来 latent predictor。
- `decoder.py`：2D Gaussian feature readout/splatting。
- `losses.py`：主 JEPA、flow 和辅助 loss。
- `model.py`：严格区分 current、target、posterior、prior 的总接口。
- `synthetic.py`：只用于结构可行性验证的受控视频特征数据。

训练与验证入口：

- `/Users/hela/Instruct-GS-World/code/scripts/test_adaptive_gaussian_wm.py`
- `/Users/hela/Instruct-GS-World/code/scripts/experiment_adaptive_gaussian_wm.py`

不修改旧 `gpstoken_wm` 和 `latent_particle_wm`，它们保留为对照。

## 15. 必须通过的验证 gate

### 15.1 结构 gate

- 输入输出 shape 与配置一致。
- 所有非 padding 样本无 NaN/Inf。
- GPSToken 有非零梯度。
- `sum activation` 随场景复杂度变化，而不是固定常数。
- Object Slot 不全部塌缩到一个 slot。

### 15.2 因果 gate

固定历史、交换未来后：

- current encoder 输出最大绝对差必须为 `0`。
- Flow Prior context 最大绝对差必须为 `0`。
- posterior action 必须发生可测变化。
- 历史 mask 分支不得因 posterior future 改变。

### 15.3 学习 gate

- 联合 latent predictor 优于 current-copy baseline。
- 对象级模型优于 flat pooled-token baseline。
- mixed `1/3` 帧训练在两种历史长度上都有效。
- 不规则物理间隔下，scale modulation 优于完全忽略间隔。

### 15.4 表征 gate

- 高复杂度/多对象样本使用更多有效 GPSToken。
- 微 token 软分配能覆盖多个对象区域。
- 对象 slot assignment 对输入 token 排列近似不变。
- 2D feature splatting 能重建目标稠密特征，且梯度到达 allocator。

### 15.5 多可能性 gate

- Posterior 能区分相同历史的不同未来。
- Prior 多样本 coverage 随样本数提高。
- 明确区分 prior center、随机单样本和 best-of-N。
- 未实现 learned ranker 前，不把 best-of-N 写成部署性能。

## 16. 小规模多样化实验

合成数据只用于回答结构是否可学，不代替真实视频结果。至少包含：

1. 单对象平移。
2. 双对象不同方向运动。
3. 局部旋转或形变。
4. 遮挡导致局部特征消失。
5. 相同历史对应两种未来。
6. 稀疏背景与高纹理/高对象密度场景。
7. 单帧与三帧历史混合。
8. 不规则未来时间间隔。

对照：

- current-copy。
- flat token Transformer。
- Object Slot 无历史 mask。
- 完整 GPSToken + Object Slot + mask + scale。

主要指标：

- held-out future latent MSE。
- 2D feature reconstruction MSE/cosine。
- GPSToken 有效数量与场景复杂度相关系数。
- slot assignment purity 和 permutation consistency。
- future-swap 因果差。
- posterior sensitivity。
- Flow Prior best-of-N coverage 曲线。

## 17. 当前风险

1. 无外部 mask 时，slot 可能按纹理而非对象分组。
2. activation 可能全开或全关，需要预算和覆盖的平衡。
3. future target slot 依赖当前 identity anchor，严重遮挡时可能错配。
4. 2D Gaussian 无法单独消除相机运动与物体运动歧义。
5. Flow Prior coverage 改善不等于单样本可部署质量。
6. 合成实验通过只证明结构可优化，真实数据还需独立 gate。

这些风险通过小规模多样化实验先暴露，不通过时不启动大规模训练。
