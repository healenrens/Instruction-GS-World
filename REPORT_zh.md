# Instruct-GS-World 研究报告(全景版)

> **任务**:给定一个 3DGS 场景 + 一句自然语言指令,预测**每个高斯球的 3D 运动**,自回归滚动出未来 —— 一个语言条件的 3D 高斯世界模型,要求在真实开放世界视频上 work。
> **本版组织方式**:逐模块、逐流程讲清**信息流**与**作用**,供全景思考。所有断言对应代码位置(可点击)。
> 更新:2026-06-12。含 §92 训练数据视觉核验、SC-GS 控制点机制澄清、旋转问题三维拆解。

---

## 0. 一页总览

| 维度 | 状态 |
|---|---|
| 语言因果(听懂指令、选对物体) | ✅ swap 1.00→0.10,selection 1.00 |
| 运动幅度(走够距离) | ✅ 0.9×(R1 gate 修复) |
| 物体完整性(不散开) | ✅ 推理期 rigid_agg 投影(散开残差 2.15→0.01cm) |
| 真实视频(AgiBot 超市) | ✅ v12 共训,真实 EPE3D 16.3→10.8cm |
| 未见名词泛化 | ✅ v13b,unseen-book EPE 49.9→15.0cm |
| **旋转** | ❌ 5°5cm=0 处处;6 次架构尝试全负;**三维卡点(§9)** |
| 精细 3D 精度 | ⚠️ EPE3D ~10cm(数据规模 ~100 clip 太小) |

**生产模型** `libero_v12_rigid`;候选 `v13b`;在跑 `v15`(词汇+旋转数据)→ `v16`(低秩基 residual)。
**当前研究焦点**:控制点(关键点)选择的专门设计 —— 旋转问题的第三维(§9.3)。

---

## 1. 端到端信息流(全景图)

```
【数据生成期】(离线)
  sim:  LIBERO parquet ──Pi3 深度/相机──> 全视频融合 canonical 3DGS ──解析逐实体刚体──> traj 伪GT
  real: AgiBot mp4 ──StV2 3D跟踪 + Pi3 升维 + openvocab分割 + 运动仲裁──> 3DGS + traj 伪GT
                └── EEF 本体感知 ──Umeyama──> 米制尺度

【一次前向】(训练/推理同构)
  clip ─┬─ 3DGS 场景 G0 [N≈26万: means/quats/scales/σ/colors] + seg_per_g + uv
        ├─ 指令 + frame0 图像
        └─ traj [K+1, N, 3] (伪GT,仅训练用)

  (A) 条件流(每 clip 编码一次,整个 rollout 共享)
      [frame0图, 指令] → 冻结 Qwen3-VL-2B → 28层 hidden [28,L,2048]
        ├→ layer_proj×28 + 16个可学习query逐层聚合 → ctx_per_block [28,16,1536]
        ├→ 指令token池化 → cond_global [1,1536]  (AdaLN 条件)
        ├→ 末层指令token → text_feats [L_t,2048] (relevance 头的 k/v)
        └→ 视觉patch网格 → 每控制点 uv 处采样 → patch_feat [M,2048]
              ├→ vis_tok / vis_film / vis_vhead  (空间接地三路)
              ├→ dyn_head → p_dyn 动静门 logit
              └→ ×text_feats 交叉注意 → r_logit 语言相关性

  (B) 几何流(K=12 步自回归循环)
      从 26万 稠密里抽 M=2048 控制点(ctrl_idx,场景不同抽法不同 §3.2)
      每步: 控制点状态 → tokenizer → token [M,1536] (+vis_tok)
            → 28×DiTBlock(自注意=控制点互看; 块j 跨注意 ctx[j]; AdaLN(cond_global+step) + vis_film)
            → 逐点输出 δ = (v, ω, dlog_s, dlogit_σ, dcolor)
            → v += vis_vhead;  v,ω → tanh 限幅 → × gate(p_dyn ⊕ r_logit)
            → 控制点流形前进 (means+v, quat←Exp(ω)⊗quat, …)
            → LBS: 稠密 26万 点按冻结的 4-近邻权重混合控制点 SE(3) → 下一帧稠密场景

  (C) 监督流(仅训练)
      主: trajectory_loss = 控制点 pos L1 + vel L1   ←伪GT traj[:, ctrl_idx](逐点独立!)
      幅度: mover_magnitude;旋转: 邻域 Kabsch chordal;门: mover-BCE;
      语言: relevance BCE + 反事实(错指令→门关);正则: render/scale-anchor/delta

  (D) 推理附加
      rigid_agg: 每实体把 (v,ω) 投票加权 Kabsch → 单个 SE(3) → 投影回每点(刚性化,仅推理)
```

三条流的分工:**条件流**决定"谁动、往哪动"(语言/视觉);**几何流**承载"怎么动"(运动场);**监督流**几乎全部压在控制点 3D 轨迹上(稠密只过低权重渲染正则)。

---

## 2. 数据模块

### 2.1 clip 统一 schema(两条管线产出同构)

| 键 | 形状 | 作用 |
|---|---|---|
| means/quats/scales/opacities/colors | [N,*] | canonical 3DGS 场景(frame-0 姿态) |
| uv | [N,2] | 每高斯在 frame-0 图像的像素坐标(视觉接地采样用) |
| seg_per_g | [N] | 实体 id(0 背景,1-7 物体,8 机械臂…)→ entity-LBS / relevance / rigid_agg 全靠它 |
| is_obj | [N] | 指令目标物体掩码(relevance BCE 的标签) |
| traj | [K+1,N,3] | **逐高斯 3D 轨迹伪 GT(监督流的全部来源)** |
| K_intr / viewmat / H / W | — | 相机(渲染与评估) |
| gt_rgb | [K+1,H,W,3] | 真实帧(渲染正则 + 视觉核验) |
| instruction | str | 指令 |

### 2.2 sim 管线(LIBERO,40 train clip)

parquet(RGB+深度+分割+目标掩码)→ Pi3 逐帧深度/相机 → **全视频融合**成完整 canonical 3DGS(§42,补洞去噪)→ 逐实体**解析刚体运动**直接搬运高斯 → traj。
**质量(§92 视觉核验)**:✅ 重建逐帧吻合真实;物体是**致密实心块,刚性平移干净**。可信的训练目标。

### 2.3 真实管线(AgiBot,66 train clip)

mp4 → **SpatialTrackerV2**(世界系 3D 轨迹,静止点位移中位 0.1cm)→ Pi3 升维 + openvocab 分割(GroundingDINO+SAM2)→ **运动仲裁**选目标实体 → 逐实体刚性化 → **EEF 本体感知 Umeyama 尺度校准**(scale 0.697,残差 1.1cm → 米制)→ 质量门(位移 5-80cm、500-40k 高斯)。
**质量(§92 视觉核验,决定性)**:
- ❌ 完整重建 = **白雾**(单目深度背景填充占 97% 高斯)——外观差但不直接伤训练(监督在 3D 不在像素);
- ⚠️ 物体单独 = **稀疏模糊点云团**(非清晰物体)——平移信号在,形状糊;
- ❌ **旋转目标是坏的**:"turn on the stove" 真转的是小旋钮(~90°),但分割出的"物体"是 11k-18k 高斯的**整个锅/灶台区域(不转)**;Kabsch 量出的 20-43° ≈ 跟踪噪声+夹爪平移+形变,**不是旋钮真旋转**。⇒ 旋转一直在对着噪声目标学/评。

### 2.4 数据定律

1. **异 fps 数据源必须按 wall-clock 对齐窗口**(fps20 用 WIN=96 而非 40;违反 → 运动尺度域错位,训练毒化,EPE 翻倍。§88 v13 验尸)。
2. **伪 GT 必须视觉核验后才能用作判据**(§92:噪声旋转目标让 v15 的旋转结论失效)。
3. 干净解析-GT 旋转源 `data/libero_goal_lerobot`(抽屉/旋钮铰接,fps20)在服务器上,**尚未进过训练 mix**。
4. **用户判断(2026-06-12)**:靠噪声真实视频数据修旋转**难 scale**——数据轴按此归档,优先架构/关键点轴。

---

## 3. 模型模块(逐模块信息流)

### 3.1 条件编码器:冻结 Qwen3-VL-2B + 逐层聚合([model_full.py:229](code/igsw/model_full.py:229))

- **输入** `[frame0 图像, 指令文本]` 一条因果序列过冻结 Qwen → 28 层 hidden `[28,L,2048]`。
- **逐层读出**:28 个独立 `layer_proj` 把每层投到 d=1536;16 个**可学习 query**(+层 id 嵌入)对每层做交叉注意聚合 → `ctx_per_block [28,16,1536]`。**作用**:把 Qwen 从浅到深的表征逐层喂给动力学的对应块(块 j ↔ 层 j),而不是只用最后一层。
- **cond_global**:仅指令 token 池化(避免图像主导)→ AdaLN 的全局条件。
- **text_feats**:末层指令 token(detach)——因果序列里图像在前,**图像 patch 永远看不到指令**,指令信息只在文本 token 里,必须逐 token 暴露给 relevance 头(池化会毁掉"点名了哪个物体")。

### 3.2 控制点:稠密的临时子集,模型 = 连续运动场(核心机制)

控制点**不是固定骨骼**,是每次从 ~26万 稠密里抽 **M=2048 个下标**([scgs.py:38](code/igsw/dynamics/scgs.py:38)),自带位置/外观/uv:

| 场景 | 抽法 | 借了什么信息 |
|---|---|---|
| 训练 | **mover-biased**:一半 GT 位移>1cm,一半静态([train_sim.py:73](code/scripts/train_sim.py:73)) | GT 运动(否则 2048/26万 仅 ~10 个 mover) |
| 评估 | 逐实体均匀 | 仅分割 |
| 裸推理 | randperm 纯随机 | 无 |

**每个 step、每个 clip 都重抽** ⇒ 模型从不依赖"哪 2048 个点",学到的是**"任意带特征点集 → 每点运动"的连续场**,控制点只是采样位置。
**推论(对关键点研究关键)**:选点策略是**采样器**不是模型的一部分——换更聪明的选点器**不需要动模型主体**;但反过来,旋转的位置信号 `‖(R−I)(x−c)‖` 正比于到转轴距离,**随机抽点 = 杠杆几何全凭运气**。

### 3.3 控制点 token 化([tokenizer.py](code/igsw/dynamics/tokenizer.py))

`FourierPE3D(归一化位置) ⊕ quat(4) ⊕ log s(3) ⊕ logit σ(1) ⊕ color(3)` → 2 层 MLP → `[M,1536]`。位置只为 PE 做尺度归一(中心/半径),实际更新仍在世界系。**quats 进 token 时 stop-grad**(`detach_state_rot`,§3.7)。

### 3.4 动力学骨干:28 块 DiT(AdaLN-Zero,[transformer.py](code/igsw/dynamics/transformer.py),1.66B)

每块([transformer.py:92](code/igsw/dynamics/transformer.py:92)):
```
x = x + gate·SelfAttn(modulate(LN(x)))      # 2048 控制点互看 —— 空间信息传播的唯一通道
x = x + CrossAttn(modulate(LN(x)), ctx[j])  # 块 j 读 Qwen 层 j 的 16 个聚合 token(常开,不门控)
x = x + gate·MLP(modulate(LN(x)))
```
AdaLN 条件 c = cond_global + step 嵌入(+逐点 vis_film);调制有界(防 ada 权重无界增长);零初始化门 → 起步恒等。**作用**:自注意让控制点之间交换信息(理论上可以协调旋转场——但损失不奖励协调,§9.1);跨注意把语言/视觉逐层注入。

### 3.5 空间接地三路 + 双门(决定"谁动")

每控制点在 frame-0 uv 处从 Qwen 视觉网格 grid_sample 出 `patch_feat`([model_full.py:285](code/igsw/model_full.py:285)),分三路注入:
- **vis_tok** [M,d]:加进 token(残差流);
- **vis_film** [M,2d]:逐点 AdaLN shift/scale(打破"全点同条件");
- **vis_vhead** [M,3]:**直接速度投票,加在 v 上(载重路径)**——DiT 输出曾把逐点方向投影进零空间,这条专用头保证接地信号不可被投影掉。

双门(相乘进运动):
- **动静门 p_dyn**:`dyn_head(patch_feat ⊕ e_sem)`;e_sem 是 16 维物体语义嵌入(Gaussian-Grouping CE 监督)——2D patch 在机械臂悬停于桌面上方时会混叠,3D 实体 id 不会。`gate_entity_pool`:门 logit 按实体池化 → **整个物体一起动/停**(防半个物体冻结)。
- **语言相关性门 r_logit**:patch_feat(q)× text_feats(k/v)交叉注意([model_full.py:317](code/igsw/model_full.py:317)),只加在物体类控制点的门上(机械臂任何指令下都动)。**反事实损失**(§4)逼它真依赖文本。

### 3.6 输出头与限幅([model.py:184](code/igsw/dynamics/model.py:184))

逐点 `δ = (v3, ω3, dlog_s3, dlogit_σ1, dcolor3)`;`v ← max_disp·tanh(v+vis_vhead投票)`,`ω ← max_rot·tanh(ω)`(0.1 尺度单位 / 0.3 rad 每步),再 ×gate。零初始化 → 起步 G_{t+1}=G_t,学残差运动。

### 3.7 流形前进与递归([manifold.py:69](code/igsw/dynamics/manifold.py:69))

`means+v;quat ← Exp(ω)⊗quat;scale ×exp(clamp(dlog_s));σ ← sigmoid(logit+δ)`。新状态生成下一步 token → **12 步展开的递归网络**。
**§86 爆炸机制与解**:ω_t → quat_{t+1} → token → DiT 的递归链是 NaN 真凶(三次在环刚性训练崩溃的共同根因);`detach_state_rot`(quat 进 token 时 stop-grad)切断它 → 训练 0 跳过。**但治住稳定 ≠ 学得对**(v14 两个旋转头稳定地学出了噪声)。

### 3.8 LBS 蒙皮:控制点 → 26万 稠密([deform.py](code/igsw/gaussians/deform.py))

**绑定(一次性,frame-0,永久冻结)**:每稠密点找**同实体内**最近 k=4 控制点(`entity_lbs`;实体无控制点 → 回退普通近邻),权重 `w_ij = exp(−d²/2σ_i²)` 归一,σ_i 自适应(4 邻域均距×2)。
**每步增量**([deform.py:81](code/igsw/gaussians/deform.py:81)):
```
x_i' = Σ_j w_ij [ R_j(x_i − p_j) + p_j + v_j ]     R_j=Exp(ω_j), p_j=控制点当前位置
q_i' = normalize(blend(w,q_j) ⊗ q_i)               外观 δ 线性加权
```
三个关键性质:
1. **杠杆效应**:ω_j 经 `R_j(x_i−p_j)` 同时移动稠密点位置——ω 噪 → 位置糊;
2. **刚体精确复现**:实体所有控制点输出同一 (R,t) 时,LBS 数学上精确还原刚体变换(p_j 项消掉)——**散开纯粹是投票不一致的症状,蒙皮无损**;
3. **权重纯几何**:只看 frame-0 距离+实体 id,**不看运动结构、不可学习、rollout 中不更新**——跨关节(抽屉↔柜体)照样平均,铰接边界必糊。

### 3.9 推理期刚性投影 rigid_agg([rigid_agg.py](code/igsw/dynamics/rigid_agg.py))

每实体把逐点运动**投票** x→x+v 做加权 Kabsch(权=p_dyn;旋转用 ω 均值,§74)→ 单个 SE(3) → 投影回每点。参数零、对已刚性场恒等。**只在推理用**:散开 2.15→0.01cm、方向/选择不变;**放进训练环 = 6 次全崩/全伤**(§7 定律 1)。

### 3.10 已拒绝的结构化运动模块(为何架构货架上没有别的了)

| 模块 | 机制 | 死因 |
|---|---|---|
| v8-ent 实体头 | 实体特征池化 → 单 SE(3) | 池化抹方向(dir +0.99→−0.18) |
| V3 v-Kabsch 在环 | 训练环里 Kabsch | SVD 反向非有限梯度 |
| R2 ω-mean 在环 | 训练环里 ω 聚合 | 递归梯度爆炸(1068 跳) |
| w_traj_rot↑ | 加大旋转损失 | 训练崩+主指标伤 |
| §87-A entity-rot 头 | 实体级 6D 注意力读出 | 注入噪声,全面变差 |
| §87-B 低秩基 pure | SoM,B=10 基替换场 | 灾难(方向 −0.14,v8-ent 同款签名) |

---

## 4. 训练模块

### 4.1 损失清单(监督到哪、起什么作用)

| 损失 | 权重(现配方) | 监督对象 | 作用 |
|---|---|---|---|
| trajectory pos+vel L1 | 1.0+1.0 | 控制点 3D 轨迹 vs `traj[:,ctrl_idx]` | **主损失;逐点独立、零耦合(散开的损失侧根源)** |
| mover_magnitude | 0.5 | mover 总位移相对误差 | 治 L1 重尾欠预测(幅度塌缩,R1) |
| rotation(邻域 Kabsch) | 0.3 | Exp(ω) vs 邻域 GT 旋转 chordal | **ω 唯一直接监督;真实数据上邻域 Kabsch=噪声** |
| mover-BCE | 1.0 | p_dyn vs GT mover 标签 | 动静门 |
| relevance BCE | 1.5 | r_logit vs is_obj | 语言→物体绑定 |
| **反事实** | 1.0 | 错指令时 p_dyn/r→0 | **语言因果的载重损失**(同图不同文必须翻转 ⇒ 纯视觉不可满足) |
| sem CE | 0.2 | e_sem vs seg_per_g | 3D 实体身份(抗 2D 混叠) |
| render photometric | 0.1 | 稠密渲染 vs gt_rgb | 仅正则(2D 指标教训后降级) |
| scale_anchor / delta_reg | 0.2 / 1e-3 | 尺度漂移 / δ 范数 | 长时域稳定 |
| InfoNCE(早期遗产) | 0.5 | 运动嵌入↔语言嵌入 | 早期反语言塌缩;作用已被 rel/反事实取代 |

### 4.2 配方与稳定性

4×A100 DDP,bf16(gsplat/SVD/rot6d fp32 岛),grad ckpt;**两组 lr**(基座 3e-4 / 新初始化头 1e-3);**NaN guard**:全局梯度范数非有限 → 整步跳过(`clip_grad_norm` 不是 guard——一个非有限梯度会经全局范数毒化全部 1.76B 参数,§30 血泪);新头一律**零初始化**(恒等起步,热启动无损);settle 阶段(短步数低 lr)防持续 lr 漂移幅度。

---

## 5. 评估模块(3D-first,§71 指标改革后)

| 指标 | 定义 | 作用 |
|---|---|---|
| EPE3D | 逐点端点误差(中位/P90) | 主精度 |
| Acc3DS / Acc3DR | ≤5cm∨5% / ≤10cm∨10% 占比 | 场景流标准 |
| **5°5cm** | 实体 Kabsch (R,t) 双阈联合 | **旋转的硬判据(处处=0 → 未解)** |
| mag-ratio 中位+P10 | 预测/GT 位移比 | 塌缩探测器(看分布不看均值) |
| GT-rot | GT 本身的实体旋转角 | 数据旋转含量审计 |
| langswap sel / dir-cos | 换指令选择/方向 | 语言因果守卫(**dir 是结构化运动的 kill 判据**) |
| 散开残差 / extent-ratio | 实体内一致性 | 刚性守卫 |

教训:corr/ratio/渲染 PSNR 都是聚合量,先后掩盖了方向、散开、幅度三连盲区——**2D/范数指标不作判据**。
视觉核验工具:`viz_traindata_verify.py`(真实|重建|仅物体 三行,§92)、`_viz_pred` 系列。

---

## 6. 研究历程(七阶段)

| 阶段 | 问题 | 解法 | 结果 |
|---|---|---|---|
| 可行性 | 架构能不能学 | 干净 sim GT + 空间接地 | corr 0.95 过拟合 + 跨任务泛化(但指标自欺) |
| 语言因果 §52-57 | 换指令照动(swap=1.00) | relevance 头 + 反事实损失 | swap→0.10,sel 0.75 |
| 刚性 §58-69 | 物体散开 | rigid_agg **推理投影**(在环=崩) | 散开 2.15→0.01cm |
| 诚实指标 §71 | 2D 掩盖一切 | eval_3d 全套 3D | 暴露塌缩 0.63×/旋转 27° |
| 幅度 §72-73 | 门误关真 mover | mover_magnitude 入配方 | 0.63→0.91×,sel→1.00 |
| 真实视频 §78-85 | 走出仿真 | StV2+EEF 校准+共训 | v12 真实 EPE 10.8cm |
| 词汇 §88-90 | 8 名词上限 | openvocab+libero_90(WIN=96) | unseen-book 49.9→15.0cm |

旋转(§86-92):六次架构尝试全负(§3.10)+ 数据目标被证伪(§2.3)→ 三维拆解(§9)。

---

## 7. 沉淀定律

1. **逐控制点平移场是唯一可靠的运动载体**;几何结构(刚性/旋转)只能做**推理期投影**——在环训练必崩,替换场必杀方向,叠加必注入噪声(6 次确认)。
2. **递归梯度是 NaN 真凶**(ω→quat→token→DiT 12 步展开);detach_state_rot 治稳定,但稳定≠学对。
3. **聚合指标系统性自欺**(方向/散开/幅度三连盲区);只有逐 clip 分布 + 3D-first 可信。
4. **异 fps 数据按 wall-clock 对齐窗口**,否则训练毒化。
5. **伪 GT 必须视觉核验**后才能作训练/评估判据(§92:噪声旋转目标使 v15 旋转结论失效)。
6. `clip_grad_norm` 不是 NaN guard;非有限梯度按步跳过。
7. 新头零初始化恒等起步;热启动后短 settle,持续高 lr 漂移幅度。

---

## 8. 能力边界(诚实版)

**能**(见过的物体类别,未见场景/实例):听懂指令选对物体(sel 1.00)、方向对(dir +0.8)、幅度对(0.9×)、刚体移动、真实超市视频可用、未见名词可泛化(15cm)。
**不能**:旋转(5°5cm=0)、厘米级精度(EPE ~10cm)、铰接/柔性(LBS 纯距离绑定)、模型原生刚性(靠推理投影,依赖 openvocab 分割质量)。

---

## 9. 未解决核心:旋转的三维拆解(当前研究焦点)

**9.1 架构耦合维**:物体要转 ⇔ 它的控制点输出**协调的切向 v 场**(对侧反向)。但主损失逐点独立(零耦合项),自注意有通道却无激励;ω 唯一监督是邻域 Kabsch(真实数据上=噪声)。架构侧的结构化方案已全部阵亡(§3.10)——**这条维度上"加头"已证死路,剩下的是改激励(损失耦合)而非改结构**。

**9.2 数据质量维(已归档,2026-06-12 用户判断)**:真实视频旋转目标被视觉核验证伪(整团锅灶 ≠ 旋钮);**靠噪声真实数据修旋转难 scale**。干净解析-GT 旋转源(libero_goal_lerobot)未用——若未来重启数据轴,从它开始;但**当前优先级让位于关键点轴**。

**9.3 关键点选择维(用户新方向,待设计)**:现状 `randperm`/mover-biased 纯随机——
- 旋转位置信号正比于**到转轴距离**:点全聚质心附近 → 信号≈0;点不在杠杆远端/转轴两侧 → ω 不可辨识;
- LBS 绑定纯距离:跨关节平均,铰接边界糊;
- 训练 mover-biased(物体 ~1024 点)vs 推理 randperm(物体可能仅几十点)→ **训推分布不一致**,推理时物体控制点稀疏、σ 大、运动被平均。
**天然接口**:模型本就是"任意点集 → 运动场"(每步重抽),选点器可**独立设计/学习**(关节点、转轴邻域、运动显著点、杠杆几何),不动模型主体;绑定权重可从纯距离升级为结构感知。**这是三维里唯一既没试过、又有干净接口的方向。**

---

## 10. 进行中实验与判读

| 实验 | 内容 | 判读 caveat |
|---|---|---|
| **v15**(跑) | v12 基座 + book96 词汇 + libero_goal 旋转数据,四重评 | 词汇轴(held90)可干净裁定;**旋转轴(heldgoal)不可**——目标本身是噪声(§2.3),5°5cm=0 不证明任何事 |
| **v16**(队列) | 同基座同数据 ± 低秩基 residual,干净 A/B | kill 判据 = langswap 方向(pure 死在 −0.14) |
| **v17**(数据备好,训练 hold) | turn-focused 强旋转(40°) | **同为噪声目标,优先级降**(§9.2 归档) |

出齐后:v13b vs v15 vs v16 生产三选一(守卫指标定夺)。

---

## 11. 模型谱系

```
stream11(AgiBot 早期,InfoNCE) → sim_gen(ManiSkill 泛化) → libero_v7_pi3
  → v8lang(语言因果) → v9lang(场景泛化) → v9lang_ov(openvocab 推理就绪)
  → v10/v11_rigid(刚性投影 + 幅度修复) → v12_rigid(sim+real 共训)★生产
  → v13b(词汇扩展)candidate → v15(+旋转数据)/ v16(低秩基 residual)[收官中]
```

**代码骨架**:`igsw/model_full.py`(总装)/ `igsw/dynamics/{model,scgs,transformer,tokenizer,manifold,rigid_agg,rot6d}.py` / `igsw/gaussians/{deform,render,…}.py` / `igsw/training/losses.py` / `scripts/{train_sim,eval_3d,eval_langswap,agibot_clip_stv2,viz_traindata_verify}.py`。流水账:`agent.md` §1-92。
