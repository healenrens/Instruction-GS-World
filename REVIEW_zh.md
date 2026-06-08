# _Instruct-GS-World 实现审查文档（zh）OLD read agent.md

> 生成时间：2026-06-07。配套权威日志见 `agent.md`（§1–§33+），本文件聚焦**实现思路 + 代码位置**，便于逐文件审查。
> ⚠️ **这是一次代码快照**（agent.md ≈ §33 时点）。代码仍在并行演进——例如 `dynamics/model.py` 已新增 §37 的"逐控制点空间 grounding"可选钩子（见 §2.6）。审查时以实际 `file:line` 为准。
> 代码根：服务器 `/mnt/pfs/public/xuhaoming/instruct_gs_world/code/`，本地镜像 `/Users/hela/Instruct-GS-World/code/`（逐文件行数一致）。所有引用写作 `相对路径:行号`。

---

## 0. 现状速览
- **目标**：给定 3DGS 场景 + 自然语言指令，预测每个高斯球的变换（3D 场景流），自回归累加到 **≥10 秒**。
- **当前在跑**：`stream13_anchor`（4×A100 DDP，~44GB/卡，0.33 it/s，约 s8200+）。配置：`--K 8 --stride 8 --M 2048 --dim 1536 --layers 28 --vlm_image 1 --use_grounding 0 --action_dim 0 --w_lang_contrast 0.5 --lang_tau 0.07 --queue_size 256 --w_scale_anchor 0.5`，从 `stream11c_infonce/ckpt_0006000.pt` 续训。
- **可行性**：已验证（v1 80M：语言条件 rollout 比静态基线 +2.5dB）。
- **两个未结指标**：①语言可控性 Δ=0.24（InfoNCE 后翻倍，中等强度）；②长程目前稳到 ~4s，5s 后曾因**尺度爆炸**崩溃，已加 `scale_anchor_loss` 修复、验证中。
- **规模**：可训练 ~1.76B（动力学 1.66B + 逐层投影 + InfoNCE 头）；Qwen3-VL 2.44B **全程冻结**。

---

## 1. 总体架构（数据流）

```
AgiBot 视频(AV1) ──decode_window(seek)──► 头相机帧窗口 [K+1,H,W,3]
   │                                            │
   │ (streaming.py 边界采样: 子任务onset过采样)   ▼
   │                                   Pi3 提升(lifting/pi3_lifter)
   │                                   → 点图/相机位姿/conf  (无标定→位姿估计)
   │                                            │
   │                              points_to_gaussians → 稠密 G0(~120k)
   │                                            │
指令文本 ──► [冻结 Qwen3-VL-2B] 全28层特征 [n_l,L,2048]      控制点子集 M=2048
              (conditioning.py)               │                 │
                                  逐层投影/聚合 → ctx_per_block   │ LBS 绑定(deform)
                                              │                 │
                                   ┌──────────▼─────────────────▼─────────┐
                                   │ 1.66B 动力学 transformer (model.py)    │
                                   │ block_j ⨯-attend Qwen layer_j + AdaLN  │
                                   │ → 每控制点 Δ(v,ω,δs,δσ,δc)  on-manifold │
                                   └──────────┬─────────────────────────────┘
                                  SC-GS rollout K 步 (scgs.py): 控制点动 + LBS 传到稠密
                                              │
            ┌──────── 监督 ──────────┬────────┴───────────┐
   CoTracker3 GT 3D 轨迹       渲染(gsplat) vs 真实未来帧    尺度锚定/反漂移/InfoNCE
   (tracking.py, 主损失)        (render.py, 辅助 w=0.1)      (losses.py)
```

**关键设计原则**：①Qwen 冻结（保泛化）+ 下游小头学"读特征→动高斯"；②动力学只在 M=2048 控制点上跑（注意力可承受），LBS 传播到 12 万稠密（保画质）；③**主损失是直接 3D 轨迹**（CoTracker3 对应），渲染只是辅助——这是 v4 redirect 后的关键改动（§13）。

---

## 2. 端到端 pipeline（按数据流，逐模块）

### 2.1 数据读取 — `igsw/data/`
- **`lerobot_agibot.py`** AgiBot LeRobot v2.1 读取器。无相机标定（被转换丢弃）→ 位姿由 Pi3 估计（docstring:13-14）。
  - `AgiBotLeRobotTask`(:65)：加载 info/tasks/episodes；`EpisodeMeta`(:55) 含 `action_config` 细粒度子任务 `{action_text,skill,start,end}`(:60-62)。
  - 子任务语义：`subtasks`(:107)/`subtask_text_at`(:122)（带回退到 episode 级语言）。
  - `read_parquet`(:167)：嵌套 list 列用 `to_pylist()+np.array` 还原多维张量（双臂 `[T,2,4]` 四元数，:183-184）。
  - 视频(PyAV/libdav1d AV1)：`decode_frames`(:209) 顺序解码（AV1 随机 seek 不可靠）；**`decode_window`(:244) 关键帧 seek + 按 pts 映射帧号**（流式用，:265-273，缺帧回退顺序解码）。
- **`streaming.py`** 流式 JIT 管线（核心扩展）。
  - `build_clip_index`(:23)：**段级索引**，每条 `(task,ep,seg_start,seg_end,action_text)`；只保留有 action_text 且长度 > span+2margin 的子任务段；无 action_config 时回退整 episode。⇒ 每个 clip 落在单一子动作内、带细粒度语言。
  - `StreamingClipDataset.__iter__`(:103)：**边界偏置采样**。`p_boundary=ratio/(ratio+1)=0.8`(:110)；onset 窗 `[lo, lo+0.35·seglen]`(:116)过采样，中段以 `middle_weight=0.3` 降权(:124)。理由：子任务起始处静态场景最不足以决定后续动作 → 语言才必要（§29）。
  - `_action_seq`(:82)：可选 `[K,8]` 末端执行器动作序列（双臂 Δpos6+夹爪Δ2）。
  - 产出 `{frames[K+1],instruction,boundary_weight,is_boundary,task,ep,f0,(actions)}`。
- **`clip_dataset.py`** 旧版缓存 `.pt`（map-style，含缓存的 `lang_hidden`）；已被 streaming 取代。

### 2.2 几何提升 — `igsw/lifting/`
- **`preprocess.py`** Pi3 严格预处理：`compute_target_size`(:20) 像素上限 255000、两边取 14 的倍数；`preprocess_frames`(:33) LANCZOS、/255、HWC→CHW、`[0,1]`（ImageNet 归一化在 Pi3 内部，不在此）。
- **`pi3_lifter.py`** `Pi3Lifter.lift`(:61, no_grad)：bf16 autocast 前向；`conf=sigmoid(logits)`(:78)，`mask=conf>0.1 & ~depth_normal_edge(rtol0.03)`(:79-82)。输出（CPU,fp32）：`points[N,H,W,3]`(全局)、`local_points`(相机系)、`conf`、`camera_poses[N,4,4]`(cam→world,OpenCV)、`images`、`mask`。
- **`to_gaussians.py`** `points_to_gaussians`(:35)：每个存活像素 1 个高斯；**尺度=右/下邻像素 3D 距离均值**(`_per_view_neighbor_scale`:21)、稳健分位裁剪(:61-64)；旋转=单位四元数、不透明度=0.8、颜色=源 RGB(SH0)；有限性过滤(:54)。返回 uv 网格用于对应。
  > ⚠️ **画质瓶颈在此**：naïve 提升同视角 PSNR ≈17，是 init 而非照片级（agent.md §8）。

### 2.3 GT 对应/轨迹（主监督来源）— `igsw/lifting/tracking.py`
- **CoTracker3**(`scaled_offline.pth`)。`track`(:25) 把 frame0 的控制点像素查询出 `[T,M,2]` 2D 轨迹 + 可见性；`sample_pointmaps_at`(:43) 用 `grid_sample` 在 Pi3 逐帧点图上取出 `[T,M,3]` **3D GT 轨迹**。高斯 1:1 来自 frame0 像素 ⇒ 其时序对应=锚点像素的 2D 轨迹。trainer 用法见 `train_stream.py:268-269`。

### 2.4 场景表示 — `igsw/gaussians/`
- **`types.py`** `GaussianSet`(:28)：`means/quats(wxyz单位)/scales(>0)/opacities(0,1)/colors[0,1]/features?`；`log_scales`(:71)、`opacity_logits`(:75) 原始视图供流形加法更新；`validate`(:79)、`save_ply`(:90)。
- **`render.py`** `render_gaussianset`(:17)：调 `gsplat.rasterization(... render_mode="RGB", sh_degree=None ...)`，世界→相机 viewmat；`psnr`(:53)。⚠️ 本文件**不显式转 fp32**，依赖调用方传 fp32（见审查重点）。
- **`cameras.py`** `intrinsics_from_local_points`(:15)：用 OpenCV 针孔关系 `u=fx·X/Z+cx` 最小二乘恢复 K（无标定时的来源）；`viewmat_from_pose`(:44)=inv(c2w)。
- **`sampling.py`** `downsample_gaussians`(:21) 固定 N（随机/体素），不足则有放回补齐。⚠️ :41 有一行死代码 `_, first = torch.unique(...), None`。
- **`deform.py`** SC-GS 线性混合蒙皮：`build_lbs_binding`(:27) kNN(k=4)+高斯权重(σ=均值邻距·2)；`lbs_step`(:62) **`x'=Σ w(R(x-p)+p+v)`**(:88-91)、四元数带半球对齐的加权融合(:93-95,`_blend_quat`:52)、尺度/不透明度/颜色按权传播。增量式（相对当前控制点位姿）⇒ 适合自回归。

### 2.5 条件编码（语言/视觉）— `igsw/dynamics/conditioning.py`
- **`QwenVLEncoder`**(:25)：加载 `Qwen3VLForConditionalGeneration`，**完全冻结**(eval/use_cache=False/requires_grad_(False) :40-43)。
- `forward`(:115, no_grad)：返回 **`(hidden_all[n_layers,L,2048], valid_mask, text_mask)`**——全 28 层逐 token 特征，从不池化；`text_mask` 排除 image+特殊 token（强制指令依赖，避免 frame0 图像主导）。
- `encode_grounded`(:69)：额外用 Qwen 自注意力做训练自由的 `rel_grid`（替代外部 SAM/DINO）。
- `forward_metaquery`(:144, **非 no_grad**)：把 N 个可学习 query 拼到冻结 Qwen 后读取 query 隐状态（MetaQuery 路径，A/B 备选；M-RoPE/visual_pos_masks/attention_mask 都正确扩展）。
  > 实证：见 `scripts/verify_qwen.py`——重编码与缓存 hidden cos=0.99996，确认 2.44B 模型真实参与。

### 2.6 动力学模型 — `igsw/model_full.py` + `igsw/dynamics/{model,transformer,tokenizer,pe}.py`
- **`InstructGSWorldModel.forward`**(`model_full.py:133`)：①VLM encode → `ctx_per_block[1,n_l,*,d]`、`cond_global`(仅指令 token 池化)、`pooled_text`；②(可选 action_emb)；③`delta_fn` 闭包(:139)：每步 `dynamics(state, ctx, mask, cond, step_idx)`；④`SCGSRollout.rollout`(:147-148) 跑 K 步；⑤输出 `means/quats/scales/opacities/colors`(逐步稠密)、`v/om/dls`(逐步控制 Δ)、`ctrl`(轨迹)；⑥InfoNCE 头：`motion_emb`(DeepSets over per-control[v.mean,v.std]→256)、`lang_emb`(`lang_proj` 作用于**纯文本**池化)(:163-167)。
  - `n_query`(:37)>0 → 可学习聚合 query 把每层特征蒸成 Q 个 token；`cond_mode`(:40) aggregator/metaquery；断言 `dyn.n_layers==VLM层数`(:45)（1:1 block↔layer 耦合）。
  - **§37 新增（可选）**：`predict_deltas` 接受 `cond_local[B,N,d]`（逐控制点视觉 token，加到 token 残差流）与 `film_local[B,N,2d]`（让 AdaLN 调制变成逐控制点 `c[B,N,d]`，不再全 N 均匀）——为"语言/视觉空间定位到具体高斯"留的接口；默认 None=旧的全局条件行为（`model.py:79-105`）。
- **`dynamics/model.py`** `GaussianDynamics`：`DynamicsConfig`(:25) d1536/16H/28L/`max_disp0.1`/`max_rot0.3`/`checkpoint_every1`；`predict_deltas`(:82) split Δ=`[v3,ω3,δlogs3,δlogitσ1,δc3]`(:101)，**`v=max_disp·tanh`,`ω=max_rot·tanh`**(:103-104) 每步有界；**零初始化 head**(:75)→恒等启动；逐层 cross-attend `ctx_per_block[:,j]`(:93)；梯度检查点(:91-98)。
- **`dynamics/transformer.py`** `DiTBlock`(:66)：self-attn(AdaLN-Zero 门控) + cross-attn(到该层语言 ctx，**常开未门控**+`norm_ca` 参数自由 LN 约束) + MLP。**§31 修复**：`sa_sc,ca_sc,mlp_sc=tanh(...)`、`sa_g,mlp_g=tanh(...)`(:93-94) 把每块 scale/gate 限到 (−1,1)，`tanh(0)=0` 保持恒等初始化、并自限梯度防 AdaLN 无界增长导致的 LN 反传 NaN。
- **`dynamics/tokenizer.py`** 每高斯 token=`FourierPE3D(归一化μ)⊕q4⊕logs3⊕logitσ1⊕c3`(:50/62)；位置仅在 PE 内归一化（gauge 无关），μ 更新仍用世界单位。
- **`dynamics/pe.py`** `FourierPE3D`(:15)：`out_dim=3·(1+2·num_freqs)=63`(默认)；唯一空间信息来源 → 置换等变、计数无关。
- **`dynamics/manifold.py`** 流形更新：`μ+v`、`q←normalize(Exp(ω)⊗q)`、`s·exp(clamp±3 δs)`、`sigmoid(logit+δσ)`、`clamp(c+δc)`。**`axis_angle_to_quat`(:36) 数值修复**：`angle=sqrt(Σω²+eps²)`（eps 在 sqrt 内 ⇒ ω=0 处梯度有限），对照 `roma` 误差 <1e-6（`scripts/test_manifold_math.py`）。

### 2.7 SC-GS rollout — `igsw/dynamics/scgs.py`
- `SCGSRollout`(:20)：控制点=稠密子集（LBS 一致）；`rollout`(:52) K 步循环：`delta_fn`→`lbs_step` 形变稠密→`apply_deltas_tensors` 推进控制点；**自由 rollout（喂自身预测前进，无 teacher forcing）**；features(relevance) 静态前传。

### 2.8 渲染 — `igsw/gaussians/render.py`（见 2.4）

---

## 3. 训练目标（损失）— `igsw/training/losses.py`
按 `train_stream.py` 实际求和顺序与默认权重：

| 损失 | 函数(行) | 含义 | CLI 权重(默认) |
|---|---|---|---|
| 轨迹位置 | `trajectory_loss`(:58) | 对 GT 3D 轨迹的可见性加权 L1（前景 `1+obj_focus·rel` 加权，按可见数归一） | `--w_traj_pos` **1.0** |
| 轨迹速度 | 同上 | 逐步速度 L1 | `--w_traj_vel` **1.0** |
| 旋转 | `rotation_loss`(:112)+`kabsch_rotation`(:98) | 邻域 Kabsch 求 GT 旋转，预测旋转的 Frobenius² | `--w_traj_rot` **0.2** |
| 语言对比 | `contrastive_infonce`(:161) | **对称 CLIP-InfoNCE**(motion↔指令)，MoCo 队列做负样本，τ=0.07 | `--w_lang_contrast` **0.5** |
| 背景静止 | `background_static_loss`(:87) | 低相关性/背景高斯的漂移惩罚 | `--w_bg_static` **0.5** |
| 渲染(辅助) | `photometric_loss`(:40) | 0.8·L1+0.2·(1−SSIM)，仅渲染 `render_steps` 步 | `--w_render` **0.1** |
| Δ正则 | `delta_reg`(:52) | v/ω/δs 的 L2 | `--w_reg` **1e-3** |
| 加速度平滑 | `velocity_smoothness`(:175) | ‖x_{t+1}−2x_t+x_{t−1}‖² | `--w_vel` **1e-3** |
| **尺度锚定** | `scale_anchor_loss`(:147) | log 空间把 rollout 尺度锚回 G0（**§33 修长程崩溃**） | `--w_scale_anchor`（live=**0.5**） |
| (legacy) | `contrastive_lang_loss`(:136) | 旧固定 margin hinge，已弃用 | — |

总损失 `/grad_accum` 后再乘 `boundary_weight`（中段 clip 降权）。

---

## 4. 训练 / 评测脚本 — `code/scripts/`
- **`train_stream.py`**（**主训练器**，412 行）：JIT 流式步=lift→relevance→CoTracker3 GT→VLM encode→SCGS rollout→上表损失→backward。
  - **DDP**(:157)：`static_graph=True, broadcast_buffers=False, gradient_as_bucket_view=True`（§后者两项是踩坑修复）。
  - **非有限梯度护栏（§30 修复）**(:356-377)：**始终 backward**（DDP lockstep）；`gnorm=clip_grad_norm_`；`if isfinite(gnorm): step() else 跳过`（gnorm 是 all-reduce 后的全局范数 ⇒ 各 rank 跳过决策一致）；跳过时打印各项损失+`meansOK/camOK`+`task/ep/f0` 便于复现。
  - **跨 rank 跳过协调**(:312-319)：`ReduceOp.MIN` 同步 ok 标志。
  - **输入有限性检查**(:303-306)：`gt_traj/g0.means/g0.scales/Ks/viewmats` 任一非有限即跳过（Pi3 偶发退化相机→NaN 渲染）。
  - **MoCo 队列**(:208-213,333-336)：预填到固定大小（DDP static_graph 需固定形状）；只入队有限 embedding。
  - **resume**(:177-204)：CPU 加载；**按形状过滤**加载（架构变化时只载匹配参数）；仅架构未变时载优化器。
  - 关键 CLI 默认：`K8/stride3/M2048/dim1536/layers28/heads16/n_query16/lr2e-4/warmup1000/checkpoint_every2/render_steps2/...`（live 覆盖见 §0）。
- **`eval_longhorizon.py`**：lift frame0→rollout N 步→静态相机渲染 vs 真实未来帧；**漂移诊断**(:95-104, 逐步 maxScale/meanScale/disp)、**语言divergence**(`--instruction2`,:106-117)、PSNR-vs-time 曲线 + mp4(GT|静态|预测)。
- **`eval_lang_sensitivity.py`**：**Δ=‖v(ℓ)−v(·)‖/‖v(ℓ)‖**（step-0 控制速度，固定 ctrl_idx 公平对比）；Δ_null(空指令)、Δ_wrong(他clip指令)。
- **`train.py`**（旧版，缓存 clip+LoRA，**无 NaN 护栏**，已被 stream 取代）。
- 工具：`verify_qwen.py`(证明2B真跑)、`test_manifold_math.py`(对照roma)、`_nanchk.py`(ckpt NaN 扫描)、`_wcmp.py`(权重差)、`overfit_clip*.py`(v1 可行性 demo)。

---

## 5. 关键工程修复 / 坑（调试史，详见 agent.md §29–§33）
1. **语言被忽略**（后验坍缩）→ 弃用饱和 hinge，改 **InfoNCE**（非饱和 MI 下界）+ 边界过采样（§29）。Δ 0.11→0.24。
2. **NaN 损毁**（s6000 一步污染全部 1.76B 参数）→ 根因 `clip_grad_norm_` 不是 NaN 护栏；加**有限范数护栏**（§30）。
3. **AdaLN 无界增长 → LN 反传 NaN**（detect_anomaly + 权重差定位）→ **tanh 限幅 scale/gate** + `norm_ca`（§31）。
4. **长程漂移=尺度爆炸**（40 步 maxScale 0.013→38万；位置漂移其实极小）→ **`scale_anchor_loss`**（§33）。当前 `scl≈0.000` 已压住。
5. **K=8→16 直接放大失败**（BPTT 梯度爆炸→模型躺平）→ 放弃，改用 K=8+尺度锚定（§33）。
6. 运维坑：conda vs venv 的 torchrun；`pkill -f train_stream` 会自杀 ssh shell；ssh `&`+重定向吞输出（用 `setsid …</dev/null &`）。

---

## 6. 审查重点 / 已知缺陷 / 待办
**建议重点审查：**
- `losses.py:trajectory_loss/rotation_loss` 的归一化与可见性掩码是否正确（主损失，最影响结果）。
- `tracking.py` + `sample_pointmaps_at`：CoTracker3 2D 轨迹经 Pi3 点图采样得到的"GT 3D 轨迹"在**动态场景**下的可靠性（Pi3 静态假设、点图噪声）——这是直接 3D 监督的根基。
- `scgs.py:rollout` 自由 rollout 的增量 LBS 一致性；`deform.py:_blend_quat` 线性四元数融合仅适合小角度。
- `train_stream.py:356-377` 护栏 + `312-319` 跨 rank 同步的 DDP 正确性。

**已知缺陷/简化（诚实记录）：**
- 画质被 naïve Pi3 提升限制（~PSNR 10–17）；未做逐帧 3DGS 优化（§9 待办）。
- 语言可控性仅中等（Δ0.24）；MetaQuery A/B 未结。
- 长程 ≥10s 尚未确认（scale-anchor 修复验证中）。
- `render.py` 未显式 fp32（依赖调用方）；`sampling.py:41` 死代码；grounding 默认关闭（用 GT 运动派生 relevance 代替）。
- action 条件当前关闭（live `--action_dim 0`），代码已就绪。

**待办（优先级）**：①确认 scale-anchor 让长程不崩→推 K/horizon 到 10s；②MetaQuery A/B（用 Δ 判定）；③逐帧 3DGS 优化提画质；④St4RTrack 更强 GT 轨迹。

---

## 7. 文件索引（quick map）
| 功能 | 文件 |
|---|---|
| 顶层模型 | `igsw/model_full.py` |
| 动力学 transformer/流形/PE/tokenizer/SCGS | `igsw/dynamics/{model,transformer,manifold,pe,tokenizer,scgs}.py` |
| 条件编码(冻结Qwen) | `igsw/dynamics/conditioning.py` |
| 高斯表示/渲染/相机/LBS/采样 | `igsw/gaussians/{types,render,cameras,deform,sampling}.py` |
| 几何提升/对应 | `igsw/lifting/{preprocess,pi3_lifter,to_gaussians,tracking}.py` |
| 数据/流式 | `igsw/data/{lerobot_agibot,streaming,clip_dataset}.py` |
| 损失 | `igsw/training/losses.py` |
| 语言grounding(可选) | `igsw/grounding/{relevance,role_masks}.py` |
| 训练/评测 | `scripts/{train_stream,eval_longhorizon,eval_lang_sensitivity}.py` |
| 决策日志 | `agent.md`（§1–§33） |
