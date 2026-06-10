# Instruct-GS-World 实验记录：训练与评估全流程

> 本文档汇总本项目至今的主要实验：每个实验**用什么数据、怎么训练、怎么评估、得到什么结论**，并给出可复现的脚本/命令。
> 完整的逐日决策日志见 [`agent.md`](../agent.md)（§ 编号与本文一致）；服务器连接与环境细节见 [`HANDOFF.md`](../HANDOFF.md)。
> 最后更新：2026-06-10（§64 之后）。

---

## 0. 项目与实验主线一览

**目标**：语言条件的 3DGS 动力学/世界模型——输入一个 3D Gaussian Splatting 场景 + 一句自然语言指令，预测每个高斯球的 3D 运动并自回归 rollout。

**模型**：冻结 Qwen3-VL-2B（Cosmos-Reason2-2B）做视觉-语言条件 + 1.66B 动力学网络（SC-GS 风格：控制点 transformer + LBS 到稠密高斯）。代码主体：`code/igsw/model_full.py`（`InstructGSWorldModel`）、`code/igsw/dynamics/`、训练入口 `code/scripts/train_sim.py`。

实验主线（每一代都由上一代的失败诊断驱动）：

| 阶段 | 实验 | 数据 | 核心结论 | agent.md |
|---|---|---|---|---|
| 可行性 | v1 / run1 | AgiBot 离线 clip | +2.5dB vs static，可行 | §20 |
| 规模化 | stream11 系列 | AgiBot 流式 | InfoNCE 修语言塌缩；NaN guard | §29-30 |
| **数据转折** | ManiSkill 干净 GT | `data/maniskill*` | 噪声 GT 是定位失败根因；spatial-grounding 必要 | §38-39 |
| 门控/语义 | dyngate1-7 | `data/maniskill_fused` | dyn-gate + sem→gate + magnitude loss | §44 |
| 换基准 | LIBERO v5-v6 | `data/libero_video_v3` | St4R 几何畸变 → 换 Pi3 | §48-51 |
| 语言诊断 | langswap on v6 | — | **模型完全忽略语言**（swap/true=1.00） | §52a |
| 后端切换 | v7-pi3 | `data/libero_pi3` | Pi3 各向同性内参，方向 +0.93 | §53 |
| **语言因果化** | v8-lang / v8-ent / v8b | `data/libero_pi3(_v2)` | relevance 头 + 反事实损失 ⇒ swap 1.00→0.10；实体头方向回归被封存 | §54-55 |
| 生产模型 | **v9-lang** | `data/libero_pi3_v2` | 双窗数据 ⇒ 场景泛化 0.25→**0.75** | §56-57 |
| 开放词汇 | openvocab + v9-lang-ov | `data/libero_pi3_v2_ov` | **openvocab 分割推理就绪**（2×2: 0.75=0.75）；OV 训练伤选择(0.50)不伤方向 | §58-64 |
| 已知未解 | 物体散开 (coherence) | — | 软刚性损失不泛化到未见场景（extent ×3.72）；调研中 | §64 后 |

---

## 1. 环境与公共基础设施

- **服务器**：4×A100-80GB；工作区 `<WS>=/mnt/pfs/public/xuhaoming/instruct_gs_world/`（连接方式见 HANDOFF.md）。本地 `/Users/hela/Instruct-GS-World/` 只镜像代码（rsync）。
- **环境**：`<WS>/.venv`（python3.11, torch 2.8+cu126, transformers 5.10.2, gsplat 1.5.3）。所有命令前缀 `.venv/bin/python` / `.venv/bin/torchrun`。
- **HF 权重缓存**：`export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；下载需代理（见 HANDOFF.md）。
- **多卡训练公共注意事项**（踩过的坑，全部已写进 `train_sim.py`）：
  - 必须用 `.venv/bin/torchrun`（系统 torchrun 缺依赖）；
  - DDP `broadcast_buffers=False, static_graph=True`（FourierPE buffer 会被默认广播破坏）；
  - **NaN guard**：永远 backward（DDP 锁步），仅当全局梯度范数有限才 `opt.step()`（`clip_grad_norm_` 本身不是 NaN 防护，§30 曾因此毁掉全部权重）；
  - 杀训练用 `pkill -9 -f "[t]rain_sim"`（`[t]` 防止杀掉自己的 ssh shell）；
  - rsync 改代码后必须清 `code/scripts/__pycache__`（mtime 倒挂会让 python 用旧字节码）+ `PYTHONDONTWRITEBYTECODE=1`。

---

## 2. 数据管线（每个实验用的数据是怎么生成的）

### 2.1 ManiSkill3 干净 GT（§38-41，`data/maniskill*`）

**动机**：真实数据 GT（Pi3 深度 + CoTracker 2D 跟踪）有遮挡噪声 → 模型连单 clip 过拟合都无法定位运动。改用仿真拿**解析精确**的逐高斯运动：frame-0 RGB-D 建 3DGS、seg mask 给每球 actor id、用 sim 的逐 actor 位姿解析移动 `X_t = T_{o,t} T_{o,0}^{-1} X_0`。

```bash
# 核心脚本：code/scripts/maniskill_gt.py（generate_episode/build_clip/validate）
# 批量生成（16 分片 × 4 GPU；--shard/--nshards 并行）：
.venv/bin/python code/scripts/gen_sim_dataset.py \
  --tasks PickCube,PushCube,StackCube --seeds 200 --seed_base 1000 \
  --held_task StackCube --held_seed_frac 0.15 \
  --window_sec 4 --control_freq 20 --random_start 1 \
  --min_val_psnr 14 --min_movefrac 0.02 \
  --shard $i --nshards 16          # launcher: code/scripts/gen_sim_launch.sh
# 整片全帧融合版（+1.2/+2.6dB）：maniskill_gt._fuse_canonical_gaussians, --fuse_stride/--fuse_voxel
```

clip 按文件名分 split：`{env}_s{seed}_{train|heldseed|heldtask}.pt`。验收：`val_psnr≥16`、`movefrac≥0.02`。

**结论**（§39）：干净 GT + per-control spatial-grounding ⇒ 过拟合 corr 0.95（vanilla 塌到 0.03）；多 clip 训练 heldseed corr≈train（零过拟合）、heldtask(StackCube 从未训过) corr 0.65→0.76。**架构成立的判据实验。**

### 2.2 LIBERO + Pi3 抬升（§50-53，`data/libero_pi3`）

数据源：HF `binhng/libero_object_lerobot_mask_depth`（500 集 libero_object，"pick up the X and place it in the basket"，RGB+深度+分割 mask+language，fps=10）。读取入口 `code/scripts/video_gt.py::load_episode_full`。

每个 clip 的构建（`code/scripts/pi3_video_gt.py`，St4RTrack 因 x 轴各向异性压缩 2.4× 被 Pi3 取代）：
1. `find_object_id`：按"质心位移最大的非机器人 id"挑被操作物体（**~12% 失败演示被 `_libero_scan_movers.py` 过滤**）；
2. `pick_window`：取 48 帧窗口（`center` 居中 / `early` 提前到接触前 10-30 帧）；
3. Pi3 对 K+1 个子采样帧一次前向 → 逐帧 pointmap + 相机位姿；竖轴焦距估计 + 针孔重建（各向同性）；
4. frame-0 高斯化（`points_to_gaussians`）；`seg_per_g` 从 mask 采样；
5. 逐实体刚性运动：物体/夹爪 3D-3D Procrustes，机械臂 k-means 运动聚类拆 2-3 刚体簇（合成子 id 50+c）；CoTracker 提供 2D 轨迹；
6. 反向锚定补洞 + teleport guard (0.15m/step) + size-depth 线索。

```bash
# 单 clip（验证用）：
.venv/bin/python code/scripts/pi3_video_gt.py --epi 0 --K 12 --win 48 \
  --window_mode center --out data/libero_pi3/clip0.pt --split train \
  --g0_png viz/g0_check.png --motion_png viz/motion_check.png
# v7 批量（单窗）：bash code/scripts/gen_libero_pi3.sh "<train epis>" "<heldtask epis>"
```

clip schema（`torch.save` dict）：`means/quats/scales/opacities/colors [N,*]`、`uv [N,2]`、`seg_per_g [N]`（0=bg,1=目标,2=篮子,3-7=干扰物,8=臂,10=夹爪,50+=臂子簇）、`traj [Kf+1,N,3]`（GT 轨迹）、`is_obj [N]`、`K_intr/viewmat(viewmats)/H/W/Kf/instruction/gt_rgb/val_psnr/n_fill/focal/backend`。

### 2.3 双窗 v2（§54，`data/libero_pi3_v2`）——破坏"夹爪邻近捷径"

§52a 发现 23/24 clip 里目标=离夹爪最近的物体 → 视觉捷径让语言失效。修复：每集生成**两个窗口**——`_e`（early：起点在接触前，夹爪还远，邻近性失效）+ `_c`（center）。

```bash
# 用法：bash code/scripts/gen_libero_pi3_v2.sh "<train>" "<heldtask>" "<heldseed>"
# v8b/v9 实际用的划分（40 train + 8 heldtask + 8 heldseed 个 clip）：
bash code/scripts/gen_libero_pi3_v2.sh \
  "0 10 30 50 70 100 110 130 150 160 180 200 210 250 260 300 310 340 350 370" \
  "410 430 450 470" \
  "40 140 240 330"
```

split 语义：**heldseed** = 见过的名词、未见的 episode（测**场景泛化**）；**heldtask** = 未见的目标名词（测**词汇泛化**）。

### 2.4 开放词汇分割版（§58-62，`data/libero_pi3_v2_ov`）

用 GroundingDINO+SAM2 取代 GT mask（`code/scripts/openvocab_seg.py`）。最终方案 `segment_frame_amg`：
- SAM2 point-grid (20×20) segment-everything → 所有实体的干净 mask；
- 机器人→id8（GroundingDINO box，管线内运动聚类拆关节）、篮子→id2（box）；
- 地板剔除用 **SAM2 floor-mask 重叠判据**（不是颜色——tan 物体在 tan 地板上是独立 region 所以保留）；
- **目标必中**：`_sam2_point` 在 GT mover 质心打单点 prompt → 目标永远以 SAM2 质量 mask 标 id1（生成期 GT-motion 仲裁，plan 允许；推理期换 CoTracker 运动仲裁即可）。
- 自验 IoU（vs GT mask）：目标 0.95-0.97、篮子 0.97-0.98、臂 0.58-0.66。

```bash
# 管线集成 = pi3_video_gt.py 的 --seg 开关（只换 widx[0]/widx[-1] 两帧 mask 来源）：
.venv/bin/python code/scripts/pi3_video_gt.py --epi 0 --K 12 --win 48 \
  --window_mode early --seg openvocab --out data/_ovtest/clip0.pt
# 批量（同 v2 划分）：bash code/scripts/gen_libero_pi3_v2_ov.sh "<train>" "<heldtask>" "<heldseed>"
# 全自动 生成→训练→评估：code/scripts/orchestrate_v9ov.sh（见 §3.4）
```

---

## 3. 训练（`code/scripts/train_sim.py`，4×A100 DDP）

### 3.1 损失构成（`code/igsw/training/losses.py`；总和见 train_sim.py:403）

| 项 | flag/权重 | 含义 | 引入 |
|---|---|---|---|
| `pos_l`/`vel_l` | `--w_traj_pos 1 --w_traj_vel 1` | 逐控制点位置/速度 L1（主监督；逐点独立） | §39 |
| `rot_l` | `--w_traj_rot 0.2` | 局部旋转（GT-KNN Kabsch） | §39 |
| `rloss` | `--w_render` | 渲染光度 (L1+SSIM)，多帧 rsteps | v1 |
| `mag_l` | `--w_mag 0.5` | mover 幅度比例损失（L1 中位数欠射修复） | §44j |
| `dyn_l` | `--dyn_gate 1 --w_dyn 1.0` | 动/静门 BCE（免费 mover 标签 disp>1cm） | §44 |
| `seg_l` | `--sem_dim 16 --w_seg 0.3` | 逐球语义 id（Gaussian-Grouping CE + 3D-NN 一致性）；`--gate_uses_sem 1` 喂给门 | §44e-i |
| `rig_l` | `--w_rigid 0.5` | **实体刚性残差**（对每实体预测自身做可微 Kabsch 拟合，惩罚偏差=散开）。软约束，训练集有效、**未见场景不泛化**（§64 后已知问题） | §49 |
| `rel_l` | `--rel_head 1 --w_rel 1.5` | relevance 头 BCE（控制点 patch 特征 × 指令 token 交叉注意力 → r_logit，只加在物体类控制点的门上） | §54 |
| `cf_l` | `--w_rel_cf 1.0` | **反事实损失（语言因果化的承重墙）**：同 patch + 错误指令 → 命名物体的 p_dyn/p_rel 必须→0 | §54 |
| `scale_a` | `--w_scale_anchor 0.2` | 尺度锚（防长时域尺度爆炸） | §35 |
| InfoNCE | `--w_lang_contrast`（v8 起 =0） | 批内对比语言损失（AgiBot 时代修语言塌缩用） | §29 |

### 3.2 各实验训练命令

**通用骨干 flags**（v7 起全部继承）：`--K 12 --M 2048 --n_query 16 --cond_mode aggregator --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 --entity_lbs 1 --w_rigid 0.5 --gate_entity_pool 1`。

```bash
cd <WS>
export HF_HOME=/mnt/pfs/public/xuhaoming/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---- ManiSkill 泛化（§39, checkpoints/sim_gen）----
.venv/bin/torchrun --nproc_per_node=4 code/scripts/train_sim.py \
  --data data/maniskill --out checkpoints/sim_gen \
  --resume checkpoints/stream11c_infonce/ckpt_0006000.pt \
  --spatial_ground 1 --vlm_image 1 --lr 3e-4 --lr_sg 1e-3

# ---- dyngate7（§44k：门+语义+幅度的定稿配方, checkpoints/dyngate7）----
.venv/bin/torchrun --nproc_per_node=4 code/scripts/train_sim.py \
  --data data/maniskill_fused --out checkpoints/dyngate7 \
  --resume checkpoints/sim_gen/ckpt_last.pt \
  --dyn_gate 1 --w_dyn 1.0 --sem_dim 16 --w_seg 0.3 --gate_uses_sem 1 \
  --obj_focus 1.5 --w_mag 0.5 --lr 5e-5 --lr_sg 5e-4 --total_steps 300
  # 注意：settled 短训（cosine→0）；持续大 LR 会让幅度漂移（§44c）

# ---- v8-lang（§54：relevance 头 + 反事实, checkpoints/libero_v8lang）----
.venv/bin/torchrun --nproc_per_node=4 code/scripts/train_sim.py \
  --data data/libero_pi3 --out checkpoints/libero_v8lang \
  --resume checkpoints/libero_v7_pi3/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 800 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --w_rigid 0.5 --gate_entity_pool 1 \
  --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 --entity_head 0 --w_resid 0

# ---- v8-ent（§54：+实体槽位 SE(3) 头；已封存——方向 +0.99→-0.18 回归）----
#   同 v8-lang，但 --entity_head 1 --w_resid 0.1 → checkpoints/libero_v8ent
# ---- v8b（双窗数据 + 双头；继承实体头回归）----
#   同 v8-ent，但 --data data/libero_pi3_v2 → checkpoints/libero_v8b

# ---- ★ v9-lang（§56-57 生产模型, checkpoints/libero_v9lang）----
#   = v8-lang 配方，换双窗数据（rel only，实体头关）：
.venv/bin/torchrun --nproc_per_node=4 --master_port=29531 code/scripts/train_sim.py \
  --data data/libero_pi3_v2 --out checkpoints/libero_v9lang \
  --resume checkpoints/libero_v7_pi3/ckpt_last.pt \
  --K 12 --M 2048 --n_query 16 --cond_mode aggregator --lr 3e-4 --total_steps 800 \
  --spatial_ground 1 --vlm_image 1 --dyn_gate 1 --sem_dim 16 --gate_uses_sem 1 \
  --entity_lbs 1 --w_rigid 0.5 --gate_entity_pool 1 \
  --rel_head 1 --w_rel 1.5 --w_rel_cf 1.0 --entity_head 0 --w_resid 0

# ---- v9-lang-ov（§61-64 诚实性测试：同配方、openvocab 数据）----
#   一键编排（生成 56 OV clip → 训练 → 三划分评估）：
setsid bash code/scripts/orchestrate_v9ov.sh &   # 日志 logs/orchestrate_v9ov.log
```

训练日志每 20 步一行，关键在线指标：`corr`（GT-pred 位移相关）、`ratio`（top-mover 幅度比）、`dcos`（**方向余弦**，§A1 补盲）、`leak`（背景泄漏）、`relSel`/`cfSup`（relevance 选对率 / 反事实抑制，→0 = 语言因果生效）、`rPSNR`、`peakGB`。

### 3.3 Warm-start 规则（所有实验共用）

- 每代 resume 上一代 `ckpt_last.pt`，`strict=False`（新头自动 reinit，日志打印 reinit 张量数）；
- 新头一律**零初始化输出层** ⇒ 第 0 步行为和上一代逐字节一致（v8 rel 头实测 |Δmeans|=3.5e-6）；
- 换任务/数据时**不加载 optimizer/step**（新 cosine 调度）。

### 3.4 自动编排脚本（无人值守跑完整链）

- `code/scripts/orchestrate_v9ov.sh`：OV 数据生成 → v9lang_ov 训练 → langswap 三划分评估，全部 detached + 日志落盘。**这是"一条命令复现一个实验"的模板**，新实验照抄改三处（数据生成命令、train flags、eval 数据目录）即可。
- `code/scripts/orchestrate_4s.sh` / `orchestrate_fused.sh`：ManiSkill 时代的同类模板（等生成完→自动开训）。

---

## 4. 评估与诊断

### 4.1 langswap（语言因果性 + 选择准确率；`code/scripts/eval_langswap.py`）

对每个 held-out clip：同一场景/g0 跑 TRUE 指令 + 4 个 SWAP 指令（换别的物体名），**逐实体均匀采样控制点**（修掉 GT-mover 偏置的 eval 泄漏）。

```bash
.venv/bin/python code/scripts/eval_langswap.py \
  --ckpt checkpoints/libero_v9lang/ckpt_last.pt \
  --data data/libero_pi3_v2 --split heldseed --n_swap 4
# split ∈ train / heldseed / heldtask；交叉评估时换 --data（如 OV 模型在 GT 数据上测）
```

**指标定义**：
- **selection accuracy**（每 clip×swap 对）= floor ∧ suppression ∧ quiet：
  - floor：TRUE 指令下真 mover 移动 ≥ 0.25× 其 GT 位移（防"全静止"作弊）；
  - suppression：SWAP 指令下真 mover 移动 ≤ 0.5× TRUE 时（语言抑制）；
  - quiet：TRUE 指令下其它物体 <2cm；
- **swap/true ratio**：换指令后命名物体运动比（→0 = 语言因果）；
- **dir-cos / endpoint-err**（§A1 补盲）：mover 实体质心位移的预测-GT 余弦/中位终点误差。**教训：corr/ratio 全是范数指标、方向盲**——实体头把方向打到 -0.18 时所有旧指标毫无反应。

### 4.2 held-out 泛化（ManiSkill 时代主评估；`code/scripts/eval_sim_generalization.py`）

```bash
.venv/bin/python code/scripts/eval_sim_generalization.py \
  --ckpt checkpoints/sim_gen/ckpt_last.pt --data data/maniskill
# 输出：train/heldseed/heldtask 的 corr、top-mover ratio、Δ_null/Δ_wrong（语言敏感度）、rollout 视频
```

注意：曾有 **eval 静默漏 flag 的 bug**（rebuild 模型时丢 `gate_entity_pool/entity_lbs/rel_head/entity_head`、不传 `seg_per_g`）→ §49/§54 的 ckpt 被错误评估。已修：所有 eval/可视化统一走 `eval_langswap.build_model`（从 ckpt 读全部 flag）。**新评估脚本一律复用 build_model，不要手写模型构建。**

### 4.3 可视化/诊断脚本（视觉验证优先于指标）

| 脚本 | 用途 |
|---|---|
| `code/scripts/_viz_ov_data.py` | 训练数据审核：RGB / openvocab seg / GT seg / 3DGS 点云 / GT 运动（红t0→青tK）五联图 |
| `code/scripts/_viz_pred.py` | **模型预测审核**：GT 运动 vs 正确指令预测 vs 错误指令预测（语言因果可视化） |
| `code/scripts/_diag_scatter.py` | **coherence 诊断**：物体 extent 比（预测尺寸/初始尺寸；GT 恒 1.00）+ 散度。§64 后发现散开问题用 |
| `code/scripts/_viz_rigidfix.py` | 每实体 Kabsch 刚性投影的离线验证（extent 3.72→1.00、方向不变） |
| `code/scripts/_libero_review_video.py` | held-out rollout 审核视频（必须用 build_model + uniform 采样） |
| `code/scripts/_libero_data_video.py` | 数据 clip 渲染审核（splat 1.5×） |
| `pi3_video_gt.py --g0_png/--motion_png` | 生成期快速目检（REAL|g0、REAL|GT-motion 网格） |

### 4.4 openvocab 自验

```bash
# IoU vs GT mask（逐实体）：openvocab_seg.py 内置 per_entity_iou；批量自验模板：
.venv/bin/python code/scripts/_ovval.py    # 5 episode：目标/篮子/臂 IoU + 物体数
# 生成期每 clip 自动打印 "[ov] target-cov / target-IoU"（gen 日志可 grep 汇总）
```

---

## 5. 主要结果汇总

### 5.1 语言因果化与泛化（langswap, heldseed=场景泛化）

| 模型 | 数据 | TRAIN sel-acc | swap/true | heldseed sel-acc | dir-cos | 结论 |
|---|---|---|---|---|---|---|
| v7-pi3 | 单窗 | 0.00 | ~1.00 | — | +0.93 | 语言被忽略 |
| v8-lang | 单窗 | 0.53 | **0.10** | 0.25 | +0.99(train) | **语言因果 ✓** |
| v8-ent | 单窗 | 0.84 | 0.00 | — | **-0.18** | 方向回归→封存 |
| **v9-lang** | 双窗 | 0.57 | ~0.1 | **0.75** | **+0.81** | **生产模型** |
| v9-lang-ov | 双窗 OV | 0.57 | ~0.1 | 0.50 | +0.78 | 见 5.2 |

heldtask（未见名词）所有模型 = 0：8 名词词汇硬限制，B3（libero_90/goal 扩词汇）是解法。

### 5.2 openvocab 诚实性 2×2（§64，heldseed sel-acc / dir-cos）

| | eval on GT 数据 | eval on OV 数据 |
|---|---|---|
| v9-lang（GT 训练） | 0.75 / +0.81 | **0.75 / +0.81** |
| v9-lang-ov（OV 训练） | 0.50 / +0.79 | 0.50 / +0.78 |

- **行内相等 ⇒ 评估数据影响=0：openvocab 分割是 GT 的完美替代（推理就绪）**——真实视频没有 GT mask 也不损失任何性能；
- **列间差 ⇒ OV 训练伤"选择"通路（0.75→0.50）但方向幸存（+0.79≈+0.81）**：物体 mask 过分割噪声（is_obj 监督目标，物体球数中位 1.03× 最高 1.77×）；修法=生成期 mask 质量门，折叠进 B3。

### 5.3 已知未解决问题（按优先级）

1. **物体散开（coherence）**：预测的物体高斯球不保持刚体（最差 extent ×3.72，cream cheese）。根因：逐控制点头独立预测 + `w_rigid` 软约束只在训练分布内有效。所有范数/方向指标对此盲视（`_diag_scatter.py` 已补）。离线已验证"每实体 Kabsch 刚性投影"可 3.72→1.00 且方向逐位不变（`_viz_rigidfix.py`）；结构化方案（vote-then-aggregate / SE(3) motion bases）调研中。
2. **未见名词泛化 = 0**：词汇限制，待 B3（IPEC libero_90: 3921 集、~20+ 新名词；fps=20 需 stride 加倍）。
3. **OV 训练的选择掉分**（5.2）：mask 质量门待实现。
4. Phase C 逐帧 viewmats 已实现未 pilot 验证（`pi3_video_gt.py`/`train_sim.py` 已支持 `viewmats[Kf+1]`，静态 clip 自动广播向后兼容）。

---

## 6. 最小复现路径（从零到 v9-lang）

```bash
# 0) 环境：HANDOFF.md（venv + HF cache + Pi3/CoTracker 权重 + binhng/libero 数据集）
# 1) 生成双窗数据（56 clips，4 GPU 并行，~1h）
bash code/scripts/gen_libero_pi3_v2.sh \
  "0 10 30 50 70 100 110 130 150 160 180 200 210 250 260 300 310 340 350 370" \
  "410 430 450 470" "40 140 240 330"
# 2) 需要 v7_pi3 起点权重（或从 dyngate7 链路重训：§3.2 自上而下）
# 3) 训练 v9-lang（§3.2 的 v9-lang 命令，800 步 ≈ 75min on 4×A100）
# 4) 评估三划分 + 出审核视频
for SP in train heldseed heldtask; do
  .venv/bin/python code/scripts/eval_langswap.py \
    --ckpt checkpoints/libero_v9lang/ckpt_last.pt --data data/libero_pi3_v2 --split $SP
done
.venv/bin/python code/scripts/_viz_pred.py checkpoints/libero_v9lang/ckpt_last.pt data/libero_pi3_v2
# 通过标准：heldseed sel-acc ≥0.6、dir-cos ≥+0.8、错误指令下命名物体 ≈0cm
```
