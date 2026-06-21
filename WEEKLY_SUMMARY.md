# GPSToken-JEPA 世界模型 · 周进展汇总（§95–§104）

> 语言条件化的 3D 高斯世界模型：给 frame0 RGB + 稀疏 2D-Gaussian token（lift 到 3D）+ 语言指令，预测每个 token 的未来 3D 运动。骨干 = 冻结 Qwen3-VL-2B（语言）+ 冻结 DINOv2-L（视觉）+ ~1.66B 可训练 DiT 预测器。

本周一句话：**在 sim 定相机上把方向预测做到了 0.92（达标、泛化、可复现）；证伪了一个过拟合假象；并把研究重心转到"从视频学（无位姿）"这条真实世界路上，跑通了 pose-free 管线，定位了它的瓶颈。**

---

## 0. 总览：本周走过的三条线

| 线 | 问题 | 结论 | 状态 |
|---|---|---|---|
| **A. sim 定相机方向泛化** | 之前以为方向已解决，其实没有 | img_loss + 干净数据 + 稳定化 → held 3D dcos **0.92** | ✅ 达成 |
| **B. 相机变化下的泛化** | 变相机会不会毁掉学习 | 变相机训练后**新相机内插 0.87**（是资产不是毒药） | ✅ 达成 |
| **C. 从视频学（无位姿 GT）** | 现实没有位姿，必须从像素得 GT | pose-free 管线跑通 ~0.45；瓶颈=GT 噪声 + 域过拟合 | 🔬 进行中 |

---

## ◆ 方法基础：GPSToken 的使用、架构改变、学习方式与目标

### A. 我们如何使用并改造 GPSToken

**原始 GPSToken（arXiv 2509.01109，空间自适应图像 tokenization）**：用一组 2D 高斯（中心 μ、尺度 σ）表示一张图，通过**熵驱动的递归分区**放置——区域复杂度 `m = h·w·H^λ`（`H` = 该区域梯度幅度直方图的熵），每次劈分**最复杂的区域**直到 L 个区域，每个区域 → 一个高斯 `(μx,μy,σx,σy)`（中心 = 区域中心，σ = 区域边长/6）。信息越丰富的地方 token 越密。原用途是图像生成/重建的自适应 tokenizer。

**我们只借用它的"放置算法（Algorithm 1）"，不做重建，并做三处改造（`gaussians/gpstoken.py` + `gpstoken_wm/tokens.py`）：**
1. **运动/任务 saliency 加权（§93）**：`m *= (1 + β·该区域平均 saliency)` → token 优先落在**会动的物体（movers）**上，而非仅纹理丰富处。**训练时**用 GT-运动 saliency（把 movers splat 到 frame0 的 uv，再高斯模糊）；**推理时换成冻结 Qwen 的 relevance grid（无需 GT）**。
2. **lift 到 3D**：每个 2D 高斯 token 在 uv 像素空间绑定到**最近的稠密 3D 高斯** → 得到该 token 的 3D 位置 `tok_xyz0`；token 本质是一个**持久的稠密索引**，其 3D 轨迹被携带、跨帧免配准。
3. **token 三元参数**：`place_tokens()` 返回 中心 `cen[M,2]`、footprint `σ[M,2]`、稠密索引 `idx[M]`（碰撞去重后 M≤L）。

### B. 架构改变（从图像 tokenizer → 语言条件化 3D 运动世界模型）

| | 原始 GPSToken | 我们的世界模型 |
|---|---|---|
| 输入 | 一张图 | frame0 RGB + 稀疏 GPSToken(lift 3D) + **语言指令** |
| 主体 | 学习式 tokenizer | **冻结 Qwen3-VL-2B**(语言)+ **冻结 DINOv2-L**(逐 token 视觉特征)+ **~1.66B 可训练 DiT 预测器** |
| token 表示 | 高斯(位置+尺度+内容) | `FourierPE3D(归一化 3D 位置) ⊕ feat_in(DINOv2 grid 特征@cen) ⊕ σ` → `tok_embed` |
| 条件化 | — | DiT 每个 block 用 cond_global(语言池化) + 逐层 ctx(Qwen 隐层聚合) |
| 输出/目标 | 重建图像 | **预测每个 token 的未来 3D 运动** |

**v2 融合（本周用户拍板，已验证有独立增益）：**
- **grounding = 门**：`gate = σ(relevance(tok_feat, text))`，`motion = gate × geom_head(h)` —— 调制运动头（哪些 token 会动），不做独立解码器、不并行另算 loss。
- **JEPA = 从运动派生**：`jepa_head(h.detach())`（stop-grad，不抢主干）预测进**原始预训练 DINOv2 latent**（`_footprint_sample` 按 token 的 σ footprint 池化，适配非均匀 patch）—— 保留"预测未来特征"的能力，作为 world-model 附加部分。

### C. 当前的学习方式与目标

**任务**：给 frame0（+ 指令），预测每个 token 从 frame0→frameK 的运动。旋转不单独训（评估时 Kabsch 读出）。

**核心损失 `img_loss`（归一化图像空间 —— 本周关键洞察）：**
- 运动 = **归一化 2D 图像位移** `smooth_L1((uv₁ᵖ−uv₀)/[W,H], (uv₁ᵍ−uv₀)/[W,H])`，即"在图像里移动了图像尺寸的几分之几"——尺度一致（固定常数归一），治好了幅度欠预测。
- 深度 = **归一化深度变化** `Δlog z` 的 smooth_L1（全 3D = 图像流 + 深度，都尺度不变）。
- **mover 加权 `w_motion`**：高位移 token 权重 ×3–10，堵"预测零运动"的幅度坍缩。

**v2 融合的辅助损失**：门 BCE(gate vs movers) + JEPA cosine(预测 vs 原始 DINOv2 未来 latent) + SIGReg(特征正则)。

**训练**：DDP 多卡；梯度累积（有效 batch）；cosine LR 衰减；从干净/一致数据训。

**GT 来源（两个 regime）**：① **sim**——用精确物体位姿算刚性轨迹（只 sim 有）；② **video**——Pi3 逐帧深度 + CoTracker 跟踪，**pose-free**（走向真实视频，现实无位姿）。

**目标指标**：held 上 **dcos→1（方向）、magR→1（幅度）** + 深度符号一致 + **泛化**（held≈train、迁移到新任务/新相机/新域）。

---

## 1. 重大纠错（方法学教训，最重要）

**"0.81 方向突破"是过拟合假象。** 之前记录的 img_loss held image-dcos 0.81 经复评（同 ckpt/数据/代码）**不可复现，真值是 train 集的数字（0.71）**，held 实际只有 **0.19**。
- 真实状态被纠正为：**幅度（magnitude）确实被 img_loss 治好了（magR~1.0）、深度符号也学到了（92–97%）；但方向（direction）才是当时真正没解决的问题。**
- **教训：必须区分 train / held，必须确定性复现，必须视觉验证。** 此后所有结论都按这个标准重做。

**数据隐藏 bug：相机不一致毒化训练。** 数据扩充（trans_v2new）里 PullCube-v1 的默认相机在工作区**对侧**（campos `[-0.5,0,0.25]` vs 规范 `[0.3,0,0.6]`），同一世界运动投影**翻转** → 干净模型在它上面 image-dcos **−0.53（反相关）**，混训把 held dcos 推成负的。→ 修复：生成器强制所有 env 用同一相机。

---

## 2. 三条线的方法与结论

### A. sim 定相机方向泛化 —— ✅ 0.92

**方法（三味药）：**
1. **img_loss（用户洞察）**：监督**归一化 2D 图像位移** `(Δu/W, Δv/H)` + 归一化深度变化 `Δlog z`，而非 3D 米。尺度一致的目标治好了幅度欠预测。
2. **干净数据 trans_v3**：440 train，全部**固定相机**（剔除相机不一致的 PullCube），把过拟合 gap 关上。
3. **稳定化**：`w_motion`（mover 加权，堵"预测零运动"的幅度坍缩）+ 梯度累积（有效 batch 8）+ **cosine LR 衰减**（压 batch 震荡）。
4. **v2 架构（用户拍板的融合）**：grounding = **门**（`gate=σ(relevance)`，调制运动头，非并行 head）；JEPA = 从**运动派生**（`jepa_head(h.detach())`，stop-grad 不抢主干）预测进**原始预训练 DINOv2 latent**（footprint 池化适配非均匀 patch）。

**结论（@1500，全部确定性复现 + 视觉验证）：**
- heldseed（同任务异种子）：**3D dcos 0.92 / magR 0.87 / image dcos 0.95**，**held ≈ train（无过拟合 gap）**。
- heldtask（**从没训过的 StackCube**）：**3D dcos 0.91**（方向迁移到新任务），magR 0.66（幅度保守）。
- **v2-fuse > img-only**，泛化上尤甚：heldtask 0.91 vs 0.75（JEPA/门有独立增益，不是搭便车）。
- 视觉验证：深度着色 2D 高斯 token 的 GT/PRED 流向重合。
- DDP 多卡正常（之前"DDP 更差"也是同一个 train/held 混淆造成的假象）。

### B. 相机变化下的泛化 —— ✅ 0.87（内插）

**方法：相机条件化 `--cam_cond`**（都 zero-init → 可从 A 的 0.92 热启动）：
- `cam_head`：全局位姿（campos + look/up + fov，13 维）→ 加进 cond，每个 DiT block 看见视角。
- `cam_tok_head`：每 token 的**相机系坐标** → 加进 hidden，把视角相关的 RGB 特征对齐到几何。
- 生成器 `--rand_cam`：半球采样相机，留出方位角带做新视角测试。

**结论：**
- **外推**（留出连续方位角弧 = 完全没见过的视角区）：cam_cond **中性**（0.56 vs 无 cam_cond 0.61）。原因：模型已拿到 world 系 token 坐标（几何本就相机无关），瓶颈是新视角 RGB 外观漂移，位姿条件化治不了。
- **内插**（Bv2，留出的相机来自训练分布 = 真实"变相机"场景）：**新相机泛化 0.87**（cam_cond 0.87 vs 无 0.85，小而稳的边际增益，主要帮幅度）。
- **净结论：在变相机上训练 → 变相机是泛化资产、不是毒药；§97 的"中毒"是 PullCube 极端对侧视角 + 定/变混训，不是视角变化本身。**

### C. 从视频学（无位姿 GT）—— 🔬 viable ~0.45，瓶颈已定位

**动机（用户重定向）**："从 video 拿 GT vs 模拟器直接拿位姿是两个世界，不能直接比较；现实没位姿，从 video 学才重要。"

**方法 —— `robotwin_video_gt.py`（pose-free 逐点 GT）：**
- **Pi3**（3D 基础模型，给逐帧深度/点图，无需内参）+ **CoTracker**（稠密网格 2D 跟踪）→ 逐点 3D 轨迹，**无 sim 位姿、无掩码**。产标准 clip dict，trainer 直接吃。
- 关键性质：img_loss 目标是**图像归一化流 + Δlog z，尺度不变** → Pi3 的仿射尺度不确定性不进损失。

**★ 核心诊断 + 修复（video-GT 的真病根）**：RoboTwin 相机**真值完全静止（0.000m）**，但 **Pi3（SfM，假设静态场景）把运动物体误判成相机自运动（估成动了 0.30 + 9.8°）** → 全局 gauge 到处是伪运动（median flow 59px）。**修复（静相机）：用逐帧 local 点图 + 固定相机帧（viewmat=I），不信 Pi3 的位姿** → median flow 59→11.6px，运动相干落在真实物体上。

**Path 1（raw 逐点）vs Path 2（rigid/合成位姿，`robotwin_rigidify.py` 运动聚类 + 逐物体 trimmed-Kabsch）：**
- 两者都 ~**0.45 dcos**、幅度尚可；**rigid 略偏方向、raw 略偏幅度，无大幅抬顶**。
- 封顶 ~0.45 **既非逐点噪声**（rigid 去噪没救）**也非过拟合**（held≈train）→ 是**数据量 + video regime 更难**（sim 当年 183→440 clip 才把 held 0.10→0.92）。
- **数据洞察**：这些任务**物体运动小**（块几乎不动、大运动是机械臂、臂从画外进 + 中途遮挡）→ 可学信号弱。**任务选择很关键**。

**FOV 多样化 + 域泛化测试（§101，用户思路）：**
- 加了 **fov67/fov73** 宽视角相机（vs D435 37°）；建了 demo_clean / demo_random 的 FOV 配置。
- 装好生成器侧 venv（mplib 0.2.1，不碰训练 venv）；**任务可靠性张力**：beat_block_hammer 规划可靠但运动弱，handover/move_can_pot 运动好但卡 mplib 规划器（start-state collision）。
- **结果（train on clean → 双测）**：clean-held **image dcos 0.66 / magR 1.8**（同域行），**demo_random 0.26 / magR 4–5×**（换杂乱随机背景就方向乱、幅度爆，且**越训越差** = 典型过拟合）。
- **结论：只在 clean 上训 → 过拟合 clean 背景外观，没学到域不变特征；FOV 多样性救不了，缺的是训练里的背景/域多样性。**

---

## 3. 代码与基础设施产出

**模型 `code/igsw/gpstoken_wm/wm_model.py`：**
- `img_loss` 分支（归一化图像流 + Δlogz 深度）。
- `fuse` 路径：门调制运动 + `jepa_head(h.detach())` 进原始 DINOv2 latent（`_footprint_sample` footprint 池化）。
- `cam_cond`：`cam_head` / `cam_tok_head` / `cam_cond_signals`（zero-init 可热启动）。

**训练器 `train_gpstoken_wm.py`**：新增 `--img_loss --w_depth --fuse --cam_cond --init_from（热启动）--w_motion --accum --lr_min_frac（cosine）`；DDP 多卡。

**评估 `_gps_imgeval.py`**：image + 3D 的 dcos / magR、深度 Δlogz 符号一致性；fuse / cam_cond 感知；`--mov_pct`（gauge 无关的相对 mover 阈值）。

**视频 GT**：`robotwin_video_gt.py`（Pi3+CoTracker pose-free GT + vis 过滤 + 静相机修复）、`robotwin_rigidify.py`（Path-2 逐物体 Kabsch 去噪）。

**sim 数据**：`maniskill_gt.py`（强制固定相机 look_at + `cam_eye` 参数）、`gen_sim_dataset.py`（`--rand_cam` 半球采样 + heldcam 划分）；`task_config/_camera_config.yml` 加 fov67/fov73。

**可视化**：`_gps_tokenviz.py`（深度着色 token 三栏）、`_gps_overlayviz.py`（GT 绿 / PRED 红 叠加箭头）。

**基础设施**：被一次 5.5T 磁盘清理误删的 Pi3 / CoTracker / SAM2 权重经代理重下；RoboTwin 生成侧 venv `/mnt/pfs/xuhaoming/xr-2/robotwin_gen_venv`（mplib 0.2.1 + sapien 3.0.0b1 + numpy<2）。

---

## 4. 关键结论

1. **sim 定相机方向泛化已解决**（0.92，泛化到新任务，确定性 + 视觉双验证）—— img_loss（归一化图像运动）+ 干净一致数据 + 稳定化 + v2 融合架构。
2. **相机变化是泛化资产**（在变相机上训练后新相机内插 0.87）；cam_cond 边际有用；新视角**外推**仍难（瓶颈是视觉外观、非位姿）。
3. **从视频学是可行的**（pose-free，~0.45，泛化 held≈train），但上限低于 sim，受 GT 噪声 + 任务运动强度限制；**Pi3 把物体运动误判成相机运动**是静相机 video-GT 的核心坑，已修。
4. **域泛化**：只在 clean 上训会过拟合 clean 外观 → **需要训练集本身有域多样性**（随机背景），FOV 多样性不够。
5. **方法学**：先验证指标再下结论（0.81 过拟合的教训）；优先用确定性 + 视觉验证。

---

## 5. 当前状态 / 下一步候选

- **当前在跑/可复用**：sim 那条 0.92 的线（A）；video pose-free 管线（C）；FOV + 生成器 + planner 环境都就绪。
- **指向的修法**：① 训练集混入 demo_random / domain-randomization（治域过拟合）；② 找"高物体运动 + 规划可靠"的任务（治弱信号 + 规划器卡死）；③ 真实视频（终极目标）。
- **未尽**：DINOv2 对非均匀 patch 的对齐（更厚 adapter）；novel-view 外推的视觉杠杆。

> 详细流水日志见 `agent.md` §95–§101。

---

# ◆ 第二周（§102–§103）：GT 生产 Pi3+CoTracker → SpaTrackerV2

> **一句话**：上周把 video 路天花板（~0.45）定位为 **GT 噪声**；本周换 tracker 重做 GT，全量 40 任务验证：**heldseed 方向 0.82 vs Pi3 0.51（GT 质量问题解决，定 SpaTracker 为正式 GT）；但 heldtask（全新任务）两个 GT 都崩 → 真正的瓶颈是零样本任务泛化，与 GT 无关。**

## 1. 诊断：Pi3 的"运动"大多是深度噪声
- 用户拉图判断 SpaTracker vs Pi3 的 GT-flow（6 任务对比图），SpaTracker 在遮挡（handover）上完胜、其余相当。
- **量化诊断**（同一 handover clip）：Pi3 "100% token 都在动"但 2D 流仅 1.5px —— 它**没跟住积木**，所谓"运动"是**深度噪声**；SpaTracker 仅 18 个 mover 但 2D 流 208px —— **干净抓住真实运动**。这解释了 Pi3 为何卡 ~0.45：方向被噪声污染。

## 2. SpaTracker GT producer（`robotwin_spatrack_clip.py`）
- SpaTrackerV2 = 联合 2D+3D 点跟踪、动态场景感知。
- **`fixed_cam=True`**：RoboTwin 相机固定 → c2w=I 全帧（`robotwin_spt_probe.py` 验证）→ track3d 运动 = **纯物体运动**（无 Pi3 的自运动误判）。
- **`traj = unproject(track2d, track3d 深度)`，viewmat=I** → `project_to_uv(traj)` 精确重现 track2d（**重投影 0.000px**），同时用 SpaTracker 的 3D 深度。输出格式与训练器完全兼容；batch + shard + resumable（读 .pt 或 hdf5）。

## 3. 指标坑 + 幅度修复
- **指标坑**：SpaTracker 的 mover **稀疏**（每 clip 仅 2–18 个真动点）。旧 `--mov_pct`（按位移取 top-%）会混入静止点**稀释** v2，误判它更差（0.45）。新增 **`--gt_flow_thr`**（按 GT 图像流幅度选 mover，跨 GT 公平）才看出真实差距（0.83）。
- **幅度修复**：稀疏大 mover 下模型回归静止 → 幅度欠预测。新增 **`--mw_cap`**（mover 上权重上限可调，原硬编码 10）。6 任务 sweep：**w_motion=30 / mw_cap=80** 把幅度 0.68→**0.90**，方向守住 0.93。

## 4. 全量对照（rtvid_multi，40 任务 / 200 train，同 config，唯一变量 = GT）

| 指标（公平 `gt_flow_thr 0.05`） | heldseed v1 Pi3 | heldseed **v2 SpaTracker** | heldtask v1 | heldtask v2 |
|---|---|---|---|---|
| 图像方向 dir-cos | 0.51 | **0.82** | −0.07 | 0.24 |
| 3D 方向 dir-cos | 0.75 | **0.85** | 0.39 | 0.19 |
| 幅度 mag-ratio | 0.30 | **0.53** | 0.11 | 0.04 |

## 5. 结论 / 下一步
1. **SpaTracker = 正式 GT**：heldseed 全面胜出，全量规模验证通过 —— GT 质量问题解决。
2. **heldtask（新任务）两 GT 都崩**（方向 ~0/随机，幅度 ~0）= **零样本任务泛化是下一个硬骨头，非 GT 问题**（两个都崩）。这是项目核心难关。
3. 幅度全量上 0.53（< 6 任务 sweep 的 0.90，多样性更难）可再调；但方向 0.82 是关键、已达标。

> 复现：producer `robotwin_spatrack_clip.py --pt_glob ... --out_dir ...`；训练 `train_gpstoken_wm.py --img_loss 1 --fuse 1 --w_motion 30 --mw_cap 80`；评估 `_gps_imgeval.py --gt_flow_thr 0.05 --split heldseed|heldtask`。ckpt：`checkpoints/wm_{mv1,mv2}/wm_002000.pt`。

---

# ◆ 第三周（§104）：曲线证伪 + 真实世界（AgiBot）数据管线与训练

> **一句话**：曲线拟合被证伪（GT 偏离是抖动非真曲线，方向反而更差）→ 保持直线目标；并首次把 SpaTracker GT 管线推到**真实世界 AgiBot**——管线就绪（需相机门控），但真机运动学习明显更难（train 0.69 / heldseed 0.27，远低于 sim 0.95/0.82），模型欠拟合复杂真机任务。两项均由并行子 agent 完成。

## 1. 曲线拟合 → 不值得做（任务1，证伪）
- 用户问：3D 环境下把直线净位移改成**完整曲线轨迹**会不会更好。
- 先量曲率（`_traj_curvature.py`）：GT 轨迹偏离直线弦 ~30%（垂距 47px / 弦 155px），但 **quad-frac 0.06** → 偏离几乎全是**逐帧跟踪抖动**，不是可拟合的平滑弯。
- 实测（`--traj_pred` 逐帧多-waypoint vs 直线 `wm_mv2`）：曲线方向**更差**（train 0.78 vs 0.95，held 0.70 vs 0.82）。（其幅度指标 0.04 是 `_gps_imgeval` 曲线分支 eval-bug，与训练日志 0.61 矛盾，已弃用——train-split 复核救了一次误报。）
- **结论：保持直线两关键帧目标。** 真机若要曲线，需先平滑去噪 GT。

## 2. 真实世界 AgiBot（任务2，= "AIGC Pro"）
**2a. 管线评估**：SpaTracker GT 管线在真机视频上能跑，但 sim 的 `fixed_cam=True` 只对**静相机** episode 成立（~2/3）。弯腰任务（洗衣机/冰箱/抽屉/扫地）动头部相机 → **重现 Pi3 自运动误判**（全帧箭头、reproj 5–8px）。相机运动**双峰**、有干净分界 → 门控：`fixed_cam=False` 读 c2w，平移 <2% 深度 且 mover<50% 才留。工具 `agibot_spatrack_eval.py` / `agibot_montage.py`。

**2b. 静相机子集训练+测试**（用户原则：删弯腰 task、只用固定相机）：门控筛出 **301 clip（220 train / 48 heldseed / 33 heldtask）**，`robotwin_spatrack_clip.py --agibot`，真机 GT 目视干净。训练 `wm_agibot`（2500 步，直线 w30c80 config）：

| 指标（`gt_flow_thr 0.05`） | train | heldseed | heldtask | （对比 sim） |
|---|---|---|---|---|
| 图像方向 dir-cos | 0.69 | 0.27 | 0.42 | sim 0.95 / 0.82 |
| 幅度 mag-ratio | 0.54 | 0.32 | 0.24 | — |

**诊断**（`_gps_predviz` 绿 GT / 红 pred）：GT 是真实相干运动（非噪声），但模型方向跑偏，连 train 都只 0.69 = **欠拟合**。真机任务（仓库分拣=多物体、双臂、快）远比 RoboTwin 单物体抓放复杂。

## 3. 结论 / 下一步
1. **数据层面真机可用**（静相机门控后 GT 干净），但**真机运动学习是下个硬骨头**（任务难度 + 数据规模），非 GT 问题。
2. 候选：扩静相机真机数据 + 加步数/容量；或从更简单真机任务起步；复查快速/形变运动的 GT 噪声。
3. **新方向（讨论中）**：本世界模型作为 **VLA backbone** —— feature 接 400–600M DiT 动作头，3D-flow 时间间隔与 action chunk 同步，先在 RoboTwin 上验证（注意 normalize + 关节角处理）。

> 复现：`agibot_spatrack_eval.py`（门控）/ `robotwin_spatrack_clip.py --agibot`（出 clip）/ `train_gpstoken_wm.py ... --data data/agibot_static` / `_gps_imgeval.py --gt_flow_thr 0.05`。ckpt `checkpoints/wm_agibot/wm_002500.pt`。
