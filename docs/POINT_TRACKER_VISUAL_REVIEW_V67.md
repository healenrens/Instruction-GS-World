# CoTracker 原始可视化评估

## 本轮只回答的问题

同一个可见表面点是否被持续追踪，遮挡时的 visibility 是否合理，重现时是否回到同一个点；相同物理时间范围内，连续帧和训练抽帧是否产生不同结果。本轮不训练模型，不把点 ID 当 object ID，不由运动相似度生成 object GT。

本地实现：`/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/review_point_tracker_v67.py`。

服务器前台入口：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/review_point_tracker_v67.sh`。

默认输入：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/multisource_real_robot_video_v53/index.json`。

默认权重：`/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/cotracker/scaled_offline.pth`，CoTracker3 offline。不下载模型，不加载 DINO、SigLIP 或历史 world-model checkpoint。默认只使用一个可见 GPU，没有 torchrun、DDP 或后台进程。联网只用于最后的 W&B 上传。

## 2026-09-20 更新：长片段与变化区域密集采点

旧版八帧、全图 jitter queries 只适合初步检查，不能覆盖完整物体变化。新版不改训练，只替换这一独立 review 的取样方式。

- 默认六源各最多 5 个 held-group clips；按 held group、episode 确定性轮换，不按 tracker 成功率选例。不足时报告实际数量，不重复补齐；这些探索样本不用于总体准确率估计。
- 每段首尾时间差至少 **10 秒**。30 Hz 时 native 输入为 **301 帧**；默认 400 ms 分支为 **26 帧**，而非原来的 8 帧。`CLIP_SECONDS` 可以增大，低于 10 时仍按 10 秒选样。短 episode 不 padding、不拼接。
- 排除腕部相机；视频路径上的 head/high/exterior/front/top 可以进入，DROID wrist 回退不能进入。Bridge 的 image_0 按其 source convention 处理；RobotWin RGB cache 按既有 cam_high builder contract 处理；未知相机不猜测为外部相机，排除数量见 selection report。
- 不重建 index。复用原 decoder 和 HY frame offset，但不再调用会跨 source 替换样本的训练 sampler。缺失 payload 由现有 index loader 跳过并计数；选中视频解码失败会暴露原路径，不悄悄换样本。

### Mask 与采点方法

1. 默认在片段的 0、2、4、6、8、10 秒附近生成提案；每个提案与前后约 0.2 秒的图像比较。图像按 decoder 返回尺寸处理，不预先缩小。
2. 使用 [OpenCV Farneback dense optical flow](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)，再用 [robust partial-affine fitting](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) 估计全局相机运动。扣除该运动后，用残差位移分割 **motion-region masks**。拟合用的规则网格不送给 CoTracker。
3. 阈值为原图残差位移 `max(0.75 px, median + 3 × 1.4826 × MAD)`；MAD 是残差与中位数之差的绝对值的中位数。3×3 closing 填小孔，保留至少 9 像素的连通区域。这些是可视化提案参数，不是经过校准的 object 标准；每例保存实际阈值、面积和相机拟合参数。
4. 默认每段总预算 **1024 点**，在非空提案时刻之间分配。每个选中连通区域先给一个点，剩余预算按面积平方根分配；区域内做确定性 farthest-point coverage。没有全图均匀/random queries。极端多碎片时预算可能不足，完整 component map 仍保存。
5. 每个点保留自己的原视频 query frame 与像素坐标。native/抽帧分支用完全相同的查询；会补入 query 帧和最终帧，因此某些来源不是严格等间距，实际 frame IDs 与间隔逐项保存，视频用原时间戳播放。
6. 每次 CoTracker 最多处理 **256 query points**，每批都看完整十秒，未在时间上切短片段。不同批次的联合 attention 不共享，官方 support queries 仍存在；预算与 points-per-pass 记录在 metadata 中。

**Mask 是变化提案，不是 SAM/实例分割，也不作为 object GT 或训练 loss。** 它可能漏掉暂时静止/低纹理的小物体，或选到机械臂、阴影、光照、相机视差；主导前景运动也可能污染相机拟合。相机拟合无结果的配对只记为 unavailable，不用未补偿全图运动替代。空 mask 仍保存原视频与可视化，不回退到全图网格，也不从报告中删掉。

这一步利用整段视频和不同时间的变化选 query，只是 **offline teacher review**，不能作为部署 student 的因果输入。原有训练采样和 checkpoint 不变。

### 比较与展示

- 原 RGB、native 轨迹、400 ms 轨迹、同时间戳左右对比、每个提案时刻的 mask+点位图、独立二值 mask 和 component map 都保留。
- 不插值生成抽帧分支不存在的轨迹；各分支共享起止时间，不再通过减少总时长控制计算量。
- 不额外过滤 tracker visibility，in-bounds 独立记录。每个点自己的 query 帧是给定值，从差异统计排除；visibility 不是 existence 或经过校准的置信概率。
- 固定坐标 crop 从采点区域选择，不追随轨迹；crop 内点过密时不叠加全部数字，完整 point ID 可从 CSV/PT 查看。
- decoder 输入不额外降采样不等于权重在原生分辨率推理：官方 CoTracker internal resize 仍存在，并明确写入 metadata。历史 RGB cache 已经发生的预处理也不会被本工具恢复。

## 结果入口

默认运行名称为 `tracker_motion_review_v67_10s_seed17_<revision前7位>`。

结果目录：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/<运行名称>/`。

| 文件 | 内容 |
|---|---|
| `index.html` | 离线视频浏览、source 过滤、人工问题记录、manual query 点选 |
| `review_bundle.zip` | 可下载的完整浏览包，解压后打开 `index.html` |
| `review.log` | 前台运行日志，含每个 case 的 decode/inference/render 进度 |
| `cases.json` | 实际样本选择、相机证据、首尾帧、短片段/腕部/未知相机排除计数 |
| `summary.json` | 运行配置、实际模型类和内部分辨率、每一组结果 |
| `wandb_run.json` | 成功上传后的 W&B run URL |
| `<case>/source.mp4` | 原始 RGB 连续片段，无轨迹叠加 |
| `<case>/sampling.json` | 各时刻 mask 阈值、相机运动拟合、连通区域面积、采点数量 |
| `<case>/queries.pt` | 每个 query 的像素坐标、原视频帧和 proposal component 标签 |
| `<case>/motion_masks/frame_<frame>_queries.png` | 原图上的橙色变化 mask 与青色实际查询点 |
| `<case>/motion_masks/frame_<frame>_mask.png` | 原尺寸二值提案 mask |
| `<case>/motion_masks/frame_<frame>_components.npz` | 无损压缩的原尺寸连通区域编号，非 object ID |
| `<case>/step_<ms>ms/native.mp4` | 连续帧推理结果和局部放大 |
| `<case>/step_<ms>ms/sampled.mp4` | 覆盖完整十秒的抽帧推理和局部放大 |
| `<case>/step_<ms>ms/comparison.mp4` | 同时间戳左右对比 |
| `<case>/step_<ms>ms/tracks.pt` | 未过滤轨迹、visibility、in-bounds、frame IDs 和查询点 |
| `<case>/step_<ms>ms/point_rows.csv` | 每个共同时间戳、每个点的两个预测和差异 |

W&B 项目：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`。
Group：`point-tracker-visual-review-v67`；视频表：`tracker_review/cases`，mask 表：`tracker_review/motion_masks`；Artifacts 中的 `tracker-review` 包含完整 ZIP。

native/sampled 距离是两次预测之间的分歧，不是相对 GT 的 tracking error。两者一致仍可能一起追错；不输出 accuracy 或 promote/reject。

## 同步、部署与运行

同步阶段允许访问 GitHub；运行阶段不读取 Git、也不下载依赖。使用交付回复中的完整 commit 运行，不需要旧的 GATE_REPORT。

同步和部署：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git fetch origin '+refs/heads/codex/tracker-visual-review-v67:refs/remotes/origin/codex/tracker-visual-review-v67' &&
git switch --detach refs/remotes/origin/codex/tracker-visual-review-v67 &&
export ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source &&
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world &&
export SOURCE_REVISION="$(git rev-parse HEAD)" &&
bash /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/deploy_continuous_predictive_object_field_v67_runtime.sh
```

前台运行（将 `SOURCE_REVISION` 设置成交付 commit；其余不依赖同步 shell 的环境）：

```bash
export SOURCE_REVISION=<交付的完整commit>
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export ROOT="${RUNTIME_ROOT}/runtime/continuous_predictive_object_field_v67/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_INDEX="${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json"
export TRACKER_CHECKPOINT="${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth"
export TRACKER_VERSION=3 CUDA_VISIBLE_DEVICES=0
export RUN_NAME="tracker_motion_review_v67_10s_seed17_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}"
export CASES_PER_SOURCE=5 CLIP_SECONDS=10 TEMPORAL_STEP_MS=400
export POINT_BUDGET=1024 POINTS_PER_PASS=256 QUERY_EVERY_SECONDS=2
export MOTION_PAIR_SECONDS=0.2 MOTION_MIN_PX=0.75 MASK_MIN_AREA=9
export SEED=17 HELD_GROUP_STRIDE=20 DISPLAY_WIDTH=640 REUSE_COMPLETED=1
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
export WANDB_NAME="${RUN_NAME}" REVIEW_STAGE=run
unset QUERIES_JSON WANDB_RUN_ID WANDB_RESUME
cd "${ROOT}"
bash "${ROOT}/code/scripts/review_point_tracker_v67.sh"
echo "REVIEW_RC=$?"
```

同一配置原样重跑会复用已完成的 mask/queries/预测结果；中途退出不会清理产物。预测已保存但视频渲染中断时，重用预测重新渲染。修改查询点、模型配置或 revision 会重新计算相应结果；不是严格训练 checkpoint resume。需要全部重算时设置 `REUSE_COMPLETED=0`。`TEMPORAL_STEP_MS=100,200,400` 仍支持三组比较，各组都看同一个十秒片段。

若视频已生成但 W&B 上传失败，复用上面的完整环境、仅将 `REVIEW_STAGE=upload` 后再次执行同一 shell。它只读已有 summary/媒体，不加载 tracker、不跑 GPU 推理。结果 ZIP 在 W&B 上传之前就已落盘。

## 人工多点复查

1. 下载 Artifacts 中的 ZIP，解压后打开离线 `index.html`。每例展开 `Select points for a manual rerun`。
2. 在原生 anchor 图像上点击多个点，分别覆盖目标表面、边缘、机械手与邻近背景。`Point group` 只是人工记录标签，不送入 tracker 或 grouping loss。
3. `Export manual queries` 导出归一化坐标和原 case 元数据。把文件放到服务器 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/manual_queries.json`。
4. 复用上面的完整前台环境，设 `QUERIES_JSON=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/manual_queries.json`，将 `RUN_NAME`、`WANDB_NAME`、`OUT` 改成独立的 `_manual` 运行后执行同一入口。只处理真正点选过的 cases；不需要重新选择视频。

浏览页新增 Region selection 的人工判断：移动物体覆盖、小物体漏选、主要是机械臂、相机运动、阴影/噪声、空 mask。先判断有没有选对区域，再记录哪个视频、哪一帧、哪个 point 漂移或 visibility 错误。人工点选只复用新版十秒、非腕部 cases，旧版八帧短片段 manual JSON 不作为新版输入。

人工问题记录通过 `Export observations` 导出。若点轨迹可靠而 grouping 不成立，应归因于 object 推导问题；若只有抽帧版本失效，先修采样；不能由视觉上好看的少数 case 声称所有 teacher targets 正确。本地只做静态检查；本版真实长片段推理与 mask 质量仍待用户在服务器运行、观看。

## 仅重绘明显移动点与完整轨迹

新增 `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/render_moving_tracker_review_v67.sh`。
它在 CPU 上读已完成 review 的 `source.mp4` 和 `tracks.pt`，不跑 CoTracker、不重新采点、不加载训练模型或权重。
原输入目录不修改，新目录保留所有原始轨迹和 masks，视频仅显示筛选后的点。中断后同参数重跑会复用已完成 case/分支。

筛选定义：排除该点自身给定的 query 帧；仅取 visible、in-bounds、有限坐标，至少 6 帧。分别计算 x/y 位置的
5% 和 95% 分位值，两轴跨度形成的包围盒对角线至少达到 `max(12 px, 短边的 2%)` 才显示。它不是首尾净位移，
因此往返运动不因回到原点而被排除；也不是累计路程，避免静止点的细微抖动累加成大移动。每点数值与选择写入
各分支的 `motion_filter.json`。例如短边 480px 时阈值为 12px，短边 1080px 时为 21.6px。

同一组 native-selected point IDs 用于 native/抽帧对照，颜色保留原始 ID。默认 `TRAILS_SECONDS=0` 表示从片段
开始至当前帧的完整轨迹；遮挡/越界时断线。`trajectories.png` 是最后一帧背景上的完整轨迹图。改成正数可缩短
可视化 trail，但不改变移动筛选和 raw tracks。旧 0.25s trail 对 400ms 抽帧几乎不连线的问题由此消除。

**这不是 mask 边界修复。** 宽区域内未明显移动的点会在主视频中消失，但相机运动/漂移仍可能通过，微小真实运动
也可能被隐藏。原橙色 proposal 图折叠在对照区，人工反馈可选 `oversized_region_background_spill`。归档 RGB 经
H264 压缩，只用于重绘背景；不计算像素 GT error。原 CSV 和 sampling consistency 仍属于全部原始点。

先按前面的同步步骤部署交付 commit。下面是独立的前台 CPU 重绘命令（无需 GPU）：

```bash
export SOURCE_REVISION=<交付的完整commit>
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export ROOT="${RUNTIME_ROOT}/runtime/continuous_predictive_object_field_v67/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export INPUT_REVIEW="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/tracker_motion_review_v67_10s_seed17_5d82a59"
export RUN_NAME="tracker_motion_review_v67_moving_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}"
export MINIMUM_MOTION_PIXELS=12 MINIMUM_MOTION_FRACTION=0.02 MINIMUM_VISIBLE_FRAMES=6
export TRAILS_SECONDS=0 DISPLAY_WIDTH=640 REUSE_COMPLETED=1
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
export WANDB_NAME="${RUN_NAME}" WANDB_DIR="${RUNTIME_ROOT}/wandb"
bash "${ROOT}/code/scripts/render_moving_tracker_review_v67.sh"
echo "RENDER_RC=$?"
```

输出为 `${OUT}/index.html`、`${OUT}/review_bundle.zip`、`${OUT}/render.log`，完整根目录是
`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/tracker_motion_review_v67_moving_<revision前7位>/`。
上传失败可同命令重跑，视频复用后重新上传，不重新跑追踪。

在下载使用的 Mac 上执行，不使用另一台机器的 `/Users/hela` 作为当前用户名：

```bash
RUN_NAME=tracker_motion_review_v67_moving_<交付commit前7位>
LOCAL_DIR="${HOME}/Downloads/${RUN_NAME}"
mkdir -p "${LOCAL_DIR}" &&
scp -P 8600 "root@10.66.0.39:/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/${RUN_NAME}/review_bundle.zip" "${LOCAL_DIR}/review_bundle.zip" &&
unzip -o "${LOCAL_DIR}/review_bundle.zip" -d "${LOCAL_DIR}" &&
open "${LOCAL_DIR}/index.html"
```

## Grounded-SAM-2：物体运动为目标，机械臂为上下文

本节是独立新入口，不覆盖前面的 moving baseline。实现位置：

- `/Users/hela/Instruct-GS-World-recovered-20260725/code/igsw/adaptive_gaussian_wm/grounded_tracker_masks_v67.py`：GroundingDINO 机械臂 boxes、SAM2 query-frame masks、crop、去重、角色证据。
- `/Users/hela/Instruct-GS-World-recovered-20260725/code/igsw/adaptive_gaussian_wm/grounded_tracker_sampling_v67.py`：区域预算、mask 内部采点、采点图和查询记录。
- `/Users/hela/Instruct-GS-World-recovered-20260725/code/igsw/adaptive_gaussian_wm/grounded_tracker_export_v67.py`：运动筛选、角色颜色及同点集训练候选导出。
- `/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/review_grounded_object_tracker_v67.py`：完整流程与中断复用。

### 具体产生什么数据

```text
原生非腕部 RGB 连续片段，至少 10 秒
 -> 每 2 秒一张 query frame
 -> GroundingDINO 机械臂/夹爪框 -> SAM2 mask，角色 robot_context
 -> SAM2 全图+重叠 crop 候选区域 -> object_candidate / unknown / scene_context
 -> 区域内采样，物体优先，保留机械臂点
 -> CoTracker 原生连续帧 + 400 ms 对照，各自追踪相同 queries
 -> 原生轨迹逐点运动筛选
 -> 同一组 IDs 的主视频 + 完整 raw/all-query 对照 + training_candidates.pt
```

这里的 Grounded-SAM-2 是 GroundingDINO 与 SAM2 的组合，使用已公开的
[HF GroundingDINO 接口](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) 和
[HF SAM2 接口](https://huggingface.co/docs/transformers/en/model_doc/sam2)，不另外引入训练模型。
推理要求既有环境提供 `Sam2Model`、`Sam2Processor` 和 `AutoModelForZeroShotObjectDetection`；脚本不会安装或升级环境。

机械臂文本是 `robot arm. robot gripper. robot hand.`。其他候选通过 SAM2 自动提示产生，不要求列出杯子/积木等物体类别。
默认每张 query frame：全图与 2x2 重叠 crop 各有 12x12 分割提示点，一次 16 prompts，复用当前图像 embedding；
这些网格点不是 tracker queries。SAM 选择每个提示评分最高的 mask，再按 SAM score>=0.7、logit stability>=0.9、
原图面积>=8 px 选 proposal。crop 边界截断的 mask 不当新物体；mask IoU>=0.8 去重，最多 48 个 object candidates
和 8 个其他候选，机械臂 mask 单独保留。以上是 proposal 参数，不是已证明正确的 object 判据。

与机械臂 mask 重叠占自身面积超过 10% 的候选先作为 unknown 保留；与机械臂几乎重复的 mask 由 robot_context 覆盖。
占整帧超过 40% 的其他大区域作为 scene_context，这是面积启发式，不是已确认的背景。其余标 object_candidate，
不等于非机械臂分类必然正确。未检测到机械臂会明确记录 `not_detected_not_proven_absent`。

2048 点预算按 object/robot/other=80%/15%/5% 分配；空上下文预算还给 object。区域轮流获得最多四个基础点，
剩余按面积平方根分配，单区域最多 96 点。采点在 mask 内部进行 farthest-point 覆盖，优先远离边界。
这是 mask 内的密集追踪，不是 dense optical flow，也不能保证 SAM 没发现的小物体被追踪。

运动筛选采用原生轨迹、排除给定 query 帧，只使用 visibility=true、in-bounds、有限坐标的至少六帧。
5%-95% x/y 范围的对角线超过 `max(1.5 px, 0.08 * 区域对角线, 3 * 二阶差分尺度)` 时为 moving。
二阶差分尺度是连续三个有效位置的二阶差分范数中位数除以 sqrt(6)，用于描述高频变化，不是 tracker confidence。
object_candidate 且 moving 才进入物体运动目标；robot、unknown、scene 都保留为上下文。
没有移动的 object candidates 不进主视频，但仍在全点视频与 raw 数据。这里没有声称消除了相机运动或阴影。

**数据与展示一致性**：主视频展示 `object_motion_target_mask | context_mask`，同一组 IDs 用于原生/抽帧两个分支。
导出数据也保留这两个 mask、所有 raw tracks、tracker visibility、in-bounds、query frame、区域来源和逐点阈值。
将来训练可用前者构造 motion supervision，后者保留交互/遮挡上下文；本次没有改动 world-model 训练或 loss。
完整视频用于分割和筛选，因此这是离线 teacher 数据，不能把未来采点、未来 visibility 或角色选择泄露给部署 student。

### 第一步：同步与部署

以下命令仅在可访问 GitHub 的同步环境执行；不删除工作区文件、不覆盖原实验产物。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source &&
git fetch origin '+refs/heads/codex/grounded-object-tracker-v67:refs/remotes/origin/codex/grounded-object-tracker-v67' &&
git switch --detach refs/remotes/origin/codex/grounded-object-tracker-v67 &&
env ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source \
  RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world \
  SOURCE_REVISION="$(git rev-parse HEAD)" \
  bash /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/deploy_continuous_predictive_object_field_v67_runtime.sh
```

### 第二步：仅准备新增权重

这一步允许下载；已有文件由 Hugging Face 下载工具复用。只下载权重及 processor/tokenizer 文件，不重装环境。
将下面两处运行命令中的 `SOURCE_REVISION` 都换成交付的完整 commit。

```bash
export SOURCE_REVISION=<交付的完整commit>
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export ROOT="${RUNTIME_ROOT}/runtime/continuous_predictive_object_field_v67/releases/${SOURCE_REVISION}"
"${RUNTIME_ROOT}/.venv/bin/python" "${ROOT}/code/scripts/prepare_grounded_tracker_models_v67.py" \
  --models_root "${RUNTIME_ROOT}/models"
```

下载目录分别是 `/mnt/pfs/public/xuhaoming/instruct_gs_world/models/grounding-dino-base` 和
`/mnt/pfs/public/xuhaoming/instruct_gs_world/models/sam2.1-hiera-large`。已有 CoTracker 使用原路径，不重新下载。

### 第三步：单卡前台重新采样、追踪与可视化

独立环境，不依赖前两条命令的 shell；运行阶段不访问 Git/GitHub，不联网获取模型。W&B 用于上传结果。
`CASE_MANIFEST` 指向用户已看过的十秒样本清单，保证同 case 比较，不按新 mask 的好坏重选样本。
需要重新从六源选样时设置 `CASE_MANIFEST=`，默认每源 5 个 held clips。

```bash
export SOURCE_REVISION=<交付的完整commit>
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export ROOT="${RUNTIME_ROOT}/runtime/continuous_predictive_object_field_v67/releases/${SOURCE_REVISION}"
export VENV_ROOT="${RUNTIME_ROOT}"
export DATA_INDEX="${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json"
export CASE_MANIFEST="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/tracker_motion_review_v67_10s_seed17_5d82a59/cases.json"
export TRACKER_CHECKPOINT="${RUNTIME_ROOT}/checkpoints/cotracker/scaled_offline.pth"
export TRACKER_VERSION=3 CUDA_VISIBLE_DEVICES=0
export GROUNDING_MODEL="${RUNTIME_ROOT}/models/grounding-dino-base"
export SAM_MODEL="${RUNTIME_ROOT}/models/sam2.1-hiera-large"
export RUN_NAME="grounded_object_tracker_v67_10s_seed17_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}"
export POINT_BUDGET=2048 POINTS_PER_PASS=256 QUERY_EVERY_SECONDS=2
export ROBOT_POINT_FRACTION=0.15 OTHER_CONTEXT_FRACTION=0.05
export CLIP_SECONDS=10 TEMPORAL_STEP_MS=400 SEED=17 REUSE_COMPLETED=1
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
export WANDB_NAME="${RUN_NAME}" WANDB_DIR="${RUNTIME_ROOT}/wandb" REVIEW_STAGE=run
unset WANDB_RUN_ID WANDB_RESUME
cd "${ROOT}"
bash "${ROOT}/code/scripts/review_grounded_object_tracker_v67.sh"
echo "REVIEW_RC=$?"
```

中断后同配置原样重跑，复用完成的 anchor masks、queries、native tracks 和分支结果。
变更任何采样配置使用新 `RUN_NAME`/`OUT` 保留对照；不回退成旧 flow sampling，不隐藏模型/文件错误。
ZIP 在 W&B 上传前落盘。若仅上传失败，同环境设置 `REVIEW_STAGE=upload` 重跑，不重新加载模型或做追踪。

### 结果与下载

统一结果根是 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/grounded_object_tracker_v67_10s_seed17_<commit前7位>`。

| 文件 | 用途 |
|---|---|
| `index.html` / `review_bundle.zip` / `review.log` | 本地浏览、完整下载包、前台进度日志 |
| `training_manifest.json` | 同批所有 case/时间分支的训练候选索引 |
| `<case>/grounded_masks/frame_<frame>.npz` / `.json` | 原尺寸 SAM masks、boxes/分类来源/采样配置 |
| `<case>/grounded_masks/frame_<frame>_queries.png` | 原图 mask 与采点覆盖 |
| `<case>/sampling.json` / `queries.pt` | 全区域与点预算、真实 query 坐标/时间/角色 |
| `<case>/step_400ms/comparison.mp4` / `trajectories.png` | 目标+上下文两路对照与完整轨迹 |
| `<case>/step_400ms/all_queries.mp4` | 包含被运动阈值隐藏的小幅/静止候选，排查漏点 |
| `<case>/step_400ms/motion_filter.json` | 逐点跨度、阈值、筛选依据 |
| `<case>/step_400ms/training_candidates.pt` | 全轨迹+target/context/display IDs，teacher-only，不是 GT |

`training_candidates.pt` 中 `native.tracks` 形状为 `[T,N,2]`，以原图像素为单位；`visibility/in_bounds` 为 `[T,N]`；
`queries.xy` 为 `[N,2]`，`queries.frames` 为 `[N]` 原视频帧号；三种 target/context/robot masks 为 `[N]`；
`display_point_ids` 为所显示的原始点索引。`T` 是真实连续帧数，`N` 是实际采得的点数，不保证固定2048。
region IDs 仅是 anchor-local 区域索引，不是物体身份真值。可见性仍是 CoTracker 的预测。

下载 Mac 上执行（不用其他 Mac 的用户名）：

```bash
RUN_NAME=grounded_object_tracker_v67_10s_seed17_<交付commit前7位>
LOCAL_DIR="${HOME}/Downloads/${RUN_NAME}"
mkdir -p "${LOCAL_DIR}" &&
scp -P 8600 "root@10.66.0.39:/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/${RUN_NAME}/review_bundle.zip" "${LOCAL_DIR}/review_bundle.zip" &&
unzip -o "${LOCAL_DIR}/review_bundle.zip" -d "${LOCAL_DIR}" &&
open "${LOCAL_DIR}/index.html"
```

重点逐例看：机械臂是否误包了被抓物体、远处小物体是否有 mask/queries、主视频丢失的点在 all-query 中是否真实运动、
阴影/相机运动是否被标为物体、遮挡后是否追到另一表面。新版本已完成静态检查，服务器推理和质量提升尚待运行确认。
