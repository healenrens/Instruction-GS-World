# CoTracker 原始可视化评估

## 本轮只回答的问题

同一个可见表面点是否被持续追踪，遮挡时的 visibility 是否合理，重现时是否回到同一个点；相同物理时间范围内，连续帧和训练抽帧是否产生不同结果。本轮不训练模型，不把点 ID 当 object ID，不由运动相似度生成 object GT。

本地实现：`/Users/hela/Instruct-GS-World-recovered-20260725/code/scripts/review_point_tracker_v67.py`。

服务器前台入口：`/mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/review_point_tracker_v67.sh`。

默认输入：`/mnt/pfs/public/xuhaoming/instruct_gs_world/data/multisource_real_robot_video_v53/index.json`。

默认权重：`/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/cotracker/scaled_offline.pth`，CoTracker3 offline。不下载模型，不加载 DINO、SigLIP 或历史 world-model checkpoint。默认只使用一个可见 GPU，没有 torchrun、DDP 或后台进程。联网只用于最后的 W&B 上传。

## 比较方式

- 默认六源各 5 个 held-group clips。按现有 sampler 选择，不按 motion/teacher confidence 筛选；这 30 个 case 是探索样本，不是总体准确率估计，也不保证涵盖全部遮挡/小物体场景。
- 复用现有 native RGB decoder 与 HY packed-video frame offset，保存实际 source、episode、group、文件路径和 replacement 信息。已有 DataLoader 对坏文件的替换会明确记录；连续帧解码的错误不被新代码吞掉。
- 请求 100/200/400 ms 三组间隔，实际 frame stride 按每个来源的 fps 四舍五入。每组恰好 8 个 sampled frames；native 分支读取这 8 帧首尾之间的全部连续帧。两个分支共享同一物理 anchor、同一批 query points、相同起止时刻。不同间隔也共享物理 anchor。
- 默认 16×16 个 query points，复用 V67 jitter sampler；颜色只表示 point ID，不表示 object。输入 RGB 不预先缩小，但 CoTracker 自身的 internal resolution 会记录在 report/W&B 中。
- 使用原始 predictor 返回的 visibility，不叠加 DINO valid、relay confidence、relation score 或剔除低置信轨迹。visibility 不等同于存在状态，也不是标定概率；query frame 是给定值，不计入两种采样的差异统计。
- 原视频、连续帧叠加、抽帧叠加、共同时间戳左右对比，以及固定原图坐标的三个 crop 放大。crop 不跟随预测移动，避免自动追随漂移而隐藏错误；crop 内显示 point ID。
- 对比视频只显示两种输入共有的 8 个时刻，不插值生成抽帧分支不存在的中间轨迹。各视频按对应真实时间间隔播放。

## 结果入口

默认运行名称为 `tracker_visual_review_v67_seed17_<revision前7位>`。

结果目录：`/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/<运行名称>/`。

| 文件 | 内容 |
|---|---|
| `index.html` | 离线视频浏览、source 过滤、人工问题记录、manual query 点选 |
| `review_bundle.zip` | 可下载的完整浏览包，解压后打开 `index.html` |
| `review.log` | 前台运行日志，含每个 case 的 decode/inference/render 进度 |
| `cases.json` | 实际样本选择，含 episode 内物理 anchor frame |
| `summary.json` | 运行配置、实际模型类和内部分辨率、每一组结果 |
| `wandb_run.json` | 成功上传后的 W&B run URL |
| `<case>/source.mp4` | 原始 RGB 连续片段，无轨迹叠加 |
| `<case>/step_<ms>ms/native.mp4` | 连续帧推理结果和局部放大 |
| `<case>/step_<ms>ms/sampled.mp4` | 8 帧抽样推理结果和局部放大 |
| `<case>/step_<ms>ms/comparison.mp4` | 同时间戳左右对比 |
| `<case>/step_<ms>ms/tracks.pt` | 未过滤轨迹、visibility、in-bounds、frame IDs 和查询点 |
| `<case>/step_<ms>ms/point_rows.csv` | 每个共同时间戳、每个点的两个预测和差异 |

W&B 项目：`healenrenss-university-of-chinese-acadmic-and-science/instruct-gs-world`。
Group：`point-tracker-visual-review-v67`；视频表：`tracker_review/cases`；Artifacts 中的 `tracker-review` 包含完整 ZIP。

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
export RUN_NAME="tracker_visual_review_v67_seed17_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/tracker_visual_reviews/${RUN_NAME}"
export CASES_PER_SOURCE=5 TEMPORAL_STEP_MS=100,200,400 GRID_SIDE=16
export SEED=17 HELD_GROUP_STRIDE=20 DISPLAY_WIDTH=640 REUSE_COMPLETED=1
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
export WANDB_NAME="${RUN_NAME}" REVIEW_STAGE=run
unset QUERIES_JSON WANDB_RUN_ID WANDB_RESUME
cd "${ROOT}"
bash "${ROOT}/code/scripts/review_point_tracker_v67.sh"
echo "REVIEW_RC=$?"
```

同一配置原样重跑会复用已完成结果；中途退出不会清理产物。预测已保存但视频渲染中断时，重用预测重新渲染。修改查询点、模型配置或 revision 会重新计算相应结果；不是严格训练 checkpoint resume。需要全部重算时设置 `REUSE_COMPLETED=0`。

若视频已生成但 W&B 上传失败，复用上面的完整环境、仅将 `REVIEW_STAGE=upload` 后再次执行同一 shell。它只读已有 summary/媒体，不加载 tracker、不跑 GPU 推理。结果 ZIP 在 W&B 上传之前就已落盘。

## 人工多点复查

1. 下载 Artifacts 中的 ZIP，解压后打开离线 `index.html`。每例展开 `Select points for a manual rerun`。
2. 在原生 anchor 图像上点击多个点，分别覆盖目标表面、边缘、机械手与邻近背景。`Point group` 只是人工记录标签，不送入 tracker 或 grouping loss。
3. `Export manual queries` 导出归一化坐标和原 case 元数据。把文件放到服务器 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/manual_queries.json`。
4. 复用上面的完整前台环境，设 `QUERIES_JSON=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/tracker_visual_reviews/manual_queries.json`，将 `RUN_NAME`、`WANDB_NAME`、`OUT` 改成独立的 `_manual` 运行后执行同一入口。只处理真正点选过的 cases；不需要重新选择视频。

浏览页的人工问题记录通过 `Export observations` 导出。先记录哪个视频、哪一帧、哪个 point 漂移或 visibility 错误，再做分来源和失败类型汇总。若点轨迹可靠而 grouping 不成立，应归因于 object 推导问题；若只有抽帧版本失效，先修采样；不能由视觉上好看的少数 case 声称所有 teacher targets 正确。
