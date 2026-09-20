# V68: Grounded Motion Data 与 RGB Object Transport

## 本次交付与边界

本地代码根为 `/Users/hela/Instruct-GS-World-recovered-20260725`，分支为 `codex/grounded-motion-data-v68`。
本次同步实现数据、reader、模型和两阶段训练；执行顺序是**先造数据并看样本，不自动启动训练**。
旧 V67 模型、数据和输出保留。没有在本地运行模型前向、训练或 smoke test；本次交付只经过静态检查。
服务器实际能否完成整条流程、背景误标是否减少、外部视角是否正确，必须由本次产物确认，不能写成已经验证。

## 数据如何产生

1. 使用已有多源 index，不重建 RGB。选择 native FPS、原尺寸、连续至少10秒的片段；RobotWin 原生30 Hz 不降采样。
   新数据包含 RobotWin、AgiBot、RoboMIND、Bridge、HY，排除 DROID。train/held 按已有 source-local group 划分，
   每20个 group 留1个 held；旧 CASE_MANIFEST 也重新应用 partition 和相机策略。
2. RoboMIND 默认只允许 camera_front_external / camera_front / camera_top。发现 wrist/handeye 等相机时，按同 episode
   的实际 LeRobot metadata 重取外部 camera 的文件、offset 和 length，不能只改路径字符串。无对应外部视频则记录排除。
   Camera key 是采集配置证据，不是视觉确认；错误命名、剪辑、转换错误尚不能由此自动解决。
3. 对每个 clip 做独立16×16参考点追踪，RANSAC 拟合背景主导的二维 homography。至少8点、55% inliers、覆盖16格中的6格
   才使用该帧的补偿。它不等于相机位姿，也不能消除所有视差；不足的帧保留为 unknown，不当成零运动。
4. 先用现有 Grounded-SAM-2 多尺度 proposal 采512个 pilot 点并追踪。机械臂角色继续用跨 anchor 的证据，
   未被检测为机械臂不等于已经证明是物体。Motion-positive pilot 点作为 SAM 正提示；邻近且长期符合背景运动的参考点
   作为负提示。重新生成支持 mask 后，再按区域面积与空间覆盖密集补点，总预算2048，单次 CoTracker256点。
5. 相对背景的运动量是每条有效轨迹 x/y 的5%-95%跨度合成长度，单位是第一帧坐标系的 native pixels。
   门槛为 `max(1.5 px, 3*二阶差分尺度, 2*背景拟合残差P90)`。这些是明确记录的工程筛选参数，不是物理定律或准确率。
6. 从另一可见时刻重新查询同一条轨迹。至少6个共同有效帧且70%以上位置相差不超过3 px，才保留一致性证据。
   排除原 query 和重查询 anchor 自身；同一个 tracker 的重查询一致性仍不是真值验证。
7. 在每个 anchor-local 合格区域内保留75%的候选，空间格轮换、格内按补偿后运动量排序。
   75%不是整张图最大位移点的75%，也不是置信度。每个候选区域独立分配，减少大运动部件挤掉小/远物体的问题。
   机械臂、scene、unknown 及被筛掉的点全部保存为 context，只有最终目标与有效帧交集进入强 motion loss。

每个 case 保存 `teacher.pt`，其中包括原始轨迹、补偿坐标、query 时间、tracker visibility、重查询误差、背景模型、
target/context masks、mask/role evidence、相机及文件映射。Track ID 不是 object ID；SAM region ID 只在当前 anchor 有效。
`target_valid` 是训练/可视化共用的监督有效性掩码，不是视觉 visibility 真值。阴影、重复错误和机器人与物体粘连仍需人工查看。

## 模块输入输出与学习目标

- **Data reader**：读固定 manifest 和离线 teacher，随机取当前时刻及独立的1–4帧历史，历史间隔0.1秒。
  读取两张未来帧：当前后1秒、3秒。3秒是请求的固定跨度，不是 episode 终局或语言 goal。
  batch 中 RGB 为 `[B,6,3,H,W]`，四个历史位置用有效性 mask 表示1–4帧，不重复当前帧；图像只 pad、不全图缩小。
  最多256个当前可见轨迹点作为监督读取位置，未来筛选不决定 RGB encoder 看见的区域、历史长度或密度。
- **RGB encoder**：复用 native CNN 的三层卷积，移除全局平均池化，得到 stride-4 可学习特征图。
  18个 recurrent slots 对有效图像特征做三轮 attention pooling / GRU 更新；16个是 object hypotheses，另外2个用于 robot 与 scene。
  输出 `[B,18,256]` latent 和 attention centroid。它们不是已证明的物理状态或语义 identity。
  teacher 坐标、mask、future RGB 不进入 `encode_history`。Student 不读 CoTracker 输出来构造状态。
- **State auxiliary readout**：在当前坐标，按各 slot 的独立 feature field 和当前 ownership 合成 DINO/SigLIP 预测。
  冻结 teacher 读取原生224像素重叠 tiles，保留 DINO1024维、SigLIP768维，不做旧版固定分组维度压缩。
  局部14/28/56像素邻域池化仅作 appearance 辅助目标；它不是 object boundary、identity 或 dynamics 成功标准。
- **State supervision**：appearance cosine、可用轨迹的跨帧 ownership JS、object/robot/scene 弱角色 CE、同 anchor-local
  区域内的 binding、不同区域明显不同相对运动的 separation。仅这些训练期 loss 可读取未来轨迹。
  同区域不保证同物体，不同运动不保证不同物体，articulation 等情形仍可能违反弱证据，必须保留逐例验证。
- **Dynamics**：先加载 V68 state checkpoint 并冻结 encoder；target encoder 拷贝其最终参数后固定。
  训练期 posterior 读取当前/未来 slot transition，得到每个 component 一个连续32维 effect，总形状 `[B,18,32]`。
  Mean/log-variance 描述变分分布，不代表空间 Gaussian；没有显式机器人 action、语言或 History Prior。
  共享4层 Transformer 更新18个 latent tokens；位置读出在**当前坐标**计算，预测1秒/3秒后的坐标。
  真实未来坐标只进入 loss，不是 decoder query。主 Transformer 不在 dense patch grid 上预测未来。
- **Two paths**：短期 effect 预测1秒；后续 effect 推进至3秒；composer 合成的 effect 直接预测3秒。
  两条3秒路径分别接受真实轨迹监督，再约束其一致性。Zero effect 通过显式差分结构得到 current-copy，
  不把这一恒等构造当成学到的能力。正确/shuffled effect 的实际误差差距才需要实验检验。

State loss 是 appearance + correspondence + 0.25 role + 0.1(binding + separation)。
Dynamics loss 是 short + long-direct + rollout 的坐标 SmoothL1，另加0.25 path、0.1 intervention、
0.1 tracker-observability BCEWithLogits、0.001 posterior KL。SmoothL1 在归一化图像坐标中计算，beta=0.01；
W&B 同时输出原生像素 EPE、P50/P90/P95 和逐案例有效点的误差数组。
EPE 是预测坐标与 tracker 目标坐标的欧氏距离，**不是人工 GT accuracy**。
Visibility 辅助仅学习 tracker 可观测输出，不训练“对象不存在”的真值。RGB reconstruction 不进入 loss。
所列权重和32维 effect 是本次实现的起点，尚无消融证明它们最优。

## 一：同步并部署（仅这一步访问 GitHub）

在服务器前台执行；这个代码块不启动数据生成或训练，不使用 shell exit。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git fetch origin '+refs/heads/codex/grounded-motion-data-v68:refs/remotes/origin/codex/grounded-motion-data-v68' &&
git switch --detach refs/remotes/origin/codex/grounded-motion-data-v68 &&
SOURCE_REVISION="$(git rev-parse HEAD)" RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world \
  bash /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source/code/scripts/deploy_grounded_motion_v68_runtime.sh
```

发布目录是 `/mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/grounded_motion_v68/releases/<commit>`。
下面每段独立从部署记录读取 revision，随后使用不可变的具体 release；运行时不访问 GitHub，不下载模型。
已有本地 GroundingDINO、SAM2、CoTracker、DINO、SigLIP 权重继续复用。

## 二：先造400个 Clip 并全部可视化

单卡、单 worker、前台；RobotWin、AgiBot、RoboMIND、Bridge、HY 各80个 episode，各取一个连续10秒窗口，目标共400个 clip。
腕部/时长不合格的 episode 不占80个名额，会继续选择该源的下一条外部视角 episode。若源中没有足够合法候选，
selection.json 如实记录实际数量，不跨源凑数，也不重复视频。轨迹效果不好/没有最终目标的 case 仍保留并可视化，
不能通过只展示成功追踪把400例变成偏置样本。这是 train-partition 的数据检查批，不是 held 指标实验。
不能解码的源片段单独记录并从同源 reserve 补位；追踪失败或零目标不补位。
旧清单不传入，重新应用相机策略；用户核验这400例之后，才讨论扩展/训练，不自动启动。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(<"${RUNTIME_ROOT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export DATA_INDEX="${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json"
export RUN_NAME="grounded_motion_v68_review400_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/data/${RUN_NAME}"
export DATA_GPUS=1 WORKERS_PER_GPU=1 DATA_PARTITION=train CASES_PER_SOURCE=80 REVIEW_CASES_PER_SOURCE=80
export CLIP_SECONDS=10 ALL_EPISODE_WINDOWS=0 MOTION_TOP_FRACTION=0.75
export PILOT_POINT_BUDGET=512 POINT_BUDGET=2048 POINTS_PER_PASS=256
export RENDER=1 REUSE_COMPLETED=1 DATA_STAGE=run OPERATION=build SEED=17
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CASE_MANIFEST CAMERA_OVERRIDES WANDB_RUN_ID
bash "${ROOT}/code/scripts/build_grounded_motion_data_v68.sh"
echo "BUILD_RC=$?"
```

同一条命令、同一 OUT、同一配置重跑即续做：已完成 case 和中间 reference/pilot/refined-query/dense/relay 阶段复用。
GPU 数、配置或代码改变时使用新 OUT，不能覆盖正在被训练读取的 teacher shards。
若仅 W&B 上传失败，计算结果已落盘；重设相同全部参数，只把 `DATA_STAGE=upload`，重跑脚本即可重传后合并。

### 恢复旧400条单卡任务

先执行上面的同步/部署步骤，再执行下面独立命令。读取旧目录 `workers.json` 恢复原数据参数和worker数，
使用新代码，显式允许复用旧revision的相同teacher配置。不会删除或重算已完成的83条；日志中的实际复用数为准。
如果采样/模型/相机参数发生变化则不复用不匹配结果。新增的reserve和失败记录不会改变已有有效clip的标签。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(<"${RUNTIME_ROOT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export OUT="${RUNTIME_ROOT}/data/grounded_motion_v68_review400_6b8a4e4_1gpu"
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
bash "${ROOT}/code/scripts/resume_grounded_motion_data_v68.sh"
echo "RESUME_RC=$?"
```

这是显式恢复入口，保留记录中的拓扑；旧400条仍单卡单worker，已运行的20,000条任务则恢复原8卡16worker。
不同revision的复用只通过此入口或显式 `REUSE_SOURCE_REVISION` 开启，普通新构建不会自动继承别版数据。

### 不等构建结束，打包已完成部分

不加载模型、不占GPU，不改 `complete.json`、case plan或构建进度。只读取完成标记，不把仅存在teacher.pt的半成品当成完成。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=/mnt/pfs/public/xuhaoming/instruct_gs_world
REV="$(<"${RT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
"${RT}/.venv/bin/python" "${RT}/runtime/grounded_motion_v68/releases/${REV}/code/scripts/pack_grounded_motion_partial_v68.py" \
  --out "${RT}/data/grounded_motion_v68_review400_6b8a4e4_1gpu"
```

输出 `partial_review_bundle.zip`、`partial_index.html`、`partial_manifest.json`，明确标记partial，不替代完整训练manifest。
本机下载并打开已有部分：

```bash
NAME=grounded_motion_v68_review400_6b8a4e4_1gpu
LOCAL_DIR="${HOME}/Downloads/${NAME}_partial"
mkdir -p "${LOCAL_DIR}"
scp -P 8600 "root@10.66.0.39:/mnt/pfs/public/xuhaoming/instruct_gs_world/data/${NAME}/partial_review_bundle.zip" "${LOCAL_DIR}/review.zip" &&
unzip -o "${LOCAL_DIR}/review.zip" -d "${LOCAL_DIR}" &&
open "${LOCAL_DIR}/partial_index.html"
```

关键产物位于 `/mnt/pfs/public/xuhaoming/instruct_gs_world/data/grounded_motion_v68_review400_<commit前7位>/`：

- `index.html`：各 shard 入口；每 case 默认展开整段 episode 概览、同 episode 相机对照、实际文件/时间映射。
- `review_bundle.zip`：离线可看的 HTML、视频、PNG、JSON，不包含大型 tensor 文件或模型权重。
- `training_manifest.json`：合并后的训练索引；`status=completed` 表示所有 worker 的构建完成，不代表 teacher 正确。
- `build.log`：前台输出副本；`shard_0000/progress.json` 等是各 worker 已完成/计划数量。
- 各 case 的 `object_after_topk.mp4` 是实际强监督点；`all_points_context.mp4` 保留机器人/背景/unknown。
  `background_reference.mp4` 和 `background_fit.json` 解释相机补偿；`refinement` PNG 展示 SAM 收紧结果。

颜色沿用 green=object candidate、orange=robot hypothesis、purple=unknown、blue=scene。
绿色仍是弱标签；优先查看瓶身覆盖、HY/RoboMIND 静态背景是否还移动、远物体是否被删、robot/object 接触边界，
再看相机概览的 SELECTED 行是不是外部视角。轨迹视频反映原图位置；补偿坐标只用于筛选和证据，不把原视频扭曲。

## 三：下载首批结果（本机执行）

不依赖本机用户名。首次同步后记录服务器输出的 RUN_NAME，下面命令也可直接读取部署 revision：

```bash
REV="$(ssh -p 8600 root@10.66.0.39 'cat /mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/grounded_motion_v68/DEPLOYED_REVISION')"
NAME="grounded_motion_v68_review400_${REV:0:7}"
LOCAL_DIR="${HOME}/Downloads/${NAME}"
mkdir -p "${LOCAL_DIR}"
scp -P 8600 "root@10.66.0.39:/mnt/pfs/public/xuhaoming/instruct_gs_world/data/${NAME}/review_bundle.zip" "${LOCAL_DIR}/review_bundle.zip" &&
unzip -o "${LOCAL_DIR}/review_bundle.zip" -d "${LOCAL_DIR}" &&
open "${LOCAL_DIR}/index.html"
```

如期间又部署了其他 revision，应使用原构建的 RUN_NAME，不下载另一版目录。

## 四：当前生产任务：20,000条数据，只可视化400条

与原400条任务使用不同输出目录，不停止或覆盖旧任务。五源各4,000个 train episodes，各一个连续10秒窗口，目标20,000条。
在全局 case plan 上先固定每源前80条作为展示集，再分片；不是每个 worker 各选400条，也不按追踪效果挑成功案例。
`REVIEW_CASES_PER_SOURCE=80` 是全局每源展示上限；设0表示展示全部选中片段，`RENDER=0` 表示不展示。
非展示样本仍完整产生训练 teacher、中间续跑缓存、mask 和逐点证据，但不渲染 PNG/MP4、不加入 review 视频包。

在已分配8张可见GPU的任务中前台执行。`DATA_GPUS=8` × `WORKERS_PER_GPU=2` = 16个独立进程；local rank 0/1用可见卡0，
2/3用卡1，以此类推。不是16卡 DDP；不做梯度同步。每个进程各自持有 SAM/GroundingDINO/CoTracker，能让一条任务的CPU解码、
磁盘写入与另一条任务的GPU推理重叠，但会增加模型副本和推理峰值显存，不保证线性加速。每个进程使用独立 shard。
不重置平台的 `CUDA_VISIBLE_DEVICES`，GPU编号是该变量映射后的逻辑编号。默认仍每卡1个 worker，下面显式启用2个。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(<"${RUNTIME_ROOT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export DATA_INDEX="${RUNTIME_ROOT}/data/multisource_real_robot_video_v53/index.json"
export RUN_NAME="grounded_motion_v68_train20k_review400_${SOURCE_REVISION:0:7}_8g2w"
export OUT="${RUNTIME_ROOT}/data/${RUN_NAME}"
export DATA_GPUS=8 WORKERS_PER_GPU=2 DATA_PARTITION=train
export CASES_PER_SOURCE=4000 REVIEW_CASES_PER_SOURCE=80 ALL_EPISODE_WINDOWS=0
export CLIP_SECONDS=10 MOTION_TOP_FRACTION=0.75 PILOT_POINT_BUDGET=512 POINT_BUDGET=2048 POINTS_PER_PASS=256
export RENDER=1 REUSE_COMPLETED=1 DATA_STAGE=run OPERATION=build SEED=17
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CASE_MANIFEST CAMERA_OVERRIDES WANDB_RUN_ID
bash "${ROOT}/code/scripts/build_grounded_motion_data_v68.sh"
echo "BUILD_RC=$?"
```

重复相同代码、OUT和全部参数即可续做，不必重新生成完成的片段；更改 worker 数时使用新 OUT，避免改变既有分片归属。
若改为每卡1个 worker，同时改成 `WORKERS_PER_GPU=1` 和新的 `RUN_NAME`（例如尾缀 `_8g1w`），不在已运行目录混用。
运行中每个 `shard_0000/progress.json` 到 `shard_0015/progress.json` 记录完成/计划片段和展示数量；日志为 `${OUT}/build.log`。
W&B 每 worker 一个 run，表格只展示预选案例；`motion_data/clips` 仍统计该 worker 的全部训练数据，不把400例平均外推为全部质量。
全部 worker 完成后 `${OUT}/training_manifest.json` 汇总 `completed_clips`、`rendered_clips`、`clips_by_source`、`review_by_source`。
样本充足时应为20,000/400，每源4,000/80；不足时如实记录，不复制凑数。HTML和 `review_bundle.zip` 只含这400例的展示媒体。
原始RGB视频和DINO/SigLIP特征不重复存盘；当前格式保留轨迹及续跑中间 tensor。
按301帧、2048点估算，20,000条主要 tensor 合计约830GB，另需 mask、逐点JSON和400例媒体空间；这是张量字节估算而非实际磁盘测量。
本次不启动模型训练。数据与人工检查完成后再固定 manifest。
源视频解码失败不再终止整个worker：沿用精确帧解码器的已知媒体错误分类，写入 `decode_failures.json`，
同worker后续不再尝试已因解码失败隔离的文件，并按同源reserve补位。没有伪造帧、置零轨迹或将模型异常当作坏视频。
reserve默认每源 `max(32, ceil(25% * CASES_PER_SOURCE))` 个额外episode，可用 `REPLACEMENT_CASES_PER_SOURCE` 显式设置。
reserve也按worker分片，补位继承原片段的展示名额；所有失败路径/原因随manifest和W&B artifact保留。
若reserve耗尽，worker报告 `incomplete`，最终manifest为partial，不谎报400或20,000条已满。
整段episode/其他相机概览若无法解码，展示明确的错误说明及源路径，不丢掉已经成功解码和追踪的10秒训练片段。
程序错误、CUDA OOM和模型失败仍自然报错，不以跳过策略隐藏。

下载这次400例展示包（本机运行；如之后部署了新版，`NAME` 使用原任务打印的名称）：

```bash
REV="$(ssh -p 8600 root@10.66.0.39 'cat /mnt/pfs/public/xuhaoming/instruct_gs_world/runtime/grounded_motion_v68/DEPLOYED_REVISION')"
NAME="grounded_motion_v68_train20k_review400_${REV:0:7}_8g2w"
LOCAL_DIR="${HOME}/Downloads/${NAME}"
mkdir -p "${LOCAL_DIR}"
scp -P 8600 "root@10.66.0.39:/mnt/pfs/public/xuhaoming/instruct_gs_world/data/${NAME}/review_bundle.zip" "${LOCAL_DIR}/review_bundle.zip" &&
unzip -o "${LOCAL_DIR}/review_bundle.zip" -d "${LOCAL_DIR}" &&
open "${LOCAL_DIR}/index.html"
```

## 五：服务器端可选整条前反向检查

这只测 reader/AMP/backward、future swap 和 zero effect，不是能力 gate，不要求训练 launcher 读取其结果。
在**有实际产出目标的首批数据完成后**运行；单 GPU，不运行在本地 Mac。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=/mnt/pfs/public/xuhaoming/instruct_gs_world
REV="$(<"${RT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/grounded_motion_v68/releases/${REV}"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${RT}/.venv/bin/python" "${ROOT}/code/scripts/verify_grounded_object_transport_v68.py" \
  --manifest "${RT}/data/grounded_motion_v68_review400_${REV:0:7}/training_manifest.json" \
  --dino_checkpoint "${RT}/models/dinov2_vitl14/model.safetensors" \
  --siglip_checkpoint "${RT}/models/siglip2-base-patch16-224"
echo "VERIFY_RC=$?"
```

## 六：State 训练（独立八卡前台命令）

这是数据构建之后的命令，本次先不执行。默认微批4、累积8、全局256，30k steps；不声称该设置已测过显存或吞吐。
训练时不运行 SAM/CoTracker，不访问仓库/模型下载服务；W&B online 只同步实验记录。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(<"${RUNTIME_ROOT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export TEACHER_MANIFEST="${RUNTIME_ROOT}/data/grounded_motion_v68_train20k_review400_${SOURCE_REVISION:0:7}_8g2w/training_manifest.json"
export STAGE=state RUN_NAME=grounded_object_transport_v68_state_r1_seed17
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export NPROC_PER_NODE=8 BATCH_PER_GPU=4 TARGET_GLOBAL_BATCH=256 WORKERS_PER_RANK=2 POINTS_PER_SAMPLE=256
export DINO_FRAME_BATCH=96 SIGLIP_FRAME_BATCH=96 STEPS=30000 LR=0.0002 SEED=17
export SAVE_EVERY=2500 RECOVERY_EVERY=500 LOG_EVERY=20
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CUDA_VISIBLE_DEVICES RESUME STATE_CHECKPOINT WANDB_RUN_ID
bash "${ROOT}/code/scripts/train_grounded_object_transport_v68.sh"
echo "TRAIN_RC=$?"
```

## 七：Dynamics 训练（State 完成并评估之后）

继承 V68 state 的 encoder，冻结它之后单独训练 latent effect + transport，不继承 V67 slot/optimizer。
不能将 source checkpoint 文件存在等同于 Object State 达标；最终仍需 held object separation、删除干预和外部观察证据。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(<"${RUNTIME_ROOT}/runtime/grounded_motion_v68/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export TEACHER_MANIFEST="${RUNTIME_ROOT}/data/grounded_motion_v68_train20k_review400_${SOURCE_REVISION:0:7}_8g2w/training_manifest.json"
export STATE_CHECKPOINT="${RUNTIME_ROOT}/outputs/grounded_object_transport_v68_state_r1_seed17/latest.pt"
export STAGE=dynamics RUN_NAME=grounded_object_transport_v68_dynamics_r1_seed17
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export NPROC_PER_NODE=8 BATCH_PER_GPU=4 TARGET_GLOBAL_BATCH=256 WORKERS_PER_RANK=2 POINTS_PER_SAMPLE=256
export DINO_FRAME_BATCH=96 SIGLIP_FRAME_BATCH=96 STEPS=30000 LR=0.0002 SEED=17
export SAVE_EVERY=2500 RECOVERY_EVERY=500 LOG_EVERY=20
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CUDA_VISIBLE_DEVICES RESUME WANDB_RUN_ID
bash "${ROOT}/code/scripts/train_grounded_object_transport_v68.sh"
echo "TRAIN_RC=$?"
```

## 八：训练断点恢复

下面恢复上面那条 state 命令；恢复 dynamics 时，首行目录改为 `/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/grounded_object_transport_v68_dynamics_r1_seed17`。
从运行目录的 run.json 恢复原代码 revision，不使用后来部署的新版本。
不会根据“目录里似乎有文件”自动恢复；必须显式指定 `RESUME`。初次运行 OUT 必须新建，避免覆盖现有实验。

```bash
export OUT=/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/grounded_object_transport_v68_state_r1_seed17
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$("${VENV_ROOT}/.venv/bin/python" -c 'import json,os; print(json.load(open(os.environ["OUT"]+"/run.json"))["args"]["source_revision"])')"
export ROOT="${RUNTIME_ROOT}/runtime/grounded_motion_v68/releases/${SOURCE_REVISION}"
export TEACHER_MANIFEST="${OUT}/dataset.json" RESUME="${OUT}/latest.pt"
export RUN_NAME="$(basename "${OUT}")" NPROC_PER_NODE=8 WORKERS_PER_RANK=2
unset CUDA_VISIBLE_DEVICES WANDB_RUN_ID
bash "${ROOT}/code/scripts/train_grounded_object_transport_v68.sh"
echo "RESUME_RC=$?"
```

恢复模型/EMA、optimizer、scheduler、数据 manifest、epoch/cursor、各 rank RNG 和 W&B run ID，保持原 rank 数和 batch 配置。
运行日志为 OUT 下的 train.log，滚动 checkpoint 为 latest.pt，里程碑为 step_0002500.pt 等，progress.json 记录实际 step。
绝不删除旧 checkpoint 来让启动命令“通过”。state 和 dynamics 是两个显式运行，不是自动连续提交。

## 下一步的判断依据

首先判断造出的数据是否满足用户看到的物体运动，而不是先看 loss：逐例核对外部视角、视频连续性、原/补偿位移、
筛掉与留下的点、机器人接触区域、真实小幅运动召回。背景 homography + SAM 提示 + relay 仍可共同出错，
不能用三者一致替代独立标注。确认后扩展离线数据，冻结数据版本再训练；不让 student 反过来重写 teacher 以降低 loss。
模型后续需要 object-local intervention 和 held 逐点误差证据，本次没有新增已完成实验结果，也没有把 tracker 一致性升级为 object 成功。
