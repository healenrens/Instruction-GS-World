# V69 Object Video Sequence：结构、目标与执行

日期：2026-09-29。开发分支：`codex/object-video-sequence-v69`。
当前实现为V69r2，architecture=`pretrained_query_object_video_sequence_v2`。审阅逐条处理见`OBJECT_ASSOCIATION_REVIEW_V69R2.md`。
状态：完整实现并进行静态代码审查；GPU执行结果待用户单卡测试。本文不是实验成功报告。
用户首轮单卡反馈：Dynamics两条路径均到step10，但resume对照在`posterior.queries`失败；整链路尚未通过。
研究主线及历史否证记录仍在 `OBJECT_WORLD_MODEL_PROGRESS_V57.md`；设计依据在 `OBJECT_VIDEO_SEQUENCE_PLAN.md`。

## 1. 这次改变什么

从“浅层CNN将几张图压成固定18个slots，再预测两个位置”改为：

```text
已观察的3秒RGB历史，16帧，保持原生分辨率
  -> Frozen DINOv3 ViT-L/16
  -> 只从历史生成16个visual queries，或由用户给定历史点query
  -> 共享Object Memory，每query一个anchor和8个可变化carriers
  -> t0状态 [B,16,9,512]
  -> 给定continuous effect的Object Dynamics
  -> 未来0.2至5秒的25个状态 [B,25,16,9,512]
  -> 对历史观测点的object-local位置作连续readout
  -> 25个时刻的位置与observability

训练专用目标：
原视频未来 + 已保存的CoTracker/relay轨迹
  -> 观测目标、对应点位置、unknown/visible/occluded证据
  -> 只进入loss、EMA目标和训练期effect posterior
```

不训练像素生成器，不使用RGB重建主loss，不增加语言、机器人action或History Prior。
这不是“纯历史即可知道5秒后唯一结果”的承诺：Dynamics阶段通过Posterior解释实际发生的未来。
部署时必须由policy/goal/语言等另一个模块选择effect；本版提供 `model.forecast`，不实现该selector。

## 2. 数据究竟如何进入

1. `prepare_object_video_manifest_v69.py`只读取V68已经原子保存的`complete.json`和`teacher.pt`，写新manifest。
   不调用SAM或CoTracker，不改旧数据；RoboTwin绝不重新追踪。未完成的20k目录不按名称视为完成。
2. 按`source/group/episode`划分train/held，默认held10%；同episode不同窗口不跨split。
3. 每epoch打乱episode，每个episode均匀选一个已记录窗口。DDP尾部为凑齐microbatch会重复少量episode，数量写入`run.json`。
   不再强行给五个来源同样权重。旧20k来自来源配额，训练sampler不能把它变成“全库均匀”。
4. 在10秒轨迹内选择容纳完整3秒history与5秒future的t0。History16帧覆盖`[-3,0]`，future25帧覆盖`[0.2,5]`。
   RGB取真实帧；轨迹保持原生帧记录。重复帧不重复计loss，history保留最后的当前帧。
   支持`history_min_seconds`和`future_min_seconds`独立配置；默认min=max，首轮固定3秒/5秒。
5. 不把图缩到224、384或24x24。每个视频单独pad到16倍数，再在batch内pad；另一张大图的padding不改变本图encoder上下文。
   视频时间使用解码PTS；老RGB cache没有PTS时明确标为`frame_index_over_manifest_fps`。外部请求时间与实际帧时间分别有据可查。
6. 从历史任意时刻有可靠观测的轨迹随机取最多256个测量点，包括当前已遮挡的点。每点reference是最后一次可靠历史位置及其时间。
   它们不是Student queries。全片段75%筛选只决定transport loss权重，不能决定输入state或Dynamics的query集合。
7. 继承所有类别轨迹的全局75%规则，不用“物体/夹爪”的不可靠分类筛点。原始可靠位置与背景补偿有效性分开。
   若motion排名完全不可用但仍有原始可靠轨迹，则明确标`uniform75_motion_ranking_unavailable`，保留均匀75%监督；不能把它称为运动筛选成功。
8. 解码错误记录并跳过。DDP中一个rank遇到坏batch时所有rank跳过该microbatch，不替换帧、不伪造数据、不重编码视频。
   为保持collective一致，同microbatch的其他可读样本也暂不用于更新；失败源和实际使用数量必须一起看。

Mask分工：`pixel_valid`只表示真实像素/真实采样帧；`teacher.valid`表示可靠测量；`point_present`表示history已有该测量；
`transport_weight`表示主75%池；`observation=-1/0/1`分别表示unknown/遮挡证据/可见证据。
模型预测observability不能关闭轨迹loss。越界和tracker失败不等于对象不存在。

## 3. 模型尺寸

下表除backbone外为按代码Linear、LayerNorm、learned tensors静态精确计数；服务器`model_inventory.json`给实际加载计数。

| Block | 结构 | 参数量 | 学习任务 |
|---|---|---:|---|
| Frozen DINOv3-L/16 | 24层、1024宽、16 heads、patch16 | 官方约300M，实际加载后记录 | 提供已有局部视觉表征，本版不更新 |
| Object Memory | width512、8 heads；4个visual cross-attention + 4个object-local memory attention；FFN2048 | 27,088,393 | 将history证据聚合为可持续读出的query状态 |
| EMA Object Memory | 同上，momentum0.996 | 27,088,393 | State阶段缓慢跟随；Dynamics阶段固定观测目标 |
| Continuous Posterior | 每query4个learned tokens；4层cross-attention；36→512相对geometry输入；512→128分布头 | 12,706,432 | 从已发生的feature/estimated-geometry transition提取4x64连续effect |
| Object Dynamics | 8个144-token interaction blocks + 8个effect cross-attention blocks；width512、8 heads | 51,290,114 | 给定source、effect与真实时间，更新未来状态 |
| Shared compositional readout | local field MLP1042→512→512；appearance1024、position2、observation1；binding head | 2,123,306 | 读出每query对测量位置的支持、外观、位置与可观测性 |
| Task model总计 | 含EMA，不含backbone | 120,296,638 | State阶段29,211,699可训练；Dynamics阶段63,996,546可训练 |

FFN均为4倍宽度。Dynamics不是“8层总数”：每一层组含一个interaction和一个effect cross-attention，各自有FFN。
默认每query9 tokens，共144tokens。K=16是并行query预算，不是检测到16个物体；不同queries可能指向同一物体。
8个carriers是完整首版的局部容量预算，不声称已经证明最优；不退回单slot压缩，也不把增加容量当作object成立的证据。

V-JEPA2.1 ViT-L/16为同量级的替换对照，1024输出、24层、16heads、tubelet2；不与DINO叠加。
它只编码截至当前时刻、最长3秒的prefix。取早期state时重新编码早期prefix，不能取全片段双向编码后的早期token。
奇数长度prefix在开头重复一次首个已观察帧以适配tubelet2，不复制future；单帧用官方image patchifier。
这保留因果边界，但成本明显高于逐帧DINO，必须实测记录。

官方接口依据：[DINOv3模型](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py)、
[V-JEPA2.1模型](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py)、
[V-JEPA官方预处理](https://github.com/facebookresearch/vjepa2/blob/main/evals/video_classification_frozen/utils.py)。
两者仅使用官方本地architecture factory、`pretrained=False`后严格加载本地预训练权重；绝非随机backbone替代。

## 4. 每一步的张量与语义

`B`是每卡microbatch；`N=ceil(H/16)*ceil(W/16)`；`K=16`；`R=9`；`P=256`。
例如1920x1080原图，pad到1920x1088，`N=120*68=8160`，不是24x24。Patch是计算单元，不是object边界或GT。

| 阶段 | 输入 | 输出 |
|---|---|---|
| Perception | RGB `[B,41,3,Hpad,Wpad]`、pixels、实际times | `[B,41,Nmax,1024]`、每图native patch坐标及valid |
| History queries | 前16帧features/坐标 | xy`[B,K,2]`、观测帧`[B,K]`、descriptor`[B,K,1024]`、valid |
| Object Memory | 逐帧feature+query+时间 | 每时刻tokens`[B,K,R,512]`、carrier centers`[B,K,R,2]` |
| Target observation | 截止各未来时刻的真实prefix | `[B,25,K,R,512]`，只作Posterior/aux target |
| Posterior | 当前及未来latent tokens、elapsed time | mean/logvar/value各`[B,K,4,64]` |
| Dynamics | 当前tokens/centers、effect、25个请求时间 | direct和rollout各`[B,25,K,R,512]`及centers |
| Binding | 历史测量位置、该处冻结feature、历史state | `[B,P,K+1]`，最后一项是unbound，不是background物体 |
| Continuous readout | object state、reference位置、历史ownership | positions`[B,25,P,2]`、observation logits`[B,25,P]` |

### Queries与Memory怎样算

自动query从第一和最后有效history帧各选8个视觉/位置有差异的tokens；是**visual proposals**，不宣称自动实例分割。
也支持人或下游策略指定已观察帧上的点。Query descriptor是冻结feature，不是未来轨迹或75%筛选结果。
共享projection把1024维转为512维，加一个anchor及8个learned carrier offsets。每帧通过cross-attention读视觉证据，
再在每个query内9个tokens间交换信息，并用learned gate更新memory。Carrier centers来自attention位置加权及同一更新gate。

Anchor feature在该段视频中保持输入条件不变；8个carriers承载变化。这个不变性是**结构定义**，不能报成学会identity。
Centers是模型的support坐标，不是物体质心GT；512维只是可读出的latent，不命名为未经证明的物理属性。
来自t0的query可回看整段已观察history，因此早期query-conditioned state是回顾性估计，不是当时的在线状态；
对外声明的因果边界是t0。Future targets则逐prefix更新。

### Effect与未来序列怎样算

Posterior读取root/carrier latent、时间、估计root相对source的位移及帧内carrier相对root的位置，不拼RGB、tracker真实center delta或机器人action。
输出Gaussian mean/logvar，采样后tanh成连续有界effect；KL约束的是tanh之前的Gaussian。
每层组先在`B*K`个独立组内，让每query的9个state tokens读取它自己的4个effect tokens，再在144个compact tokens间interaction。
归属由张量分组保证，不再依赖anchor feature是否可区分；interaction后允许物体间相互影响。
每次用请求时间与本次elapsed time更新feature/center，保留query anchor。
Direct每次从t0预测请求时刻；rollout从前一步预测继续，**不使用真实未来state teacher forcing**。

### Readout怎样算

对每个历史测量点，记录其最后可靠观测处相对各query中心的坐标和ownership；这个reference可早于t0。
未来读出继续使用同一个reference object-local坐标，按局部carrier距离softmax池化feature，和anchor及位置Fourier特征一起进入共享MLP。
位置输出为未来support center加reference local坐标再加learned residual；减去reference时刻的读出偏差后加回真实历史位置。
各query独立产生field，仅通过reference ownership混合；不在measurement点之间做attention。
删除component时不重新归一化其余components，才能测量删掉哪部分证据造成误差。
这不是把object投回固定patch格；连续坐标可以跨patch，但小于patch的视觉细节仍受backbone限制，不能靠插值声称恢复。

## 5. Loss与它能证明什么

所有主要误差先在每个case内按有效样本归一，再求batch均值；报告另保留逐点、逐时间的原值、p50/p90/p95。

| Loss | 计算和目标 | 不能推导出的结论 |
|---|---|---|
| Observed/sequence transport | 预测位置与可靠对应点的原生像素Charbonnier距离；报告未平滑pixel EPE | 对同tracker误差小不等于人工GT准确 |
| Appearance辅助 | 对应点冻结DINO feature的cosine error | 不是RGB质量，不是object identity成立 |
| Weak binding | 冻结query/点feature affinity相对本帧分布标准化后形成soft target，KL监督ownership | 是weak appearance evidence，不是实例标签 |
| Track correspondence | 同一可靠点相邻时刻ownership的JS | 单独使用仍允许uniform捷径 |
| Relative motion | 邻近测量点之间相对位移的pixel error | 抵消共享translation，不是3D相机运动恢复 |
| Future latent辅助 | 预测carrier tokens与固定EMA观测tokens的LayerNorm距离 | 不单独用这个loss证明object语义 |
| Path consistency | rollout与stop-gradient direct在同一时刻的位置一致 | 两条路径都必须受真实轨迹约束 |
| Observability | 对known tracker证据的BCEWithLogits；unknown不参与 | tracker不可见不能解释成不存在 |
| Effect rate | FP32 Gaussian KL，均值仍用于正则，额外记录clip总和/每query总和 | 不是离散codebook，也不单独证明effect有用或实际bitrate |

State：appearance + weak binding + correspondence + observed transport + 0.1 observation。
Dynamics：direct transport + rollout transport + 0.25 relative motion + 0.25 latent + 0.25 path + 0.001 KL + 0.1 observation。
这些权重是明确的首轮配置，不是已选优结论。Shuffled/zero effect只记录对照，不用强制margin把同类运动判成错误。
Zero effect不被硬编码为静止。没有RGB主loss、全图dense未来DINO loss或任意拼feature后命名的identity loss。

State阶段先学习已观察视频中的可读状态；未来RGB在这个阶段属于观测重建，不冒充预测。
Dynamics阶段冻结Object Memory/readout并继承其EMA，单独训练Posterior和Dynamics。没有自动课程跳过独立state评估。

## 6. 交付的证据工具

| 入口 | 做什么 |
|---|---|
| `test_object_video_sequence_v69.sh` | 单卡完整模型，真实多源视频，两训练阶段、断点恢复与未中断参数对照、因果输入、evaluator和视频导出 |
| `review_motion_richness_v69.py` | 已有10秒轨迹的逐窗口原视频、raw/补偿motion、relay、人工盲评表；默认400clips，不重追踪 |
| `analyze_motion_richness_v69.py` | 人工左右窗口选择与各单项统计对照，train/held分开；不自动改采样分布 |
| `calibrate_motion_teacher_v69.py` | 独立人工点/遮挡标注检验tracker和relay 1/2/3/5/8px阈值；不改生产阈值 |
| `compare_pretrained_encoders_v69.py` | 相同原图/时间/点比较DINOv3与V-JEPA的定位、对应和两已观察帧的线性inverse probe；含zero/train-mean基线 |
| `evaluate_object_video_sequence_v69.sh` | 400held clips、逐例绝对轨迹误差、direct/rollout/zero/shuffle、历史顺序/末帧对照、视频、W&B |
| `upload_object_video_report_v69.py` | W&B上传失败后重传已保存产物，不重算模型 |

Encoder probe会用未来图像和对应点位置提取feature，**不是forecasting**。独立标注为空时标`not_measured`，不能拿tracker自评冒充独立GT。
Held evaluator会生成`annotation_template.jsonl`；人填写query所属物体、稀疏点位置和visible/unknown后，独立binding、删除component、
same/different object关系才有GT。先按独立object标签合并query概率，再评测uniform、merge-all、all-unbound反例；正确的same-object split queries不再误罚。这是评测器的否证能力，
不是已经证明训练目标会排除这些反例。没有人工标签时不造GT。
反转历史的V-JEPA对照会从变换后的RGB重新编码，不能用已包含完整history的缓存token。
所有视频标出HISTORY INPUT/FUTURE TARGET；yellow是transport池，cyan是辅助/context，red是预测，不再用颜色假称物体/机械臂。

## 7. 服务器资源需求

主方案必须有下面两个本地资产，路径可以用环境变量改。当前**尚未确认服务器是否存在**：

```text
/mnt/pfs/public/xuhaoming/instruct_gs_world/third_party/dinov3/hubconf.py
/mnt/pfs/public/xuhaoming/instruct_gs_world/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

它们分别是完整官方DINOv3源码目录和官方ViT-L/16预训练原始state dict；不能用Hugging Face另一种格式目录顶替文件。
V-JEPA对照另需`third_party/vjepa2/`和`models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt`，主方案不依赖它。
Python使用现有`.venv`的CUDA PyTorch、torchvision、numpy、Pillow、PyAV、OpenCV、wandb，以及所选官方encoder源码的依赖。
官方DINOv3依赖还列出omegaconf、ftfy、regex、scikit-learn、submitit、termcolor、torchmetrics；V-JEPA完整依赖见其源码`requirements.txt`。
不在训练任务里pip安装，不升级现有torch。旧V68 teacher读取不需要在训练中加载SAM/CoTracker权重。
单卡测试用80GB GPU、完整3s/5s、B=1；不是通过缩分辨率或缩模型来通过。CPU内存需承载原始41帧及checkpoint，建议64GB以上。
测试会写完整backbone/模型/optimizer checkpoint，多阶段及未中断对照合计需预留约30GB；实际显存/耗时进入report，不预先承诺。

如缺资产，仅在有网络的准备环境单独执行下载脚本。DINO链接需用户接受官方license后取得：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
RT=/mnt/pfs/public/xuhaoming/instruct_gs_world
"${RT}/.venv/bin/python" code/scripts/prepare_object_video_assets_v69.py \
  --runtime_root "${RT}" --dinov3_weights_url "填写官方授权下载URL"
```

此准备步骤允许网络；下列运行步骤不访问代码仓库或模型下载服务。

## 8. 同步与发布，独立于运行

代码推送后在可联网的登录环境执行；这里切换分支，不覆盖本地修改，不执行reset/clean。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git fetch origin refs/heads/codex/object-video-sequence-v69
git switch --detach FETCH_HEAD
export SOURCE_REVISION="$(git rev-parse HEAD)"
export RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world
bash code/scripts/deploy_object_video_sequence_v69_runtime.sh
```

之后固定release运行。`DEPLOYED_REVISION`只用来找到本次release；正在运行的进程不重新读取它。

## 9. 准备数据清单，独立命令

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=$PWD
REV="$(cat "${RT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/object_video_sequence_v69/releases/${REV}"
"${RT}/.venv/bin/python" "${ROOT}/code/scripts/prepare_object_video_manifest_v69.py" \
  --input "${RT}/data/grounded_motion_v68_train20k_allpoints75_recovered_r1" \
  --output "${RT}/data/object_video_sequence_v69/manifest.json"
```

这是索引快照，不是重建视频或追踪。后续生成更多clip时写另一份manifest；训练run会复制本次manifest并保持不变。

## 10. 单卡完整测试，前台独立命令

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD
export VENV_ROOT="${RUNTIME_ROOT}"
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
export MANIFEST="${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json"
export ENCODER=dinov3_vitl16
export ENCODER_REPOSITORY="${RUNTIME_ROOT}/third_party/dinov3"
export ENCODER_WEIGHTS="${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export ENCODER_FRAME_BATCH=2
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME="object_video_v69_full_single_gpu_${SOURCE_REVISION:0:7}_$(date +%Y%m%d_%H%M%S)"
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export WANDB_MODE=online
export WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset RESUME STATE_CHECKPOINT MODEL_CONFIG WANDB_RUN_ID WANDB_RESUME
bash "${ROOT}/code/scripts/test_object_video_sequence_v69.sh"
TEST_RC=$?
echo "TEST_RC=${TEST_RC}  REPORT=${OUT}/test_report.json  LOG=${OUT}/test.log"
```

每次完整测试用独立OUT；内部会自行测试strict resume。默认每源一个真实clip，State/Dynamics各`2*来源数`个optimizer steps，
另跑未中断版本比较模型参数，并调用一次evaluator。这个样本量只判断工程能否运行，不判断研究方法是否成立。
W&B最终run含报告、module inventory、运行指标、evaluator视频；不上传模型checkpoint。
后台命令、Git访问、自动下载、额外gate文件均不在这条链路里。测试失败会自然打印具体异常，SSH不会被脚本exit关闭。

### Resume比较修订后的简短重跑入口

单卡测试的两条路径现在都显式启用deterministic algorithms、固定cuBLAS workspace、关闭TF32/cuDNN benchmark，
任务attention使用显式QKV计算；未缩小模型或分辨率，也未放宽原`atol=1e-6, rtol=1e-4`。
冻结backbone不全局切换到展开全图attention的math实现，避免无谓放大原生图像显存。
生产训练默认仍是fast kernels；strict resume指训练状态恢复，不等同于fast CUDA执行必然逐位相同。
需要生产确定性模式时显式设置`DETERMINISTIC=1`，它随checkpoint args恢复，可能降低吞吐。

测试会在每个microbatch保存case/frame/point/teacher、CPU/CUDA RNG、source state、posterior实际采样noise及梯度；
比较模型、optimizer、scheduler、数据游标和最终RNG，并在assert之前保存、上传逐步差异。
RNG及posterior noise要求exact equality，浮点参数沿用原阈值。报告在`state_resume_comparison.json`和`dynamics_resume_comparison.json`。
旧日志没有差异幅度，不能仅凭它确定究竟是CUDA非确定性还是恢复错误；新报告用于区分，未提前宣称修复已通过GPU。

在可联网的测试机器执行以下完整命令，无外层括号，失败不关闭交互SSH：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
git fetch origin refs/heads/codex/object-video-sequence-v69 &&
git switch --detach FETCH_HEAD &&
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world TEST_GPU=0 \
  bash code/scripts/prepare_and_test_object_video_sequence_v69.sh
TEST_RC=$?
echo "TEST_RC=${TEST_RC}"
```

此入口自动发布当前revision、复用已有manifest并使用新的test OUT，不覆盖旧测试。它只在测试机读取本地Git HEAD；
训练入口不调用该脚本，不访问GitHub。encoder本地路径不同仍可通过`ENCODER_REPOSITORY/ENCODER_WEIGHTS`覆盖。

## 11. 八卡训练与恢复

只在用户确认单卡结果后执行。State从预训练backbone与新任务头开始，不继承V68或测试optimizer。
以下是完整State命令；不会在本次交付时自动执行：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD VENV_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
export MANIFEST="${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json"
export ENCODER=dinov3_vitl16
export ENCODER_REPOSITORY="${RUNTIME_ROOT}/third_party/dinov3"
export ENCODER_WEIGHTS="${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export STAGE=state NPROC_PER_NODE=8 BATCH_PER_GPU=2 TARGET_GLOBAL_BATCH=256
export ENCODER_FRAME_BATCH=2 WORKERS_PER_RANK=2 STEPS=30000 LR=0.0002
export RECOVERY_EVERY=250 SAVE_EVERY=2500 LOG_EVERY=20
export RUN_NAME="object_video_v69_state_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CUDA_VISIBLE_DEVICES RESUME STATE_CHECKPOINT MODEL_CONFIG STOP_AFTER WANDB_RUN_ID WANDB_RESUME
bash "${ROOT}/code/scripts/train_object_video_sequence_v69.sh"
TRAIN_RC=$?
echo "TRAIN_RC=${TRAIN_RC}  LOG=${OUT}/train.log"
```

B=2指41张原生图像的两段video，不等于2张224图；八卡accum16得到有效256。具体显存以单卡report为据，不靠盲目增加frame batch宣称吞吐提升。
Dynamics不自动跟随State结束启动：先看held state/独立binding。通过后使用下列独立命令（V69 state路径需指向实际选定checkpoint）：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD VENV_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
export MANIFEST="${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json"
export ENCODER=dinov3_vitl16
export ENCODER_REPOSITORY="${RUNTIME_ROOT}/third_party/dinov3"
export ENCODER_WEIGHTS="${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export STAGE=dynamics NPROC_PER_NODE=8 BATCH_PER_GPU=2 TARGET_GLOBAL_BATCH=256
export ENCODER_FRAME_BATCH=2 WORKERS_PER_RANK=2 STEPS=30000 LR=0.0002
export RECOVERY_EVERY=250 SAVE_EVERY=2500 LOG_EVERY=20
export STATE_CHECKPOINT="${RUNTIME_ROOT}/outputs/object_video_v69_state_${SOURCE_REVISION:0:7}/latest.pt"
export RUN_NAME="object_video_v69_dynamics_${SOURCE_REVISION:0:7}"
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset CUDA_VISIBLE_DEVICES RESUME MODEL_CONFIG STOP_AFTER WANDB_RUN_ID WANDB_RESUME
bash "${ROOT}/code/scripts/train_object_video_sequence_v69.sh"
TRAIN_RC=$?
echo "TRAIN_RC=${TRAIN_RC}  LOG=${OUT}/train.log"
```

Strict resume单独执行，checkpoint指定哪个run，就恢复哪个run；不要用新OUT冒充resume：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD VENV_ROOT=$PWD
export SOURCE_REVISION="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${SOURCE_REVISION}"
export RESUME="${RUNTIME_ROOT}/outputs/object_video_v69_state_${SOURCE_REVISION:0:7}/latest.pt"
unset CUDA_VISIBLE_DEVICES STOP_AFTER WANDB_RUN_ID WANDB_RESUME
bash "${ROOT}/code/scripts/train_object_video_sequence_v69.sh"
RESUME_RC=$?
echo "RESUME_RC=${RESUME_RC}"
```

恢复读取checkpoint的原revision、本地release、world size、microbatch/accum、manifest、模型/冻结backbone/EMA、optimizer、scheduler、
sampler epoch/cursor、各rank RNG与W&B ID。不能用8卡checkpoint宣称单卡strict resume；模型内容与数据配置也不从新环境偷偷覆盖。
每个新run会本地复制官方encoder源码到`OUT/encoder_source`，resume读取该快照，不受外部源码目录后续更改影响；不访问网络。
新拓扑或改变模型需要新实验，不自动降级为warm-start。当前run最新产物：`latest.pt`、`step_XXXXXXX.pt`、`run.json`、`progress.json`、
`metrics.jsonl`、`cases_rankXXXX.jsonl`、`model_inventory.json`、`train.log`。

## 12. 现在不能宣称的结果

- 尚无本版单卡/八卡GPU运行结果；静态审查不替代实际环境、显存、官方权重兼容性验证。
- 尚无证据自动visual queries稳定对应完整物体，weak affinity能否克服uniform/merge-all仍需独立评估。
- 原生输入不意味着小物体自动可分辨，DINO patch内部信息损失与V-JEPA成本仍需实测。
- CoTracker/relay/SAM依然是外部证据，不是真实object/visibility GT；新架构没有把错误teacher自动变正确。
- Dynamics的posterior结果不等于可部署意图选择；没有无条件多未来校准结果。
- 当前最早未完成研究项仍是G0；此版本提供完整实现与证据工具，不宣布已越过G0。

## 13. 其他证据入口的独立命令

以下各代码块互不依赖前一块的shell变量。路径中的`human_annotations.jsonl`是实际人工填写的标签，不是自动生成的GT。
模型下载仍须先单独准备完成。不会重新调用SAM/CoTracker。

### 400例motion-richness原视频检查

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=$PWD
REV="$(cat "${RT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/object_video_sequence_v69/releases/${REV}"
"${RT}/.venv/bin/python" "${ROOT}/code/scripts/review_motion_richness_v69.py" \
  --manifest "${RT}/data/object_video_sequence_v69/manifest.json" \
  --out "${RT}/outputs/v69_motion_review_${REV:0:7}" --items 400 --visualize 400 \
  --wandb_name "v69_motion_review_${REV:0:7}"
```

看`outputs/v69_motion_review_<revision前7位>/index.html`。默认2秒窗口是10秒已追踪clip内部的对比单位，不是新的训练时长。
看原视频后填写`blind_comparisons.json`的winner/reason/confound；不要先看统计挑符合分数的片段。

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=$PWD
REV="$(cat "${RT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/object_video_sequence_v69/releases/${REV}"
OUT="${RT}/outputs/v69_motion_review_${REV:0:7}"
"${RT}/.venv/bin/python" "${ROOT}/code/scripts/analyze_motion_richness_v69.py" \
  --report "${OUT}/report.json" --labels "${OUT}/blind_comparisons.json" --output "${OUT}/human_rank_analysis.json"
```

### 单卡encoder对照

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=$PWD
REV="$(cat "${RT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/object_video_sequence_v69/releases/${REV}"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"${RT}/.venv/bin/python" "${ROOT}/code/scripts/compare_pretrained_encoders_v69.py" \
  --manifest "${RT}/data/object_video_sequence_v69/manifest.json" \
  --out "${RT}/outputs/v69_encoder_comparison_${REV:0:7}" --items 400 --probe_train_items 64 \
  --dinov3_repository "${RT}/third_party/dinov3" \
  --dinov3_weights "${RT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
  --vjepa_repository "${RT}/third_party/vjepa2" \
  --vjepa_weights "${RT}/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
```

有独立标签时加`--annotations <标签文件>`。每点记录位移与可选human object extent，不能用tracker span代替真实物体尺寸。

### 训练后的held评测

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
export RUNTIME_ROOT=$PWD VENV_ROOT=$PWD
REV="$(cat "${RUNTIME_ROOT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
export ROOT="${RUNTIME_ROOT}/runtime/object_video_sequence_v69/releases/${REV}"
export MANIFEST="${RUNTIME_ROOT}/data/object_video_sequence_v69/manifest.json"
export CHECKPOINT="${RUNTIME_ROOT}/outputs/object_video_v69_state_${REV:0:7}/latest.pt"
export ENCODER=dinov3_vitl16
export ENCODER_REPOSITORY="${RUNTIME_ROOT}/third_party/dinov3"
export ENCODER_WEIGHTS="${RUNTIME_ROOT}/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export CUDA_VISIBLE_DEVICES=0 ITEMS=400 VISUALIZE=40 ENCODER_FRAME_BATCH=2
export RUN_NAME="v69_state_held_${REV:0:7}_$(date +%Y%m%d_%H%M%S)"
export OUT="${RUNTIME_ROOT}/outputs/${RUN_NAME}"
export WANDB_MODE=online WANDB_PROJECT=instruct-gs-world
export WANDB_ENTITY=healenrenss-university-of-chinese-acadmic-and-science
unset ANNOTATIONS WANDB_RUN_ID WANDB_RESUME
bash "${ROOT}/code/scripts/evaluate_object_video_sequence_v69.sh"
echo "EVAL_RC=$?  REVIEW=${OUT}/index.html"
```

Dynamics用同一入口但换CHECKPOINT为实际Dynamics checkpoint。Stage从checkpoint读取，不靠目录名称猜测。
人填好template后，使用新OUT并设置`ANNOTATIONS`为该JSONL重新评估，才会产生独立object指标。

### Tracker/visibility独立校准

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world
RT=$PWD
REV="$(cat "${RT}/runtime/object_video_sequence_v69/DEPLOYED_REVISION")"
ROOT="${RT}/runtime/object_video_sequence_v69/releases/${REV}"
"${RT}/.venv/bin/python" "${ROOT}/code/scripts/calibrate_motion_teacher_v69.py" \
  --manifest "${RT}/data/object_video_sequence_v69/manifest.json" \
  --annotations "${RT}/data/object_video_sequence_v69/human_annotations.jsonl" \
  --out "${RT}/outputs/v69_teacher_calibration_${REV:0:7}"
```

只对明确填写`tracker_point_id`的人工点测校准；没有该映射不会猜。位置和visible必须人工/仿真独立提供，不能复制tracker输出。

## 14. 导出独立迁移数据包

`export_portable_object_video_v69.sh`按现有V69 manifest快照导出，默认全量、四个CPU文件复制线程，不需要GPU。
媒体按解析后的源文件路径去重，复制原视频/原RGB cache的完整文件，不转码、不降分辨率、不插值、不重跑SAM/CoTracker。
某些MP4包含多个episode，包中会包含其中未使用的帧，文件体积可能较大；开头打印去重后的media/teacher字节数。
默认同时产生独立目录与未压缩tar，导出盘约需两份数据空间。已有压缩视频不再用gzip重复压缩。

```text
exports/object_video_v69_portable/
  manifest.json             root="."，包内相对路径
  media/00000000.mp4        原始视频或RGB cache，实际扩展名保留
  clips/00000000/teacher.pt 原始轨迹/relay/筛选等，case.record.path改为包内路径
  export_plan.json          固定复制计划与原始路径来源记录
  export_report.json        实际样本数、来源、字节数及缺失跳过
  progress.json             当前复制/归档进度，仅工作目录中
  README.md
exports/object_video_v69_portable.tar
```

运行命令可独立执行，不访问GitHub：

```bash
cd /mnt/pfs/public/xuhaoming/instruct_gs_world_v28_source
RUNTIME_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world \
VENV_ROOT=/mnt/pfs/public/xuhaoming/instruct_gs_world \
MANIFEST=/mnt/pfs/public/xuhaoming/instruct_gs_world/data/object_video_sequence_v69/manifest.json \
PORTABLE_OUT=/mnt/pfs/public/xuhaoming/instruct_gs_world/exports/object_video_v69_portable \
COPY_WORKERS=4 EXPORT_ITEMS=0 PLAN_ONLY=0 CREATE_ARCHIVE=1 \
bash code/scripts/export_portable_object_video_v69.sh
EXPORT_RC=$?
echo "EXPORT_RC=${EXPORT_RC}"
```

同一命令中断后重跑，复用已完成文件；未完成单个文件重新复制。复制计划第一次写入后固定，原数据新增不会混入同一包。
需要新的快照时换`PORTABLE_OUT`。只看容量计划可设`PLAN_ONLY=1`；使用rsync迁移目录时设`CREATE_ARCHIVE=0`，避免第二份tar占用。
缺失teacher/媒体的案例明确列出并跳过，不制造空视频。这里只复制媒体字节，不进行视频质量/解码验证；已有解码失败处理仍由loader负责。

在迁移目标机器解压，例如：

```bash
mkdir -p /data/datasets
tar -xf /data/transfers/object_video_v69_portable.tar -C /data/datasets
export MANIFEST=/data/datasets/object_video_v69_portable/manifest.json
```

使用本次支持portable路径的V69代码，并让训练命令采用这个`MANIFEST`。新run会把解析后的数据根写进自己的manifest快照；
不依赖原`/mnt/pfs/public/...`目录，原始路径只留在provenance中，不会作为运行输入读取。
数据包不包含DINO权重、Python环境或训练checkpoint。若还迁移旧checkpoint，checkpoint的run OUT、dataset快照和encoder源码路径
属于另一项恢复迁移，不能仅改新的`MANIFEST`就声称旧run完成strict resume。
