# Object Video Sequence 改造计划

日期：2026-09-27；2026-09-28更新。状态：V69代码已实现并做静态审查，尚未执行GPU实验；审阅基线为V68 `991099e`。
本文对应永久账本第15.43节，不另开研究主线。用户固定的时间边界：history最长3秒，future最长5秒。
完整实现的模块尺寸、loss及独立运行指令见[执行文档](OBJECT_VIDEO_SEQUENCE_V69_RUNBOOK.md)。

## 1. 总体决定

- 使用已有预训练视觉encoder，不重新训练一个像素CNN或随机初始化ViT作为视觉基础。
- 主方案为冻结DINOv3 ViT-L/16；V-JEPA 2.1 ViT-L/16是一个可替换的受控对照，不同时堆叠两套encoder。
- SigLIP2暂不进入主训练链路。现有DINOv2/SigLIP2结果作为历史对照保留，不静默修改旧数据或checkpoint。
- 输入是一段history，目标是一段future object-state/trajectory sequence，不再只有两个端点。
- 维持纯视频、无显式机器人action、无语言、无RGB重建主目标。连续latent-effect posterior保留，部署selector仍后置。
- 原始RGB、现有10秒点轨迹、背景/relay诊断、坏视频跳过机制保留。RoboTwin只重处理已有轨迹，不重新追踪。

## 2. 当前实现审阅

| 问题 | 代码证据 | 影响 |
| --- | --- | --- |
| 预训练模型只作为外观teacher，Student仍从像素学习浅层CNN | grounded_object_transport_v68.py:34；grounded_appearance_teacher_v68.py:44 | 已有视觉先验没有直接用于Student输入编码 |
| history只有1-4帧、最多约0.3秒，future只有1秒/3秒两个位置 | grounded_motion_dataset_v68.py:34 | 长轨迹数据没有变成长时序训练样本 |
| 固定18个global slots，不是账本中的query-conditioned object state | grounded_object_transport_v68.py:38、114 | slot数不等于物体数；同一index不证明同一物体 |
| 角色与region标签全unknown后，两项loss实际失效 | grounded_motion_dataset_v68.py:70；grounded_object_transport_v68.py:169 | 不能继续把旧角色/区域监督当成有效约束 |
| 归属JS允许各点均匀分配的解 | grounded_object_transport_v68.py:165 | 所有点在18个slots上均匀分配时，跨帧JS为0；不同点归属点积为1/18。是目标捷径，不是本轮已观察到的训练结果 |
| 查询集合及当前位置来自全视频tracker和未来75%筛选 | grounded_motion_dataset_v68.py:43 | history encoder未读未来，不代表部署查询接口已独立于未来；评测还有选择偏差 |
| 只从当前可见点建立查询集合 | grounded_motion_dataset_v68.py:44 | history中见过、当前遮挡、future重新出现的点缺少直接训练覆盖 |
| visibility辅助目标把不可见、越界、非有限位置混为false | grounded_object_transport_v68.py:206 | 不可观测不等于不存在，tracker失败不等于真实遮挡 |
| 运动排名用背景补偿坐标，transport监督用原始图像坐标 | grounded_motion_dataset_v68.py:81；grounded_object_transport_v68.py:195 | 目前实际预测包含相机运动，不是纯物体相对运动 |
| zero effect被结构性固定为零位移 | grounded_object_transport_v68.py:137 | 胜过zero不单独证明学会使用effect；必须看shuffled和held干预 |
| 来源配额、group轮换、中间截窗和训练来源重复 | grounded_motion_sources_v68.py:176；grounded_motion_dataset_v68.py:124 | 不是全体episode等概率，也不能证明运动丰富度筛选有效 |

本轮是静态审阅，不把上述风险写成已观测的服务器实验结果。最早未完成项仍是G0的teacher/目标有效性。

## 3. Encoder选择依据

### DINOv3作为主方案

选择官方预训练ViT-L/16，约300M参数，先冻结。官方定位包含dense features、局部对应、分割和视频分割追踪；
这里需要的是保留可定位的局部证据，而不是先获得一句场景语义。逐帧提取使任意时间前缀可独立编码，
future target可以复用同一帧特征，不需要每个目标时刻重复运行整段双向video encoder。

这是任务适配判断，不是已经证明DINOv3在我们的五源数据上优于V-JEPA。DINOv3是image encoder，
仍需学习一个小型causal temporal object aggregator；不把静态语义相似度当成运动表征。

输入保留原生分辨率和宽高比，仅做必要padding；分块按帧调度冻结encoder，不把全图强制缩到224/384。
如果原生整帧资源成本无法接受，使用覆盖全视野的重叠native-scale tiles，并单独验证接缝与跨tile关系，
不能未经检查就把裁剪feature当成整图feature。小于一个patch的细节不能靠插值声称恢复，需用小物体实测确认。

参考：[官方实现](https://github.com/facebookresearch/dinov3)、[模型说明](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)。

### 为什么比较V-JEPA 2.1而不是直接叠加

V-JEPA 2.1已经发布ViT-L/16 384预训练权重，并明确改进dense和时序一致性。它是有实质理由的video encoder对照，
不应沿用旧V-JEPA 2局部特征不足的结论替它下判决。但公开任务成绩不等于我们的小物体、原生尺度和时序状态任务成绩。
若使用它，当前state只能编码以当前时刻结束的历史clip；需要早期时刻state时必须用那个时刻的前缀，
不能取整个8秒video双向编码后的早期token冒充因果state。历史clip内部的双向attention可以使用，因为全部已经观察到。

对照只替换encoder，固定数据、查询、分辨率映射、聚合头、监督和评测。报告参数、token数、延迟和显存；
不把不同分辨率或不同读出容量导致的差异记成encoder能力。优先比较冻结feature的定位、跨帧对应、小物体可读性及简单动态probe。

参考：[官方权重列表](https://github.com/facebookresearch/vjepa2#models)、[V-JEPA 2.1论文](https://arxiv.org/abs/2603.14482)。

SigLIP2 NaFlex具有语义与分辨率适配能力，但这里不需要语言接口，也没有证据需要把两套appearance目标平均成一个loss。
NaFlex按token预算调整分辨率，并不等于保留所有原生像素。第一轮不叠加；后续仅在独立语义需求出现时引入。
参考：[官方模型卡](https://huggingface.co/google/siglip2-so400m-patch16-naflex)、[接口说明](https://huggingface.co/docs/transformers/model_doc/siglip2)。

## 4. 数据与时间协议

### 视频选择

新增/未来数据按独立episode等概率抽样，episode内先均匀选起始位置；MP4可能打包多个episode，不能按文件数平衡。
训练在已收录episode之间等概率，不再通过重复小来源来强制五源各20%。来源数量分布照实记录。
现有20k候选是旧配额选出来的；修改训练sampler只能在已有集合内均匀，不能追溯声称它是全库均匀样本。
这一点不要求重做已有RoboTwin。现有生成任务和新采样实验分开版本化，运行中的清单不修改。

### Motion-richness先证实再启用

冻结一批跨五源、含小物体/手臂/相机运动/静止/遮挡/漂移的原始片段，报告每个候选窗口的原视频、轨迹、
相对运动幅度分布、覆盖范围与持续时间，不先将这些量随意合成单个分数。盲评窗口对“哪个含更多有效物体变化”，
再检查候选排序与人工判断是否一致、错误类型及被丢弃的真实小运动。留独立片段确认，不能只看挑选后的好看视频。
确认后只在同episode内改变窗口偏好，保留均匀窗口作为覆盖对照；静止、等待和遮挡不能全部被清除。
当前75%是点级loss采样政策，不是已经验证的video-level motion-richness评分。

### 时序样本

首轮默认在有效8秒窗口内安排history 3秒、future 5秒，可复用10秒轨迹。history使用16帧，覆盖t0-3至t0，
future使用25个时刻，覆盖t0+0.2至t0+5.0秒；16/25是初始计算预算，不是最优帧率结论。
保留原视频时间戳和原生逐帧轨迹，RGB按请求时间选择真实帧，不制造插值GT。数据帧率不足或时间戳重复要如实呈现。
需要训练不同观测长度时，从较短历史到3秒独立抽取，不能退化为默认单帧；future请求长度独立抽取到5秒。
不根据未来运动、tracker成功率、终局或剩余episode长度决定history帧数/间隔。只在具有所需观测/监督窗口的锚点上取样，
请求跨度显式传入，真实终局剩余时间不传入。未来缺测使用loss mask，不据此改变history。

## 5. 单一Student的数据流

```text
RGB history (observed only) + current point/region query
    -> frozen pretrained DINOv3 features + spatial coordinates + actual timestamps
    -> shared query-conditioned temporal Object Memory
    -> compact state at t0
    -> latent-effect-conditioned object sequence Transformer
    -> future object states over 0-5s
    -> shared compositional transport/support readout

Full RGB/track clip (training only)
    -> frozen correspondence and observation evidence
    -> future supervision at readout coordinates; never history inputs
```

Object query的作用是指定当前需要绑定和持续表示的实体，不是任意一个固定index。query来自已观察到的点/区域输入，
同时记录其历史帧时间；可以由部署策略指定，或由只读已观察图像的proposal方式提供，不使用全视频motion排序决定student query。
history里见过但当前已遮挡的实体保留query和memory，当前观测为unknown；不因当前不可见直接删掉它。
从未在history出现的新实体不在首轮可查询对象范围内，不把其缺失误称为完整世界状态预测成功。
同一共享网络处理多个queries，不拆成两个Student，不给不同对象建立独立encoder。
原来固定的16+2角色划分退出语义约定；K表示并行实体查询数，不声称恰有K个物体。
这项本身是对账本主线的恢复，不是声称已证明同一物体的多个queries能合并。

设B为batch，Th/Tf为history/future采样时刻数，N为一帧视觉token数，K为当前实体queries数，P为监督测量点数：

| 模块 | 输入 | 输出与含义 |
| --- | --- | --- |
| Frozen perception | [B,Th,3,H,W]，原生图像与padding mask | DINOv3-L视觉feature [B,Th,N,1024]及原生坐标；不是object GT |
| Query temporal aggregation | feature、时间、current queries及其观测支持 | [B,16,9,512]；每query一个anchor与8个变化carriers，不给任意维度强命名物理属性 |
| Target observation | 截止各未来时刻的已观察prefix和冻结/EMA聚合头 | [B,25,16,9,512]；同一个显式query条件，不把相同index自称为真实物体ID |
| Effect posterior | 当前与匹配future latent序列 | 每个实体少量连续effect tokens；训练期允许看未来，不拼接GT坐标增量或RGB差 |
| Object sequence Dynamics | compact state、effect [B,16,4,64]和请求时间 | [B,25,16,9,512]；时间展开在object tokens上，不在全图patch上展开 |
| Transport readout | 预测state、当前测量点及其当前绑定 | [B,Tf,P,2]预测位置及单独的observability预测 |

把固定18个slots扩成更多并不能替代容量验证。完整实现采用root+8个局部carrier作为首轮容量预算，
它尚未通过实际可读信息验证，不宣称最优；不同时重启Gaussian/RGB重建路线。

## 6. Teacher、mask与因果边界

- Student object queries和任何部署时必需的support，只能由history/current得到。全视频teacher仅提供目标、测量点和loss权重。
- 为复用旧轨迹，可以在teacher保存的点上评估预测field，但这些点不能参与构造Student state，点集之间也不能经readout attention把选择信息传给Dynamics。
- 独立评测使用事先固定的current queries，不用未来75%筛选来挑容易评估的点。报告原始候选数、已观测数、teacher覆盖和各类缺测。
- SAM区域仍是采点候选，不作为完整物体GT；当前没有可靠部件合并证据，禁止仅因接触就做mask union。
- 分开pixel padding、history presence、tracker evidence、loss weight、predicted visibility。只把观测证据明确的情况监督可见/不可见；证据冲突为unknown，不当absent。
- 背景homography和relay仅是诊断证据。现有3像素阈值先与人工稀疏位置、遮挡标注比较，不把一致性率当准确率。
- 原始像素轨迹保留为可测量的主目标，相对局部位移为辅助；相机运动单列评估。背景配准失败不自动抹掉所有原始轨迹监督，不声称已恢复真实相机运动或3D物理位移。
- encoder替换、teacher配置和采样版本与当前任务隔离。运行命令只加载本地代码/权重，不访问GitHub；W&B记录保持在线。

## 7. 优化目标与训练顺序

不一次堆出大量命名loss。每项对应可观察问题：

1. **序列transport**：在所有有可靠目标的未来时刻比较预测与对应点位置，保留逐时刻像素误差；不是只比较端点或相对persistence收益。
2. **Query binding/persistence**：验证query关联的支持在时间上是否持续，以及明显无关支持能否被排除；uniform、merge-all、track-per-object都是必须比合理解更差的反例。
   不用slot index自监督，不用仅全slot cosine定义identity；可疑relation只作弱证据，不强制为物体标签。
3. **冻结外观辅助**：在实际对应位置比较预训练feature，帮助保留外观；不把任意MLP后的距离改名为object语义，不重建整幅dense未来feature图。
4. **Observability**：有证据才监督visible/occluded，unknown不参与负类；visibility不能让模型自主关闭主要trajectory loss。
5. **Effect与rollout**：正确posterior、错配effect和context-only读出共享相同容量比较；去掉“zero effect被代码强制不动”作为主要成功依据。
   直接与分段路径都先受真实序列约束，再加同一时间点的路径一致性，不能只让两条错误路径相互接近。

先训练query-conditioned视频state和对应读出，固定backbone；在state确有独立binding/保持证据之后冻结或低LR保持state，
训练posterior-conditioned未来序列。rollout使用自身预测state继续，不把每一步真实未来state喂回Dynamics并称为部署性能。
未来真实RGB仅用于target和posterior，不能跨越attention或KV cache边界进入Student。History-only Prior、语言和显式action不在本轮加入。
5秒未来有多解；posterior重建能力和history-only预测能力分开报告，不承诺无意图条件能唯一预测操作。

## 8. 执行顺序与判断材料

### A. 数据与encoder证据包

在当前已完成轨迹上做重处理，RoboTwin不重新追踪。五源逐例材料包含原视频、history/future分界、3s/5s采样时间、
原始与保留点、relay两次轨迹、失败位置、候选窗口运动统计。不能因整段坏视频就将其他来源停止。
选一批只用于评测的稀疏人工点和遮挡标注，不用于训练；冻结DINOv3-L与V-JEPA2.1-L比较小物体定位、对应、
遮挡重现和简单动态可读性，按来源/物体像素尺度/运动幅度分组，同时报告算力。论文排名不替代这一步。

### B. 时序state实现

替换视觉前端、重写sequence loader、实现query-conditioned causal aggregation；保留现有readout作明确对照。
检查history-only输入不随future变、时间顺序/历史长度确有可测量影响，以及uniform assignment不是低损失捷径。
删除query/state component后的误差区域由外部选定点衡量，不由模型自己的mask定义成功。

### C. 未来序列学习

只在B说明state可用后启动一次明确的序列训练：最长3s history、5s future、连续effect posterior，
比较每一未来时刻的绝对轨迹误差、误差分位数、遮挡/重现、直接/rollout以及错配effect。
除了同tracker held结果，还保留人工稀疏GT；没有独立证据时只能声称teacher拟合改善。
报告总encoder+aggregator+Dynamics耗时/显存和单次latent rollout成本，不能只用token少宣称整个系统计算更少。

### 实现边界

计划新增/替换：episode_uniform_sampler、motion_richness_review、object_video_sequence_dataset、pretrained_visual_encoder、
query_object_video_encoder、object_sequence_dynamics、object_sequence_objective、object_sequence_evaluator。
复用现有decoder、原视频映射、离线teacher读取、W&B/前台launcher和resume工程。下一实现checkpoint升级，
不把V68的CNN/18-slot optimizer直接resume为新架构；旧结果始终保留为baseline。权重与依赖下载仅在独立准备阶段进行。

本版还提供完整单卡两阶段训练/恢复/评测测试、400例motion证据包、独立人工点标注评测和encoder对照代码。
实现不意味着encoder已选优、teacher已正确、数据已全部完成，或任一新模块已经在服务器验证。
