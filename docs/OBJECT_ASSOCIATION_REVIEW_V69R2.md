# V69r2：Object Association审阅、修正与验证

日期：2026-09-29。分支保持`codex/object-video-sequence-v69`；已提交八卡任务的命令不变。
当前架构名为`pretrained_query_object_video_sequence_v2`，属于V69接口修订，不另开研究主线。
本轮只有实现和静态审查；没有新的服务器运行或学习效果结论。

## 逐条判定

| 审阅意见 | 判定 | 处理 |
|---|---|---|
| 2.1 weak affinity/同track一致性不等于完整物体关联 | 成立，是科学风险；目前没有已验证的同实例teacher | 增加独立多query/多测量位置诊断，不用新的未经验证聚类强造same-object标签 |
| 2.2 identical anchors下全局effect attention对交换归属不敏感 | 明确结构反例 | 每层组先query-local effect conditioning，再跨query interaction；不再依赖anchor可区分 |
| 3.1 Posterior漏掉state显式centers | 明确接口不对称，但不能据此说tokens完全无motion | 加相对source root变化和帧内carrier布局，保留tokens-only受控开关 |
| 4.1 应有对象条件下的测量，不只overall EPE | 成立；原版已经有逐case/point EPE，但没有这些对象条件诊断 | 外部object/region标签下测common-location一致性、per-query/region EPE及额外测点 |
| 4.2 effect使用与effect归属要分开 | 成立 | 联合tuple重排、仅交换effect、单query替换分别记录；不要求交互后其他物体完全不变 |
| 4.3 query probability点积偏好单query集中 | 明确评测错误 | 先按独立entity标签合并query概率，unbound/unknown不作为共同物体，且不重归一化剩余mass |
| 4.4 z不必完全object-independent；mean KL不是总rate | 成立，但平均KL作为正则本身没有算错 | 不强制latent相等或identity剥离；保留原loss尺度，另记clip总KL、query KL及同实体query总KL |

另需澄清：“第一阶段codec”的术语不能覆盖当前两个阶段。**State阶段冻结Posterior和Dynamics**；新增几何输入和local effect routing
用于之后的Dynamics/变化编码阶段。当前排队的State训练仍使用原来的State学习目标，不能因为后验接口被修好就宣称query–object语义已解决。

## 修改后的计算

### Geometry-complete Posterior

对每个query k、时刻tau、carrier r：

$$g_{\tau,k,r}=\left[c_{\tau,k,0}-c_{0,k,0},\;c_{\tau,k,r}-c_{\tau,k,0}\right]\in\mathbb R^4.$$

4维geometry经同一Fourier编码变36维，再由Linear36→512加入latent/time context。
第一项保留整体translation；第二项保留帧内local组织。它来自Object Memory自己估计的state，不是tracker GT flow或真实物体center delta。
将整个source/target序列同时平移不会改变这个几何输入，单独移动future root会改变它。

新增18,944参数。Posterior共12,706,432；Dynamics仍51,290,114；含EMA、不含backbone的任务模型共120,296,638参数。
State可训练29,211,699，Dynamics阶段可训练63,996,546。Latent预算仍是每query4×64，不是每物理物体已经确认的256维。

### Local effect，再interaction

每个层组先执行：

$$\widetilde U^k=\mathrm{CrossAttention}(U^k,\mathrm{Embed}(z^k)),$$

再对所有queries的tokens做interaction。第一步把`[B,K,R,D]`变为`[B*K,R,D]`，effect context为`[B*K,4,D]`；
不同queries在这一层不会读到彼此的effect。下一步恢复`[B,K*R,D]`建模交互。
相同anchor不影响effect路由；k只标识当前输入tuple的对应关系，不是永久物体ID。
这个结构允许改变A的effect经interaction影响B；不能把physical interaction误判成绑定泄漏。

### 多query的entity评测

人工标签给出query k属于哪个entity。先计算：

$$P_{p,o}=\sum_{k:\;label(k)=o}p_{p,k},\qquad A_{ij}=\sum_o P_{i,o}P_{j,o}.$$

两个点在同物体的两个queries上各分0.5时，合并后都是1，不再被错误惩罚。
Unbound/unknown不进入entity sum，不把null mass重归一化掉；background/scene作为context，不自动定义成一个物理物体。
另外保留unbound mass与可评测点覆盖，避免不同实体pair得低agreement就掩盖all-unbound退化。
反例包括merge-all、uniform-query、all-unbound；正确的same-entity split queries作为零误罚参照，不当坏解。

## 新诊断具体计算什么

1. **同一物体多个queries**：在相同历史reference位置与相同future测量点上，读出各query自己的continuous field。
   比较各自对GT的像素误差和彼此差异，不比较latent z是否数值相等，也不要求不同位置具有相同位移。
2. **物体内不同区域**：人工`region_id`如cap/body，分别报告query→各区域的原始EPE分布。
3. **新增测点**：`measurement_set=primary/additional`共享已编码state和effect；不重跑encoder/Posterior。
   对相同点子集的读出差异是接口一致性检查，额外点的GT误差才是泛化证据。没有把tracks送入encoder再拆一半。
4. **Effect归属**：固定state，单独替换一个query effect，及交换两个不同实体代表query的effect；对所有有独立标签的实体记录影响和GT误差。
   不假设不同物体effect完全可交换，也不把其改变后的输出当作已知物理counterfactual GT。
5. **静止案例**：用人工`case_tags=static`标识实际无变化片段，记录真实位移和正确Posterior重建误差；不规定z=0必须静止。
6. **Rate与采样**：训练额外记录`effect_clip_kl_nats`、`effect_query_kl_nats`、`effect_valid_query_count`；
   evaluator记录每query KL、每实体的query数量/总KL、默认4次posterior采样的重建与spread。
   KL在FP32中计算，采用`expm1(logvar)-logvar`避免小值相减损失精度；原mean-KL正则权重不改。
   所有rate都是tanh前Gaussian KL的nats，不是真实压缩文件bitrate；spread不是不确定性校准证明。

每query reconstruction是单query条件field的能力诊断，区别于正常ownership mixture的部署输出；两者在报告中分开命名。
由reference校准保证完全重合的那一帧不计入关联重建成绩，避免把已给定的reference位置作为独立成功证据。
没有独立labels时对象诊断标`not_measured_no_annotations`，不会从tracker或模型mask生成替代GT。

## 小型独立诊断集

不要求标注全部训练数据。使用仅用于评测的少量case，至少覆盖：

- 同一刚性物体两个以上可见位置的queries，含translation与rotation；
- 两个外观相似、但变化不同且相互分离的物体；
- 接触/耦合运动，不能要求跨对象影响为零；
- 实际静止片段与遮挡重现；
- 同一物体primary和additional测量位置，不仅标原来的训练teacher点。

沿用evaluator生成的`annotation_template.jsonl`，新增可选`region_id`、`measurement_set`、`case_tags`字段。
Query必须来自已观察history；unknown位置/visibility保持unknown，标签来源必须明确为人工或仿真而非训练tracker。
W&B保存query/region误差、common-location一致性、effect干预、entity grouping和rate表，完整逐点值在同步artifact中。

## 工程测试与受控对照

单卡整链路测试新增同anchor effect交换、local条件化隔离、joint tuple permutation、geometry-only posterior变化、
共同translation不变性、tokens-only geometry不敏感、多query同实体归并、unbound排除以及固定state测点子集一致性。
这些用完整模型模块验证接口，不证明真实objects已学成；原3s/5s真实数据两阶段训练、resume对照和evaluator仍运行。

Geometry对照从**同一新版State checkpoint**分别启动Dynamics，固定数据/seed/latent维度/readout：
`POSTERIOR_GEOMETRY=on`与`POSTERIOR_GEOMETRY=off`，使用不同OUT。Off分支保留同样参数容量，但geometry贡献为0；
这不是直接把训练好的geometry Posterior在推理时去掉输入就当作受控训练对照。Resume读取checkpoint保存的选择，不从环境偷偷切换。

## 已提交八卡任务与checkpoint

- 继续使用原V69 shell入口、环境变量、八卡数、原生数据与默认训练阶段；无需重提任务。
- 测试机拉取后，`prepare_and_test_object_video_sequence_v69.sh`先发布完整新release，再原子更新`DEPLOYED_REVISION`。
  未启动任务实际开始时读取新版；已经启动的进程不会被热修改。
- 新的fresh run使用architecture v2。已有v1 checkpoint按原release恢复，不把旧checkpoint重命名成新版；本轮不作隐式warm-start。
- State loss权重、encoder、分辨率、3s/5s、latent维度都没有因为审阅而做调参；当前最早未解决的科学问题仍是query–object关联有效性。
