# V69代码审查记录

日期：2026-09-28。审查对象：`codex/object-video-sequence-v69`相对V68 `991099e`的实现。
方式：本轮实现者逐模块静态审查；不是独立审稿人，也没有本地或服务器GPU运行结果。
结论：本轮发现的实现问题已修正，可交付用户做完整单卡运行；不宣称端到端测试已通过或object学习已成立。

## 已修正项

| 级别 | 问题 | 修正后的行为 |
|---|---|---|
| P1 | 无归属的effect token集合会对对象间effect置换保持不变 | Dynamics将每个effect与对应query anchor绑定，且显式输入carrier位置 |
| P1 | Future field沿用旧全局像素坐标会改变其object-local含义 | reference时刻固定local坐标、ownership和读出偏差，随未来state读出 |
| P1 | normalized坐标loss使高分辨率小运动梯度被缩小 | 主transport直接优化native-pixel Charbonnier，报告native EPE |
| P1 | V-JEPA完整clip早期token或末帧缓存可包含不应有的时间信息 | 每时刻使用真实prefix；reversed/last-frame对照重新编码各自RGB |
| P1 | DINO feature有效性错误地决定raw轨迹是否受监督 | 位置loss只使用原始轨迹/point/motion证据；外观loss单独用feature validity |
| P1 | 随机DataLoader iterator创建消耗模型RNG，破坏strict resume | 独立loader generator，恢复模型RNG、sampler epoch及cursor；单卡测试与未中断版本对比 |
| P1 | Dynamics初始化若把EMA重置为online，会丢掉已建立target | 继承state阶段保存的EMA并冻结 |
| P1 | 视同所有shuffled effect为错误GT会惩罚相同真实变化 | 只作诊断，不加任意negative-margin训练项 |
| P2 | Duplicate采样可能丢掉最后当前帧 | history/future分别保留重复序列最后一次，current始终保留 |
| P2 | 不存在的独立标注可能被报告成通过 | 标记not_measured；仅明确人工/仿真标签进入独立评测 |
| P2 | 删除component只看任意feature差无法解释局部贡献 | 报告在人工分组的当前点上，对实际冻结feature目标误差增加的内外分布 |
| P2 | 变更encoder名字但仍默认使用DINO路径 | test/train/eval三个launcher均按backend选择本地默认路径 |
| P2 | 运行中更改官方encoder源码会破坏恢复复现 | 每run本地复制encoder源码；resume使用原快照；模型权重也写进checkpoint |

## 审查覆盖

- 数据构建和训练采样分离；episode split、原始坐标、真实PTS、pixel/teacher/loss masks、未知标签和坏视频路径。
- DINOv3及V-JEPA2.1官方factory、weight key、forward返回结构；没有网络fallback或随机backbone替代。
- Query、Memory、Posterior、Dynamics、readout的shape/dtype、梯度连接、effect归属及未来因果边界。
- Loss目标来源与单位；BCEWithLogits、无全slot identity假设、无Gaussian矩阵求逆。
- 单卡/八卡前台launcher、梯度累积、DDP共同跳过坏batch、原子checkpoint、manifest快照、恢复W&B ID与数据位置。
- Held逐例评测、独立人工点、错误binding反例、motion盲评、teacher校准、encoder probe及W&B产物。
- 删除未使用且可能误用缓存video token的旧history-ablation helper；不清理或改写历史V28–V68实验模块。

静态检查：新Python及共享decoder通过AST parse；Ruff `F,E9`通过；四个shell入口通过`bash -n`；提交前执行`git diff --check`。
静态计数：27个Python文件被解析，未import模型、未运行forward、未做本地训练。
只有一个整链路单卡集成测试入口，没有为每模块堆叠阻断生产的gate。

## 仍需实际证据

1. 官方权重/源码和现有Python环境是否兼容、80GB GPU原生输入的真实峰值显存和吞吐。
2. 单卡训练/恢复/未中断对照/evaluator能否全部完成。单卡不能替代八卡NCCL执行结果。
3. Weak appearance affinity是否仍有uniform、merge-all、fragmentation捷径；本版不把静态接口称为成功object监督。
4. 原始tracker的accuracy与visibility可靠性，小物体/相机运动/远距离与腕部视角遗留问题。
5. 旧配额数据上的episode-uniform不是原始全库uniform；已有轨迹不能补回未追踪区域。
6. V-JEPA prefix重算的计算成本与小物体质量；不从公开benchmark推定其在本项目优于DINOv3。
7. Posterior-conditioned reconstruction不是部署期effect选择，未实现selector不伪装成自主未来预测。

这些是待测结论，不是静态审查可以签发的能力证明。运行及解释见`OBJECT_VIDEO_SEQUENCE_V69_RUNBOOK.md`。
