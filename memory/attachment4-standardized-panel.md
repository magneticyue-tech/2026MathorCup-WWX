---
name: attachment4-standardized-panel
description: 附件 4 是原始评分按品评员 × 维度标准化后的啤酒均值；仅凭附件 4 无法还原 0–6 分
type: constraint
---

附件 4 由原始评分按“品评员 × 维度”标准化（使用该品评员的全部评分，样本标准差 ddof=1），再对每款啤酒的全部评分取平均得到。在某维度上评分恒为 0 的品评员，附件 4 对其赋了逐维常数，赋值规则未识别。

**依据：** 用 Zenodo 23624915 的原始评分按上述规则重建，复现 11,274/12,500 个单元（90.2%，误差中位数 2.8e-10）；其余 1,226 个单元全部来自 15 个零方差的“品评员 × 维度”组合（`experimentalCode/q1_raw_target/q1_raw_data_checks.py` 检查 B、C）。

**如何应用：** 标准化去掉了 0–6 的位置和尺度，没有原始评分时，无法在不加假设的情况下还原 0–6 分。Q1、Q2 主线以原始评分为感官目标；附件 4 在 Q1 只用于对照 C1、C2（`model/q1_methodology.md` §1.3），在 Q2 只用于敏感性臂（`model/q2_methodology.md` §7.3）。

相关：[[data-provenance-zenodo]]
