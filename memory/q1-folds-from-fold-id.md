---
name: q1-folds-from-fold-id
description: 下游复用 Q1 外层划分时读 q1_main_oof.npz 的 fold_id，不调用 outer_splits 重算；另记 mathorcup 环境须设 MKL_THREADING_LAYER=SEQUENTIAL
type: constraint
---

问题 2、问题 3 复用问题 1 的外层划分时，从 `code/data/processed/q1/q1_main_oof.npz` 的 `fold_id`（5 × 250）重建训练/测试下标，并与 `chosen_params` 的 `{重复}_{折}_{组}` 键配对，不调用 `src.q1.splits.outer_splits` 重算。

`mathorcup` conda 环境（sklearn 1.9.1、numpy 2.4.6、MKL）运行前须设 `MKL_THREADING_LAYER=SEQUENTIAL`。不设时，首次调用 BLAS 的进程直接退出，返回码 127，不打印错误。

**依据：** 2026-10-29 Q2 正式重跑。在 sklearn 1.9.1 下调用 `outer_splits(date_group, style, 5, 5, seed=0)`，第 0 次重复第 0 折已与 Q1 的 `fold_id` 不一致；Q1 正式运行使用 sklearn 1.7.2（`q1_main_oof.npz` 的 `meta`）。`StratifiedGroupKFold` 的打乱随版本变化，而 Q1 的逐折参数只对原划分有效。MKL 问题由 `python -X faulthandler` 定位，崩溃点为 `numpy.corrcoef`，异常码 0xc06d007f；改用 `SEQUENTIAL` 后正常，`INTEL`、`GNU` 仍崩溃。

**如何应用：** 跨问题复用划分或逐折参数时，以 Q1 产物中保存的划分为准，运行时核对每折训练、测试日期组不相交。换环境或升级 sklearn 不得改变下游划分。在 `mathorcup` 环境运行任何求解脚本前先导出该变量；返回码 127 且无输出时先检查此项。

相关：[[method-decision-closure]]
