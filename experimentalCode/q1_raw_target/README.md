# 问题 1 原始评分目标的探索性试验

`model/q1_methodology.md` 中标为“预运行”的数字由本目录脚本生成。输入为 `data/raw/`，输出写入 `experimentalCode/results/q1_raw_target/`。试验使用固定超参数（深度 3、学习率 0.05、200 轮、叶节点 20、l2=1）、`GroupKFold(5, shuffle)` 按日期组划分、3 次重复（种子 0–2）；正式运行按方法文档 §5 重新计算。

在项目根目录运行：

| 顺序 | 脚本 | 内容 | 方法文档 |
|---|---|---|---|
| 1 | `q1_raw_data_checks.py` | 检查 A–G：编号对应、附件 4 重建、零方差品评员、barnyard、日期组、出席偏移、信度 | §1–§2、§5.5 |
| 2 | `exp_targets.py` → `eval_targets.py` | 训练目标比较：$\bar x$、$\alpha$、$z$ 映射、B0、B1 | §3.1、§5.4 |
| 3 | `exp_discretize.py` → `eval_discretize.py` | 按评分数加权 | §2.2 |
| 4 | `check_reference.py` | 线性参照与真值、信度 | §5.3、§5.5 |
| 5 | `run_fold.py REP K`（15 折）→ `merge_folds.py` → `eval_twostage.py` | 一阶段与 C1、C2；混合参照 R 的误差 | §5.3 |
| 6 | `run_fold_1s.py REP K`（15 折）→ `eval_rules_1s.py` | 离散规则比较 | §4 |
| 7 | `run_fold_k2.py REP K`（15 折）→ `eval_k2.py` | K2 对照、混淆矩阵、RPS、对数损失、区间覆盖 | §4.1、§5.3 |

逐折脚本可并行：`for rep in 0 1 2; do for k in 0 1 2 3 4; do echo "$rep $k"; done; done | OMP_NUM_THREADS=1 xargs -P 15 -L 1 sh -c 'python experimentalCode/q1_raw_target/run_fold_k2.py $0 $1'`。`exp_twostage.py` 直接运行时使用 joblib 多进程，在本机出现工作进程被终止，改用 `run_fold.py` 逐折运行。
