# 官方原始输入核验与 PC-Gym CSTR 开发评估

## 目的与边界

1. 在原 ICDS meta_val 固定行上，核验官方 TabICL 的自然原始输入与已有 typed 输入。
2. 将现有 hr0 CAQ / TabICL-FT best.pt 接入独立方程来源的工业过程，测量无需梯度更新的迁移表现。

本次不训练神经网络，不使用 external_test 选择模型。已有外部数据及旧版模型结果曾用于历史研究，因此 external_val 是开发评估，不能称为从未接触的最终外部测试。

## 输入与共同预测任务

ICDS：context=960，query=256，seed=42，meta_val 26 worlds；完全复用原 fixed_tasks 的行号和 typed 目标（连续状态增量、count/binary 下一值）。

- `official_raw`：状态原值、实际动作维度、可见外部条件、公开设备参数与可用性标记、dt；若 observation 与 state 不同则保留 observation，避免减少原协议的公开信息。没有 selector、anchor、重复缩放状态列或8维动作补零；标签使用未标准化物理目标，交给官方 fit 自行处理。
- `official_typed`：复现 compare_three.py 的官方8-estimator包装与既有预处理。
- `caq` / `tabicl_ft`：保留各自训练输入协议，加载 hr0 的 best.pt，不更新权重。
- `ridge_raw`：同一原始输入，在context上拟合 StandardScaler + Ridge(alpha=1e-5)，float64、SVD求解。与旧脚本未标准化设计矩阵的 ridge 不应混为同一数值基线。
- `boosting_raw`：逐目标 HistGradientBoostingRegressor，100轮、15叶、min_samples_leaf=10、seed42，不根据query调参。
- `persistence`：连续状态变化量预测0，count/binary预测当前值。

官方模型检查点、集成数8、随机种子42固定。所有模型在同一任务上看到相同context标签和相同query输入；并非强行要求相同数值编码。

## 外部任务筛选

| 候选 | 信息条件 | 本轮选择 |
|---|---|---|
| PC-Gym CSTR | 完整浓度/温度，单个冷却温度动作，连续非线性动力学 | 选定：接口清晰、两目标可解释、可核验数值积分 |
| PC-Gym multistage extraction | 完整10维浓度、两动作 | 可作为后续任务，本轮不用于选择有利结果 |
| Industrial Benchmark | 部分可观测、随机系统，需要历史 | 暂不优先：与目前full-state模型存在额外信息条件差异 |

PC-Gym 来源：

- 作者仓库：https://github.com/MaximilianB2/pc-gym
- 论文：https://arxiv.org/abs/2410.22093
- 本地冻结 commit：`d26cbc0690024efef488dc6a304d37654934e525`
- 数据 manifest：`/home/lizitao/project/Industrial_synthetic_data/data/external-m4-v2/manifest.json`

数据使用PC-Gym原始CSTR方程、独立采样/积分脚本。它不是ICDS thermal/tank方程，也不是PC-Gym官方Gym奖励基准。反应物浓度Ca和温度T为公开当前状态，冷却液温度Tc为动作；q、V、UA等变化参数保留隐藏。输入原始表为 `[Ca, T, Tc, dt_seconds]`。

积分步长0.05分钟，统一模型时间特征为3秒。该转换不改变原始转移或动作轨迹。温度用K、浓度用mol/L。目标为下一状态减当前状态。

## 外部评估协议

- 仅CSTR external_val的6个world，参数实例0–5；每个world原有6条episode，每条768步。
- 3条episode作为context池，另外3条作为query池，禁止episode/source交叉。
- context=128/512/960；context抽样seed=42/43/44；每任务query=256。
- 同一world的所有context长度、抽样种子、模型使用完全相同query；同一种子下context前缀嵌套。
- 总计54个任务、7种方法、378条任务级评估记录。
- 当前CSTR只有两个状态，低于ICDS训练配置的4维下限，也是本次迁移差异之一。
- 两个hr0模型均只使用ICDS训练，没有在这6个world上更新权重。

主指标为每个目标按预定义参考尺度归一化后的MSE，再按world宏平均；CSTR参考尺度为0.1 mol/L与1 K。另报各目标RMSE/MAE。3次context抽样先在world内平均，不视为3个训练seed或18个独立world。不同任务的绝对归一化误差不能直接混合判断难度。

## 运行、结果与核验

启动：`bash experiments/exp6_CAQ-ICDS/run_raw_and_cstr.sh`，在mytmux1:allenv19_flat、GPU1运行。

结果根目录：`experiments/exp6_CAQ-ICDS/checkpoints/raw-input-cstr-audit/`。

每个子目录保存协议/源码/检查点哈希、任务行号、逐任务指标、全部预测和真实标签。报告在全部方法完成后才标记completed。

核验：`python experiments/exp6_CAQ-ICDS/verify_raw_and_cstr.py`。检查完整记录数、预测重算误差、query与标签一致、数据与源码哈希、旧typed与两模型结果复现；另外使用独立Radau积分对CSTR的18个样本进行数值核验。核验结果写入 `verification.json`。

本轮仅开发评估，不能建立真实工厂迁移结论，也不能建立闭环控制性能或长期递归稳定性结论。

## 已完成结果（2026-10-08）

两套评估均为completed。ICDS为26个任务×7方法=182条记录；CSTR为54个任务×7方法=378条记录。没有新增梯度训练，CAQ/FT均加载正式hr0 best.pt。

### ICDS原始输入核验

| 方法 | 总体固定尺度MSE | discrete | hybrid_line | physical_ode |
|---|---:|---:|---:|---:|
| 官方原始输入 | 4.635190 | 0.001169 | 9.625288 | 5.312741 |
| 官方typed输入 | 6.565050 | 0.001014 | 19.363258 | 3.513937 |
| CAQ | 0.176994 | 0.000342 | 0.499144 | 0.110475 |
| TabICL-FT | 0.179654 | 0.000335 | 0.421440 | 0.171789 |
| Ridge原始输入 | 5.479098 | 0.001370 | 10.256945 | 7.064561 |
| HistGB原始输入 | 14.975987 | 0.007423 | 6.939774 | 34.073043 |
| Persistence | 221.240461 | 0.040253 | 9.847230 | 568.295911 |

官方原始输入比typed总体误差低约29.4%，但并非每个机制族都改善。预训练后两模型的优势没有因改用原始输入而消失。CAQ比FT总体低约1.48%，仍只是单训练seed的开发结果。此处的原始输入指原单位公开特征；监督目标保持原任务定义（连续增量、count/binary下一值），不是另行改成所有目标都预测绝对next-state。

### CSTR独立方程来源的迁移

| 方法 | context=128 | context=512 | context=960 |
|---|---:|---:|---:|
| CAQ | 0.049588 | 0.003493 | 0.001852 |
| TabICL-FT | 0.071634 | 0.002996 | 0.001270 |
| 官方原始输入 | 0.078141 | 0.015516 | 0.019231 |
| 官方typed输入 | 0.058571 | 0.006722 | 0.008675 |
| Ridge原始输入 | 0.546565 | 0.218768 | 0.172658 |
| HistGB原始输入 | 0.283528 | 0.151728 | 0.181475 |
| Persistence | 0.299671 | 0.299671 | 0.299671 |

context=960时CAQ比官方raw低约90.4%、比更强的官方typed低约78.6%。这支持整个工业预训练方案在此过程上的迁移价值，不足以把收益归因于CAQ结构。

CAQ在context=128时比FT总体低约30.8%；512/960时总体分别高约16.6%/45.8%。960时CAQ实际上在6个world中的4个获胜，但其余world的较大误差使宏平均落后，因此不能把平均差值理解成每个world都落后。

探索性统计：先在world内平均3次context抽样，再对6个world作50,000次配对bootstrap。CAQ减FT的均值差95%区间分别为：128时[-0.05645,-0.000107]；512时[-0.000137,0.001609]；960时[-0.000365,0.002125]。后两者跨零，128的上界也接近零。这只是一个机制族、6个参数world、一个训练seed下未作多重比较校正的探索性区间，不能宣布稳定的架构优越性。

context=960时，按任务计算RMSE后平均：CAQ浓度0.000160 mol/L、温度0.03044 K；FT浓度0.000175 mol/L、温度0.02948 K。它们不是先池化全部平方误差再开方的RMSE，与主指标的聚合顺序不同。

### 核验与产物

- `checkpoints/raw-input-cstr-audit/verification.json`：逐预测核验、输入/行号/源码/检查点哈希、所有指标及汇总核验。
- CAQ和FT精确复现旧typed主指标；官方typed与旧值绝对差4.92e-8。
- PC-Gym源码及6个CSTR shard哈希一致；18条转移用独立Radau求解核验，最大参考尺度归一化差1.14e-13。
- `checkpoints/raw-input-cstr-audit/analysis.json`：配对world区间与物理单位误差。
- `checkpoints/raw-input-cstr-audit/comparison.png`：ICDS比较及CSTR context曲线。
- `summarize_raw_and_cstr.py`：重新生成统计摘要和图。

初次运行在Ridge单目标输出形状处中断；修复为二维输出并用float64 SVD后恢复。恢复验证参数、任务行号及检查点哈希，仅复用完整模型组，原协议和中断前记录保留于输出目录。未重写已有神经模型预测。

## 下一步建议

先扩展第二个独立过程开发评估，并在相同context档位比较两模型；少context优势只是待验证假设。保留官方raw和typed结果，避免选择有利包装。HistGB当前只是未调参的快速基线，不能代表充分调参的树模型性能。最终论文还需要独立训练seed及未参与开发决策的测试协议，不建议仅根据这一轮直接扩大预训练规模。
