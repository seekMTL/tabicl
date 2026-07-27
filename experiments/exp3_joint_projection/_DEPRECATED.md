# ⚠️ 已废弃

本目录的实验方案 **"全向量多维目标投影 (Joint Vector Target Projection)"** 已被废弃。

## 原因

将 `Linear(1, 128)` 替换为 `Linear(12, 128)` + 权重复制 `cw.repeat(1, 12) * (1/12)` 的全向量联合投影方案，
在 epoch 280 上 Overall MSE 退化 2.07x（原始为 2.15x），远超可接受范围。

## 根因

1. 权重复制稀释：每个输入维度的信号强度降为 1/12
2. StandardScaler 归一化后 reward 和 delta 同等权重，稀释效应被暴露
3. 加性注入表达力有限，无法做特征级差异化调制

## 被取代

- **分组加性注入**（Joint, backupv2/2_group）：将 reward 和 delta 分组建模，Overall 降至 1.75x
- **FiLM 条件化调制**（experiments/exp4_film/）：Feature-wise Linear Modulation，同样因纯推理模式下无法激活而废弃

## 目录结构

- `backupv1/` — 原始全向量方案快照（代码 + 报告）
- `backupv2/1_combine/` — 分组加性注入方案快照
- `backupv2/2_group/` — 分组变体方案快照
- `eval_joint_vs_baseline.py` — 评测脚本（当前使用的分组方案）
