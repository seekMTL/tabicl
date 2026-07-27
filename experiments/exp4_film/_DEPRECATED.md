# ⚠️ 已废弃

本目录的实验方案 **"FiLM 条件化调制 (Feature-wise Linear Modulation)"** 已被废弃。

## 原因

FiLM 在纯推理模式下，γ 分量（`fc_gamma`）零初始化导致乘法门控未激活，
实际退化为 `src + β ≈ src + y_emb`，与分组加性注入（Joint）方案在精度上完全等价（Overall ~1.75x）。

未能体现 FiLM 论文中乘法调制的核心优势。

## 根因

1. 纯推理模式下 FiLM generator 无法训练，γ 始终为 0
2. 从预训练权重初始化 β 只能复现加性注入行为
3. 需要 fine-tune 才能激活乘法门控，但当前未实施

## 保留价值

- `eval_film_vs_baseline.py` — 可复用的多输出对比评测脚本（含自检机制）
- 代码改动（`src/tabicl/_model/embedding.py` 中的 `MultiDimTargetFiLM` 类等）已合入主代码，
  其中 `num_outputs=1` 时向后兼容原版 TabICL

## 目录内容

- `eval_film_vs_baseline.py` — 评测脚本
- `REPORT.md` — 实验报告（含原始记录 + 分析）
- `*.py` (learning/regressor/sklearn_utils/tabicl) — 实验时的代码快照备份
