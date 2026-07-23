# TabICL 多输出 + CAIM 实验报告 v2

> 日期: 2026-07-18 ~ 2026-07-19
> 数据集: MBPO Hopper-v5 经验回放池 (15 个累积快照, 6K~286K 样本)
> 基线: 原始 12×TabICLRegressor (n_estimators=4) + Ensemble MLP (7 networks, 5 elites)
> 预训练模型: `tabicl-regressor-v2-20260212.ckpt` (jingang/TabICL)

## 目录

1. [背景与目标](#1-背景与目标)
2. [Target-Aware 机制解释](#2-target-aware-机制解释)
3. [实验 0: 共享 ColEmbedding 多输出模型](#3-实验-0-共享-colembedding-多输出模型)
4. [实验 1: 双流分离 + MLP ActionEncoder](#4-实验-1-双流分离--mlp-actionencoder)
5. [突破: 每维度独立 + 原始预处理](#5-突破-每维度独立--原始预处理)
6. [实验 2 (CAIM): 分离 state/action + CausalFusion + Masked Action Modeling](#6-实验-2-caim-分离-stateaction--causalfusion--masked-action-modeling)
7. [代码变更清单](#7-代码变更清单)
8. [当前瓶颈与下一步](#8-当前瓶颈与下一步)

---

## 1. 背景与目标

### 1.1 原始 baseline

原始实验使用 **12 个独立的 TabICLRegressor**，每个预测 1 个输出维度：

```python
for d in range(12):
    reg = TabICLRegressor(n_estimators=4, use_amp=True)
    reg.fit(x_tr, y_tr[:, d])      # 每个模型用自己维度的 target
    pred = reg.predict(x_va)       # 零样本推理
```

输入 `[state(11)|action(3)]` = 14 特征列，标签 `[reward(1)|delta_state(11)]` = 12 维。

每个模型内部: `ColEmbedding → RowInteraction → ICL → decoder`。

### 1.2 目标

1. 将 12 个独立模型合并为统一架构，消除重复计算
2. 添加 ActionEncoder + 因果机制，突出 action 的因果干预作用
3. 为多 RL 环境预训练奠定基础

---

## 2. Target-Aware 机制解释

这是理解所有实验结果的关键。

### 2.1 机制

ColEmbedding 的 `_compute_embeddings` 方法（[embedding.py:416-418](src/tabicl/_model/embedding.py#L416)）：

```python
y_emb = self.y_encoder(y_train.unsqueeze(-1))          # 标量 y → Linear(1,128) 嵌入
src[:, :, :train_size, :] += y_emb                      # 只加到训练样本上
src = self.tf_col(src, train_size=...)                   # Set Transformer 处理
```

**效果**: 训练样本的特征嵌入被注入标签值 y，测试样本没有。Set Transformer 利用 y 的数值来调节列间注意力——y 的具体数值（是 2.5 还是 0.003）直接决定产出的列嵌入。

### 2.2 关键推论

**不同 y 分布 → 不同的列嵌入 → 一个 ColEmbedding 前向不能同时服务 12 个不同的 y 分布。**

- y=2.5 (reward 值) 的嵌入和 y=0.003 (delta_state 值) 的嵌入导向完全不同的注意力模式
- 用 reward 值冒充 delta_state 的 target → 列嵌入完全错位 → **170x MSE 退化**（消融实验 B）

---

## 3. 实验 0: 共享 ColEmbedding 多输出模型

### 3.1 设计

```
[state|action](14列) → ColEmbedding(1次, y=reward) → RowInteraction(1次)
                              ↓
                    12× [y_enc_d + ICL_d + dec_d]
```

**改动**: 修改 `ICLearning`（[learning.py](src/tabicl/_model/learning.py)），添加 `num_outputs=12`，替换单 `y_encoder/decoder` 为 `ModuleList`。ColEmbedding + RowInteraction 各跑 1 次，ICL 跑 12 次。

### 3.2 零样本评估结果

| Epoch | 样本 | MSE | vs OrigTabICL | vs Ensemble |
|-------|------|-----|---------------|-------------|
| 0 | 6K | 0.00699 | 2.63x | 0.48x |
| 5 | 11K | 0.00492 | 1.97x | 0.24x |
| 10 | 16K | 0.00453 | 2.18x | 0.52x |
| 20 | 26K | 0.00329 | 2.07x | 0.45x |
| 50 | 56K | 0.00354 | 2.19x | 0.76x |

**结论: 平均 ~2.2x vs OrigTabICL，始终远超 Ensemble MLP。**

### 3.3 消融实验：定位性能差距

| 实验 | ColEmbedding target | MSE | vs Baseline |
|------|---------------------|-----|-------------|
| A: 每维度正确 y | 12 个模型各自正确 target | 0.00324 | **1.22x** |
| B: 全用 reward y | delta 维度用 reward 冒充 | 0.45305 | **170x** |
| C: target_aware=False | 关掉 target-aware | 0.00657 | 2.47x |
| D: 当前共享方案 | 共享 ColEmbedding + reward | ~0.006 | ~2.2x |

**结论**:
1. 架构正确（实验 A: 1.22x）
2. target_aware 是性能关键（A vs B: 140x 差距）
3. 共享 ColEmbedding 必然导致 11/12 维度 target 错误 → 不可行

---

## 4. 实验 1: 双流分离 + MLP ActionEncoder

### 4.1 设计

```
state(11列) → ColEmbedding → RowInteraction → state_repr (frozen)
action(3列) → ActionEncoder(MLP) → action_repr (trainable)
                    ↓
            concat + Linear → combined
                    ↓
            12× [ICL_d + dec_d]
```

**新建**: [action_encoder.py](src/tabicl/_model/action_encoder.py)，修改 [tabicl.py](src/tabicl/_model/tabicl.py) 添加 `use_action_encoder` 模式。

### 4.2 结果

| Epoch | Zero-shot | Fine-tuned(50ep) | vs Orig |
|-------|-----------|------------------|---------|
| 0 | 6.53x | **1.94x** | 1.94x |
| 5 | 7.71x | 2.22x | 2.22x |
| 20 | 8.58x | 3.60x | 3.60x |

**问题**: 
- Zero-shot 不可用（ActionEncoder + fusion 随机初始化）
- Fine-tune 受限于 GPU 显存，被迫子采样 4000 样本
- 大数据集下退化（epoch 20: 3.60x）

---

## 5. 突破: 每维度独立 + 原始预处理

### 5.1 核心发现

回到实验 0 消融实验 A 的思路：**每维度独立 target_aware + 原始 sklearn 预处理 + Ensemble**。

```python
# 每维度用单输出 TabICL + 正确的 y_d + Ensemble 4 视图
for d in range(12):
    m = TabICL(num_outputs=1)
    m.load_state_dict(sd)
    for each ensemble view:
        pred = m(X_view, y_train=y_tr[:, d:d+1])  # target_aware 正确!
    dim_preds[d] = ensemble_mean
```

与原始 12 个 `TabICLRegressor` 完全等价，只是组织为一个代码流程。

### 5.2 结果

| Epoch | 样本 | MSE | vs Orig | vs Ens |
|-------|------|-----|---------|--------|
| 0 | 6K | 0.00296 | **1.11x** | 0.21x |
| 10 | 16K | 0.00203 | **0.98x** ✅ | 0.25x |
| 30 | 36K | 0.00183 | 1.06x | 0.19x |
| 50 | 56K | 0.00170 | 1.05x | 0.32x |
| 100 | 106K | 0.00136 | 1.03x | 0.50x |

**结论**: **~1.06x vs OrigTabICL，始终远超 Ensemble MLP（3-50x）。**

这是整个实验周期中唯一达到 baseline 水平的方案。三大要素缺一不可：
1. 每维度正确的 target_aware
2. 原始 sklearn 预处理 (TransformToNumerical + EnsembleGenerator)
3. Ensemble 4 视图平均

---

## 6. 实验 2 (CAIM): 分离 state/action + CausalFusion + Masked Action Modeling

### 6.1 设计

```
state(11列) → ColEmbedding(y=y_d) → RowInteraction → state_repr_d (frozen)
action(3列) → ActionEncoder(MLP) → CausalFusion → fused_d (trainable)
                    ↓                        ↓
            MaskedActionModeling     12× [ICL_d + dec_d]
```

**新建**:
- [causal_fusion.py](src/tabicl/_model/causal_fusion.py): CausalCrossAttention + MaskedActionModeling
- [caim.py](src/tabicl/_model/caim.py): CAIM 主模型

### 6.2 训练配置

- 数据: epoch 280 (286K 样本)，训练 276K / 验证 10K
- 每 epoch 随机子采样 4000 训练样本
- Quantile loss + Masked Action Modeling 辅助损失
- AMP 混合精度
- 逐维度梯度累积

### 6.3 训练历史（1200 epochs = 200 + 1000）

| 累计 Epoch | MSE | vs Orig | Reward MSE | Delta MSE |
|-----------|-----|---------|------------|-----------|
| 1 | 0.216 | 284x | 0.114 | 0.225 |
| 200 | 0.120 | 158x | 0.0040 | 0.130 |
| 400 | 0.115 | 152x | 0.0029 | 0.126 |
| 600 | 0.114 | 150x | 0.0033 | 0.124 |
| 800 | 0.114 | 150x | 0.0032 | 0.124 |
| 1000 | 0.113 | 149x | 0.0030 | 0.123 |
| 1200 | 0.113 | 149x | - | - |

### 6.4 收敛分析

**已完全收敛**。epoch 600→1200 的 600 个 epoch 中，MSE 从 0.1141→0.1132，仅下降 0.8%。

- **Reward 预测**: 0.114→0.003（38x 改善），但 baseline=0.0005，仍差 6x
- **Delta 预测**: 0.225→0.123（仅 1.8x 改善），baseline=0.00078，差 158x

### 6.5 失败根因

```
预训练时 ColEmbedding 看到:
  [state_0, ..., state_10 | action_0, action_1, action_2]  ← 14 列统一起
  Set Transformer 学到的列间注意力: "action_dim0 和 state_dim2 有交互..."

CAIM 分离后 ColEmbedding 看到:
  [state_0, ..., state_10]  ← 只有 11 列
  Set Transformer: "action 列去哪了？我的注意力模式失效了"

CausalFusion 背负了所有:
  - 要从零学习 state-action 交互（Set Trasformer 已学过的知识）
  - 只有 138 万参数 vs backbone 的 2700 万
  - 每个 epoch 只有 4000 样本
```

### 6.6 反事实/Counterfactual 的尝试

实验 2 过程中尝试了多种反事实机制：
1. **Contrastive Loss (cosine similarity)**: 平凡解 attn_out≈0
2. **MSE Loss on attn_out**: 平凡解 attn_out≈0
3. **Negate strategy**: action 取反 → 仍然无效
4. **Masked Action Modeling**: 从 attn_out 重建被 mask 的 action 值 → 最终采用的方案

**结论**: 在分离架构下，反事实机制无法发挥作用，因为 state 信息（frozen backbone 输出）主导了表示空间，action 贡献被其淹没。

---

## 7. 代码变更清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/tabicl/_model/action_encoder.py` | Action 编码器 (MLP/Transformer 双模式) |
| `src/tabicl/_model/causal_fusion.py` | CausalCrossAttention + MaskedActionModeling |
| `src/tabicl/_model/caim.py` | CAIM 主模型 |
| `experiments/exp0_multi_output/*` | 实验 0 脚本 |
| `experiments/exp1_action_encoder/*` | 实验 1 脚本 |
| `experiments/exp2_caim/*` | 实验 2 脚本 |
| `experiments/EXPERIMENT_REPORT_v2.md` | 本报告 |

### 已修改文件

| 文件 | 改动 |
|------|------|
| `src/tabicl/_model/learning.py` | 添加 `num_outputs` 多输出支持 |
| `src/tabicl/_model/tabicl.py` | 添加 `num_outputs`, `use_action_encoder`, `state_dim`, `action_dim` 参数 |

---

## 8. 当前瓶颈与下一步

### 8.1 核心瓶颈

**预训练模型的 target_aware 机制与多输出任务存在根本性矛盾。**

- ColEmbedding 的 target_aware 要求每个前向传播有唯一的 target y
- 12 个输出维度有 12 个不同的 y 分布
- 共享 ColEmbedding → target 错误 → 性能崩溃（170x）
- 每维度独立 ColEmbedding → 等同 12 个独立模型（当前最优 1.06x）
- 分离 state/action → 破坏预训练列间注意力 → 无法收敛（149x）

**"添加 Action Encoder 且不改动 backbone"这个约束，在当前预训练模型的架构假设下，无法通过分离 state/action 来实现。** 预训练模型在 Set Transformer 层面已经把 state 和 action 当做统一的表格列来处理了。

### 8.2 可行的下一步: 路线 B

```
[state|action] → ColEmbedding → RowInteraction → backbone_repr (frozen, 1.06x)
                                                        │
                                                  fine-tune: y_encoders + decoders
```

保持 backbone 完全不变（state+action 统一输入），只在 12 个 decoder head 上做 fine-tune。用 epoch 280 的 276K 样本，quantile loss，目标从 1.06x 进一步逼近 1.0x 或更优。

### 8.3 更长期的方向

1. **从头预训练多输出 TabICL**: 在合成 tabular 数据上预训练时就支持多输出
2. **分离架构预训练**: 从零开始预训练 state-only ColEmbedding + ActionEncoder，而不是在单输出 checkpoint 上改造
3. **更轻量的 action 辅助模块**: 在 unified backbone 基础上加小型辅助模块（如 ActionEncoder 输出的 embedding 直接加到 ICL 输入上），仅微调 decoder heads

### 8.4 关键教训

1. **不要破坏预训练模型的输入分布**: ColEmbedding 的 Set Transformer 对输入列结构敏感
2. **target_aware 是决定性的**: 它是 1.06x 和 170x 之间的唯一差异
3. **零样本 > 错误训练**: 预训练模型的零样本能力极强（1.06x），远好于错误架构下的任何训练（149x）
4. **复杂的因果/反事实机制必须在正确的架构基础上才能发挥作用**: 如果基础表示已经错误，上层机制无法纠正
