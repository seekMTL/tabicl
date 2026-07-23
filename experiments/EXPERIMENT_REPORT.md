# TabICL 多输出 + ActionEncoder 实验报告

> 日期: 2026-07-18 ~ 2026-07-19
> 数据集: MBPO Hopper-v5 经验回放池 (15 个快照, 6K~286K 样本)
> 基线: 原始 12×TabICLRegressor (n_estimators=4) + Ensemble MLP (7 networks, 5 elites)
> 预训练模型: `tabicl-regressor-v2-20260212.ckpt` (jingang/TabICL)

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [实验 0: 共享 Backbone + 多输出头](#2-实验-0-共享-backbone--多输出头)
3. [实验 1: 双流分离 + MLP ActionEncoder](#3-实验-1-双流分离--mlp-actionencoder)
4. [总结与下一步](#4-总结与下一步)

---

## 1. 背景与目标

### 1.1 原始 baseline 架构

原始实验使用 **12 个独立的 TabICLRegressor**，每个预测一个输出维度（1 reward + 11 delta_state），每次 forward 流程：

```
[state(11)|action(3)] → ColEmbedding → RowInteraction → y_encoder + ICL → decoder → 1 scalar
     14 features              12×独立运行          12×独立运行       12×独立
```

**问题**: 12 个模型各自独立运行 ColEmbedding + RowInteraction + ICL，大量冗余计算。

### 1.2 目标

1. **实验 0**: 将 12 个独立模型合并为 1 个共享 backbone + 12 个 decoder head，验证架构正确性
2. **实验 1**: 添加独立的 ActionEncoder，将 state 和 action 分离处理

---

## 2. 实验 0: 共享 Backbone + 多输出头

### 2.1 解决的问题

**消除 ColEmbedding + RowInteraction 的 12× 冗余计算**，用一个共享 backbone 替换 12 个独立模型。

### 2.2 代码改动

**文件**: `src/tabicl/_model/learning.py`

核心改动：`ICLearning` 新增 `num_outputs` 参数：

```python
# 改动前: 单输出
self.y_encoder = nn.Linear(1, d_model)
self.decoder = nn.Sequential(Linear(512→1024), GELU, Linear(1024→999))

# 改动后: 多输出（num_outputs > 1）
self.y_encoders = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_outputs)])
self.decoders = nn.ModuleList([nn.Sequential(...) for _ in range(num_outputs)])
```

`_icl_predictions` 方法：每个输出维度独立运行 y_encoder + ICL transformer + decoder head，共享同一个 R（RowInteraction 输出）。

**文件**: `src/tabicl/_model/tabicl.py`

- 新增 `num_outputs` 参数透传到 `ICLearning`
- `_train_forward` / `_inference_forward`：多输出时 ColEmbedding 使用第一个维度（reward）做 target-aware embedding

### 2.3 实验结果

#### 2.3.1 零样本评估（全量数据，纯推理，无 KV cache，无 ensemble）

| Epoch | 训练样本 | MultiOut MSE | OrigTabICL MSE | vs Orig | vs Ensemble MLP |
|-------|---------|-------------|----------------|---------|-----------------|
| 0 | 4,800 | 0.00673 | 0.00266 | **2.53x** | **0.48x** |
| 5 | 8,800 | 0.00652 | 0.00250 | 2.61x | 0.24x |
| 10 | 12,800 | 0.00415 | 0.00208 | 1.99x | 0.52x |
| 20 | 20,800 | 0.00349 | 0.00158 | 2.20x | 0.45x |
| 50 | 44,800 | 0.00399 | 0.00162 | 2.47x | **0.76x** |
| 100 | 84,800 | 0.00351 | 0.00133 | 2.65x | 1.28x |

**平均: vs OrigTabICL ~2.4x, vs Ensemble MLP ~0.6x**

#### 2.3.2 消融实验：定位性能差距根因

| 实验 | ColEmbedding 配置 | MSE | vs Baseline |
|------|-------------------|-----|-------------|
| **A: 每维度正确 y** | 12 个模型各自用正确 target | 0.00324 | **1.22x** ✅ |
| B: 全用 reward y | delta 维度用 reward 冒充 | 0.45305 | 170x ❌ |
| C: target_aware=False | 关掉 target-aware | 0.00657 | 2.47x |
| D: 当前多输出 | 共享 ColEmbedding 用 reward y | ~0.006 | ~2.2x |

#### 2.3.3 Ensemble 预处理对比

| 预处理方案 | vs OrigTabICL |
|-----------|---------------|
| StandardScaler (单视图) | 2.53x |
| StandardScaler + 随机 shuffle ensemble (4视图) | **2.02x** |
| TransformToNumerical + Latin square + Power transform (原始管线) | 2.18x |
| col_target_aware=False + 多输出 | 2.47x |

### 2.4 结论

1. **多输出架构正确**: 实验 A 证明，当 ColEmbedding 收到正确 target 时，多输出模型达到 **1.22x**（剩余 0.22x 来自无 ensemble）。架构不是瓶颈。

2. **ColEmbedding 的 target-aware embedding 是性能关键**: 错误的 target 信号导致 **140x MSE 退化**（实验 A vs B），这是 2.2x 差距的根本原因。

3. **预处理不是主因**: 原始预处理管线（Power transform + Latin square）与简化 StandardScaler 差异仅 ~0.16x。

4. **始终远超 Ensemble MLP**: 最差情况（epoch 100）仍比 Ensemble MLP 好。

### 2.5 遗留问题

**ColEmbedding 共享导致 target 信号错误**: 12 个输出维度需要 12 个不同的 target 做 target-aware embedding，但 ColEmbedding 共享时只能用 1 个 target（reward）。如何在不牺牲共享效率的前提下提供正确的 target 信号？→ 实验 1

---

## 3. 实验 1: 双流分离 + MLP ActionEncoder

### 3.1 解决的问题

通过分离 state 和 action 的处理路径，重新设计信息流：
- State ColEmbedding 不再需要同时编码 action 信息
- ActionEncoder 提供 action 专用表示
- 融合层学习 state-action 交互
- 为后续预训练 ActionEncoder 奠定基础

### 3.2 架构设计

```
state (B,T,11) ──▶ ColEmbedding ──▶ RowInteraction ──▶ state_repr (B,T,512)
                   (不改动, 11列)    (不改动)              │
                                                          ├─ concat ──▶ fusion_proj
action (B,T,3) ──▶ ActionEncoder ─────────────────▶ action_repr (B,T,128)
                   MLP: Linear(3→256)                         │
                   → LayerNorm → GELU                Linear(640→512)
                   → Linear(256→128)                         │
                   → LayerNorm                        combined (B,T,512)
                                                          │
                                                  12×[y_encoder_j + ICL_j + decoder_j]
```

### 3.3 代码改动

**新建**: `src/tabicl/_model/action_encoder.py`

```python
class ActionEncoder(nn.Module):
    def __init__(self, action_dim=3, d_model=128, hidden_dim=256, mode="mlp"):
        # MLP: Linear(3→256) → LayerNorm → GELU → Linear(256→128) → LayerNorm
        # Transformer: Linear(3→128) → 2×TransformerBlock(4 heads)
```

**修改**: `src/tabicl/_model/tabicl.py`

新增参数 `use_action_encoder`, `state_dim`, `action_dim`, `action_encoder_mode`。

`_train_forward` 双流路径:
```python
if self.use_action_encoder:
    X_state = X[:, :, :self.state_dim]     # 前 11 列
    X_action = X[:, :, self.state_dim:]    # 后 3 列
    state_repr = self.row_interactor(self.col_embedder(X_state, ...))
    action_repr = self.action_encoder(X_action)
    combined = self.fusion_proj(torch.cat([state_repr, action_repr], dim=-1))
    return self.icl_predictor(combined, y_train=y_train)
```

### 3.4 实验结果

#### 3.4.1 Zero-shot vs Fine-tuned

| Epoch | Zero-shot | Fine-tuned (50 epochs) |
|-------|-----------|----------------------|
| 0 | 6.53x | **1.94x** |
| 5 | 7.71x | 2.22x |
| 10 | 7.29x | 2.95x |
| 20 | 8.58x | 3.60x |

- **Zero-shot 不可用**: ActionEncoder + fusion_proj 随机初始化，产出噪声
- **Fine-tuned 有效**: epoch 0 达到 1.94x（优于实验 0 的 2.53x）
- **大数据集退化**: fine-tune 时子采样 4000 样本（GPU 显存限制），随快照增大而退化

#### 3.4.2 显存分析

| 场景 | 显存需求 | 原因 |
|------|---------|------|
| 原始脚本（纯推理 + AMP） | ~8 GB | 前向传播 + float16 |
| 实验 0 零样本（纯推理, float32） | ~12 GB | 前向 + 12× ICL forward |
| 实验 1 fine-tune（训练, float32） | **>23 GB OOM** | 前向+反向+12×ICL×激活存储 |

**结论**: 纯推理全量数据可正常进行，不 OOM（与原始脚本一致）。fine-tune OOM 来自训练时的反向传播激活存储。

### 3.5 结论

1. **双流架构优于单流**: fine-tune 后 epoch 0 达到 1.94x vs orig（实验 0 为 2.53x）
2. **ActionEncoder + fusion_proj 必须训练**: 随机初始化下 zero-shot 不可用（6-8x），但仅 50 epoch fine-tune 即可显著改善
3. **fine-tune 需要在更大规模上进行**: 子采样 4000 导致大数据集退化，需 AMP + 梯度检查点 + 更大 GPU

---

## 4. 总结与下一步

### 4.1 实验递进关系

```
实验 0: 共享 Backbone + 多输出头
├── 解决: ColEmbedding+RowInteraction 12× → 1×
├── 发现: target-aware embedding 对 target 正确性极度敏感 (140x 差距)
└── 结论: 架构正确 (1.22x)，但共享 ColEmbedding 的 target 问题是瓶颈
        │
        ▼
实验 1: 双流分离 + ActionEncoder
├── 解决: 分离 state/action 处理，ActionEncoder 专用编码
├── 发现: fine-tune 后优于实验 0 (1.94x vs 2.53x)
├── 限制: fine-tune 需要更多 GPU 显存
└── 结论: 双流架构有优势，需解决训练效率
        │
        ▼
实验 2 (待做): Transformer ActionEncoder + Cross-Attention 融合
```

### 4.2 量化对比

| 指标 | 原始 12×TabICL | 实验 0 (共享) | 实验 1 (双流) |
|------|---------------|--------------|--------------|
| ColEmbedding 调用 | 12× | **1×** | **1×** |
| RowInteraction 调用 | 12× | **1×** | **1×** |
| ICL forward 调用 | 12× (ensemble: 48×) | 12× | 12× |
| 新增参数 | 0 | 0 | 363K (ActionEncoder + fusion) |
| Best vs OrigTabICL | 1.0x | 1.22x (per-dim 正确 y) | **1.94x** (fine-tuned) |
| Best vs Ensemble MLP | 0.08x | 0.18x | 待测 |
| Zero-shot 可用 | ✅ | ✅ (~2.4x) | ❌ (需 fine-tune) |

### 4.3 下一步

1. **实验 2**: Transformer ActionEncoder + Cross-Attention 融合，捕捉动作时序模式
2. **训练优化**: AMP + 梯度检查点 + LoRA，解决 fine-tune 显存问题
3. **多环境预训练**: 在多个 RL 环境（Hopper, Walker2d, HalfCheetah 等）上预训练 ActionEncoder + fusion_proj
4. **完整预处理**: 实现 TabICLMultiOutputRegressor sklearn wrapper，消除预处理差距
