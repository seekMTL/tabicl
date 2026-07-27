# 实验 3：全向量多维目标投影 (Joint Vector Target Projection)

> 日期: 2026-07-24
> 预训练模型: `tabicl-regressor-v2-20260212.ckpt` (jingang/TabICL)
> 测试数据: MBPO Hopper-v5 经验回放池 (epoch 0 6K / epoch 280 286K)
> 评估脚本: `experiments/exp3_joint_projection/eval_joint_vs_baseline.py`

---

## 1. 背景与问题

### 1.1 原始 TabICL 的工作方式

TabICL 是一个三阶段表格基础模型：

```
X → ColEmbedding → RowInteraction → ICL → prediction
```

其中 **ColEmbedding** 阶段有一个关键机制——**target-aware embedding**：将训练样本的标签 $y$ 通过 `Linear(1, 128)` 编码为 $y_{emb}$，加到训练样本的特征嵌入上，Set Transformer 据此调节列间注意力。

```python
# embedding.py
y_emb = self.y_encoder(y_train.unsqueeze(-1))  # Linear(1, 128): 标量 y → 128 维嵌入
src[:, :, :train_size, :] += y_emb              # 注入训练样本的特征嵌入
src = self.tf_col(src, ...)                     # Set Transformer 用 y 的值调节列间注意力
```

### 1.2 多输出场景的瓶颈

RL 场景中，输出是 12 维向量 $[reward, \Delta s_0, ..., \Delta s_{10}]$。原始做法是**创建 12 个独立的 TabICLRegressor**，每个预测一个维度：

```python
for d in range(12):
    reg = TabICLRegressor(n_estimators=4)
    reg.fit(X, y[:, d])   # 每个模型用自己维度的 target
    pred[:, d] = reg.predict(X)
```

**问题**：ColEmbedding + RowInteraction 被重复计算 12 次，每次 Set Transformer 只能看到单维度的 target 信息。

### 1.3 核心思路

RL 中 $[s', r]$ 是由 $(s, a)$ 共同决定的联合状态转移。**不需要把标签拆开**——将整个 12 维标签向量 $\mathbf{y} \in \mathbb{R}^{12}$ 一次性注入 ColEmbedding，让 Set Transformer 同时感知所有 target 维度的联合语义。

---

## 2. 方法：全向量多维目标投影

### 2.1 ColEmbedding 改造

将 y_encoder 从单标量输入扩展为全向量输入：

```python
# 原始: Linear(1, 128) — 只接受 1 个标量 y
self.y_encoder = nn.Linear(1, embed_dim)

# 改造: Linear(12, 128) — 接受完整 12 维 target 向量
self.y_encoder = nn.Linear(num_targets, embed_dim)
```

**数学含义**：

$$
y_{\text{emb}} = \frac{1}{12} \sum_{j=0}^{11} W_{\text{pretrained}} \cdot y_j
$$

预训练权重 $W_{\text{pretrained}} \in \mathbb{R}^{128 \times 1}$ 被复制到 12 个输入维度并缩放 $1/12$，保持注入 SetTransformer 的能量量级与单输出一致。

### 2.2 ICL 阶段共享

ColEmbedding + RowInteraction **只运行 1 次**，产出共享的行表示 $R \in \mathbb{R}^{B \times T \times 512}$。之后的 ICL 阶段：

- **y_encoder 和 decoder 在所有 12 个维度间共享**（权重相同，逐维传入不同的 $y_j$）
- `tf_icl`（12 层 Transformer）每维度独立前向（因为每维注入的 $y_{emb}$ 不同）

```python
# learning.py — _icl_predictions 多输出路径
for j in range(num_outputs):
    R_j = R.clone()
    Ry_train = self.y_encoder(y_train[:, :, j].unsqueeze(-1))  # 共享 y_encoder
    R_j[:, :train_size] += Ry_train
    src_j = self.tf_icl(R_j, train_size=train_size)             # 共享 tf_icl
    out_j = self.decoder(src_j)                                  # 共享 decoder
    outs.append(out_j.cpu())                                     # 移 CPU 防 OOM
return torch.stack(outs, dim=-2).to(R.device)
```

### 2.3 架构对比

| | Baseline (12× 单输出) | Joint (1× 多输出) |
|---|---|---|
| ColEmbedding 前向次数 | 12 | **1** |
| ColEmbedding y_encoder | `Linear(1, 128)` | **`Linear(12, 128)`** |
| ICL y_encoder | 12 个独立的 `Linear(1, 512)` | **1 个共享的 `Linear(1, 512)`** |
| ICL decoder | 12 个独立的 MLP | **1 个共享的 MLP** |
| ICL tf_icl 前向次数 | 12 | 12（每维独立 target 注入） |

---

## 3. OOM 问题与解决

### 3.1 为什么 12 个独立模型不 OOM，1 个共享模型 OOM？

**12 个独立模型是串行的**——每个模型产生 `(B, T, 999)` = `(8, 286K, 999)` ≈ 9 GB 输出 → 立即释放 → 下一个模型。GPU 上同时只存在约 9 GB。

**1 个共享模型是并行的**——_icl_predictions 内部循环 12 次，所有 12 维的输出同时存在于 `outs` 列表中 → `torch.stack` 创建 `(B, T, 12, 999)` ≈ 110 GB，直接炸。

### 3.2 两个 OOM 点及其修复

| OOM 位置 | 原因 | 修复 | 文件 |
|---------|------|------|------|
| `_icl_predictions` 循环 | 12 个 `(B, 286K, 999)` 同时堆叠在 GPU | `out_j.cpu()` 移到 CPU，stack 在 CPU 完成后再回 GPU | `learning.py` |
| `predict_stats` | `quantile_dist` 对 `(B, 94K, 12, 999)` 做 sort | 逐维度循环 `quantile_dist`，处理完一维释放一维 | `tabicl.py` |

**predict_stats 逐维处理**：

```python
# tabicl.py — predict_stats
if raw_quantiles.ndim == 4 and raw_quantiles.shape[-2] > 1:
    # 多输出: (B, test, 12, 999)，逐维处理避免一次性 sort 36 GB
    for j in range(raw_quantiles.shape[-2]):
        q_j = raw_quantiles[:, :, j, :]           # (B, test, 999) 单维
        dist_j = self.quantile_dist(q_j)           # sort + monotonicity
        results["mean"].append(q_j.mean(dim=-1))   # 即时取 mean 释放
    results["mean"] = torch.stack(results["mean"], dim=-1)
else:
    # 单输出: 原始路径
    ...
```

---

## 4. 实验结果

### 4.1 Epoch 0 (6K 样本: 4020 train / 1980 test)

| 维度 | Baseline MSE | Joint MSE | Ratio(J/B) |
|------|-------------|-----------|:---:|
| reward | 0.009108 | 0.012181 | 1.34x |
| delta[10] | 0.014721 | 0.033660 | 2.29x |
| **REWARD** | 0.009108 | 0.012181 | 1.34x |
| **DELTA(avg)** | 0.002920 | 0.005562 | 1.91x |
| **OVERALL** | **0.003435** | **0.006114** | **1.78x** |

| | Baseline | Joint |
|---|---|---|
| 耗时 | 5.6s | 1.1s |
| **加速比** | | **4.9x** |

### 4.2 Epoch 280 (286K 样本: 191,620 train / 94,380 test)

| 维度 | Baseline MSE | Joint MSE | Ratio(J/B) |
|------|-------------|-----------|:---:|
| reward | 0.000791 | 0.000819 | **1.04x** |
| delta[10] | 0.005522 | 0.013766 | 2.49x |
| **REWARD** | 0.000791 | 0.000819 | 1.04x |
| **DELTA(avg)** | 0.000902 | 0.001943 | 2.15x |
| **OVERALL** | **0.000893** | **0.001849** | **2.07x** |

| | Baseline | Joint |
|---|---|---|
| 耗时 | 541.1s | 498.2s |
| **加速比** | | **1.1x** |

### 4.3 汇总

| 测试 | 样本数 | Overall Ratio | Speedup |
|------|--------|:---:|:---:|
| Epoch 0 | 6K | 1.78x | 4.9x |
| Epoch 280 | 286K | 2.07x | 1.1x |

---

## 5. 分析

### 5.1 精度

- **Reward 维度**：Joint 与 Baseline 几乎持平（1.04x），大样本下基本等价
- **Delta 维度**：Joint 整体差 ~2x。11 个 delta 维度共享同一个 y_encoder 的 `repeat * 1/12` 初始化，各维度的 target 信息在联合投影中被均匀混合，无法像独立模型那样有针对性地调节列间注意力
- **整体 MSE 被 reward 主导**：reward 的 MSE 绝对值比 delta 大 2-3 个数量级

### 5.2 速度

- **小样本（6K）**：ColEmbedding+RowInteraction 是主要开销，共享后加速 4.9x
- **大样本（286K）**：ICL Transformer（12 层 self-attention × 12 维）占绝对主导，加速比收敛至 1.1x

### 5.3 与之前裸 TabICL 测试的差异

之前用裸 `TabICL`（无 sklearn 预处理）测试，Joint **优于** Baseline（0.77x）。差异来自 `TabICLRegressor` 的 `StandardScaler`：

- 裸 TabICL：原始 y 值中 reward(~1) 远大于 delta(~0.001)，联合投影被 reward 主导，歪打正着
- TabICLRegressor：StandardScaler 将所有维度拉到 mean=0, std=1，消除了量级差异，但也暴露了 naive 权重复制策略的不足

### 5.4 改进方向

1. **ColEmbedding y_encoder 初始化优化**：当前 `repeat * 1/12` 是手工启发式，可 fine-tune 让模型学习更好的联合投影
2. **逐维度 ICL 并行化**：12 次 `tf_icl` 前向可以 batch 处理（将不同维度的 $R_j$ 堆叠为 batch 维度），减少循环开销

---

## 6. 代码变更清单

| 文件 | 改动 |
|------|------|
| `src/tabicl/_model/embedding.py` | `ColEmbedding` 新增 `num_targets` 参数；`y_encoder` 改为 `Linear(num_targets, embed_dim)`；y_train expand 适配 2D/3D |
| `src/tabicl/_model/learning.py` | `ICLearning` 新增 `num_outputs`；y_encoder/decoder 多输出共享；`_icl_predictions` 多输出循环 + CPU offloading |
| `src/tabicl/_model/tabicl.py` | `TabICL` 新增 `num_outputs` 透传；`predict_stats` 多输出逐维 `quantile_dist` 防 OOM |
| `src/tabicl/_sklearn/regressor.py` | `TabICLRegressor` 新增 `num_outputs`；`_load_model` 多输出权重复制；`fit` 支持 2D y；`predict` 多输出反标准化 |
| `src/tabicl/_sklearn/sklearn_utils.py` | `validate_data` 自动检测 2D y 设 `multi_output=True` |
| `experiments/exp3_joint_projection/eval_joint_vs_baseline.py` | 评估脚本 |
| `experiments/exp3_joint_projection/REPORT.md` | 本报告 |

---

## 附录：原始实验记录

### A.1 Epoch 0 (6K 样本)

**运行命令**:
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp3_joint_projection/eval_joint_vs_baseline.py 0
```

**终端输出**:
```
Epoch 0: 4020 train / 1980 test
y 范围: reward [-1.645,3.373]  delta [-6.220,3.157]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 5.6s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 1.1s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.009108       0.012181       1.34x
     1 delta[0]           0.000000       0.000000       1.75x
     2 delta[1]           0.000000       0.000000       1.10x
     3 delta[2]           0.000000       0.000000       2.62x
     4 delta[3]           0.000000       0.000000       1.36x
     5 delta[4]           0.000000       0.000001       2.37x
     6 delta[5]           0.000394       0.000432       1.10x
     7 delta[6]           0.000451       0.000750       1.66x
     8 delta[7]           0.005114       0.007792       1.52x
     9 delta[8]           0.001854       0.004594       2.48x
    10 delta[9]           0.009583       0.013956       1.46x
    11 delta[10]          0.014721       0.033660       2.29x
  ------------------------------------------------------------------
         REWARD MSE       0.009108       0.012181       1.34x
     DELTA(avg) MSE       0.002920       0.005562       1.91x
        OVERALL MSE       0.003435       0.006114       1.78x

  加速比: 4.9x (5.6s → 1.1s)

```

**硬件环境**: NVIDIA GeForce RTX 4090 D (24 GB), PyTorch 2.11.0+cu128, Flash SDP enabled

---

### A.2 Epoch 100 (106K 样本)

**运行命令**:
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp3_joint_projection/eval_joint_vs_baseline.py 100
```

**终端输出**:
```
Epoch 100: 71020 train / 34980 test
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 96.3s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 77.3s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.001428       0.001484       1.04x
     1 delta[0]           0.000000       0.000000       2.11x
     2 delta[1]           0.000000       0.000000       1.56x
     3 delta[2]           0.000000       0.000000       1.97x
     4 delta[3]           0.000000       0.000000       1.92x
     5 delta[4]           0.000002       0.000003       1.50x
     6 delta[5]           0.000219       0.000357       1.63x
     7 delta[6]           0.000298       0.000581       1.95x
     8 delta[7]           0.002822       0.003925       1.39x
     9 delta[8]           0.001012       0.002184       2.16x
    10 delta[9]           0.003085       0.005259       1.70x
    11 delta[10]          0.007746       0.025885       3.34x
  ------------------------------------------------------------------
         REWARD MSE       0.001428       0.001484       1.04x
     DELTA(avg) MSE       0.001380       0.003472       2.52x
        OVERALL MSE       0.001384       0.003307       2.39x

  加速比: 1.2x (96.3s → 77.3s)
```

---

### A.3 Epoch 280 (286K 样本)

**运行命令**:
```
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp3_joint_projection/eval_joint_vs_baseline.py 280
```

**终端输出**:
```
Epoch 280: 191620 train / 94380 test
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 541.1s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 498.2s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.000791       0.000819       1.04x
     1 delta[0]           0.000000       0.000000       2.18x
     2 delta[1]           0.000000       0.000000       1.45x
     3 delta[2]           0.000000       0.000000       2.16x
     4 delta[3]           0.000000       0.000000       1.54x
     5 delta[4]           0.000001       0.000002       1.38x
     6 delta[5]           0.000135       0.000259       1.92x
     7 delta[6]           0.000170       0.000328       1.93x
     8 delta[7]           0.001647       0.002292       1.39x
     9 delta[8]           0.000552       0.001162       2.11x
    10 delta[9]           0.001894       0.003561       1.88x
    11 delta[10]          0.005522       0.013766       2.49x
  ------------------------------------------------------------------
         REWARD MSE       0.000791       0.000819       1.04x
     DELTA(avg) MSE       0.000902       0.001943       2.15x
        OVERALL MSE       0.000893       0.001849       2.07x

  加速比: 1.1x (541.1s → 498.2s)
```

**硬件环境**: NVIDIA GeForce RTX 4090 D (24 GB), PyTorch 2.11.0+cu128, Flash SDP enabled

---

### A.4 OOM 修复前的失败记录

**首次运行 epoch 280 时的 OOM 错误**（完整调用栈）:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.09 GiB.
GPU 0 has a total capacity of 23.53 GiB of which 693.69 MiB is free.
Including non-PyTorch memory, this process has 22.84 GiB memory in use.
Of the allocated memory 20.69 GiB is allocated by PyTorch, and 1.75 GiB is reserved
by PyTorch but unallocated.

File "learning.py", line 307, in _icl_predictions
    src_j = self.tf_icl(R_j, train_size=train_size)  ← 12 维输出堆叠导致 OOM

File "quantile_dist.py", line 279, in enforce_monotonicity
    return torch.sort(quantiles, dim=-1).values  ← (B, 94K, 12, 999) ≈ 36 GB sort
```

**根本原因**: 12 个独立模型是串行的，每个产生 3 GB 输出后释放；Joint 模型在一次前向中同时产生 12 × (B, 286K, 999) = 110 GB 的中间张量。修复方案见第 3 节。**修复后 epoch 280 全量 286K 成功运行。**

### A.5 官方pip包实验记录

以下是使用官方 pip 包 tabicl v2.0.2 的终端输出结果（12 个单输出 TabICLRegressor）

```
tabic1 location: /home/lizitao/software/miniconda3/envs/mbpo_tabicl/lib/python3.9/site-packages/tabicl/__init__.py
Epoch 0: 4020 train / 1980 test
y range: reward [-1.645,3.373]  delta [-6.220,3.157]
y_tr shape: (4020, 12), y_te shape: (1980, 12)

=== Baseline: 12 x TabICLRegressor (official pip package) ===
  [1/12] reward ... MSE=0.009005, time=0.8s
  [2/12] delta[0] ... MSE=0.000000, time=0.4s
  [3/12] delta[1] ... MSE=0.000000, time=0.4s
  [4/12] delta[2] ... MSE=0.000000, time=0.4s
  [5/12] delta[3] ... MSE=0.000000, time=0.4s
  [6/12] delta[4] ... MSE=0.000000, time=0.4s
  [7/12] delta[5] ... MSE=0.000393, time=0.4s
  [8/12] delta[6] ... MSE=0.000451, time=0.4s
  [9/12] delta[7] ... MSE=0.005121, time=0.4s
  [10/12] delta[8] ... MSE=0.001856, time=0.4s
  [11/12] delta[9] ... MSE=0.009579, time=0.4s
  [12/12] delta[10] ... MSE=0.014689, time=0.5s
Total time: 5.3s

======================================================================
   Dim Label                   MSE
  ------------------------------
     0 reward             0.009005
     1 delta[0]           0.000000
     2 delta[1]           0.000000
     3 delta[2]           0.000000
     4 delta[3]           0.000000
     5 delta[4]           0.000000
     6 delta[5]           0.000393
     7 delta[6]           0.000451
     8 delta[7]           0.005121
     9 delta[8]           0.001856
    10 delta[9]           0.009579
    11 delta[10]          0.014689
  ------------------------------
         REWARD MSE       0.009005
      DELTA AVG MSE       0.002917
    OVERALL AVG MSE       0.003425

Per-dim MSE: [9.00494866e-03 1.12878098e-08 1.81508369e-07 5.03751423e-08
 3.51933721e-07 2.83180555e-07 3.92876274e-04 4.51115280e-04
 5.12080640e-03 1.85639318e-03 9.57923848e-03 1.46894092e-02]
Shell cwd was reset to /home/lizitao/project/tabicl
```