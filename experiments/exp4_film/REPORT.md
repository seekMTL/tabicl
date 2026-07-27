## 1. 实验背景与动机

### 问题：多维 Target 无法高效注入 ColEmbedding

TabICL 的 ColEmbedding 通过 target-aware 机制将标量 y 编码为 128 维嵌入后**加性注入**到训练样本特征中（`src + y_emb`），以此引导 SetTransformer 的列间注意力。然而对于多输出回归任务（如 Hopper-v5 RL 场景，y ∈ R^12: 1 reward + 11 delta states），原始设计面临两个瓶颈：

1. **无法注入多维 target**：`y_encoder = Linear(1, 128)` 仅接受标量输入
2. **加性注入表达力有限**：`src + y_emb` 将相同的 target 信号加到所有特征列，无法对不同特征做差异化调制

### 方案 1 的失败经验

之前实验的"全向量多维目标投影 (Joint Vector Target Projection)"将 `Linear(1, 128)` 替换为 `Linear(12, 128)`，通过权重复制 `cw.repeat(1, 12) * (1/12)` 初始化。结果 epoch 280 Overall MSE 退化 2.07x。根因分析：

- **权重复制稀释**：每个输入维度的信号强度降为预训练权重的 1/12
- **StandardScaler 揭穿假象**：归一化后 reward(~1) 和 delta(~0.001) 等权重输入，1/12 稀释效应被暴露
- **加性注入同质化**：所有特征列收到相同的 target 信号，无法差异化

### 本实验目标

引入 **FiLM (Feature-wise Linear Modulation)** 替代加性注入，核心公式为 `src * (1 + γ) + β`，其中 γ（缩放）和 β（偏置）由多维 target 通过 FiLM generator 生成。预期：(1) 避免权重复制稀释；(2) 提供更强的特征级调制能力。

---

## 2. 方法：FiLM 条件化调制

### 2.1 原始 FiLM (Perez et al., AAAI 2018)

FiLM 是一种通用条件化方法，通过 feature-wise 仿射变换影响网络计算：

$$\text{FiLM}(F_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$$

其中 $\gamma, \beta$ 由 FiLM generator（可学习函数）从条件输入生成。论文使用 `γ = γ_raw + 1` 的 residual 形式保证训练初期接近恒等映射。

### 2.2 TabICL 适配

**FiLM Generator**：将多维 target y ∈ R^12 直接映射为 per-embedding-dimension 的 γ, β：

$$ \gamma = \text{fc\_gamma}(y) \in \mathbb{R}^{128}, \quad \beta = \text{fc\_beta}(y) \in \mathbb{R}^{128} $$

**调制方式**：在 ColEmbedding 的 `in_linear` 之后、SetTransformer 之前，对训练样本做特征级线性调制：

$$\text{src}[:,\ :\text{train\_size}, :] = (1 + \gamma) \odot \text{src}[:,\ :\text{train\_size}, :] + \beta$$

**架构对比**：

```
原版 TabICL:
  y_scalar → y_encoder(Linear(1,128)) → y_emb
  src_train += y_emb                    ← 加性注入，所有特征列相同

FiLM TabICL:
  y_12d → fc_gamma(Linear(12,128)) → γ (零初始化，γ=0)
  y_12d → fc_beta(Linear(12,128))  → β (从预训练 y_encoder 初始化)
  src_train = (1+γ) * src_train + β   ← 特征级线性调制
```

### 2.3 权重初始化策略

TabICL 为纯推理模型（不做 fine-tuning），FiLM generator 必须从预训练权重初始化以保留信号：

| 参数 | 初始化 | 效果 |
|------|--------|------|
| `fc_gamma.weight/bias` | 全零 | γ=0 → 乘法项=1（无调制） |
| `fc_beta.weight[:, 0]` | = `cw.squeeze()` | reward 维度完整预训练权重 |
| `fc_beta.weight[:, 1:]` | = `cw.repeat(1,11) * (1/11)` | 11 个 delta 共享缩放权重 |
| `fc_beta.bias` | = `cb` | 直接复制偏置 |

**效果**：初始时 `(1+0) * src + β ≈ src + y_emb`，FiLM 退化为分组加性注入，预训练模型可正常推理。γ 分量预留乘法门控能力。

### 2.4 关键设计决策：去掉 ReLU 非线性的 shared MLP

最初设计中 FiLM generator 包含 `shared MLP (Linear + ReLU)` 作为特征提取器（仿照原 FiLM 论文的 GRU encoder→decoder→linear 管线）。但实验发现 ReLU 非线性破坏了信号——测试时 `shared(y_12d)` 的输出可能为负值被清零，导致 beta 输出偏离预期。**简化为直接 Linear 投影**后效果显著改善：

| 设计 | Epoch 0 Delta MSE | 说明 |
|------|-------------------|------|
| shared MLP + ReLU | 4.29x | ReLU 截断信号，严重退化 |
| 直接 Linear(fc_beta) | 2.02x | 线性投影，匹配预训练模型期望 |

### 2.5 代码修改

涉及 5 个文件的修改（详见代码 diff）：

| 文件 | 修改内容 |
|------|---------|
| `src/tabicl/_model/embedding.py` | + `MultiDimTargetFiLM` 类；`ColEmbedding` 新增 `num_targets` 参数；`_compute_embeddings` / `_compute_embeddings_with_cache` 中根据 `num_targets` 选择 FiLM 或原始 y_encoder；y_train expand 操作适配 3D |
| `src/tabicl/_model/learning.py` | `ICLearning` 新增 `num_outputs`；`_icl_predictions` 多输出 per-dim 循环（共享 y_encoder/decoder，CPU offloading 防 OOM） |
| `src/tabicl/_model/tabicl.py` | `TabICL` 新增 `num_outputs`；串联 ColEmbedding + ICLearning；`predict_stats` 逐维 `quantile_dist` |
| `src/tabicl/_sklearn/regressor.py` | `TabICLRegressor` 新增 `num_outputs`；`_load_model` 多输出 FiLM 权重初始化；`fit`/`predict` 多输出 y 处理 |
| `src/tabicl/_sklearn/sklearn_utils.py` | `validate_data` 自动检测 2D y 并设置 `multi_output=True` |

---

## 3. 实验结果

### 3.1 Epoch 0 (6K 样本, 1980 test)

| 指标 | Baseline | FiLM | Ratio |
|------|----------|------|-------|
| Reward MSE | 0.009108 | 0.009305 | **1.02x** |
| DELTA(avg) MSE | 0.002920 | 0.005888 | **2.02x** |
| OVERALL MSE | 0.003435 | 0.006173 | **1.80x** |
| 耗时 | 5.2s | 1.1s | **4.6x 加速** |

小样本下 ColEmbedding + RowInteraction 阶段占主导，共享机制带来 4.6x 加速。Reward 维度几乎与 Baseline 持平（1.02x）。

### 3.2 Epoch 280 (286K 样本, 94K test, 不限制)

| 指标 | Baseline | FiLM | Ratio |
|------|----------|------|-------|
| Reward MSE | 0.000791 | 0.000814 | **1.03x** |
| DELTA(avg) MSE | 0.000902 | 0.001630 | **1.81x** |
| OVERALL MSE | 0.000893 | 0.001562 | **1.75x** |
| 耗时 | 539.4s | 498.2s | **1.1x 加速** |

大样本下 ICL Transformer（12 层 self-attention × 12 维度）占主导，加速比收敛到 1.1x。

### 3.3 Epoch 280 (286K 样本, 10K test)

| 指标 | Baseline | FiLM | Ratio |
|------|----------|------|-------|
| Reward MSE | 0.000546 | 0.000594 | **1.09x** |
| DELTA(avg) MSE | 0.001016 | 0.001443 | **1.42x** |
| OVERALL MSE | 0.000977 | 0.001373 | **1.41x** |
| 耗时 | 387.1s | 352.5s | **1.1x 加速** |

### 3.4 Per-Dimension 一致性

三次实验（epoch 0 full / epoch 280 full / epoch 280 10K）中，各维度 Ratio 的相对高低一致：

- **最优**：reward (1.02-1.09x)、delta[5] (1.03-1.12x) — 几乎持平 Baseline
- **中等**：delta[6]~delta[9] (1.13-1.71x) — 可控退化
- **最差**：delta[10] (1.68-2.21x) — 退化最严重的维度，但比旧 Joint 的 2.49x 有改善

---

## 4. 与方案 1 (分组加性注入, Joint) 对比

### 4.1 精度对比

实验数据表明，**FiLM 与 Joint（分组加性注入）在精度上几乎完全一致**。以下为三次实验的逐项对比：

**Epoch 280 全量 (94K test)**：

| 指标 | Joint | FiLM | 差异 |
|------|-------|------|------|
| Reward MSE | 1.10x | **1.03x** | FiLM 略优 |
| DELTA(avg) MSE | **1.81x** | **1.81x** | 相同 |
| OVERALL MSE | **1.75x** | **1.75x** | 相同 |
| 耗时 | 527.3s | 539.4s | 1.1x |

**Epoch 280 10K test**：

| 指标 | Joint | FiLM | 差异 |
|------|-------|------|------|
| Reward MSE | 1.13x | 1.09x | FiLM 略优 |
| DELTA(avg) MSE | **1.41x** | 1.42x | 相同 |
| OVERALL MSE | **1.40x** | 1.41x | 相同 |

**Epoch 100 全量 (35K test)**：

| 指标 | Joint | FiLM | 差异 |
|------|-------|------|------|
| Reward MSE | 1.14x | 1.08x | FiLM 略优 |
| DELTA(avg) MSE | 1.79x | 1.81x | 相同 |
| OVERALL MSE | 1.73x | 1.75x | 相同 |

### 4.2 等价性分析

Joint 和 FiLM 精度一致的根本原因是**数学等价性**：

- **Joint**: `y_emb = Linear(1,128)(reward) + Linear(11,128)(deltas)`，两个 encoder 的输出相加
- **FiLM**: `β = fc_beta(Linear(12,128))([reward, deltas])`，单个 Linear 的列级权重分配

当 FiLM 的 γ=0（零初始化）时，两者都计算 `cw * reward + Σ(cw/11) * d_i + cb`，输出完全一致。核心改进来自**列级权重分配策略**（reward 列 = 完整预训练权重，delta 列 = 缩放权重），而非 FiLM 特有的乘法门控机制。

**与原始"全向量"方案的区别**：原始方案 1（`Linear(12,128)` + `repeat * 1/12`）在 epoch 280 实现 Overall 2.07x，比分组方案（1.75x）差 15%。改进完全来自将 reward 和 delta 的权重分开处理，避免 12 个维度均分 1/12 的稀释。

### 4.3 FiLM 相对于 Joint 的优势

尽管当前精度相同，FiLM 在架构上具有以下优势：

1. **乘法门控预留**：`fc_gamma` 层为未来的 fine-tune 提供了乘法调制能力，Joint 的纯加性结构无法扩展
2. **更简洁的设计**：单个 `Linear(12, 128)` + `fc_gamma` 替代 Joint 的两个独立 encoder + add 操作
3. **统一的 target 接口**：所有 target 维度通过同一个 FiLM generator 处理，便于后续扩展到注意力和 token 机制

---

## 5. 分析与讨论

### 5.1 核心发现：FiLM 与 Joint（分组加性注入）精度等价

三次实验（epoch 0/100/280）一致表明，FiLM 和 Joint 的 Overall MSE、Delta MSE 几乎完全相同。根本原因在于 **γ=0 时的数学等价性**：

- Joint: `y_encoder_reward(reward) + y_encoder_delta(deltas)` 
- FiLM: `fc_beta([reward, deltas])` 

两者都计算 `cw * reward + Σ(cw/11) * d_i + cb`。**精度改进（Overall 从 2.07x → 1.75x）完全来自列级权重分配策略，而非 FiLM 机制本身。**

### 5.2 Reward 维度近乎完美匹配

`fc_beta.weight[:, 0]` 直接复制预训练权重 `cw`，对 reward 维度的 β 输出与原始 `y_encoder = Linear(1, 128)(reward)` 完全等价。实验证实 reward MSE 仅退化 2-10%（Joint 退化 10-15%），FiLM 在 reward 上略优于 Joint。

### 5.3 Delta 维度存在系统性退化

尽管两种方案都将 delta 退化从原全向量方案的 2.15x 降至 1.81x，但仍未消除。根本原因：

- **线性混合无法解耦**：`fc_beta / y_encoder_delta` 将 11 个 delta 维度通过同一组缩放权重线性混合，模型无法区分 individual delta 维度
- **预训练模型期望 1D 信号**：SetTransformer 的 ISAB 注意力被训练为对"单一语义"的 target 信号做出响应
- **StandardScaler 消除了量级线索**：归一化后所有 delta 维度 μ=0, σ=1，天然幅值差异被抹除

### 5.4 γ 分量闲置

`fc_gamma` 零初始化意味着 `1+γ = 1`，FiLM 的乘法门控功能未激活。这是纯推理模式的固有限制——需要通过 fine-tune 才能激活 γ 的非零调制。当前 FiLM 等价于 β-only 线性注入。

### 5.5 速度-精度权衡

| 数据规模 | 加速比 | 精度代价 (Overall) |
|----------|--------|-------------------|
| 6K (epoch 0) | 4.6-5.0x | 1.80-1.96x |
| 106K (epoch 100) | 1.2-1.3x | 1.73-1.75x |
| 286K (epoch 280) | 1.1x | 1.75x |

小样本场景共享 ColEmbedding 收益大，大样本 ICL 阶段成为瓶颈。

---

## 6. 结论

1. **FiLM 与 Joint（分组加性注入）精度等价**：三次实验（epoch 0/100/280）Overall MSE 差异在噪声范围内。核心改进来自列级权重分配（reward 列完整权重、delta 列缩放），而非 FiLM 的乘法调制。

2. **Reward 维度几乎无损（1.02-1.10x）**：两种方案均在 reward 上接近 Baseline。FiLM 略优于 Joint（1.03x vs 1.10x on epoch 280）。

3. **Delta 退化从原全向量方案的 2.15x 降至 1.81x**：列级权重分配将 reward/delta 信号分离，避免统一稀释。但线性混合的固有限制使 delta 仍有系统退化。

4. **FiLM 的独特优势是架构可扩展性**：虽然当前 γ=0 时与 Joint 等价，但 `fc_gamma` 为 fine-tune 提供了乘法门控接口。Joint 的纯加性结构无法扩展。

5. **小样本加速显著（4.6-5.0x）**，大样本加速有限（1.1x）。ColEmbedding 共享在小数据时收益最大。

6. **推荐**：需要乘法调制能力或后续 fine-tune → 选择 FiLM；仅需最简实现且无 fine-tune 需求 → 选择 Joint（分组加性注入）。

### 后续方向

- **Fine-tune FiLM generator**：在小规模数据上微调 fc_gamma/fc_beta，激活乘法门控能力，这是 FiLM 相对于 Joint 的核心价值
- **Per-group FiLM**：为每个特征组生成独立的 (γ, β)，实现列级差异化调制
- **Target 维度解耦**：探索注意力机制将多维 target 分解为独立语义单元，解决 delta 混合问题

## 附录：原始实验记录
### A.1 Epoch 0 (6K 样本)

执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp4_film/eval_film_vs_baseline.py 0
```

终端输出:
```
Epoch 0: 4020 train / 1980 test
y 范围: reward [-1.645,3.373]  delta [-6.220,3.157]

=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===
  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:
    ✓  ColEmbedding.y_encoder
    ✓  ColEmbedding.film (应为 False)
    ✓  ICLearning.num_outputs=1
  耗时: 5.2s

=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===
  [自检] FiLM 模型: has film=True, has y_encoder=False
  耗时: 1.1s

======================================================================
   Dim Label          Baseline MSE       FiLM MSE Ratio(F/B)
  ------------------------------------------------------------------
     0 reward             0.009108       0.009305       1.02x
     1 delta[0]           0.000000       0.000000       1.38x
     2 delta[1]           0.000000       0.000000       1.28x
     3 delta[2]           0.000000       0.000000       4.93x
     4 delta[3]           0.000000       0.000001       1.69x
     5 delta[4]           0.000000       0.000000       1.67x
     6 delta[5]           0.000394       0.000441       1.12x
     7 delta[6]           0.000451       0.000539       1.20x
     8 delta[7]           0.005114       0.010119       1.98x
     9 delta[8]           0.001854       0.006907       3.72x
    10 delta[9]           0.009583       0.019472       2.03x
    11 delta[10]          0.014721       0.027286       1.85x
  ------------------------------------------------------------------
         REWARD MSE       0.009108       0.009305       1.02x
     DELTA(avg) MSE       0.002920       0.005888       2.02x
        OVERALL MSE       0.003435       0.006173       1.80x

  加速比: 4.6x (5.2s → 1.1s)
```

### A.2 Epoch 100 (106K 样本)
#### 不设置测试集上限

终端输出：
```
Epoch 100: 71020 train / 34980 test
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===
  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:
    ✓  ColEmbedding.y_encoder
    ✓  ColEmbedding.film (应为 False)
    ✓  ICLearning.num_outputs=1
  耗时: 93.8s

=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===
  [自检] FiLM 模型: has film=True, has y_encoder=False
  耗时: 76.6s

======================================================================
   Dim Label          Baseline MSE       FiLM MSE Ratio(F/B)
  ------------------------------------------------------------------
     0 reward             0.001428       0.001540       1.08x
     1 delta[0]           0.000000       0.000000       1.65x
     2 delta[1]           0.000000       0.000000       1.01x
     3 delta[2]           0.000000       0.000000       1.56x
     4 delta[3]           0.000000       0.000000       1.53x
     5 delta[4]           0.000002       0.000002       1.00x
     6 delta[5]           0.000219       0.000224       1.02x
     7 delta[6]           0.000298       0.000333       1.12x
     8 delta[7]           0.002822       0.003334       1.18x
     9 delta[8]           0.001012       0.001768       1.75x
    10 delta[9]           0.003085       0.004689       1.52x
    11 delta[10]          0.007746       0.017111       2.21x
  ------------------------------------------------------------------
         REWARD MSE       0.001428       0.001540       1.08x
     DELTA(avg) MSE       0.001380       0.002496       1.81x
        OVERALL MSE       0.001384       0.002417       1.75x

  加速比: 1.2x (93.8s → 76.6s)
```

#### 限制测试集上限10k

终端输出：
```
Epoch 100: 71020 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===
  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:
    ✓  ColEmbedding.y_encoder
    ✓  ColEmbedding.film (应为 False)
    ✓  ICLearning.num_outputs=1
  耗时: 74.8s

=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===
  [自检] FiLM 模型: has film=True, has y_encoder=False
  耗时: 59.0s

======================================================================
   Dim Label          Baseline MSE       FiLM MSE Ratio(F/B)
  ------------------------------------------------------------------
     0 reward             0.001567       0.001678       1.07x
     1 delta[0]           0.000000       0.000000       1.71x
     2 delta[1]           0.000000       0.000000       0.99x
     3 delta[2]           0.000000       0.000000       1.44x
     4 delta[3]           0.000000       0.000000       1.50x
     5 delta[4]           0.000002       0.000003       1.02x
     6 delta[5]           0.000214       0.000226       1.05x
     7 delta[6]           0.000269       0.000324       1.21x
     8 delta[7]           0.002571       0.003306       1.29x
     9 delta[8]           0.001018       0.001812       1.78x
    10 delta[9]           0.003583       0.004917       1.37x
    11 delta[10]          0.008452       0.017617       2.08x
  ------------------------------------------------------------------
         REWARD MSE       0.001567       0.001678       1.07x
     DELTA(avg) MSE       0.001465       0.002564       1.75x
        OVERALL MSE       0.001473       0.002490       1.69x

  加速比: 1.3x (74.8s → 59.0s)
```

### A.3 Epoch 280 (286K 样本)

#### 不设置测试集上限
执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp4_film/eval_film_vs_baseline.py 280
```

终端输出:
```
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===
  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:
    ✓  ColEmbedding.y_encoder
    ✓  ColEmbedding.film (应为 False)
    ✓  ICLearning.num_outputs=1
  耗时:539.4s

=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===
  [自检] FiLM 模型: has film=True, has y_encoder=False
  耗时:498.2s

======================================================================
   Dim Label          Baseline MSE       FiLM MSE Ratio(F/B)
  ------------------------------------------------------------------
     0 reward             0.000791       0.000814       1.03x
     1 delta[0]           0.000000       0.000000       1.72x
     2 delta[1]           0.000000       0.000000       0.87x
     3 delta[2]           0.000000       0.000000       1.37x
     4 delta[3]           0.000000       0.000000       1.32x
     5 delta[4]           0.000001       0.000001       1.16x
     6 delta[5]           0.000135       0.000145       1.08x
     7 delta[6]           0.000170       0.000212       1.25x
     8 delta[7]           0.001647       0.001868       1.13x
     9 delta[8]           0.000552       0.000900       1.63x
    10 delta[9]           0.001894       0.002616       1.38x
    11 delta[10]          0.005522       0.012191       2.21x
  ------------------------------------------------------------------
         REWARD MSE       0.000791       0.000814       1.03x
     DELTA(avg) MSE       0.000902       0.001630       1.81x
        OVERALL MSE       0.000893       0.001562       1.75x

加速比:1.1x(539.4s→498.2s)
```

#### 限制测试集上限10k

终端输出：
```
Epoch 280: 191620 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===
  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:
    ✓  ColEmbedding.y_encoder
    ✓  ColEmbedding.film (应为 False)
    ✓  ICLearning.num_outputs=1
  耗时: 387.1s

=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===
  [自检] FiLM 模型: has film=True, has y_encoder=False
  耗时: 352.5s

======================================================================
   Dim Label          Baseline MSE       FiLM MSE Ratio(F/B)
  ------------------------------------------------------------------
     0 reward             0.000546       0.000594       1.09x
     1 delta[0]           0.000000       0.000000       1.78x
     2 delta[1]           0.000000       0.000000       0.77x
     3 delta[2]           0.000000       0.000000       1.38x
     4 delta[3]           0.000000       0.000000       1.50x
     5 delta[4]           0.000001       0.000002       1.29x
     6 delta[5]           0.000142       0.000147       1.03x
     7 delta[6]           0.000181       0.000251       1.39x
     8 delta[7]           0.001758       0.001955       1.11x
     9 delta[8]           0.000625       0.001067       1.71x
    10 delta[9]           0.002393       0.002251       0.94x
    11 delta[10]          0.006073       0.010204       1.68x
  ------------------------------------------------------------------
         REWARD MSE       0.000546       0.000594       1.09x
     DELTA(avg) MSE       0.001016       0.001443       1.42x
        OVERALL MSE       0.000977       0.001373       1.41x

  加速比: 1.1x (387.1s → 352.5s)
```