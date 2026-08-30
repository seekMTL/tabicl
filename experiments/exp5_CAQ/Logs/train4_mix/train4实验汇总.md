# CAQ-TabICL 混合 RL/SCM 训练实验汇总

> 覆盖 `train4_mix` 全部训练实验、`train3_parallel` 的 Ant 诊断实验、以及三组评估拆解实验。
> 所有结论均来自对应日志记录（各日志开头含执行命令）；标注了不可直接对比之处。
> 汇总日期：2026-08-15。

---

## 一、训练实验总表（6 个）

| # | 实验/日志 | 脚本 | 环境 | 训练部件 | 冻结 | TA | action 通道 | SCM replay | 训练 ctx/query | eval ctx | lr | 验证假设 | 结果 (CAQ/Baseline) |
|---|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 1 | `4env_ctx1024.log` | train_large4_mixed.py | 4 环境 | Col+Row+CAQ (1.44M, 5.0%) | ICL (27.3M) | ❌ 关 | 仅 causal | ✅ 80/20 | 1024/512 | 20K | 1e-5 | 基线 | Hopper 0.83x / Walker2d 1.25x / HalfCheetah 0.96x / Ant 2.40x → **OVERALL 1.24x**（3.2h） |
| 2 | `ant_v4_single_ctx1024.log` | train_large4_mixed.py | Ant 单环境 | 同上 | 同上 | ❌ 关 | 仅 causal | ✅ | 1024/512 | 20K | 1e-5 | 环境干扰 | Ant **2.02x**（4.1h）→ 假设 ❌ 排除 |
| 3 | `mixed2_block4_ctx1024.log` | train_large4_mixed2.py | 4 环境 | 同上 | 同上 | ❌ 关 | 仅 causal | ✅ | 1024/512 | 20K | 1e-5 | 块状轮流 (block=4) | Hopper 0.84x / Walker2d 1.23x / HalfCheetah 1.03x / Ant 2.58x → **1.28x**（3.9h）→ 假设 ❌ 排除（略差） |
| 4 | `4env_ctx3000.log` | train_large4_mixed2.py | 4 环境 | 同上 | 同上 | ❌ 关 | 仅 causal | ✅ | **3000**/512 | 20K | 1e-5 | context | Hopper **0.68x** / Walker2d **1.15x** / HalfCheetah **0.82x** / Ant 2.01x → **1.08x**（16.8h, 49G）→ ✅ 成立，三环境历史最佳 |
| 5 | `train3_parallel/ant_ctx1024_eval50k.log` | train_large3_multenv.py | Ant 单环境 | **仅 CAQ** (167K) | **Col+Row+ICL 全冻结** | ❌ 关 | 仅 causal | **无 replay** | 1024/512 | **50K** | **1e-4** | 架构诊断 | Ant **3.52x**（2.2h） |
| 6 | `actionfeat_ant_ctx1024.log` | train_large4_actionfeat.py | Ant 单环境 | Col+Row+CAQ | ICL | ❌ 关 | **特征 (35列) + causal** | ✅ | 1024/512 | 20K | 1e-5 | action 入特征 | Ant **2.66x**（7.4h）→ TA-off 条件下未修复 |

### 脚本差异要点

- **train_large4_mixed.py**（v1）：环境每步轮流；无 `--env_block_size`/`--scm_loss_weight` 参数。
- **train_large4_mixed2.py**（v2）：块状轮流（`--env_block_size`，默认 4；实验 4 显式设 1 恢复每步轮流）+ `--scm_loss_weight` + 最终报告含 best/last。
- **train_large4_actionfeat.py**：基于 mixed2 + action 拼入特征表（与 state 同管线 z-score）+ causal 路径保留。
- **train_large3_multenv.py**：仅训 CAQ 模块（Col/Row/ICL 全冻结）、无 SCM replay、lr 1e-4、eval 默认 100K（本实验用 50K）。
- **关键事实**：所有 CAQ 模型构造时 `target_aware` 被硬编码关闭（caq_model.py），即实验 1-6 全部在 TA-off（无 y 条件列嵌入）下运行。

---

## 二、评估实验总表（3 个脚本，9 项，均 Ant-v4、eval ctx 20K）

| # | 对照项 | 脚本 | 冻结/适配 | TA | action 通道 | OVERALL | dims 20-27 |
|---|------|------|------|------|------|------|------|
| 7 | Baseline n_estimators=4 | eval_targetaware.py | — | ✅ | 特征（baseline 自带） | **1.00x** | ~1.0x |
| 8 | Baseline n_estimators=1 | eval_targetaware.py | — | ✅ | 特征 | 1.01x | ~1.0-1.08x |
| 9 | 冻结 + TA-off + action 特征 | eval_frozen_actionfeat.py | 全冻结 | ❌ 关 | 特征 | **9.19x** | 9-49x |
| 10 | 冻结 + TA-on + action 特征 | eval_targetaware2.py | 全冻结 | ✅ 开 | 特征 | **1.01x** | 1.0-1.08x |
| 11 | 冻结 + TA-on + state-only | eval_targetaware2.py | 全冻结 | ✅ 开 | **无 action 信息** | **17.86x** | 28-97x |
| 12 | ant_v4_single 适配 + TA-on + state-only | eval_targetaware2.py | 适配（TA-off 训练） | ✅ 开（评估时） | 仅 causal（训练过） | 2.13x | 2.9-7.4x |
| 13 | actionfeat 适配 + TA-on + action 特征 | eval_targetaware2.py | 适配（TA-off 训练） | ✅ 开（评估时） | 特征+causal | 1.48x | 1.45-3.43x |
| 14 | actionfeat 适配 + TA-off（训练时评估） | actionfeat 训练日志 | 适配 | ❌ 关 | 特征+causal | 2.66x | — |

---

## 三、可靠结论

**1. target_aware 是被发现的系统性缺陷，且是最强的单因素。**
对比 #9 与 #10：同一冻结模型、同一数据，仅开/关 target-aware，9.19x → 1.01x。
CAQ 模型构造时硬编码关闭它（为了共享嵌入并行训练），导致训练实验 1-6 全部运行在
残缺信息流下（包括 Hopper 0.68x 等胜利——它们的含金量因此更高）。

**2. action 信息绝对必需，且"特征通道"显著强于"causal 通道"。**
- 无 action 信息（#11）= 17.86x：dims 20-27 达 28-97x，dims 6/8/10/12 也崩到 7-11x
- 特征通道（#10）= 1.01x ≈ baseline
- causal 通道最优形态（#12）= 2.13x——CAQ 路径**有效**（从 17.86x 救回），
  但信息承载能力约为特征通道的一半

**3. "冻结 + TA-on + action 特征 = baseline"（#10）是核心等式。**
预训练管线零适配即达 baseline 水平；ensemble 增益几乎为零（#7 vs #8）。

**4. 已排除的假设（TA-off 世界内）：**
环境干扰（#2：单环境仍 2.02x）、块状轮流（#3：1.28x 略差）。
**已验证的杠杆：** context（#4：1.24x → 1.08x，Hopper/Walker2d/HalfCheetah 三环境历史最佳）。

**5. 已裁决：TA-on 下从头适配是有效的（详见第四节）。**
#13（1.48x）与 #12（2.13x）是"TA-off 训练权重 + TA-on 评估"的**组合实验**，
只能说明该组合不兼容，不能裁决适配本身。V1/V2b 实验给出裁决：
action 特征 + TA-on 适配 = **0.94x（超越 baseline）**；state-only + TA-on 适配 = 1.00x。

### 对比边界声明（避免误导）

- train3 的 3.52x 是 eval ctx **50K**，与 20K 系列只能定性对比。
- actionfeat 的 2.66x 与 ant_v4_single 的 2.02x 之间存在**两个变量**（action 特征通道 + action z-score 管线），差值不能单独归因。
- #12/#13 为"TA-off 训练权重 + TA-on 评估"组合实验，不构成对"适配本身"的裁决。

---

## 四、TA-on 裁决实验（V1 / V2b，均已完成，2026-08-17）

### 实验设置与结果

| 项 | V1（actionfeat+TA-on） | V2b（state-only+TA-on） |
|---|------|------|
| 脚本 | `train_large5_targetaware.py` | `train_large5_taon_stateonly.py` |
| action 通道 | 特征（35 列）+ causal | 仅 causal（action_encoder→CLS 调制） |
| 设置 | Ant-v4 单环境，ctx1024/512，100步×300epoch，lr 1e-5，rl_ratio 0.8，eval ctx 20K | 同左 |
| 分块 | 49G，chunk=14，从 epoch 80 恢复 | 本地 24GB，chunk=7，从头 |
| 日志 | `Logs/train5_TA/antv4_ctx1024.log` | `Logs/train5_TA/antv4_ctx1024_stateonly.log` |
| 耗时 | 13.6h | 12.9h |
| **最终 avg** | **0.080668（last）/ 0.080431（best）** | 0.085619（last）/ 0.085423（best） |
| **vs baseline (0.085206)** | **0.94x / 0.95x** | **1.00x / 1.00x** |
| forget_score | 0.8~1.0 全程健康 | 0.7~1.1 全程健康 |

**逐维对比（vs baseline）**：

| 维度组 | V1（特征通道） | V2b（causal 通道） |
|------|------|------|
| reward (dim0) | 0.92x | **0.88x** |
| dims 14-19（state 驱动尾维） | 0.86~0.92x | 0.84~0.94x（略优） |
| **dims 20-27（action 驱动维，历史失败区）** | **0.91~1.13x** | 1.06~1.27x |

### 最终裁决

**1. "TA-on 下从头适配"的裁决：<1.0x 成立。** V1 = 0.94x，Ant 首次超越 baseline。
此前"适配是净破坏"的结论被推翻——TA-off 时代的破坏是**信息流缺陷（关 target-aware）
的产物**，不是适配本身。在正确信息流下，适配从冻结基线 1.01x 提升到 0.94x。

**2. 通道对比（TA-on 训练后）：特征通道（0.94x）> causal 通道（1.00x）。**
与冻结评估的排序一致（1.01x vs 17.86x），但差距从"数量级"缩小到"6%"。
causal 通道在 TA-on 下从 TA-off 的 2.13x 恢复到 1.00x——CAQ 路径的能力在
正确信息流下完全恢复，达到 baseline 水平但无法超越。

**3. 两通道各有强项**：V2b 赢 reward（0.88x vs 0.92x）和 state 驱动维；
V1 赢 action 驱动维（0.91~1.13x vs 1.06~1.27x）。V1 双通道并行时 reward 略逊于
V2b 单通道，提示 causal 调制对 reward 维可能有轻微干扰（可做 causal 消融验证）。

**4. 历史失败区（dims 20-27）被彻底解决**：从 TA-off 时代的 2~97x 劣化，
到 TA-on 时代的 0.91~1.27x——action 驱动维度的预测能力恢复。

### 过程记录（显存与工程）

- TA-on 全维批处理（B=28）实测 OOM：24GB 卡峰值 ~22.7GB、49G 卡（47.4GiB）
  峰值 ~47.2GiB。实现 `--ta_dim_chunk` 维度分块（随机维排列 → 逐块前向+立即反向，
  梯度按块权重累积等效全维，图随块释放）；ta_dim_chunk 越大越快（并行效率），
  选择原则为"显存允许的最大 chunk"
- SCM 生成器偶发 NaN y 批次 → target-aware 分支的 `int(y_train.max())` 前向内部
  崩溃（先于既有跳过保护）。修复：数据层拦截（scm_train_step/eval_scm_loss 内
  非有限 y 重采样最多 3 次 + 跳过兜底），**不修改 tabicl 源码**
- 恢复逻辑改进：从最新 `ckpt_epoch*.pt` 恢复权重/优化器，历史最优 best_mse 从
  best.pt 继承

### 下一步

1. **TA-on 4 环境混合（ctx3000）**：TA-off 时代 Hopper 0.68x / HalfCheetah 0.82x /
   Walker2d 1.15x 均在残缺信息流下取得，TA-on 重跑预计全面超越历史最佳
2. **causal_block 消融**（actionfeat+TA-on 去掉 causal 路径）：验证第 3 条的干扰假说，
   并决定 CAQ 模块在论文中的定位（辅助增强 vs 可移除）
3. Hopper/HalfCheetah 的 TA-on 单环境验证（与 Ant 0.94x 同协议）
