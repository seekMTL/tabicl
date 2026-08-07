### CAQ-TabICL 实验设计与代码修改方案

本方案Causal-Action Querying TabICL（CAQ-TabICL）基于 **“保留原生 $N$ 层 Block + 新增第 $N+1$ 层进行纯动作因果检索”** 的终极架构落地。

#### 1. 阶段 2 (`RowInteraction`) 数据流与张量规格对照表

假设输入特征组数为 $G$（对应 11 个状态特征列在 `group_size=3` 下的分组），嵌入维度为 **128**，预留 CLS 数量为 **4**。  

| **步骤**   | **操作模块**                  | **输入 Shape**      | **输出 Shape**     | **物理与数学意义**                                           |
| ---------- | ----------------------------- | ------------------- | ------------------ | ------------------------------------------------------------ |
| **Step 1** | **前 $N-1$ 层原生 Blocks**    | `(B, T, G+4, 128)`  | `(B, T, G+4, 128)` | 状态列之间进行充分的行内自注意力交互                         |
| **Step 2** | **切片缓存特征组 ($K, V$)**   | `x[:, :, 4:, :]`    | `(B, T, G, 128)`   | 提取后 $G$ 个带有局部物理语义的特征组，作为后续检索的 Key 和 Value |
| **Step 3** | **第 $N$ 层原生 Block**       | `(B, T, G+4, 128)`  | `(B, T, G+4, 128)` | 通过非对称注意力，将全表状态压缩汇聚到前 **4** 个 CLS 向量中 |
| **Step 4** | **提取基础状态 `cls_1`**      | `x[:, :, :4, :]`    | `(B, T, 4, 128)`   | 获得完备且纯净的当前系统宏观全局状态缩影                     |
| **Step 5** | **动作因果 Query 构造 ($Q$)** | `cls_1` + `MLP(a)`  | `(B, T, 4, 128)`   | 将 **3** 维动作映射为 **4 × 128** 矩阵并加法注入：$Q = \text{LayerNorm}(\text{cls\_1} + A_{\text{emb}})$ |
| **Step 6** | **新增第 $N+1$ 层 Block**     | $Q, K, V$           | `(B, T, 4, 128)`   | 执行因果交叉注意力检索，配套残差连接与 FFN 融合，得到因果更新后的 `cls_2` |
| **Step 7** | **维度扁平化 (Flatten)**      | `cls_2.flatten(-2)` | `(B, T, 512)`      | 将 **4 × 128** 拼接为标准的 **512** 维行表示，严密对接阶段 3 |

#### 2. 关键代码落地修改指南 (PyTorch 实现)

请直接定位到 TabICL 源码中的 `RowInteraction` 模块（通常位于 `src/tabicl/_model/interaction.py` 或相关网络结构文件中），将其替换/重构为以下实现：

```python
import torch
import torch.nn as nn

class CausalActionEncoder(nn.Module):
    """将连续或离散动作映射为与 CLS slots 匹配的特征空间 (4, 128)"""
    def __init__(self, action_dim=3, num_cls=4, d_model=128):
        super().__init__()
        self.num_cls = num_cls
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_cls * d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, action):
        # action shape: (B, T, action_dim)
        B, T, _ = action.shape
        a_emb = self.mlp(action).view(B, T, self.num_cls, self.d_model)
        return self.norm(a_emb)


class CausalInterventionBlock(nn.Module):
    """新增的第 N+1 层：负责用 Action+CLS 作为 Query 检索特征组"""
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, Q, KV):
        """
        Q (cls_2): (B, T, 4, 128) -> 需展平 B*T 进行并行注意力计算
        KV (features): (B, T, G, 128)
        """
        B, T, num_cls, d = Q.shape
        G = KV.shape[2]
        
        # 展平批次与序列维度以适配 MultiheadAttention
        q_flat = Q.view(B * T, num_cls, d)
        kv_flat = KV.view(B * T, G, d)
        
        # 1. 因果交叉注意力与残差
        attn_out, _ = self.cross_attn(query=q_flat, key=kv_flat, value=kv_flat)
        x = self.norm1(q_flat + attn_out)
        
        # 2. FFN 与残差
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        
        return out.view(B, T, num_cls, d)


class CAQ_RowInteraction(nn.Module):
    """改造后的终极 RowInteraction 模块"""
    def __init__(self, native_blocks, action_dim=3, num_cls=4, d_model=128):
        super().__init__()
        # native_blocks: 原生 TabICL RowInteraction 中包含的 N 个 Block 列表
        self.blocks_except_last = nn.ModuleList(native_blocks[:-1])
        self.last_native_block = native_blocks[-1]
        
        # 新增模块
        self.action_encoder = CausalActionEncoder(action_dim, num_cls, d_model)
        self.causal_block = CausalInterventionBlock(d_model=d_model)
        self.num_cls = num_cls

    def forward(self, x, action):
        """
        x: (B, T, G+4, 128) - 从阶段 1 ColEmbedding 输出的特征
        action: (B, T, action_dim) - 动作数据
        """
        # Step 1: 正常执行前 N-1 个原生 Blocks
        for block in self.blocks_except_last:
            x = block(x)
            
        # Step 2: 截取并保存后 G 个特征组 Token 作为 K, V
        kv_features = x[:, :, self.num_cls:, :] # shape: (B, T, G, 128)
        
        # Step 3 & 4: 执行原生第 N 层 Block，聚合生成 cls_1[cite: 5]
        x_native_last = self.last_native_block(x)
        cls_1 = x_native_last[:, :, :self.num_cls, :] # shape: (B, T, 4, 128)
        
        # Step 5: 构造因果控制 Query
        a_emb = self.action_encoder(action)
        cls_2 = cls_1 + a_emb # 可以直接加和，causal_block 内部会做 Norm
        
        # Step 6: 执行第 N+1 层因果交叉检索
        cls_final = self.causal_block(Q=cls_2, KV=kv_features)
        
        # Step 7: Flatten 为 512 维输出给阶段 3[cite: 5]
        row_representation = cls_final.flatten(-2) # shape: (B, T, 512)[cite: 5]
        
        return row_representation
```

#### 3. 实验验证路线与实施参数

为清晰验证 CAQ-TabICL 改造效果，建议按以下路线开展实验：

##### 阶段 I：快速验证与注意力权重检查 (Sanity Check)

1. **数据配置**：选取 Hopper-v5 数据集的一个切片（例如使用 **4000** 个训练上下文样本）[cite: 5]。
2. **验证监控**：
   - 除了观察 Loss 下降曲线外，在 `CausalInterventionBlock` 中临时打印或保存 `self.cross_attn` 返回的注意力权重矩阵 `attn_weights`（形状为 **4 × G**）。
   - **成功标志**：对于不同的动作输入 $a$，注意力权重矩阵应当展现出**明显的稀释与动态变化**（例如：某个动作改变时，前 4 个 Query Token 对第 2、第 5 个特征组的注意力显著增加），这证明模型确实学到了动作对特定状态物理特性的因果检索。

##### 阶段 II：全量对齐与对照实验

建议继续采用两组核心对照进行实验最终定档：

- **对比架构 1 (Baseline)**：原封不动运行 12 个独立的原生 TabICL 模型（分别预测 **1** 维 Reward + **11** 维 $\Delta \text{State}$），记录目标基线误差。
- **对比架构 2 (本方案 CAQ-TabICL)**：直接运行此套改造后的新逻辑。
- **训练超参数建议**：
  - **学习率 (LR)**：保持 **1e-4**（如果带有 Cosine 退火策略则效果更佳）。
  - **特征分组**：保持原生默认参数 `feature_group=True, group_size=3`[cite: 5]。
  - **正则化辅助**：在总损失函数 $\mathcal{L} = \text{MSE}(\hat{y}, y)$ 基础上，可以添加一组微弱的动量或权重衰减（Weight Decay = **1e-5**），确保新增的第 $N+1$ 层网络在初期能够平稳接收原生层传来的梯度。