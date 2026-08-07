"""CAQ-TabICL: Causal-Action Querying RowInteraction 模块

基于"保留原生 N 层 Block + 新增第 N+1 层进行纯动作因果检索"的架构。
"""

from __future__ import annotations

from typing import Optional
from functools import partial

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .inference import InferenceManager
from .inference_config import MgrConfig


class CausalActionEncoder(nn.Module):
    """将连续动作映射为与 CLS slots 匹配的特征空间 (num_cls, d_model)。

    动作维度（如 Hopper 的 3 维连续动作）通过两层 MLP 映射为
    num_cls × d_model 的矩阵，每个 CLS token 获得一个动作感知的偏置。
    """

    def __init__(self, action_dim: int = 3, num_cls: int = 4, d_model: int = 128):
        super().__init__()
        self.num_cls = num_cls
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_cls * d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        # 零初始化最后一层 Linear，确保 action=0 时 a_emb ≈ 0
        # 这样未训练的 CAQ 输出与原版 RowInteraction 一致
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: (B, T, action_dim) - 动作数据
        Returns:
            a_emb: (B, T, num_cls, d_model) - 动作嵌入
        """
        B, T, _ = action.shape
        a_emb = self.mlp(action).view(B, T, self.num_cls, self.d_model)
        return self.norm(a_emb)


class CausalInterventionBlock(nn.Module):
    """新增的第 N+1 层：用 Action+CLS 作为 Query 检索特征组。

    执行因果交叉注意力：CLS token（注入了动作信息）作为 Query，
    原生 Block 输出的特征组作为 Key/Value，实现动作对状态特征的因果检索。
    配套残差连接与 FFN 融合。
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # 零初始化输出投影：确保训练初期 causal_block 输出 ≈ 0
        # 这样残差连接 cls_caq + cls_1 ≈ cls_1，保留预训练 CLS 质量
        # 与 MultiheadAttentionBlock.init_weights() 保持一致
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-2].weight)  # FFN 最后一个 Linear 层
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(self, Q: Tensor, KV: Tensor) -> Tensor:
        """
        Args:
            Q (cls + action):  (B, T, num_cls, d_model)
            KV (features):     (B, T, G, d_model)
        Returns:
            out: (B, T, num_cls, d_model) - 因果更新后的 CLS token

        采用标准 Pre-Norm 架构，与原生 MultiheadAttentionBlock 一致。
        零初始化输出投影确保训练初期该 Block 为恒等映射，不破坏预训练 CLS。
        """
        B, T, num_cls, d = Q.shape
        G = KV.shape[2]

        # 展平 B*T 维度以适配 MultiheadAttention 的 batch_first 格式
        # 使用 reshape 而非 view，避免非连续内存导致的 stride 错误
        q_flat = Q.reshape(B * T, num_cls, d) # (B*T, num_cls, d)
        kv_flat = KV.reshape(B * T, G, d) # (B*T, G, d)

        # Pre-norm 交叉注意力 + 残差
        attn_out, _ = self.cross_attn(
            query=self.norm1(q_flat), key=kv_flat, value=kv_flat
        )
        x = q_flat + attn_out

        # Pre-norm FFN + 残差
        ffn_out = self.ffn(self.norm2(x))
        out = x + ffn_out

        return out.view(B, T, num_cls, d)


class CAQ_RowInteraction(nn.Module):
    """改造后的 Causal-Action Querying RowInteraction 模块。

    核心数据流（对照设计方案 Step 1-7）：

    - Step 1: 前 N-1 层原生 Blocks 进行标准行内自注意力
    - Step 2: 切片缓存后 G 个特征组作为 K, V
    - Step 3: 第 N 层原生 Block 执行非对称注意力，聚合 CLS token
    - Step 4: 提取 cls_1（前 num_cls 个 token）
    - Step 5: 动作编码 + cls_1 构造因果 Query
    - Step 6: 新增第 N+1 层因果交叉检索
    - Step 7: Flatten 输出 (B, T, num_cls * d_model)
    """

    def __init__(
        self,
        native_blocks: nn.ModuleList,
        action_dim: int = 3,
        num_cls: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        rope=None,
    ):
        super().__init__()
        if len(native_blocks) < 2:
            raise ValueError(
                f"CAQ_RowInteraction requires at least 2 native blocks, "
                f"got {len(native_blocks)}"
            )

        self.num_cls = num_cls
        self.d_model = d_model
        self.embed_dim = d_model
        self.norm_first = norm_first
        self.recompute = recompute
        self.rope = rope  # RoPE 位置编码（从原生 Encoder 继承）

        # 前 N-1 个原生 Block（保持预训练权重）
        self.blocks_except_last = nn.ModuleList(native_blocks[:-1])
        # 第 N 个原生 Block（非对称注意力 CLS→features）
        self.last_native_block = native_blocks[-1]

        # 新增模块（随机初始化，需要训练）
        self.action_encoder = CausalActionEncoder(action_dim, num_cls, d_model)
        self.causal_block = CausalInterventionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # CLS token（可学习参数，从预训练权重加载）
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, d_model))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        # 输出层归一化
        self.out_ln = (
            nn.LayerNorm(d_model, bias=not bias_free_ln)
            if norm_first
            else nn.Identity()
        )

        # 推理管理器
        self.inference_mgr = InferenceManager(
            enc_name="tf_row", out_dim=d_model * num_cls, out_no_seq=True
        )

    def _aggregate_embeddings(
        self,
        embeddings: Tensor,
        action: Tensor,
        key_mask: Optional[Tensor] = None,
        rope=None,
    ) -> Tensor:
        """执行 CAQ 增强的嵌入聚合。

        Args:
            embeddings: (B, T, G+num_cls, d_model) - 特征嵌入 + CLS 占位
            action: (B, T, action_dim) - 动作数据
            key_mask: 可选，空特征掩码
            rope: RoPE 位置编码

        Returns:
            cls_final: (B, T, num_cls * d_model) - 展平后的行表示
        """
        # Step 1: 前 N-1 个原生 Block（标准自注意力）
        for block in self.blocks_except_last:
            if self.recompute:
                embeddings = checkpoint(
                    partial(block, key_padding_mask=key_mask, rope=rope),
                    embeddings,
                    use_reentrant=False,
                )
            else:
                embeddings = block(embeddings, key_padding_mask=key_mask, rope=rope)

        # Step 2: 截取后 G 个特征组 Token 作为 K, V（在最后一个 block 之前）
        kv_features = embeddings[:, :, self.num_cls:, :]  # (B, T, G, d_model)

        # Step 3 & 4: 执行原生第 N 层 Block（非对称注意力），获得 cls_1
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: self.last_native_block(
                    q=emb[..., : self.num_cls, :],
                    k=emb,
                    v=emb,
                    key_padding_mask=key_mask,
                    rope=rope,
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = self.last_native_block(
                q=embeddings[..., : self.num_cls, :],
                k=embeddings,
                v=embeddings,
                key_padding_mask=key_mask,
                rope=rope,
            )

        del embeddings

        # cls_1: (B, T, num_cls, d_model) - 完备的全局状态缩影
        cls_1 = cls_outputs  # 已经是 (B, T, num_cls, d_model)，last block 输出只有 CLS

        # Step 5: 构造因果控制 Query = cls_1 + ActionEmbedding
        a_emb = self.action_encoder(action)  # (B, T, num_cls, d_model)
        cls_2 = cls_1 + a_emb  # 动作信息作为偏置注入 CLS token

        # Step 6: 执行第 N+1 层因果交叉检索
        # causal_block 为零初始化 Pre-Norm 架构，训练初期为恒等映射
        # cls_caq ≈ cls_2 = cls_1 + a_emb，未训练时 ≈ cls_1（预训练质量）
        cls_caq = self.causal_block(Q=cls_2, KV=kv_features)

        # 输出归一化
        cls_final = self.out_ln(cls_caq)

        # Step 7: Flatten 为 (B, T, num_cls * d_model)
        return cls_final.flatten(-2)

    def _train_forward(
        self, embeddings: Tensor, action: Tensor, d: Optional[Tensor] = None
    ) -> Tensor:
        """训练模式前向传播。"""
        B, T, HC, E = embeddings.shape
        device = embeddings.device

        # 注入可学习的 CLS Token
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(device)

        # 空特征掩码
        if d is None:
            key_mask = None
        else:
            d_padded = d + self.num_cls
            indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)
            key_mask = indices >= d_padded.view(B, 1, 1)

        representations = self._aggregate_embeddings(
            embeddings, action, key_mask, self.rope
        )
        return representations  # (B, T, num_cls * d_model)

    def _inference_forward(
        self,
        embeddings: Tensor,
        action: Tensor,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """推理模式前向传播（简化版，不使用 InferenceManager）。"""
        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        # col_embedder 在 eval 模式下可能输出 float16，显式转 float32 避免
        # 与 float32 native blocks 的 LayerNorm 发生 dtype 不匹配
        embeddings = embeddings.float()

        return self._aggregate_embeddings(embeddings, action, None, self.rope)

    def forward(
        self,
        embeddings: Tensor,
        action: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """
        Args:
            embeddings: (B, T, G+num_cls, d_model) - 来自 ColEmbedding 的特征
            action:     (B, T, action_dim) - 动作数据
            d:          可选，每张表的特征数
            mgr_config: 推理配置

        Returns:
            representations: (B, T, num_cls * d_model) - 行表示
        """
        if self.training:
            return self._train_forward(embeddings, action, d)
        else:
            return self._inference_forward(embeddings, action, mgr_config)


def build_caq_row_interaction(
    row_interactor,
    action_dim: int = 3,
    dropout: float = 0.0,
    nhead: int = 8,
    dim_feedforward: int = 512,
) -> CAQ_RowInteraction:
    """从预训练的 RowInteraction 构建 CAQ_RowInteraction。

    提取原生 Block 权重和 RoPE，创建 CAQ_RowInteraction 并加载匹配的预训练参数。

    Args:
        row_interactor: 预训练的 RowInteraction 模块
        action_dim: 动作维度
        dropout: dropout 概率
        nhead: 注意力头数
        dim_feedforward: FFN 维度

    Returns:
        CAQ_RowInteraction 实例，原生 Block、CLS token、out_ln 权重已加载
    """
    native_blocks = row_interactor.tf_row.blocks
    rope = row_interactor.tf_row.rope  # 继承 RoPE 位置编码

    # 推断 bias_free_ln：检查原生 out_ln 是否为无 bias 的 LayerNorm
    bias_free_ln = False
    if isinstance(row_interactor.out_ln, nn.LayerNorm):
        bias_free_ln = row_interactor.out_ln.bias is None

    caq = CAQ_RowInteraction(
        native_blocks=native_blocks,
        action_dim=action_dim,
        num_cls=row_interactor.num_cls,
        d_model=row_interactor.embed_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=row_interactor.norm_first,
        bias_free_ln=bias_free_ln,
        recompute=row_interactor.recompute,
        rope=rope,
    )

    # 加载 CLS token 权重
    caq.cls_tokens.data.copy_(row_interactor.cls_tokens.data)

    # 加载输出层归一化权重（如果是 LayerNorm）
    if isinstance(caq.out_ln, nn.LayerNorm) and isinstance(
        row_interactor.out_ln, nn.LayerNorm
    ):
        caq.out_ln.load_state_dict(row_interactor.out_ln.state_dict())

    return caq
