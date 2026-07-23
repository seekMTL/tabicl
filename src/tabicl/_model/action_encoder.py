"""ActionEncoder: 连续动作序列的专用编码器。

与 ColEmbedding 并行工作，将动作特征编码为与 state 表示同维度的嵌入，
通过融合层与 state 表示组合后送入 ICLearning。

支持两种模式:
- MLP: 逐帧独立编码（默认）
- Transformer: 跨时序自注意力编码（实验 2）
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn, Tensor


class ActionEncoder(nn.Module):
    """将连续动作序列编码为固定维度表示。

    Parameters
    ----------
    action_dim : int
        动作空间维度（Hopper-v5 为 3）。
    d_model : int
        输出嵌入维度，应与 state 表示的 embed_dim 一致（默认 128）。
    hidden_dim : int, default=256
        MLP 隐藏层维度。
    mode : str, default="mlp"
        编码器类型：
        - ``"mlp"``: 两层 MLP + LayerNorm + GELU，逐帧独立编码。
        - ``"transformer"``: 轻量 Transformer encoder（实验 2 用）。
    num_blocks : int, default=2
        Transformer 层数（仅 mode="transformer" 时有效）。
    nhead : int, default=4
        注意力头数（仅 mode="transformer" 时有效）。
    dropout : float, default=0.0
        Dropout 概率。
    """

    def __init__(
        self,
        action_dim: int = 3,
        d_model: int = 128,
        hidden_dim: int = 256,
        mode: Literal["mlp", "transformer"] = "mlp",
        num_blocks: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.mode = mode

        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )
        elif mode == "transformer":
            from .encoders import Encoder
            self.in_proj = nn.Linear(action_dim, d_model)
            self.encoder = Encoder(
                num_blocks=num_blocks,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
                use_rope=False,  # 动作序列位置信息由时序自注意力隐式捕获
            )
            self.out_ln = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, actions: Tensor) -> Tensor:
        """编码动作序列。

        Parameters
        ----------
        actions : Tensor
            动作张量，shape (B, T, action_dim)，值域 [-1, 1]。

        Returns
        -------
        Tensor
            动作表示，shape (B, T, d_model)。
        """
        if self.mode == "mlp":
            return self.net(actions)  # (B, T, d_model)
        else:
            # Transformer mode: (B, T, action_dim) → (B, T, d_model)
            src = self.in_proj(actions)  # (B, T, d_model)
            # Encoder expects (B, T, d_model)
            src = self.encoder(src)
            return self.out_ln(src)  # (B, T, d_model)
