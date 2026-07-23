"""Causal Fusion Module: 因果方向 state-action 融合 + Masked Action Modeling 预训练。

核心理念:
- State 是"上下文"，Action 是"干预"
- 因果方向: state → action → outcome
- Phase 1 预训练: Masked Action Modeling — 随机 mask action 维度，从 state + 剩余 action 重建

包含:
- CausalCrossAttention: state 查询 action 的交叉注意力融合
- MaskedActionModeling: 从融合表示重建被 mask 的 action 值
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CausalCrossAttention(nn.Module):
    """因果交叉注意力融合: state 查询 action。"""

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 128,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, state_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self, state_repr: Tensor, action_repr: Tensor, return_attn: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """因果融合。"""
        action_proj = self.action_proj(action_repr)  # (B, T, state_dim)

        attn_out, _ = self.cross_attn(
            query=state_repr, key=action_proj, value=action_proj,
        )  # (B, T, state_dim)

        fused = self.out_proj(state_repr + attn_out)

        if return_attn:
            return fused, attn_out
        return fused


class MaskedActionModeling(nn.Module):
    """Masked Action Modeling: 随机 mask 部分 action 维度，从 fused 表示重建。

    迫使 ActionEncoder + CausalFusion 学习编码有意义的 action 信息，
    无法通过平凡解（如 attn_out≈0）规避。
    """

    def __init__(self, fused_dim: int = 512, action_dim: int = 3, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.action_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, action_dim),
        )

    def forward(self, fused: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """计算 Masked Action Modeling 损失。

        Parameters
        ----------
        fused : (B, T, fused_dim)
            CausalFusion 的纯 action 贡献（attn_out）。
        actions : (B, T, action_dim)
            原始 action 值。

        Returns
        -------
        loss : 重建损失（仅在 mask 位置）
        mask : 可视化用的 mask
        """
        B, T, A = actions.shape
        mask = (torch.rand(B, T, A, device=actions.device) < self.mask_ratio).float()

        # 从 fused 重建 action
        reconstructed = self.action_decoder(fused)  # (B, T, action_dim)

        # 仅在 mask 位置计算 MSE
        diff = (reconstructed - actions) ** 2
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        return loss, mask
