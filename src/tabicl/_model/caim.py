"""CAIM: Causal Action Intervention Model.

在 TabICL backbone 之上加入因果 action 模块，专为强化学习场景设计。

架构:
  state ──▶ ColEmbedding ──▶ RowInteraction ──▶ state_repr (frozen backbone)
                                                  │ (query)
  action ─▶ ActionEncoder ──────────────────▶ CausalCrossAttention ──▶ fused
             (trainable)                        (trainable)              │
                                                                12×[y_enc_j + tf_icl + dec_j]
                                                                        │
                                                                [reward, delta_state]

特性:
- TabICL backbone (ColEmbedding, RowInteraction, tf_icl) 冻结
- ActionEncoder + CausalFusion + per-dim heads 可训练
- 反事实训练: 对比观测 action vs 随机替换 action 的表示
- 每维度使用正确的 target_aware，保证零样本兼容
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .action_encoder import ActionEncoder
from .causal_fusion import (
    CausalCrossAttention,
    MaskedActionModeling,
)
from .encoders import Encoder


class CAIM(nn.Module):
    """Causal Action Intervention Model.

    Parameters
    ----------
    checkpoint_path : str
        预训练 TabICL checkpoint 路径。
    num_outputs : int, default=12
        输出维度数（1 reward + 11 delta_state）。
    state_dim : int, default=11
        状态维度。
    action_dim : int, default=3
        动作维度。
    action_encoder_mode : str, default="mlp"
        ActionEncoder 类型 ("mlp" 或 "transformer")。
    counterfactual_alpha : float, default=0.1
        反事实损失权重。
    counterfactual_beta : float, default=0.1
        平滑损失权重。
    freeze_backbone : bool, default=True
        是否冻结 TabICL backbone。

    预训练权重加载:
    - ColEmbedding, RowInteraction: 从 checkpoint 加载（frozen）
    - tf_icl (12-layer Transformer): 从 checkpoint 加载（frozen）
    - y_encoders, decoders: 从 checkpoint 单输出权重复制初始化（trainable）
    - ActionEncoder, causal_fusion: 随机初始化（trainable）
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_outputs: int = 12,
        state_dim: int = 11,
        action_dim: int = 3,
        action_encoder_mode: str = "mlp",
        counterfactual_alpha: float = 0.1,
        counterfactual_beta: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.freeze_backbone = freeze_backbone

        # 加载预训练 checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        state_dict = ckpt["state_dict"]

        embed_dim = config["embed_dim"]        # 128
        num_cls = config["row_num_cls"]         # 4
        icl_dim = embed_dim * num_cls            # 512
        out_dim = config["num_quantiles"]        # 999

        # ── TabICL Backbone (frozen) ──
        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=config["col_num_blocks"],
            nhead=config["col_nhead"],
            dim_feedforward=embed_dim * config["ff_factor"],
            num_inds=config["col_num_inds"],
            dropout=config["dropout"],
            activation=config["activation"],
            norm_first=config["norm_first"],
            bias_free_ln=config.get("bias_free_ln", False),
            affine=config["col_affine"],
            feature_group=config["col_feature_group"],
            feature_group_size=config["col_feature_group_size"],
            target_aware=config["col_target_aware"],
            max_classes=config["max_classes"],
            reserve_cls_tokens=num_cls,
            ssmax=config["col_ssmax"],
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=config["row_num_blocks"],
            nhead=config["row_nhead"],
            dim_feedforward=embed_dim * config["ff_factor"],
            num_cls=num_cls,
            rope_base=config["row_rope_base"],
            rope_interleaved=config["row_rope_interleaved"],
            dropout=config["dropout"],
            activation=config["activation"],
            norm_first=config["norm_first"],
            bias_free_ln=config.get("bias_free_ln", False),
        )

        # ICL Transformer (共享权重，每维度独立前向)
        self.tf_icl = Encoder(
            num_blocks=config["icl_num_blocks"],
            d_model=icl_dim,
            nhead=config["icl_nhead"],
            dim_feedforward=icl_dim * config["ff_factor"],
            dropout=config["dropout"],
            activation=config["activation"],
            norm_first=config["norm_first"],
            bias_free_ln=config.get("bias_free_ln", False),
            ssmax=config["icl_ssmax"],
        )

        self.icl_ln = nn.LayerNorm(icl_dim, bias=not config.get("bias_free_ln", False))

        # ── CAIM 可训练模块 ──
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            d_model=embed_dim,
            hidden_dim=embed_dim * 2,
            mode=action_encoder_mode,
        )

        self.causal_fusion = CausalCrossAttention(
            state_dim=icl_dim,
            action_dim=embed_dim,
        )

        self.masked_action_modeling = MaskedActionModeling(
            fused_dim=icl_dim,
            action_dim=action_dim,
            mask_ratio=0.3,
        )

        # 每维度独立的 y_encoder + decoder（从预训练复制权重初始化）
        self.y_encoders = nn.ModuleList([
            nn.Linear(1, icl_dim) for _ in range(num_outputs)
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(icl_dim, icl_dim * 2),
                nn.GELU(),
                nn.Linear(icl_dim * 2, out_dim),
            )
            for _ in range(num_outputs)
        ])

        # ── 加载预训练权重 ──
        self._load_backbone_weights(state_dict, num_outputs)

        # 冻结 backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _load_backbone_weights(self, state_dict: dict, num_outputs: int):
        """从预训练 checkpoint 加载 backbone 权重，并复制单输出 head 权重到所有维度。"""
        # 过滤出 backbone 的 key
        backbone_keys = {k: v for k, v in state_dict.items()
                         if not k.startswith("icl_predictor.y_encoder")
                         and not k.startswith("icl_predictor.decoder")}

        # 加载 backbone（ColEmbedding + RowInteraction + tf_icl + ln）
        model_state = self.state_dict()
        for k, v in backbone_keys.items():
            # 映射 key: icl_predictor.tf_icl... → tf_icl...
            mapped_k = k.replace("icl_predictor.", "")
            if mapped_k in model_state:
                model_state[mapped_k].copy_(v)

        # 复制单输出 y_encoder 权重 → 所有 y_encoders
        y_enc_w = state_dict.get("icl_predictor.y_encoder.weight")
        y_enc_b = state_dict.get("icl_predictor.y_encoder.bias")
        if y_enc_w is not None:
            for j in range(num_outputs):
                self.y_encoders[j].weight.data.copy_(y_enc_w)
                if y_enc_b is not None:
                    self.y_encoders[j].bias.data.copy_(y_enc_b)

        # 复制单输出 decoder 权重 → 所有 decoders
        for sfx in ["0.weight", "0.bias", "2.weight", "2.bias"]:
            src_key = f"icl_predictor.decoder.{sfx}"
            src = state_dict.get(src_key)
            if src is not None:
                for j in range(num_outputs):
                    tgt_key = f"decoders.{j}.{sfx}"
                    tgt = dict(self.named_parameters()).get(tgt_key)
                    if tgt is not None:
                        tgt.data.copy_(src)

    def _freeze_backbone(self):
        """冻结 TabICL backbone。"""
        for p in self.col_embedder.parameters():
            p.requires_grad = False
        for p in self.row_interactor.parameters():
            p.requires_grad = False
        for p in self.tf_icl.parameters():
            p.requires_grad = False
        for p in self.icl_ln.parameters():
            p.requires_grad = False

    def _encode_state(self, X_state: Tensor, y_target: Tensor) -> Tensor:
        """编码 state 为行表示。

        Parameters
        ----------
        X_state : (B, T, state_dim)
        y_target : (B, train_size)
            用于 target_aware 的标签（每维度用正确的值）。

        Returns
        -------
        state_repr : (B, T, icl_dim)
        """
        state_emb = self.col_embedder(X_state, y_train=y_target)
        return self.row_interactor(state_emb)  # (B, T, icl_dim)

    def _predict_dimension(
        self, fused: Tensor, y_j: Tensor, dim_idx: int
    ) -> Tensor:
        """对单个输出维度进行 ICL 预测。

        Parameters
        ----------
        fused : (B, T, icl_dim)
            CausalFusion 融合后的表示。
        y_j : (B, train_size)
            该维度的目标值。
        dim_idx : int
            维度索引。

        Returns
        -------
        pred : (B, T, out_dim)
        """
        train_size = y_j.shape[1]
        R_j = fused.clone()
        y_emb = self.y_encoders[dim_idx](y_j.unsqueeze(-1))  # (B, train_size, icl_dim)
        R_j[:, :train_size] = R_j[:, :train_size] + y_emb

        src = self.tf_icl(R_j, train_size=train_size)
        src = self.icl_ln(src)
        return self.decoders[dim_idx](src)  # (B, T, out_dim)

    def forward(
        self,
        X_state: Tensor,
        X_action: Tensor,
        y_train: Tensor,
    ) -> Tensor:
        """前向传播。

        Parameters
        ----------
        X_state : (B, T, state_dim)
            状态特征。
        X_action : (B, T, action_dim)
            动作特征。
        y_train : (B, train_size, num_outputs)
            训练标签。每个维度使用自己的正确 target。

        Returns
        -------
        preds : (B, test_size, num_outputs, out_dim)
        """
        train_size = y_train.shape[1]

        # Action 编码（共享，1 次前向）
        action_repr = self.action_encoder(X_action)  # (B, T, embed_dim)

        # 每维度独立: ColEmbedding(target_aware 正确) → RowInteraction → Fusion → ICL → decoder
        outputs = []
        for j in range(self.num_outputs):
            y_j = y_train[:, :, j]  # (B, train_size) — 该维度的正确 target

            # State 编码（每维度用正确 target，保证 target_aware 语义正确）
            state_repr = self._encode_state(X_state, y_j)  # (B, T, icl_dim)

            # 因果融合
            fused = self.causal_fusion(state_repr, action_repr)  # (B, T, icl_dim)

            # ICL 预测
            out_j = self._predict_dimension(fused, y_j, j)  # (B, T, out_dim)
            outputs.append(out_j[:, train_size:])  # 只保留测试部分

        return torch.stack(outputs, dim=-2)  # (B, test_size, num_outputs, out_dim)

    def _forward_single_dim(
        self, X_state: Tensor, X_action: Tensor, y_j: Tensor, dim_idx: int
    ) -> Tensor:
        """对单个输出维度的完整前向传播（用于评估）。

        Parameters
        ----------
        X_state : (B, T, state_dim)
        X_action : (B, T, action_dim)
        y_j : (B, train_size, 1) 或 (B, train_size)
            该维度的目标值。
        dim_idx : 维度索引。

        Returns
        -------
        pred : (B, T, out_dim)
        """
        if y_j.dim() == 2:
            y_j_2d = y_j  # (B, train_size)
        else:
            y_j_2d = y_j.squeeze(-1)  # (B, train_size)

        train_size = y_j_2d.shape[1]

        # State 编码（frozen, per-dim correct target）
        with torch.no_grad():
            state_repr = self._encode_state(X_state, y_j_2d)

        # Action 编码（trainable, shared）
        action_repr = self.action_encoder(X_action)

        # CausalFusion（trainable）
        fused, _ = self.causal_fusion(state_repr, action_repr, return_attn=True)

        # ICL + decoder
        return self._predict_dimension(fused, y_j_2d, dim_idx)

    @property
    def trainable_parameters(self):
        """返回可训练参数（用于优化器）。"""
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def trainable_param_count(self):
        """可训练参数数量。"""
        return sum(p.numel() for p in self.trainable_parameters)


def quantile_loss(pred_quantiles: Tensor, target: Tensor, num_quantiles: int = 999) -> Tensor:
    """Pinball / quantile loss。"""
    tau = torch.linspace(
        0.5 / num_quantiles, 1.0 - 0.5 / num_quantiles, num_quantiles,
        device=pred_quantiles.device, dtype=pred_quantiles.dtype,
    )
    target = target.unsqueeze(-1)
    errors = target - pred_quantiles
    return torch.max(tau * errors, (tau - 1) * errors).mean()
