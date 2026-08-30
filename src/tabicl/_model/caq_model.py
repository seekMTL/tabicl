"""CAQ-TabICL 模型：集成预训练 TabICL + Causal-Action Querying 机制。

加载预训练 TabICL 权重，保留原生 ColEmbedding、RowInteraction Block、ICLearning 权重，
新增第 N+1 层因果交叉检索模块（CausalActionEncoder + CausalInterventionBlock）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn, Tensor

from .tabicl import TabICL
from .caq_interaction import CAQ_RowInteraction, build_caq_row_interaction


class CAQTabICL(nn.Module):
    """Causal-Action Querying TabICL 模型。

    在原生 TabICL 的 RowInteraction 之后插入动作因果检索层，
    使 CLS token 能够根据动作信息动态检索相关状态特征组。

    架构变更（对照设计文档 Step 1-7）：
    - ColEmbedding: 保持不变，处理状态特征
    - RowInteraction: 替换为 CAQ_RowInteraction（状态嵌入 + 独立动作输入 + 新增 N+1 层）
    - ICLearning: 保持不变
    """

    def __init__(
        self,
        pretrained_tabicl: TabICL,
        action_dim: int = 3,
        freeze_col_embedder: bool = True,
        freeze_native_blocks: bool = False,
        freeze_icl: bool = False,
        target_aware: bool = False,
    ):
        """
        Args:
            pretrained_tabicl: 已加载预训练权重的 TabICL 模型
            action_dim: 动作空间维度（Hopper-v5 为 3）
            freeze_col_embedder: 是否冻结 ColEmbedding 参数
            freeze_native_blocks: 是否冻结原生 RowInteraction Block 参数
            freeze_icl: 是否冻结 ICLearning 参数
            target_aware: 是否启用 ColEmbedding 的 target-aware y 条件嵌入
                （预训练原生设置；开启时训练步需逐维 y 条件批处理）
        """
        super().__init__()

        # 复制关键配置
        self.max_classes = pretrained_tabicl.max_classes
        self.num_quantiles = pretrained_tabicl.num_quantiles
        self.embed_dim = pretrained_tabicl.embed_dim
        self.action_dim = action_dim  # 用于 X_action=None 时构造零动作

        # Stage 1: ColEmbedding（保持预训练权重）
        self.col_embedder = pretrained_tabicl.col_embedder
        # 默认关闭 target-aware（共享嵌入并行训练需要 CLS 与 y 无关）；
        # 开启时恢复预训练原生的 y 条件嵌入（评估实验证实其必不可少，
        # 关闭时冻结 ICL 的输入形态与预训练不匹配，Ant 劣化 9 倍）
        self.col_embedder.target_aware = target_aware
        if freeze_col_embedder:
            for p in self.col_embedder.parameters():
                p.requires_grad = False

        # Stage 2: CAQ_RowInteraction（替换原生 RowInteraction）
        row = pretrained_tabicl.row_interactor
        self.row_interactor = build_caq_row_interaction(
            row,
            action_dim=action_dim,
            dropout=pretrained_tabicl.dropout,
            nhead=pretrained_tabicl.row_nhead,
            dim_feedforward=pretrained_tabicl.embed_dim * pretrained_tabicl.ff_factor,
        )

        if freeze_native_blocks:
            for p in self.row_interactor.blocks_except_last.parameters():
                p.requires_grad = False
            for p in self.row_interactor.last_native_block.parameters():
                p.requires_grad = False

        # Stage 3: ICLearning（保持预训练权重）
        self.icl_predictor = pretrained_tabicl.icl_predictor
        if freeze_icl:
            for p in self.icl_predictor.parameters():
                p.requires_grad = False

        # 量化分布转换器（回归任务专用）
        if self.max_classes == 0:
            self.quantile_dist = pretrained_tabicl.quantile_dist

        self._cache = None

    @property
    def has_cache(self) -> bool:
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """清除内部缓存。"""
        self._cache = None

    def _forward_embeddings(
        self,
        X_state: Tensor,
        X_action: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        """执行阶段 1+2：列嵌入 + CAQ 行交互。

        Args:
            X_state:  (B, T, state_dim) - 状态特征
            X_action: (B, T, action_dim) - 动作特征
            y_train:  (B, train_size) - 训练标签
            d:         可选，每张表的实际特征数
            embed_with_test: 如果为真，则允许训练样本在嵌入过程中关注测试样本

        Returns:
            representations: (B, T, num_cls * embed_dim) - 行表示 (默认 512 维)
        """
        # Stage 1: ColEmbedding（仅处理状态特征）
        col_embeddings = self.col_embedder(
            X_state, y_train=y_train, d=d, embed_with_test=embed_with_test
        )  # (B, T, G+num_cls, embed_dim)

        # Stage 2: CAQ RowInteraction（状态嵌入 + 独立动作输入）
        representations = self.row_interactor(
            col_embeddings, X_action, d=d
        )  # (B, T, num_cls * embed_dim)

        return representations

    def forward(
        self,
        X_state: Tensor,
        X_action: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        """训练/推理前向传播。

        Args:
            X_state:  (B, T, state_dim) - 状态特征
            X_action: (B, T, action_dim) - 动作特征。若为 None（SCM 模式），
                      自动构造零动作：零初始化保证 a_emb ≈ 0，CAQ 模块退化为
                      原生 RowInteraction 行为，用于 SCM replay 防遗忘。
            y_train:  (B, train_size) - 训练标签
            d:         可选，每张表的实际特征数
            embed_with_test: 如果为真，则允许训练样本在嵌入过程中关注测试样本

        Returns:
            predictions: (B, test_size, out_dim)
                       - 回归: out_dim = num_quantiles (999)
                       - 分类: out_dim = max_classes
        """
        assert y_train is not None, "y_train is required"

        if X_action is None:
            # SCM 模式：零动作 → ActionEncoder 输出 ≈ 0（零初始化）→
            # cls_2 ≈ cls_1，CAQ 退化为原生 RowInteraction
            B, T = X_state.shape[:2]
            X_action = torch.zeros(
                B, T, self.action_dim, device=X_state.device, dtype=X_state.dtype
            )

        representations = self._forward_embeddings(
            X_state, X_action, y_train, d, embed_with_test
        )

        out = self.icl_predictor(representations, y_train=y_train)
        return out

    def trainable_parameters(self):
        """返回所有可训练参数（用于优化器配置）。"""
        return [p for p in self.parameters() if p.requires_grad]

    def summary(self) -> Dict[str, int]:
        """返回各模块的参数量统计。

        Returns:
            dict: 包含每个子模块的总参数量和可训练参数量
        """
        stats: Dict[str, int] = {}

        modules = [
            ("col_embedder", self.col_embedder),
            ("row.blocks_except_last", self.row_interactor.blocks_except_last),
            ("row.last_native_block", self.row_interactor.last_native_block),
            ("row.action_encoder", self.row_interactor.action_encoder),
            ("row.causal_block", self.row_interactor.causal_block),
            ("icl_predictor", self.icl_predictor),
        ]

        for name, module in modules:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            stats[name] = total
            stats[f"{name}_trainable"] = trainable

        # CLS token 单独统计
        stats["row.cls_tokens"] = self.row_interactor.cls_tokens.numel()
        stats["row.cls_tokens_trainable"] = (
            stats["row.cls_tokens"]
            if self.row_interactor.cls_tokens.requires_grad
            else 0
        )

        stats["total"] = sum(
            v for k, v in stats.items() if not k.endswith("_trainable")
        )
        stats["total_trainable"] = sum(
            v for k, v in stats.items() if k.endswith("_trainable")
        )
        return stats


def load_caq_model(
    checkpoint_path: str | Path,
    action_dim: int = 3,
    device: torch.device | str = "cpu",
    freeze_col_embedder: bool = True,
    freeze_native_blocks: bool = False,
    freeze_icl: bool = False,
    target_aware: bool = False,
) -> CAQTabICL:
    """从预训练 checkpoint 加载 CAQTabICL 模型。

    这是构建 CAQTabICL 的推荐入口函数：
    1. 从 checkpoint 加载标准 TabICL 及其全部预训练权重
    2. 提取原生 Block、RoPE、CLS token、out_ln 权重
    3. 构建 CAQ_RowInteraction（新增模块随机初始化）
    4. 组装为 CAQTabICL

    Args:
        checkpoint_path: 预训练 TabICL checkpoint 路径（.ckpt 文件）
        action_dim: 动作空间维度（Hopper-v5 为 3）
        device: 目标设备
        freeze_col_embedder: 是否冻结 ColEmbedding
        freeze_native_blocks: 是否冻结原生 RowInteraction Blocks
        freeze_icl: 是否冻结 ICLearning
        target_aware: 是否启用 ColEmbedding 的 target-aware y 条件嵌入
            （默认 False 保持旧 checkpoint 兼容；开启时训练步需逐维 y 条件批处理）

    Returns:
        CAQTabICL 模型实例，已加载到目标设备
    """
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "Checkpoint 缺少 'config'"
    assert "state_dict" in checkpoint, "Checkpoint 缺少 'state_dict'"

    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    # 创建标准 TabICL 并加载所有权重
    pretrained = TabICL(**config)
    pretrained.load_state_dict(state_dict)
    pretrained = pretrained.float()  # 确保 float32（checkpoint 可能是 AMP float16）
    pretrained.eval()

    # 从预训练模型构建 CAQTabICL
    model = CAQTabICL(
        pretrained_tabicl=pretrained,
        action_dim=action_dim,
        freeze_col_embedder=freeze_col_embedder,
        freeze_native_blocks=freeze_native_blocks,
        freeze_icl=freeze_icl,
        target_aware=target_aware,
    )

    model.to(device)
    return model
