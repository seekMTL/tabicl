from __future__ import annotations

from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of blocks used in the encoder.

    nhead : int
        Number of attention heads of the encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the encoder.

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row.

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    dropout : float, default=0.0
        Dropout probability used in the encoder.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.recompute = recompute

        self.tf_row = Encoder(
            num_blocks=num_blocks, # 包含 3 个 MultiheadAttentionBlock
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            recompute=recompute,
        )

        # 1.torch.empty(num_cls, embed_dim)：创建一个形状为(num_cls, embed_dim)的张量。
        # 关键是 empty 不会初始化值，内存中原来是什么就是什么（通常是垃圾值/随机旧数据）。这样比 torch.zeros 或 torch.rand 略快，因为省去填充步骤——反正下一句马上会覆盖掉。
        # 2.nn.Parameter()：把这个张量包装成 nn.Parameter。这一步的意义是：
        # 将张量注册为模块的可学习参数（learnable parameter）；
        # 调用 model.parameters() 时会被包含进去，从而被优化器更新；默认 requires_grad=True，会参与反向传播
        # 3.赋值给 self.cls_tokens — 作为模块的属性存储，PyTorch 的 nn.Module 会自动追踪它。
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        # 对上面创建的参数做截断正态分布初始化（truncated normal initialization）。从正态分布 N(mean=0, std=0.02) 中采样，但截断在 [-2σ, 2σ] 范围内（即 [-0.04, 0.04]），超出范围的值会被丢弃并重新采样。
        # 小方差（std=0.02）：这是 Transformer 的经典做法（来自 GPT/BERT）。初始值接近0，训练初期 CLS token 对注意力的干扰小，模型可以从数据中慢慢学到有意义的聚合模式。
        # 截断而非纯正态：避免极端离群值，让训练更稳定。
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        # CLS token 会在前向传播时被拼接到输入序列前面，经过 Transformer 编码后，用来聚合整行/整个表格的表示（见_aggregate_embeddings方法）

        
        # 条件	            self.out_ln 的值	 作用
        # norm_first=True	nn.LayerNorm(...)	对 Pre-Norm Transformer 最后补一层归一化
        # norm_first=False	nn.Identity()	    恒等映射，Post-Norm 已归一化过，什么都不做

        # Transformer 的两种架构范式：
        #            Post-Norm（norm_first=False）	Pre-Norm（norm_first=True）
        # Block内部  Attn → LN → FFN → LN	        LN → Attn → LN → FFN
        # 输出后	 不需要额外 LN	                 需要一个额外 LN
        # Post-Norm（原始 Transformer/BERT）的结构：x → Attention → + residual → LayerNorm → FFN → + residual → LayerNorm
        # Pre-Norm（现代主流 Transformer，如 GPT、LLaMA）的结构：x → LayerNorm → Attention → + residual → LayerNorm → FFN → + residual

        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)
 
    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """Process a batch of rows through a transformer encoder.

        This method:

        1. Processes embeddings through the transformer
        2. Extracts only the class token representations and applies normalization if pre-norm
        3. Concatenates the class tokens into a single vector per row

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        key_mask : Optional[Tensor], default=None
            Boolean mask of shape (B, T, H+C) where True indicates positions
            to ignore during attention (empty feature slots).

        Returns
        -------
        Tensor
            Flattened class token outputs of shape (B*T, C*E).
        """
        rope = self.tf_row.rope # RoPE（旋转位置编码）：每个特征在行中有一个位置（第0个特征、第1个特征...），RoPE 编码了这个位置信息。这使得模型知道每个嵌入对应的是哪个特征列 

        # 步骤 A: 前 N-1 个 Transformer Block(标准自注意力)
        # 每个 block 执行标准的 Pre-Norm Transformer:
        # x → LayerNorm → Multi-Head Attention → + residual
        # x → LayerNorm → FeedForward → + residual

        # Process all blocks except the last
        if self.recompute: # recompute（梯度检查点）：为节省显存，前向传播时不保存中间激活值，反向传播时重新计算。用时间换空间
            for block in self.tf_row.blocks[:-1]:
                embeddings = checkpoint(
                    partial(block, key_padding_mask=key_mask, rope=rope), embeddings, use_reentrant=False
                )
        else:
            for block in self.tf_row.blocks[:-1]:
                # key_padding_mask：屏蔽空特征位置，让注意力不关注它们
                embeddings = block(embeddings, key_padding_mask=key_mask, rope=rope)

        # 步骤 B: 最后一个 Block(CLS 交叉注意力) 使用非对称注意力
        # q（Query）：只有 CLS token 的嵌入 embeddings[..., :C, :]，形状 (B, T, C, E); k, v（Key, Value）：整个序列 embeddings，形状 (B, T, H+C, E)
        # 即CLS token 能看到所有特征嵌入（通过注意力聚合信息）；特征嵌入不参与这个 block 的输出计算（只作为 k/v 被 CLS 读取）。这是一种高效的"读出"机制——CLS token 作为"摘要器"，从所有特征中提取关键信息
        # 普通 Block:
        #   q = [CLS, CLS, CLS, CLS, feat1, feat2, ..., featH]
        #   k = [CLS, CLS, CLS, CLS, feat1, feat2, ..., featH]
        #   v = [CLS, CLS, CLS, CLS, feat1, feat2, ..., featH]
        #   → 所有位置互相 attend
        # 最后一个 Block:
        #   q = [CLS, CLS, CLS, CLS]           ← 只有 CLS
        #   k = [CLS, CLS, CLS, CLS, feat1, ..., featH]  ← 全序列
        #   v = [CLS, CLS, CLS, CLS, feat1, ..., featH]  ← 全序列
        #   → 只有 CLS attend 到特征，特征不被更新

        # Last block: q = CLS tokens, k/v = full sequence
        last_block = self.tf_row.blocks[-1]
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :], k=emb, v=emb, key_padding_mask=key_mask, rope=rope
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = last_block(
                q=embeddings[..., : self.num_cls, :], k=embeddings, v=embeddings, key_padding_mask=key_mask, rope=rope
            )

        # 步骤 C：后处理与展平
        del embeddings # 显式释放前序嵌入的显存（Python GC 不保证及时回收）
        cls_outputs = self.out_ln(cls_outputs) # LayerNorm，稳定输出分布

        # 展平后，C 个 CLS token 的 E 维嵌入被拼接成一个 C*E 维的向量，作为这一行的最终表示, 所以输出是：每行一个 C*E 维的向量，编码了该行所有特征的交互信息
        return cls_outputs.flatten(-2)  # 将 (B, T, C, E) 的最后两维展平为 (B, T, C*E)

    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """Transform feature embeddings into row representations for training.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """

        B, T, HC, E = embeddings.shape
        device = embeddings.device

        # 注入可学习的 CLS Token
        # CLS Token 的作用：它们不对应任何具体特征，而是作为"信息聚合器"。在 Transformer 的注意力机制中，CLS token 会 attend 到所有特征嵌入，从而吸收整行的信息
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim) # self.cls_tokens 是形状(C, E)的可学习参数，初始化为截断正态分布N(0, 0.02)；expand(B, T, C, E) 不复制数据，只是增加视图维度（共享内存）
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device) # 直接赋值到 embeddings 的前 C 个位置。注意：这里用的是 in-place 赋值，原始 embeddings 张量会被修改

        # 场景：batch 中不同数据集可能有不同数量的特征（如数据集A有10个特征，数据集B有15个特征）。为了让它们能放在同一个 batch 中处理，需要将少特征的数据集填充到相同维度。填充的位置就是"空特征"，需要在注意力计算时被屏蔽
        # Create mask to prevent from attending to empty features
        if d is None:
            key_mask = None # 所有数据集特征数相同，不需要掩码
        else:
            d = d + self.num_cls
            indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)
            key_mask = indices >= d.view(B, 1, 1)  # (B, T, HC)
            # 在 Transformer 的注意力计算中，key_padding_mask(key_mask)=True 的位置会被设为 -inf，softmax 后变为 0，即不参与注意力

        representations = self._aggregate_embeddings(embeddings, key_mask)  # (B, T, C*E)

        return representations  # (B, T, C*E)

    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
        """Transform feature embeddings into row representations for inference.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)

        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )

        return representations  # (B, T, C*E)

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
        """Transform feature embeddings into row representations.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """

        if self.training:
            representations = self._train_forward(embeddings, d)
        else:
            representations = self._inference_forward(embeddings, mgr_config)

        return representations  # (B, T, C*E)
