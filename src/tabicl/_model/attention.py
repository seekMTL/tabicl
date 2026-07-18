from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3

    HAS_FLASH_ATTN3 = True
except ImportError:
    HAS_FLASH_ATTN3 = False

_use_flash_attn3 = True


@contextmanager
def flash_attn3_toggle(enabled: bool):
    """Context manager to temporarily enable or disable Flash Attention 3.

    Used by ``InferenceManager._run_forward()`` in ``inference.py`` to control
    whether Flash Attention 3 is used during each forward pass based on the
    ``use_fa3`` configuration.
    """
    global _use_flash_attn3
    old = _use_flash_attn3
    _use_flash_attn3 = enabled
    try:
        yield
    finally:
        _use_flash_attn3 = old


def sdpa_with_flattened_batch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None,
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's ``scaled_dot_product_attention`` and then reshaping the
    output back to the original shape. This flattening is necessary to properly
    trigger Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:

        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len).

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input.
    """

    # 把最后 3 维（head 数、序列长度、head 维度）保持不动，前面所有 batch 维度展平为 1 维，触发 Flash Attention 优化
    q_shape = q.shape # (2, 3, 8, 7, 16)
    q = q.reshape(-1, *q.shape[-3:]) # (6, 8, 7, 16)
    k = k.reshape(-1, *k.shape[-3:]) # (6, 8, 7, 16)
    v = v.reshape(-1, *v.shape[-3:]) # (6, 8, 7, 16)
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

    if ssmax_layer is not None:
        src_len = k.size(-2) # 7
        q = ssmax_layer(q, src_len) # 应用 SSMax（如果提供）

    # FlashAttn3 doesn't support dropout and custom attention mask
    if HAS_FLASH_ATTN3 and _use_flash_attn3 and q.is_cuda and attn_mask is None and dropout_p == 0.0:
        # FlashAttention only supports fp16, bf16, and fp8_e4m3
        # Convert to bf16 if needed, then convert back to original dtype
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            fa_dtype = torch.float16
        else:
            fa_dtype = orig_dtype

        flat_bs, nheads, seqlen_q, headdim = q.shape
        seqlen_k = k.shape[-2]
        q_fa = q.transpose(1, 2).reshape(flat_bs * seqlen_q, nheads, headdim).contiguous().to(fa_dtype)
        k_fa = k.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        v_fa = v.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        cu_seqlens_q = torch.arange(0, (flat_bs + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (flat_bs + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        # 路径A: Flash Attention 3 (CUDA 且无 dropout 无 mask)
        out = flash_attn3(q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        out = out.view(flat_bs, seqlen_q, nheads, headdim).transpose(1, 2).to(orig_dtype)
    else:
        # 路径B: PyTorch 标准 scaled_dot_product_attention，内部计算公式：
        # $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right) \cdot V$$
        # 按我们的例子：
        # Q @ K^T:
        #   (6, 8, 7, 16) @ (6, 8, 16, 7) → (6, 8, 7, 7)

        #   每对 (6, 8) 产生一个 7×7 的注意力矩阵:
        #         Key位置 → (CLS0 CLS1 CLS2 CLS3 Feat0 Feat1 Feat2)
        #   Q位置
        #   CLS0   [ w    w    w    w    w     w     w   ]
        #   CLS1   [ w    w    w    w    w     w     w   ]
        #   CLS2   [ w    w    w    w    w     w     w   ]
        #   CLS3   [ w    w    w    w    w     w     w   ]
        #   Feat0  [ w    w    w    w    w     w     w   ]
        #   Feat1  [ w    w    w    w    w     w     w   ]
        #   Feat2  [ w    w    w    w    w     w     w   ]

        # 每个 w = (q @ k^T) / sqrt(16) = (q @ k^T) / 4

        # softmax 后，被 mask 的位置 → 0
        # 然后加权求和 V：(6, 8, 7, 7) @ (6, 8, 7, 16) → (6, 8, 7, 16)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    need_kv: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Multi-head attention with support for rotary position embeddings.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim).

    num_heads : int
        Number of attention heads.

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections.

    in_proj_bias : Tensor
        Combined bias vector for input projections.

    dropout_p : float
        Dropout probability applied to attention weights.

    out_proj_weight : Tensor
        Output projection weight matrix.

    out_proj_bias : Tensor
        Output projection bias vector.

    key : Optional[Tensor], default=None
        Key tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    value : Optional[Tensor], default=None
        Value tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    cached_kv : Optional[KVCacheEntry], default=None
        Pre-computed key and value projections for caching. When provided:

        - key and value parameters are ignored
        - Only query projection is computed
        - cached_kv.key shape: (..., num_heads, src_len, head_dim)
        - cached_kv.value shape: (..., num_heads, src_len, head_dim)
        - RoPE is applied only to queries (keys should already have RoPE applied)

    training : bool, default=True
        Whether the model is in training mode (affects dropout).

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:

        - For binary masks: True values indicate positions to ignore.
        - For float masks: Values are directly added to attention scores.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (tgt_len, src_len) or
        (..., num_heads, tgt_len, src_len).

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    need_kv : bool, default=False
        If True and ``cached_kv`` is None, also returns the computed K and V
        projections along with the attention output. Useful for caching K/V for
        subsequent calls.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        If ``need_kv`` is False or ``cached_kv`` is provided:
            Attention output tensor of shape (..., tgt_len, embed_dim).
        If ``need_kv`` is True and ``cached_kv`` is None:
            Tuple of (attn_output, k, v) where:

            - attn_output: shape (..., tgt_len, embed_dim)
            - k: shape (..., num_heads, src_len, head_dim)
            - v: shape (..., num_heads, src_len, head_dim)
    """

    # 假设输入qkv的形状(B,T,H+C,embed_dim)
    # 例如 B=2 (2张表), T=3 (每表3行), H+C=7 (3个特征组+4个CLS token)
    # embed_dim = 128, num_heads = 8，head_dim = 128/8 = 16

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape # 提取形状。支持任意 batch 维度
    head_dim = embed_dim // num_heads # 每个头的维度  128 // 8 = 16
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}" # 确保 embed_dim 能被 num_heads 整除

    # Q/K/V 投影 + 多头重塑

    if cached_kv is None: # 无缓存——标准三路投影
        # Standard: project Q, K, V jointly
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        src_len = key.shape[-2]
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        # in_proj_weight 形状: (3*128, 128) = (384, 128)，是三个权重矩阵拼接，F._in_projection_packed 一次性计算 Q=X@Wq, K=X@Wk, V=X@Wv
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        # 按例则 q 重塑为 (2, 3, tgt_len=7, num_heads=8, head_dim=16) → (2, 3, 8, 7, 16)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None: # 应用 RoPE（如果提供）
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
    else: # 使用缓存——仅投影 Q
        # Use cached K/V, project Q only
        k, v = cached_kv.key, cached_kv.value # 从缓存获取 K/V（形状: (2,3,8,src_len,16)）
        src_len = k.shape[-2]
        # 从联合权重中提取仅 Q 的投影权重
        q_proj_weight = in_proj_weight[:embed_dim] # 前128行, 形状 (128, 128)
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias) # 只投影 Q  (2, 3, 7, 128)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2) # 重塑为多头格式 → (2, 3, 8, 7, 16)
        if rope is not None: # 只对 Q 应用 RoPE（K 已在缓存时应用过）
            q = rope.rotate_queries_or_keys(q)

    # 处理注意力掩码

    # Disable dropout during evaluation
    if not training: # 评估模式下禁用 dropout
        dropout_p = 0.0

    # Process attention mask
    correct_2d_shape = (tgt_len, src_len)
    correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        # 如果 attn_mask 是 2D (tgt_len, src_len) = (7, 7)，则扩展为 (2, 3, 8, 7, 7)
        if attn_mask.dim() == 2:
            if attn_mask.shape != correct_2d_shape:
                raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
            attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
        elif attn_mask.dim() == len(correct_nd_shape):
            if attn_mask.shape != correct_nd_shape:
                raise ValueError(
                    f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                    f"but got {attn_mask.shape}"
                )
        else:
            raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

    # Process key padding mask
    # key_padding_mask 形状: (2, 3, 7)，标明哪些特征位置是填充的
    if key_padding_mask is not None:
        if key_padding_mask.shape != (*batch_shape, src_len):
            raise ValueError(
                f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
            )
        # 扩展为 (2, 3, 1, 1, 7) → (2, 3, 8, 7, 7)
        key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
            *batch_shape, num_heads, tgt_len, src_len
        )

        # 与 attn_mask 合并
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # 缩放点积注意力，输出: (2, 3, 8, 7, 16)
    attn_output = sdpa_with_flattened_batch(
        q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer
    )  # (..., nh, tgt_len, hs)

    # Reshape and project output
    # 将多头输出合并回 embed_dim
    # (2, 3, 8, 7, 16) → transpose(-3,-2) → (2, 3, 7, 8, 16) → contiguous → view → (2, 3, 7, 128)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
    # 输出投影
    # out_proj_weight: (128, 128), 将拼接后的结果再做一次线性变换，输出: (2, 3, 7, 128)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    # 按需返回
    if need_kv and cached_kv is None:
        return attn_output, k, v # 同时返回输出和 K/V 投影（用于缓存）

    return attn_output # 只返回注意力输出
