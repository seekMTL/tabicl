"""方案1：全向量多维目标投影 (Joint Vector Target Projection)

核心思路：
  将 ColEmbedding 的 y_encoder 从 Linear(1, embed_dim) 改为 Linear(d, embed_dim)，
  一次性将完整的多维标签向量 y ∈ R^d 投影为统一的 y_emb，
  使得 Set Transformer 能够同时感知所有目标维度的联合语义来调节列间注意力。

对比实验：
  - Baseline: 12 个独立的 single-output TabICL（ColEmbedding + RowInteraction 重复 12 次）
  - Joint:    1 个 multi-output TabICL（ColEmbedding + RowInteraction 仅运行 1 次）

用法:
  python eval_joint_vs_baseline.py [epoch_idx] [--test]
  - epoch_idx: 数据快照编号，默认 280（286K 样本全集）
  - --test:    仅用 200 样本快速测试
"""

import sys
import time
import numpy as np
import torch
from src.tabicl._model.tabicl import TabICL


# ============================================================================
# 工具函数
# ============================================================================

def make_tabiclk(config, num_outputs):
    """根据预训练配置创建 TabICL 模型。

    Parameters
    ----------
    config : dict  预训练 checkpoint 中的配置字典。
    num_outputs : int  输出维度数。1 = 原始单输出，12 = RL 多输出。
    """
    return TabICL(
        max_classes=config['max_classes'],  # 0 = 回归
        num_quantiles=config['num_quantiles'],  # 999 分位数
        num_outputs=num_outputs,   # 核心参数：控制 ColEmbedding 和 ICLearning
        embed_dim=config['embed_dim'],  # 128
        col_num_blocks=config['col_num_blocks'],  # Set Transformer 层数
        col_nhead=config['col_nhead'],
        col_num_inds=config['col_num_inds'],  # ISAB 诱导点数量
        col_affine=config['col_affine'],
        col_feature_group=config['col_feature_group'],  # 'same'：循环排列分组
        col_feature_group_size=config['col_feature_group_size'],
        col_target_aware=config['col_target_aware'],  # True：注入标签信息
        col_ssmax=config['col_ssmax'],
        row_num_blocks=config['row_num_blocks'],  # 行交互 Transformer 层数
        row_nhead=config['row_nhead'],
        row_num_cls=config['row_num_cls'],  # 4 个 CLS token
        row_rope_base=config['row_rope_base'],
        row_rope_interleaved=config['row_rope_interleaved'],
        icl_num_blocks=config['icl_num_blocks'],  # ICL Transformer 12 层
        icl_nhead=config['icl_nhead'],
        icl_ssmax=config['icl_ssmax'],
        ff_factor=config['ff_factor'],
        dropout=config['dropout'],
        activation=config['activation'],
        norm_first=config['norm_first'],
        bias_free_ln=config['bias_free_ln'],
        recompute=config['recompute'],
    )


def load_joint_model(config, state_dict):
    """加载预训练权重到联合多输出模型。

    权重映射策略：
    1. Backbone (SetTransformer + RowInteraction + ICL Transformer)：直接加载
    2. ColEmbedding.y_encoder: Linear(1,128) → Linear(12,128)
       - 将预训练的 (128,1) 权重沿输入维度重复 12 份，缩放 1/12
       - 含义: y_emb = (1/12) * Σ_j pretrained_weight * y_j，保持注入量级稳定
    3. ICL y_encoders[j] / decoders[j]: 单输出 → 逐维度独立复制
    """
    m = make_tabiclk(config, num_outputs=12)
    ms = m.state_dict()

    # ── 加载 Backbone 权重（跳过单输出 icl_predictor head）──
    for k, v in state_dict.items():
        if k.startswith('icl_predictor.y_encoder') or k.startswith('icl_predictor.decoder'):
            continue
        if k in ms:
            ms[k].copy_(v)

    # ── ColEmbedding y_encoder: Linear(1, 128) → Linear(12, 128) ──
    # 单输出权重 (128, 1) 复制到所有 12 个输入维度，缩放 1/12
    cw = state_dict['col_embedder.y_encoder.weight']  # (128, 1)
    cb = state_dict['col_embedder.y_encoder.bias']    # (128,)
    m.col_embedder.y_encoder.weight.data.copy_(cw.repeat(1, 12) * (1.0 / 12))
    m.col_embedder.y_encoder.bias.data.copy_(cb)

    # ── ICL per-dim heads：单输出权重复制到 12 个 head ──
    # y_encoders: 每维度独立的 Linear(1, 512)，形状与预训练一致
    for j in range(12):
        m.icl_predictor.y_encoders[j].weight.data.copy_(
            state_dict['icl_predictor.y_encoder.weight'])
        m.icl_predictor.y_encoders[j].bias.data.copy_(
            state_dict['icl_predictor.y_encoder.bias'])
    # decoders: 两层 MLP (512→1024→999)，每层权重复制
    for sfx in ['0.weight', '0.bias', '2.weight', '2.bias']:
        src = state_dict[f'icl_predictor.decoder.{sfx}']
        for j in range(12):
            tgt = dict(m.icl_predictor.named_parameters()).get(f'decoders.{j}.{sfx}')
            if tgt is not None:
                tgt.data.copy_(src)
    return m


def load_hopper_data(epoch_idx):
    """加载 Hopper-v5 经验回放池，构造 X=[state|action], y=[reward|delta]。

    X: (N, 14) = [state(11) | action(3)]
    y: (N, 12) = [reward(1) | delta_state_0..10(11)], delta = next_state - state
    """
    path = f'/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble/env_pool_epoch_{epoch_idx:04d}.npz'
    data = np.load(path)
    states, actions = data['states'], data['actions']
    X = np.concatenate([states, actions], axis=1).astype(np.float32)
    delta = (data['next_states'] - states).astype(np.float32)
    y = np.concatenate([data['rewards'][:, None], delta], axis=1).astype(np.float32)
    return X, y


# ============================================================================
# 主实验
# ============================================================================

def main():
    # ── 解析命令行参数 ──
    epoch_idx = 280  # 默认 epoch 280（286K 样本）
    quick_test = False
    for arg in sys.argv[1:]:
        if arg.startswith('--test'):
            quick_test = True
        elif arg.isdigit():
            epoch_idx = int(arg)

    # ── 加载预训练权重 ──
    ckpt_path = '/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    config, state_dict = ckpt['config'], ckpt['state_dict']

    # ── 加载数据 ──
    X_all, y_all = load_hopper_data(epoch_idx)
    if quick_test:
        # 快速测试：只用 200 样本验证代码正确性
        np.random.seed(42)
        idx = np.random.choice(len(X_all), 200, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    n_total = len(X_all)
    n_train = int(n_total * 0.67)  # 2/3 训练，1/3 测试
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    X_all, y_all = X_all[idx], y_all[idx]
    X_tr, X_te = X_all[:n_train], X_all[n_train:]
    y_tr, y_te = y_all[:n_train], y_all[n_train:]

    print(f"Epoch {epoch_idx}: X={X_all.shape}, y={y_all.shape}")
    print(f"Train: {n_train}, Test: {n_total - n_train}")
    print(f"y ranges: reward [{y_all[:, 0].min():.4f}, {y_all[:, 0].max():.4f}], "
          f"delta [{y_all[:, 1:].min():.4f}, {y_all[:, 1:].max():.4f}]")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Baseline: 12 个独立单输出模型 ──
    # 每个模型用自己维度的正确 target，ColEmbedding+RowInteraction 运行 12 次
    print("\n" + "=" * 60)
    print("Baseline: 12 single-output TabICL (ColEmbedding × 12)")
    print("=" * 60)
    preds = []
    t_start = time.time()
    X_batch = torch.FloatTensor(np.concatenate([X_tr, X_te], axis=0)).unsqueeze(0).to('cuda')
    for d in range(12):
        m = make_tabiclk(config, num_outputs=1)
        m.load_state_dict(state_dict, strict=False)
        m.eval().to('cuda')
        y_batch = torch.FloatTensor(y_tr[:, d]).unsqueeze(0).to('cuda')
        with torch.no_grad():
            out = m(X_batch, y_train=y_batch)
        preds.append(out.squeeze(0).cpu().numpy().mean(axis=-1))  # 999 分位数 → 均值
        del m, out
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_baseline = time.time() - t_start
    preds_baseline = np.column_stack(preds)  # (test_size, 12)
    print(f"Completed in {t_baseline:.1f}s")
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Joint: 1 个多输出模型 ──
    # ColEmbedding + RowInteraction 仅运行 1 次，ICL per-dim 运行 12 次
    print("\n" + "=" * 60)
    print("Joint: 1 multi-output TabICL (ColEmbedding × 1)")
    print("=" * 60)
    m_joint = load_joint_model(config, state_dict)
    m_joint.eval().to('cuda')
    y_batch = torch.FloatTensor(y_tr).unsqueeze(0).to('cuda')  # (1, train_size, 12)
    t_start = time.time()
    with torch.no_grad():
        out_joint = m_joint(X_batch, y_train=y_batch)
    torch.cuda.synchronize()
    t_joint = time.time() - t_start
    pred_joint = out_joint.squeeze(0).cpu().numpy().mean(axis=-1)  # (test_size, 12)
    print(f"Completed in {t_joint:.1f}s")
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── 精度对比 ──
    print("\n" + "=" * 70)
    mode_str = "QUICK TEST (200 samples)" if quick_test else f"FULL ({n_total} samples)"
    print(f"  {mode_str}")
    print(f"  Train: {n_train}, Test: {n_total - n_train}")
    print("=" * 70)
    header = (f"{'Dim':>6} {'Label':<15} {'Baseline MSE':>14} "
              f"{'Joint MSE':>14} {'Ratio(J/B)':>10}")
    print(header)
    print("-" * 70)
    for d in range(12):
        mse_b = np.mean((preds_baseline[:, d] - y_te[:, d]) ** 2)
        mse_j = np.mean((pred_joint[:, d] - y_te[:, d]) ** 2)
        label = 'reward' if d == 0 else f'delta[{d - 1}]'
        print(f"{d:>6} {label:<15} {mse_b:>14.6f} {mse_j:>14.6f} "
              f"{mse_j / mse_b:>10.2f}x")

    mse_b = np.mean((preds_baseline - y_te) ** 2)
    mse_j = np.mean((pred_joint - y_te) ** 2)
    print("-" * 70)
    print(f"{'':>6} {'OVERALL':<15} {mse_b:>14.6f} {mse_j:>14.6f} "
          f"{mse_j / mse_b:>10.2f}x")

    # ── 速度对比 ──
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"  Baseline (12 × single): {t_baseline:.1f}s")
    print(f"  Joint    (1  × multi ): {t_joint:.1f}s")
    print(f"  Speedup: {t_baseline / t_joint:.2f}x")
    print(f"  Time saved by sharing ColEmbedding+RowInteraction: "
          f"{t_baseline - t_joint:.1f}s")


if __name__ == '__main__':
    main()
