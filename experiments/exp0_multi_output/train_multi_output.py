#!/usr/bin/env python3
"""
实验 0：共享 Backbone + 多输出头

将 TabICL 从 12 个独立模型（每维度一个）替换为 1 个共享 backbone + 12 个 decoder head 的模型。
- ColEmbedding + RowInteraction + ICL Transformer 共享
- 每输出维度独立的 y_encoder + decoder head
- 从预训练单输出 checkpoint 加载权重并复制到所有 head

用法:
  python experiments/exp0_multi_output/train_multi_output.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot_dir", type=str,
                   default=str(_PROJECT_ROOT.parent / "other" / "sample_collect" / "Hopper-v5-ensemble"),
                   help="快照文件目录")
    p.add_argument("--snapshot_name", type=str, default="env_pool_epoch_0000.npz",
                   help="使用的快照文件名（默认从最小开始测试）")
    p.add_argument("--ckpt_path", type=str,
                   default=str(_PROJECT_ROOT.parent / "mbpo_pyt_tabpfn" / "ckpt" / "tabiclv2"
                               / "tabicl-regressor-v2-20260212.ckpt"),
                   help="预训练 TabICL checkpoint 路径")
    p.add_argument("--output_dir", type=str,
                   default=str(_PROJECT_ROOT / "experiments" / "exp0_multi_output" / "outputs"),
                   help="输出目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=8,
                   help="训练 batch size（每个 batch 是一个完整的 table）")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_snapshot(npz_path: Path):
    """加载快照并构造 inputs/labels"""
    data = np.load(npz_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    delta = next_states - states
    inputs = np.concatenate([states, actions], axis=-1)       # (N, 14)
    labels = np.concatenate([rewards, delta], axis=-1)        # (N, 12)
    metadata = {
        "epoch": int(data["epoch"][0]),
        "num_samples": int(data["num_samples"][0]),
    }
    return inputs, labels, metadata


def train_val_split(inputs, labels, holdout_ratio, rng):
    n = inputs.shape[0]
    n_val = int(round(n * holdout_ratio))
    n_val = min(max(n_val, 1), n - 1)
    perm = rng.permutation(n)
    return (inputs[perm[n_val:]], labels[perm[n_val:]],
            inputs[perm[:n_val]], labels[perm[:n_val]])


def compute_mse(pred, target, reward_size=1):
    """pred/target: (N, num_outputs)"""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse_all = float(np.mean(diff ** 2))
    mse_r = float(np.mean(diff[:, :reward_size] ** 2))
    mse_d = float(np.mean(diff[:, reward_size:] ** 2))
    return mse_all, mse_r, mse_d


def load_model_with_multi_output(ckpt_path: str, num_outputs: int, device: torch.device):
    """从单输出 checkpoint 加载模型，扩展为多输出。"""
    from tabicl._model.tabicl import TabICL

    log.info(f"加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    # 单输出 checkpoint 的 key 映射
    log.info(f"原始 config: max_classes={config.get('max_classes')}, "
             f"embed_dim={config.get('embed_dim')}, "
             f"row_num_cls={config.get('row_num_cls')}")

    # 创建多输出模型
    config["num_outputs"] = num_outputs
    model = TabICL(**config)

    # 加载 state_dict，strict=False 跳过不匹配的新增 head 参数
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(f"加载预训练权重完成。missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # 将单输出 y_encoder 和 decoder 的权重复制到所有 head（更好的初始化）
    _init_multi_heads_from_single(model, state_dict, num_outputs)

    model.to(device)
    model.train()
    return model, config


def _init_multi_heads_from_single(model, state_dict, num_outputs):
    """用预训练单输出权重初始化所有多输出 head。"""
    # y_encoders: 复制 y_encoder.weight/bias → y_encoders.{j}.weight/bias
    y_enc_weight = state_dict.get("icl_predictor.y_encoder.weight")
    y_enc_bias = state_dict.get("icl_predictor.y_encoder.bias")
    if y_enc_weight is not None:
        for j in range(num_outputs):
            model.icl_predictor.y_encoders[j].weight.data.copy_(y_enc_weight)
            if y_enc_bias is not None:
                model.icl_predictor.y_encoders[j].bias.data.copy_(y_enc_bias)
        log.info(f"已用预训练 y_encoder 权重初始化 {num_outputs} 个 y_encoders")

    # decoders: decoder.0 → decoders.{j}.0, decoder.2 → decoders.{j}.2
    for suffix in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        src_key = f"icl_predictor.decoder.{suffix}"
        if src_key in state_dict:
            src_weight = state_dict[src_key]
            for j in range(num_outputs):
                tgt_key = f"icl_predictor.decoders.{j}.{suffix}"
                target = dict(model.named_parameters()).get(tgt_key)
                if target is not None:
                    target.data.copy_(src_weight)
    log.info(f"已用预训练 decoder 权重初始化 {num_outputs} 个 decoder heads")


def quantile_loss(pred_quantiles: torch.Tensor, target: torch.Tensor, num_quantiles: int = 999):
    """Pinball / quantile loss.

    Args:
        pred_quantiles: (B, test_size, num_quantiles) 预测的分位数
        target: (B, test_size) 真实值
        num_quantiles: 分位数数量

    Returns:
        scalar loss
    """
    # 分位数级别: [1/(2K), 3/(2K), ..., (2K-1)/(2K)]
    quantile_levels = torch.linspace(
        0.5 / num_quantiles, 1.0 - 0.5 / num_quantiles, num_quantiles,
        device=pred_quantiles.device, dtype=pred_quantiles.dtype,
    )  # (K,)

    # target: (B, test_size, 1), pred: (B, test_size, K)
    target = target.unsqueeze(-1)  # (B, test_size, 1)
    errors = target - pred_quantiles  # (B, test_size, K)
    loss = torch.max(quantile_levels * errors, (quantile_levels - 1) * errors)
    return loss.mean()


def train_epoch(model, X, y_train, optimizer):
    """单步训练：每个 batch 是一个完整的 table（B=1 或 B=batch_size）。"""
    model.train()
    optimizer.zero_grad()

    B, T, H = X.shape
    train_size = y_train.shape[1]
    num_outputs = y_train.shape[2] if y_train.dim() == 3 else 1

    # 前向传播
    pred = model(X, y_train=y_train)  # (B, test_size, num_outputs, num_quantiles)

    # 测试样本的真实标签
    y_test = X.new_zeros(B, T - train_size, num_outputs)  # placeholder
    # 实际训练中，我们需要知道测试标签来计算 loss
    # 这里 y_test 在后文中从完整 labels 中切分

    return pred


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode='w'),
        ],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 1. 加载数据
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    snapshot_path = snapshot_dir / args.snapshot_name
    log.info(f"加载快照: {snapshot_path}")
    inputs, labels, metadata = load_snapshot(snapshot_path)
    num_outputs = labels.shape[1]  # 12
    state_size = labels.shape[1] - 1  # 11 (delta_state dims)
    log.info(f"  epoch={metadata['epoch']}, samples={metadata['num_samples']}, "
             f"inputs={inputs.shape}, labels={labels.shape}, num_outputs={num_outputs}")

    # 2. 加载模型（多输出）
    model, config = load_model_with_multi_output(args.ckpt_path, num_outputs, device)
    log.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 划分训练/验证集
    x_tr, y_tr, x_va, y_va = train_val_split(inputs, labels, args.holdout_ratio, rng)
    log.info(f"train={x_tr.shape}, val={x_va.shape}")

    # 4. 转换为 Tensor
    # TabICL 输入: (B, T, H)，在训练时将训练和测试拼在一起
    # 这里 B=1, T = train_size + 少量测试样本
    # 为简单起见，每次训练使用全量训练集 + 一小批测试样本
    train_size = x_tr.shape[0]
    test_size = min(x_va.shape[0], 256)  # 每次训练用 256 个测试样本

    # 4. 训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # 每 epoch 随机采样测试样本
        test_indices = rng.choice(x_va.shape[0], size=test_size, replace=False)

        # 构造输入: (1, train_size + test_size, 14)
        X_batch = np.concatenate([x_tr, x_va[test_indices]], axis=0)
        X_tensor = torch.from_numpy(X_batch).unsqueeze(0).to(device)  # (1, T, 14)

        # 构造标签: (1, train_size + test_size, 12)
        y_batch = np.concatenate([y_tr, y_va[test_indices]], axis=0)
        y_tensor = torch.from_numpy(y_batch).unsqueeze(0).to(device)  # (1, T, 12)

        # 训练标签只给训练部分
        y_train_tensor = y_tensor[:, :train_size, :]  # (1, train_size, 12)

        # 前向传播
        pred = model(X_tensor, y_train=y_train_tensor)
        # pred: (1, test_size, num_outputs, num_quantiles)

        # 计算 quantile loss (每个输出维度独立计算后求和)
        y_test_tensor = y_tensor[:, train_size:, :]  # (1, test_size, 12)
        loss = 0.0
        for j in range(num_outputs):
            loss = loss + quantile_loss(
                pred[:, :, j, :],  # (1, test_size, num_quantiles)
                y_test_tensor[:, :, j],  # (1, test_size)
            )
        loss = loss / num_outputs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{args.num_epochs}, loss={loss.item():.6f}")

    # 5. 评估：在完整验证集上计算 MSE
    log.info("评估模型...")
    model.eval()

    all_preds = []
    eval_batch_size = 1024
    with torch.no_grad():
        for start in range(0, x_va.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, x_va.shape[0])
            x_val_chunk = x_va[start:end]

            # 构造输入: train + test_chunk
            X_eval = np.concatenate([x_tr, x_val_chunk], axis=0)
            X_eval_t = torch.from_numpy(X_eval).unsqueeze(0).to(device)

            y_tr_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)
            # y_train: (1, train_size, 12) for multi-output
            y_train_eval = y_tr_t  # (1, train_size, 12)

            pred = model(X_eval_t, y_train=y_train_eval)
            # pred: (1, chunk_size, 12, 999)
            pred_mean = pred.squeeze(0).mean(dim=-1)  # (chunk_size, 12)
            all_preds.append(pred_mean.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (val_size, 12)
    # Ensure we only take the first x_va.shape[0] predictions
    all_preds = all_preds[:x_va.shape[0]]

    mse_all, mse_r, mse_d = compute_mse(all_preds, y_va)
    log.info(f"=== 实验结果 ===")
    log.info(f"Overall MSE:  {mse_all:.8f}")
    log.info(f"Reward MSE:   {mse_r:.8f}")
    log.info(f"Delta S MSE:  {mse_d:.8f}")

    # 与 Ensemble baseline 对比（从用户之前实验的 CSV 中获取）
    log.info(f"\n对比 baseline (来自 distribution_tabicl2_mse 实验):")
    log.info(f"  TabICL (12个独立模型) 在相同快照的 MSE: 参考 mse_comparison_results.csv")

    return mse_all, mse_r, mse_d


if __name__ == "__main__":
    main()
