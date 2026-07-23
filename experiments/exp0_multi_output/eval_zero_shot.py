#!/usr/bin/env python3
"""
实验 0：Zero-shot 评估脚本（含数据归一化）

验证多输出模型在 zero-shot ICL 下是否与 12 个独立模型表现一致。
关键：复现 sklearn wrapper 的 StandardScaler 预处理。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot_dir", type=str,
                   default=str(_PROJECT_ROOT.parent / "other" / "sample_collect" / "Hopper-v5-ensemble"))
    p.add_argument("--snapshot_name", type=str, default="env_pool_epoch_0000.npz")
    p.add_argument("--ckpt_path", type=str,
                   default=str(_PROJECT_ROOT.parent / "mbpo_pyt_tabpfn" / "ckpt" / "tabiclv2"
                               / "tabicl-regressor-v2-20260212.ckpt"))
    p.add_argument("--output_dir", type=str,
                   default=str(_PROJECT_ROOT / "experiments" / "exp0_multi_output" / "outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout_ratio", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_snapshot(npz_path):
    data = np.load(npz_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    delta = next_states - states
    inputs = np.concatenate([states, actions], axis=-1)
    labels = np.concatenate([rewards, delta], axis=-1)
    metadata = {"epoch": int(data["epoch"][0]), "num_samples": int(data["num_samples"][0])}
    return inputs, labels, metadata


def compute_mse(pred, target, reward_size=1):
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse_all = float(np.mean(diff ** 2))
    mse_r = float(np.mean(diff[:, :reward_size] ** 2))
    mse_d = float(np.mean(diff[:, reward_size:] ** 2))
    return mse_all, mse_r, mse_d


def load_multi_output_model(ckpt_path, num_outputs, device):
    from tabicl._model.tabicl import TabICL

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    config["num_outputs"] = num_outputs
    model = TabICL(**config)
    model.load_state_dict(state_dict, strict=False)

    # 复制单输出权重到所有 head
    y_enc_weight = state_dict["icl_predictor.y_encoder.weight"]
    y_enc_bias = state_dict["icl_predictor.y_encoder.bias"]
    for j in range(num_outputs):
        model.icl_predictor.y_encoders[j].weight.data.copy_(y_enc_weight)
        model.icl_predictor.y_encoders[j].bias.data.copy_(y_enc_bias)

    for suffix in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        src_key = f"icl_predictor.decoder.{suffix}"
        src_weight = state_dict[src_key]
        for j in range(num_outputs):
            target = dict(model.named_parameters())[f"icl_predictor.decoders.{j}.{suffix}"]
            target.data.copy_(src_weight)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 加载数据
    snapshot_path = Path(args.snapshot_dir) / args.snapshot_name
    log.info(f"加载快照: {snapshot_path}")
    inputs, labels, metadata = load_snapshot(snapshot_path)
    num_outputs = labels.shape[1]
    log.info(f"epoch={metadata['epoch']}, samples={metadata['num_samples']}, "
             f"inputs={inputs.shape}, labels={labels.shape}, num_outputs={num_outputs}")

    # 划分训练/验证
    n = inputs.shape[0]
    n_val = int(round(n * args.holdout_ratio))
    n_val = min(max(n_val, 1), n - 1)
    perm = rng.permutation(n)
    x_tr_raw, y_tr_raw = inputs[perm[n_val:]], labels[perm[n_val:]]
    x_va_raw, y_va_raw = inputs[perm[:n_val]], labels[perm[:n_val]]

    # ---- 关键：复现 sklearn wrapper 的预处理 ----
    # 1. 特征归一化（仿照 TransformToNumerical + StandardScaler）
    x_scaler = StandardScaler()
    x_tr = x_scaler.fit_transform(x_tr_raw).astype(np.float32)
    x_va = x_scaler.transform(x_va_raw).astype(np.float32)

    # 2. 目标归一化（仿照 TabICLRegressor.fit 中的 y_scaler_）
    y_scalers = []
    y_tr = np.zeros_like(y_tr_raw, dtype=np.float32)
    y_va = np.zeros_like(y_va_raw, dtype=np.float32)
    for d in range(num_outputs):
        scaler = StandardScaler()
        y_tr[:, d] = scaler.fit_transform(y_tr_raw[:, d].reshape(-1, 1)).ravel()
        y_va[:, d] = scaler.transform(y_va_raw[:, d].reshape(-1, 1)).ravel()
        y_scalers.append(scaler)

    log.info(f"train={x_tr.shape}, val={x_va.shape}")
    log.info(f"y_tr range: [{y_tr.min():.2f}, {y_tr.max():.2f}] (after scaling)")
    log.info(f"x_tr range: [{x_tr.min():.2f}, {x_tr.max():.2f}] (after scaling)")

    # 加载模型
    log.info("加载多输出模型...")
    model = load_multi_output_model(args.ckpt_path, num_outputs, device)

    # 零样本推理
    log.info("Zero-shot 推理中...")
    all_preds_scaled = []
    eval_bs = 256
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, x_va.shape[0], eval_bs):
            end = min(start + eval_bs, x_va.shape[0])
            x_chunk = x_va[start:end]

            X = np.concatenate([x_tr, x_chunk], axis=0)
            X_t = torch.from_numpy(X).unsqueeze(0).to(device)

            y_tr_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)

            pred = model(X_t, y_train=y_tr_t)
            pred_mean = pred.squeeze(0).mean(dim=-1)  # (chunk_size, 12)
            all_preds_scaled.append(pred_mean.cpu().numpy())

    elapsed = time.time() - t0
    preds_scaled = np.concatenate(all_preds_scaled, axis=0)[:x_va.shape[0]]

    # 逆归一化：从 scaled 空间回到原始空间
    preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
    for d in range(num_outputs):
        preds_raw[:, d] = y_scalers[d].inverse_transform(
            preds_scaled[:, d].reshape(-1, 1)
        ).ravel()

    mse_all, mse_r, mse_d = compute_mse(preds_raw, y_va_raw)
    log.info(f"\n=== Zero-shot 多输出模型结果（含 StandardScaler 预处理）===")
    log.info(f"Overall MSE:  {mse_all:.8f}")
    log.info(f"Reward MSE:   {mse_r:.8f}")
    log.info(f"Delta S MSE:  {mse_d:.8f}")
    log.info(f"推理耗时: {elapsed:.1f}s")

    # 对比 baseline
    log.info(f"\n=== 对比 Baseline (12 个独立 TabICLRegressor, zero-shot) ===")
    b_all = 0.00265815
    b_r = 0.00550285
    b_d = 0.00239954
    log.info(f"Baseline Overall MSE: {b_all:.8f}")
    log.info(f"Baseline Reward MSE:  {b_r:.8f}")
    log.info(f"Baseline Delta S MSE: {b_d:.8f}")
    log.info(f"\nRatio (multi-output / baseline):")
    log.info(f"  Overall: {mse_all / b_all:.4f}")
    log.info(f"  Reward:  {mse_r / b_r:.4f}")
    log.info(f"  Delta S: {mse_d / b_d:.4f}")


if __name__ == "__main__":
    main()
