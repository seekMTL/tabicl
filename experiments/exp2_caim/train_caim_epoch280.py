#!/usr/bin/env python3
"""
CAIM 训练: epoch 280 快照 (286K 样本), state/action 分离架构。

关键设计:
- ColEmbedding(state-only, 11列) + per-dimension 正确 target_aware (frozen)
- ActionEncoder(action-only, 3列) (trainable, 共享)
- CausalFusion 融合 (trainable)
- tf_icl (frozen) + y_encoders/decoders (trainable)
- AMP + 逐维度梯度累积 + 每 epoch 随机子采样 4000 训练样本

用法:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp2_caim/train_caim_epoch280.py
"""

from __future__ import annotations

import argparse, csv, logging, sys, time
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
SNAPSHOT = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble/env_pool_epoch_0280.npz")
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp2_caim" / "outputs"
BASELINE_CSV = "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3/distribution_tabicl2_mse/mse_comparison_results.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_subset", type=int, default=4000, help="每 epoch 训练样本数")
    p.add_argument("--test_subset", type=int, default=256, help="每 epoch 测试样本数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=20, help="每 N epoch 评估一次")
    p.add_argument("--resume", type=str, default=None, help="从保存的 .pt 权重恢复训练")
    return p.parse_args()


def load_data():
    d = np.load(SNAPSHOT)
    S = np.asarray(d["states"], dtype=np.float32)
    A = np.asarray(d["actions"], dtype=np.float32)
    R = np.asarray(d["rewards"], dtype=np.float32).reshape(-1, 1)
    NS = np.asarray(d["next_states"], dtype=np.float32)
    Y = np.concatenate([R, NS - S], -1)
    log.info(f"Loaded: states={S.shape}, actions={A.shape}, labels={Y.shape}")
    return S, A, Y


def quantile_loss(pred_q, target, nq=999):
    """Pinball loss on 999 quantiles."""
    tau = torch.linspace(0.5 / nq, 1.0 - 0.5 / nq, nq, device=pred_q.device, dtype=pred_q.dtype)
    e = target.unsqueeze(-1) - pred_q
    return torch.max(tau * e, (tau - 1) * e).mean()


def load_caim_model(device, resume_path=None):
    """加载 CAIM: 冻结 backbone + 可训练 ActionEncoder/CausalFusion/heads。"""
    from src.tabicl._model.caim import CAIM

    model = CAIM(
        checkpoint_path=CKPT,
        num_outputs=12,
        state_dim=11,
        action_dim=3,
        action_encoder_mode="mlp",
        freeze_backbone=True,
    )

    if resume_path:
        w = torch.load(resume_path, map_location=device)
        model.action_encoder.load_state_dict(w["action_encoder"])
        model.causal_fusion.load_state_dict(w["causal_fusion"])
        model.y_encoders.load_state_dict(w["y_encoders"])
        model.decoders.load_state_dict(w["decoders"])
        log.info(f"Resumed weights from: {resume_path}")

    model.to(device)
    log.info(f"Trainable: {model.trainable_param_count:,} / Total: {sum(p.numel() for p in model.parameters()):,}")
    return model


def train_epoch(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va_s, optimizer, scaler, device, rng, args):
    """单 epoch 训练: 逐维度反向传播 + 梯度累积。"""
    model.train()
    total_loss = 0.0
    num_outputs = model.num_outputs
    train_size = S_tr.shape[0]

    # 随机子采样测试样本
    test_idx = rng.choice(S_va.shape[0], min(args.test_subset, S_va.shape[0]), replace=False)

    # 构造 ICL 输入
    X_s = torch.from_numpy(np.concatenate([S_tr, S_va[test_idx]], 0)).unsqueeze(0).to(device)
    X_a = torch.from_numpy(np.concatenate([A_tr, A_va[test_idx]], 0)).unsqueeze(0).to(device)
    Y_te = torch.from_numpy(Y_va_s[test_idx]).unsqueeze(0).to(device)

    # 每维度独立前向 + 反向
    for j in range(num_outputs):
        optimizer.zero_grad()

        y_j = Y_tr_s[:, j:j+1].copy()  # (train_size, 1), correct target for dim j
        y_j_t = torch.from_numpy(y_j).unsqueeze(0).to(device)  # (1, train_size, 1)

        with autocast():
            # 1. Action 编码（trainable, 共享）
            action_repr = model.action_encoder(X_a)  # (1, T, 128)

            # 2. State 编码（frozen backbone, per-dim correct target_aware）
            with torch.no_grad():
                state_repr = model._encode_state(X_s, y_j_t.squeeze(-1))  # (1, T, 512)

            # 3. CausalFusion（trainable）
            fused, attn_out = model.causal_fusion(state_repr, action_repr, return_attn=True)

            # 4. ICL + decoder
            out_j = model._predict_dimension(fused, y_j_t.squeeze(-1), j)
            pred_j = out_j[:, train_size:, :]  # (1, test_size, 999)

            # 5. Loss
            loss_pred = quantile_loss(pred_j, Y_te[:, :, j])
            loss_mam, _ = model.masked_action_modeling(attn_out, X_a)
            loss = loss_pred + 0.1 * loss_mam

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / num_outputs


def evaluate(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va_s, y_scalers, Y_va_raw, device, rng, max_val=2000, max_train=4000):
    """评估: 子采样训练集以避免 Cross-Attention OOM。"""
    model.eval()
    num_outputs = model.num_outputs

    # 子采样训练集（Cross-Attention 需要 O(train²) 显存）
    train_idx = rng.choice(S_tr.shape[0], min(max_train, S_tr.shape[0]), replace=False)
    S_tr_sub, A_tr_sub, Y_tr_s_sub = S_tr[train_idx], A_tr[train_idx], Y_tr_s[train_idx]
    train_size = S_tr_sub.shape[0]

    all_preds = []
    t0 = time.time()
    n_va = min(S_va.shape[0], max_val)

    with torch.no_grad():
        for start in range(0, n_va, 128):
            end = min(start + 128, n_va)
            X_s = torch.from_numpy(np.concatenate([S_tr_sub, S_va[start:end]], 0)).unsqueeze(0).to(device)
            X_a = torch.from_numpy(np.concatenate([A_tr_sub, A_va[start:end]], 0)).unsqueeze(0).to(device)

            dim_preds = []
            for j in range(num_outputs):
                y_j = torch.from_numpy(Y_tr_s_sub[:, j:j+1]).unsqueeze(0).to(device)
                pred = model._forward_single_dim(X_s, X_a, y_j, j)
                dim_preds.append(pred[:, train_size:, :].mean(-1).cpu().numpy().reshape(-1))

            all_preds.append(np.column_stack(dim_preds))

    elapsed = time.time() - t0
    preds_scaled = np.concatenate(all_preds, 0)[:n_va]
    preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
    for d in range(num_outputs):
        preds_raw[:, d] = y_scalers[d].inverse_transform(preds_scaled[:, d].reshape(-1, 1)).ravel()

    diff = preds_raw.astype(np.float64) - Y_va_raw[:n_va].astype(np.float64)
    mse_all = float(np.mean(diff ** 2))
    mse_r = float(np.mean(diff[:, :1] ** 2))
    mse_d = float(np.mean(diff[:, 1:] ** 2))
    return mse_all, mse_r, mse_d, elapsed


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"train_caim_280_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 加载数据
    S, A, Y = load_data()
    num_outputs = Y.shape[1]

    # 划分 80/20
    n = S.shape[0]
    n_val = int(round(n * 0.2))
    n_val = min(n_val, 10000)
    perm = rng.permutation(n)
    S_tr, A_tr, Y_tr = S[perm[n_val:]], A[perm[n_val:]], Y[perm[n_val:]]
    S_va, A_va, Y_va = S[perm[:n_val]], A[perm[:n_val]], Y[perm[:n_val]]
    log.info(f"Train: {S_tr.shape[0]:,}, Val: {S_va.shape[0]:,}")

    # 标准化
    ss, as_ = StandardScaler(), StandardScaler()
    S_tr = ss.fit_transform(S_tr).astype(np.float32); S_va = ss.transform(S_va).astype(np.float32)
    A_tr = as_.fit_transform(A_tr).astype(np.float32); A_va = as_.transform(A_va).astype(np.float32)

    y_scalers, Y_tr_s, Y_va_s = [], np.zeros_like(Y_tr, dtype=np.float32), np.zeros_like(Y_va, dtype=np.float32)
    for d in range(num_outputs):
        s = StandardScaler()
        Y_tr_s[:, d] = s.fit_transform(Y_tr[:, d].reshape(-1, 1)).ravel()
        Y_va_s[:, d] = s.transform(Y_va[:, d].reshape(-1, 1)).ravel()
        y_scalers.append(s)

    # 加载模型
    model = load_caim_model(device, resume_path=args.resume)
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=args.lr, weight_decay=0.01)
    scaler = GradScaler()

    # Baseline
    import csv as csv_mod
    baseline = {}
    try:
        with open(BASELINE_CSV) as f:
            for r in csv_mod.DictReader(f):
                if int(r["epoch"]) == 280:
                    baseline = {"orig": float(r["tabicl_mse_all"]), "ens": float(r["ensemble_mse_all"])}
    except Exception:
        pass

    log.info(f"Baseline OrigTabICL: {baseline.get('orig', 'N/A')}")

    # 训练循环
    best_mse = float("inf")
    for epoch in range(args.num_epochs):
        # 每 epoch 随机子采样训练集
        train_idx = rng.choice(S_tr.shape[0], min(args.train_subset, S_tr.shape[0]), replace=False)
        S_tr_sub, A_tr_sub, Y_tr_s_sub = S_tr[train_idx], A_tr[train_idx], Y_tr_s[train_idx]

        loss = train_epoch(model, S_tr_sub, A_tr_sub, Y_tr_s_sub, S_va, A_va, Y_va_s,
                           optimizer, scaler, device, rng, args)

        if (epoch + 1) % max(1, args.num_epochs // 20) == 0:
            log.info(f"Epoch {epoch+1}/{args.num_epochs}, loss={loss:.6f}")

        # 评估
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            mse_all, mse_r, mse_d, et = evaluate(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va_s, y_scalers, Y_va, device, rng)
            vs_o = mse_all / baseline.get("orig", 1) if baseline.get("orig", 0) > 0 else 0
            log.info(f"  Eval epoch {epoch+1}: MSE={mse_all:.6g}, reward={mse_r:.6g}, delta={mse_d:.6g}, vs orig: {vs_o:.2f}x, t={et:.0f}s")

            if mse_all < best_mse:
                best_mse = mse_all
                torch.save({
                    "action_encoder": model.action_encoder.state_dict(),
                    "causal_fusion": model.causal_fusion.state_dict(),
                    "y_encoders": model.y_encoders.state_dict(),
                    "decoders": model.decoders.state_dict(),
                }, OUTPUT_DIR / "caim_epoch280_best.pt")
                log.info(f"  Saved best model (MSE={best_mse:.6g})")

    log.info(f"Best MSE: {best_mse:.6g}")


if __name__ == "__main__":
    main()
