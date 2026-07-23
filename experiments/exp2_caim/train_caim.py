#!/usr/bin/env python3
"""
CAIM 反事实预训练 + 有监督微调。

用法:
  # Phase 1: 反事实预训练（仅训练 ActionEncoder + CausalFusion）
  python experiments/exp2_caim/train_caim.py --phase 1 --num_epochs 100

  # Phase 2: 有监督微调（训练所有可训练参数）
  python experiments/exp2_caim/train_caim.py --phase 2 --num_epochs 50

  # 评估
  python experiments/exp2_caim/train_caim.py --eval_only
"""

from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

CKPT_PATH = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
SNAPSHOT_DIR = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp2_caim" / "outputs"

BASELINE_CSV = "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3/distribution_tabicl2_mse/mse_comparison_results.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, default=1, choices=[1, 2],
                   help="1=反事实预训练, 2=有监督微调")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1, help="每 batch 一个完整 table")
    p.add_argument("--snapshot_name", type=str, default="env_pool_epoch_0000.npz")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_snapshot(npz_path):
    data = np.load(npz_path)
    s = np.asarray(data["states"], dtype=np.float32)
    a = np.asarray(data["actions"], dtype=np.float32)
    r = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    ns = np.asarray(data["next_states"], dtype=np.float32)
    return s, a, np.concatenate([r, ns - s], -1)


def compute_mse(p, t, rs=1):
    d = p.astype(np.float64) - t.astype(np.float64)
    return float(np.mean(d**2)), float(np.mean(d[:,:rs]**2)), float(np.mean(d[:,rs:]**2))


def main():
    args = parse_args()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"train_phase{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ],
    )
    log.info(f"Log file: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 加载数据
    sp = SNAPSHOT_DIR / args.snapshot_name
    log.info(f"Loading: {sp}")
    S, A, Y = load_snapshot(sp)
    num_outputs = Y.shape[1]
    log.info(f"Data: states={S.shape}, actions={A.shape}, labels={Y.shape}")

    # 划分训练/验证
    n = S.shape[0]
    n_val = int(round(n * 0.2))
    n_val = min(max(n_val, 1), n - 1)
    perm = rng.permutation(n)
    S_tr, A_tr, Y_tr = S[perm[n_val:]], A[perm[n_val:]], Y[perm[n_val:]]
    S_va, A_va, Y_va = S[perm[:n_val]], A[perm[:n_val]], Y[perm[:n_val]]

    # 标准化
    s_scaler = StandardScaler(); a_scaler = StandardScaler()
    S_tr = s_scaler.fit_transform(S_tr).astype(np.float32)
    S_va = s_scaler.transform(S_va).astype(np.float32)
    A_tr = a_scaler.fit_transform(A_tr).astype(np.float32)
    A_va = a_scaler.transform(A_va).astype(np.float32)

    y_scalers, Y_tr_s = [], np.zeros_like(Y_tr, dtype=np.float32)
    for d in range(num_outputs):
        s = StandardScaler()
        Y_tr_s[:, d] = s.fit_transform(Y_tr[:, d].reshape(-1, 1)).ravel()
        y_scalers.append(s)

    train_size = S_tr.shape[0]
    log.info(f"train={train_size}, val={S_va.shape[0]}")

    # 加载模型
    from src.tabicl._model.caim import CAIM
    model = CAIM(CKPT_PATH, num_outputs=num_outputs, state_dim=11, action_dim=3)
    model.to(device)
    log.info(f"Trainable params: {model.trainable_param_count:,}")

    if args.eval_only:
        log.info("Eval only mode")
        evaluate(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va, y_scalers, Y, device, rng)
        return

    # 优化器
    if args.phase == 1:
        # Phase 1: 只训练 ActionEncoder + CausalFusion
        trainable = (list(model.action_encoder.parameters()) +
                     list(model.causal_fusion.parameters()))
        log.info(f"Phase 1: training ActionEncoder + CausalFusion only ({sum(p.numel() for p in trainable):,} params)")
    else:
        # Phase 2: 训练所有可训练参数
        trainable = model.trainable_parameters
        log.info(f"Phase 2: training all trainable params ({model.trainable_param_count:,} params)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # 构造固定 ICL 输入
    test_size = min(128, S_va.shape[0])
    test_idx = rng.choice(S_va.shape[0], test_size, replace=False)

    X_s = torch.from_numpy(np.concatenate([S_tr, S_va[test_idx]], 0)).unsqueeze(0).to(device)
    X_a = torch.from_numpy(np.concatenate([A_tr, A_va[test_idx]], 0)).unsqueeze(0).to(device)
    Y_full = torch.from_numpy(
        np.concatenate([Y_tr_s, Y_tr_s[test_idx]], 0)  # placeholder for test y in cf training
    ).unsqueeze(0).to(device)
    # Actually need correct y_test for CF training
    Y_te_t = torch.from_numpy(Y_tr_s[test_idx]).unsqueeze(0).to(device)

    y_train_t = Y_full[:, :train_size, :]
    # y_test placeholder - we'll use the real test labels for loss computation

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        from src.tabicl._model.causal_fusion import MaskedActionModeling

        for j in range(num_outputs):
            optimizer.zero_grad()

            # 编码 action
            action_obs = model.action_encoder(X_a)

            y_j = y_train_t[:, :, j]

            # State 编码（backbone frozen → no_grad）
            with torch.no_grad():
                state_repr = model._encode_state(X_s, y_j)

            # 因果融合：获取纯 action 贡献 attn_out
            fused_obs, attn_obs = model.causal_fusion(state_repr, action_obs, return_attn=True)

            if args.phase == 1:
                # Masked Action Modeling: 从 attn_out 重建被 mask 的 action
                loss, _ = model.masked_action_modeling(attn_obs, X_a)
            else:
                # ICL 预测
                out_j = model._predict_dimension(fused_obs, y_j, j)
                train_size = y_j.shape[1]
                pred_j = out_j[:, train_size:, :].mean(dim=-1)
                loss_pred = torch.nn.functional.mse_loss(pred_j, Y_te_t[:, :, j])
                # 辅助任务: Masked Action Modeling
                loss_mam, _ = model.masked_action_modeling(attn_obs, X_a)
                loss = loss_pred + 0.1 * loss_mam

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()

        loss_val = total_loss / num_outputs

        if (epoch + 1) % max(1, args.num_epochs // 5) == 0:
            log.info(f"Epoch {epoch+1}/{args.num_epochs}, loss={loss_val:.6f}")

    # 评估
    log.info("Evaluating...")
    evaluate(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va, y_scalers, Y, device, rng)

    # 保存模型
    save_path = output_dir / f"caim_phase{args.phase}_epoch{args.num_epochs}.pt"
    torch.save({
        "action_encoder": model.action_encoder.state_dict(),
        "causal_fusion": model.causal_fusion.state_dict(),
        "y_encoders": model.y_encoders.state_dict(),
        "decoders": model.decoders.state_dict(),
    }, save_path)
    log.info(f"Model saved to: {save_path}")


def evaluate(model, S_tr, A_tr, Y_tr_s, S_va, A_va, Y_va, y_scalers, Y_raw, device, rng):
    """在验证集上计算 MSE。"""
    import csv as csv_mod
    model.eval()
    all_preds = []
    infer_bs = 256
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, S_va.shape[0], infer_bs):
            end = min(start + infer_bs, S_va.shape[0])
            X_s = torch.from_numpy(np.concatenate([S_tr, S_va[start:end]], 0)).unsqueeze(0).to(device)
            X_a = torch.from_numpy(np.concatenate([A_tr, A_va[start:end]], 0)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(Y_tr_s).unsqueeze(0).to(device)
            pred = model(X_s, X_a, y_t)
            all_preds.append(pred.squeeze(0).mean(dim=-1).cpu().numpy())

    elapsed = time.time() - t0
    preds_scaled = np.concatenate(all_preds, axis=0)[:S_va.shape[0]]
    preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
    for d in range(preds_scaled.shape[1]):
        preds_raw[:, d] = y_scalers[d].inverse_transform(preds_scaled[:, d].reshape(-1, 1)).ravel()

    mse_all, mse_r, mse_d = compute_mse(preds_raw, Y_va)
    log.info(f"MSE: all={mse_all:.6g}, reward={mse_r:.6g}, delta={mse_d:.6g}")

    # 对比 baseline
    try:
        with open(BASELINE_CSV) as f:
            for row in csv_mod.DictReader(f):
                if int(row["epoch"]) == 0:
                    b = float(row["tabicl_mse_all"])
                    log.info(f"vs OrigTabICL: {mse_all/b:.2f}x (baseline={b:.6g})")
                    break
    except Exception:
        pass

    log.info(f"Time: {elapsed:.1f}s")
    return mse_all, mse_r, mse_d


if __name__ == "__main__":
    main()
