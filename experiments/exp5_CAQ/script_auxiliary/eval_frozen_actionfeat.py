"""冻结模型 + action 特征表的评估对照。

验证假设："actionfeat 训练结果的劣化全部来自 Col/Row 适配，而非信息通道/架构"。

做法：加载全冻结的预训练模型（action_encoder/causal_block 为零初始化未训练
→ a_emb=0 且 causal_block 恒等 → 模型输出 = 原生冻结 TabICL 的输出），
用与 train_large4_actionfeat 相同的数据布局（35 列表格：27 state + 8 action，
同一 z-score 管线）评估 Ant-v4，并在同一脚本、同一数据下逐维对比 Baseline
（TabICLRegressor, ctx 20K）。

预期：若冻结模型 ≈ Baseline，则差距 100% 来自适配。
用法:
  python eval_frozen_actionfeat.py --device cuda
"""

import sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exp5_CAQ 目录（训练脚本所在）

from train_large4_actionfeat import (
    ENV_CONFIGS, load_data, preprocess_data, evaluate_env, evaluate_baseline_full,
    MAX_ACTION_DIM, format_mse,
)
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_context_eval", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    env_name = "Ant-v4"
    cfg = ENV_CONFIGS[env_name]

    # ── 数据加载与划分（与训练脚本完全一致）──
    print(f"📦 加载 {env_name} 数据...")
    X_state, X_action, y = load_data(cfg["dir"], cfg["epoch"])
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X_state))
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    n_train = int(len(X_state) * 0.8)
    n_test = min(args.n_test, len(X_state) - n_train)
    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:n_train + n_test]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:n_train + n_test]
    y_tr, y_te = y[:n_train], y[n_train:n_train + n_test]

    (X_state_tr_pp, X_state_te_pp, X_action_tr_pp, X_action_te_pp,
     y_tr_pp, y_te_pp, y_stats) = preprocess_data(
        X_state_tr, X_state_te, X_action_tr, X_action_te,
        y_tr, y_te, norm_method="none", seed=args.seed,
    )

    env_data = {
        "X_state_tr_pp": X_state_tr_pp,
        "X_state_te_pp": X_state_te_pp,
        "X_action_tr": X_action_tr_pp,   # z-scored action（与训练一致）
        "X_action_te": X_action_te_pp,
        "y_tr_pp": y_tr_pp,
        "y_te_pp": y_te_pp,
        "y_tr_raw": y_tr,
        "y_te_raw": y_te,
        "y_stats": y_stats,
        "output_dims": y.shape[1],
        "X_tr_raw": np.concatenate([X_state_tr, X_action_tr], axis=1),
        "X_te_raw": np.concatenate([X_state_te, X_action_te], axis=1),
    }

    # ── Baseline（同一脚本、同一数据）──
    print(f"\n📊 Baseline 评估中（ctx={args.n_context_eval}）...")
    mse_base = evaluate_baseline_full(
        env_data["X_tr_raw"], env_data["X_te_raw"], env_data["y_tr_raw"], env_data["y_te_raw"],
        device=str(device), n_context_eval=args.n_context_eval,
    )
    for d in range(y.shape[1]):
        print(f"    [{d:2d}] MSE={format_mse(mse_base[d])}")
    print(f"    avg MSE = {format_mse(np.mean(mse_base))}")

    # ── 冻结模型（CAQ 模块未训练 → 等价原生冻结 TabICL）──
    print(f"\n🧬 加载全冻结模型（action 特征表评估）...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=True, freeze_native_blocks=True, freeze_icl=True,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_trainable:,}（action_encoder/causal_block 零初始化未训练）")

    criterion = nn.MSELoss()
    print(f"\n🧊 冻结模型评估中（ctx={args.n_context_eval}，35 列表格）...")
    mse_caq = evaluate_env(model, env_data, device, criterion, args.n_context_eval)

    print(f"\n{'='*70}")
    print(f"  📊 冻结模型 vs Baseline（{env_name}, eval ctx={args.n_context_eval}）")
    print(f"  {'Dim':>4} {'Baseline':>12} {'Frozen':>12} {'Ratio':>10}")
    print(f"  {'-'*46}")
    for d in range(y.shape[1]):
        r = mse_caq[d] / mse_base[d] if mse_base[d] > 0 else float("inf")
        print(f"  {d:>4} {format_mse(mse_base[d]):>12} {format_mse(mse_caq[d]):>12} {r:>10.2f}x")
    print(f"  {'-'*46}")
    print(f"  {'OVERALL':>4} {format_mse(np.mean(mse_base)):>12} "
          f"{format_mse(np.mean(list(mse_caq.values()))):>12} "
          f"{np.mean(list(mse_caq.values())) / np.mean(mse_base):>10.2f}x")


if __name__ == "__main__":
    main()
