"""因素拆解评估：target_aware / ensemble / 适配对 Ant action 维度的贡献。

背景：
- 冻结模型 + action 特征表（target_aware=False）评估 = 9.19x，远超 baseline
- baseline TabICLRegressor 是满配集成（多归一化 + 特征排列 + target_aware=True）
- 本脚本逐项拆解，回答：差距主要来自哪个因素？

对照项（同一数据、同一 eval ctx=20K）：
  1. Baseline n_estimators=4（复现 0.085）
  2. Baseline n_estimators=1（单成员：隔离 ensemble 增益）
  3. 冻结模型 + target_aware=True（隔离 target_aware 贡献）
  4. actionfeat 训练 checkpoint + target_aware=True（适配 + target_aware 能否超 baseline）

用法:
  python eval_targetaware.py --device cuda
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
    ENV_CONFIGS, load_data, preprocess_data, evaluate_env,
    MAX_ACTION_DIM, format_mse,
)
from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
ACTIONFEAT_CKPT = str(_PROJECT_ROOT / "experiments/exp5_CAQ/checkpoints/train4_mix/actionfeat_ant_ctx1024/best.pt")


def run_baseline(X_tr, X_te, y_tr, y_te, device, n_ctx, n_estimators):
    mse = np.zeros(y_tr.shape[1])
    X_ctx, y_ctx = X_tr[:n_ctx], y_tr[:n_ctx]
    for d in range(y_tr.shape[1]):
        reg = TabICLRegressor(n_estimators=n_estimators, model_path=CKPT,
                              device=device, random_state=42)
        reg.fit(X_ctx, y_ctx[:, d])
        pred = reg.predict(X_te)
        mse[d] = np.mean((pred - y_te[:, d]) ** 2)
    return mse


def print_table(name, mse, mse_base4):
    mse = np.asarray(list(mse.values()) if isinstance(mse, dict) else mse)
    print(f"\n  {'='*64}")
    print(f"  {name}")
    print(f"  {'Dim':>4} {'Base(4)':>12} {'本项':>12} {'Ratio':>10}")
    print(f"  {'-'*46}")
    for d in range(len(mse)):
        r = mse[d] / mse_base4[d] if mse_base4[d] > 0 else float("inf")
        print(f"  {d:>4} {format_mse(mse_base4[d]):>12} {format_mse(mse[d]):>12} {r:>10.2f}x")
    print(f"  {'-'*46}")
    print(f"  {'OVERALL':>4} {format_mse(np.mean(mse_base4)):>12} "
          f"{format_mse(np.mean(mse)):>12} {np.mean(mse) / np.mean(mse_base4):>10.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_context_eval", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 数据（与训练脚本一致）──
    cfg = ENV_CONFIGS["Ant-v4"]
    print("📦 加载 Ant-v4 数据...")
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
        "X_action_tr": X_action_tr_pp,
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
    criterion = nn.MSELoss()
    n_ctx = args.n_context_eval

    # ── 1. Baseline 4-ensemble ──
    print(f"\n📊 [1/4] Baseline n_estimators=4（ctx={n_ctx}）...")
    mse_base4 = run_baseline(env_data["X_tr_raw"], env_data["X_te_raw"],
                             env_data["y_tr_raw"], env_data["y_te_raw"],
                             str(device), n_ctx, 4)
    print_table("Baseline n_estimators=4", mse_base4, mse_base4)

    # ── 2. Baseline 1-ensemble ──
    print(f"\n📊 [2/4] Baseline n_estimators=1（单成员）...")
    mse_base1 = run_baseline(env_data["X_tr_raw"], env_data["X_te_raw"],
                             env_data["y_tr_raw"], env_data["y_te_raw"],
                             str(device), n_ctx, 1)
    print_table("Baseline n_estimators=1", mse_base1, mse_base4)

    # ── 3. 冻结模型 + target_aware=True ──
    print(f"\n🧊 [3/4] 冻结模型 + target_aware=True（action 特征表）...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=True, freeze_native_blocks=True, freeze_icl=True,
    )
    # 恢复预训练时的 target-aware 设置（CAQ 模型构造时被关闭）
    model.col_embedder.target_aware = True
    mse_ta = evaluate_env(model, env_data, device, criterion, n_ctx)
    print_table("冻结 + target_aware=True", mse_ta, mse_base4)
    del model
    torch.cuda.empty_cache()

    # ── 4. actionfeat 训练 checkpoint + target_aware=True ──
    print(f"\n🔥 [4/4] actionfeat 训练模型 + target_aware=True...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False, freeze_icl=True,
    )
    ckpt = torch.load(ACTIONFEAT_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.col_embedder.target_aware = True
    mse_af = evaluate_env(model, env_data, device, criterion, n_ctx)
    print_table("actionfeat 训练 + target_aware=True", mse_af, mse_base4)

    print("\n✅ 因素拆解评估完成")


if __name__ == "__main__":
    main()
