"""因素拆解评估（补全版）：补上 state-only（action 仅经 causal 路径）的对照格。

上一版 eval_targetaware.py 遗漏了"冻结 + TA-on + state-only"格子，
以及 ant_v4_single checkpoint 的 TA-on 评估。本脚本补全（同一数据、同一协议）：

  1. Baseline n_estimators=4（基准）
  2. 冻结 + TA-on + action 特征表（35 列）→ 上一版 = 1.01x
  3. 冻结 + TA-on + state-only（27 列，action 仅经零初始化 causal 路径）
  4. ant_v4_single checkpoint（state-only 适配，TA-off 训练）+ TA-on 评估
  5. actionfeat checkpoint（action 特征适配，TA-off 训练）+ TA-on 评估

注意：3 是纯冻结对照；4、5 是"TA-off 训练的适配权重 + TA-on 评估"的组合实验，
用于与 2 对比，不能单独用于裁决"适配本身是否有害"（需要 TA-on 下从头训练）。

用法:
  python eval_targetaware2.py --device cuda
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
    ENV_CONFIGS, load_data, preprocess_data, evaluate_env, pad_action,
    MAX_ACTION_DIM, format_mse,
)
from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
ACTIONFEAT_CKPT = str(_PROJECT_ROOT / "experiments/exp5_CAQ/checkpoints/train4_mix/actionfeat_ant_ctx1024/best.pt")
ANTSINGLE_CKPT = str(_PROJECT_ROOT / "experiments/exp5_CAQ/checkpoints/train4_mix/only_ant_ctx1024/best.pt")


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


def evaluate_env_stateonly(model, env_data, device, criterion, n_context_eval):
    """state-only 评估：特征表不含 action 列（action 仅走 causal 路径）。

    即 train_large4_mixed.py 原版 evaluate_env 的逻辑。
    """
    X_state_tr = env_data["X_state_tr_pp"]
    X_action_tr = env_data["X_action_tr"]
    y_tr_pp = env_data["y_tr_pp"]
    X_state_te = env_data["X_state_te_pp"]
    X_action_te = env_data["X_action_te"]
    y_te_pp = env_data["y_te_pp"]
    y_te_raw = env_data["y_te_raw"]
    y_tr_raw = env_data["y_tr_raw"]
    y_stats = env_data["y_stats"]
    output_dims = env_data["output_dims"]

    n_ctx = min(n_context_eval, len(X_state_tr))
    model.eval()
    mse_dims = {}
    with torch.no_grad():
        for d in range(output_dims):
            eval_state = np.concatenate([X_state_tr[:n_ctx], X_state_te])
            eval_action = pad_action(np.concatenate([X_action_tr[:n_ctx], X_action_te]))
            eval_y = np.concatenate([y_tr_pp[:n_ctx, d], y_te_pp[:, d]])

            X_state_t = torch.from_numpy(eval_state).float().unsqueeze(0).to(device)
            X_action_t = torch.from_numpy(eval_action).float().unsqueeze(0).to(device)
            yb = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
            y_train = yb[:, :n_ctx]

            out = model(X_state_t, X_action_t, y_train=y_train)
            pred = out.mean(dim=-1).squeeze(0)

            mean, std = y_stats[d]
            pred = pred * std + mean
            raw = np.concatenate([y_tr_raw[:n_ctx, d], y_te_raw[:, d]])
            target = torch.from_numpy(raw).float().to(device)[n_ctx:]
            mse_dims[d] = criterion(pred, target).item()

    return mse_dims


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
    # 注意：actionfeat 的 preprocess_data 里 state 列与 action 列各自独立 z-score，
    # 所以 X_state_tr_pp 与"只预处理 state"的结果完全相同，state-only 评估可复用。

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
    print(f"\n📊 [1/5] Baseline n_estimators=4（ctx={n_ctx}）...")
    mse_base4 = run_baseline(env_data["X_tr_raw"], env_data["X_te_raw"],
                             env_data["y_tr_raw"], env_data["y_te_raw"],
                             str(device), n_ctx, 4)
    print_table("Baseline n_estimators=4", mse_base4, mse_base4)

    # ── 2. 冻结 + TA-on + action 特征表 ──
    print(f"\n🧊 [2/5] 冻结 + TA-on + action 特征表（35 列）...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=True, freeze_native_blocks=True, freeze_icl=True,
    )
    model.col_embedder.target_aware = True
    mse2 = evaluate_env(model, env_data, device, criterion, n_ctx)
    print_table("冻结 + TA-on + action特征", mse2, mse_base4)

    # ── 3. 冻结 + TA-on + state-only ──
    print(f"\n🧊 [3/5] 冻结 + TA-on + state-only（27 列，action 仅经零初始化 causal）...")
    mse3 = evaluate_env_stateonly(model, env_data, device, criterion, n_ctx)
    print_table("冻结 + TA-on + state-only", mse3, mse_base4)
    del model
    torch.cuda.empty_cache()

    # ── 4. ant_v4_single checkpoint + TA-on + state-only ──
    print(f"\n🔥 [4/5] ant_v4_single 适配模型 + TA-on + state-only...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False, freeze_icl=True,
    )
    ckpt = torch.load(ANTSINGLE_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.col_embedder.target_aware = True
    mse4 = evaluate_env_stateonly(model, env_data, device, criterion, n_ctx)
    print_table("ant_v4_single 适配 + TA-on + state-only", mse4, mse_base4)
    del model
    torch.cuda.empty_cache()

    # ── 5. actionfeat checkpoint + TA-on + action 特征表 ──
    print(f"\n🔥 [5/5] actionfeat 适配模型 + TA-on + action 特征表...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False, freeze_icl=True,
    )
    ckpt = torch.load(ACTIONFEAT_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.col_embedder.target_aware = True
    mse5 = evaluate_env(model, env_data, device, criterion, n_ctx)
    print_table("actionfeat 适配 + TA-on + action特征", mse5, mse_base4)

    print("\n✅ 补全拆解评估完成")


if __name__ == "__main__":
    main()
