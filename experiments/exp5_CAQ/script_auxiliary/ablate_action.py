"""causal 通道消融：验证 action 调制（action_encoder → causal_block）是否对预测有因果贡献。

三种 action 模式 × 两种模型（训练后 best.pt / 冻结初始）:
  normal        原始 action（与训练评估协议一致，= 日志里的评估数字）
  zero          action 全 0（z-score 后 = 平均动作，无动作特异性信息）
  shuffle_query 仅 query 行 action 随机重排（context 完好，破坏 query 行
                action 与 state/δ 的配对）

判据（对照冻结模型）:
  - 训练后 zero/shuffle 相对 normal 的 MSE 上升 > 冻结模型的对应上升
    → causal 通道在训练中学会了 action→δ
  - 训练后 normal 明显优于冻结 normal，但消融无差异
    → 0.87x 来自 Col/Row 对数据分布的适应，与因果调制无关

用法:
  python experiments/exp5_CAQ/script_auxiliary/ablate_action.py --device cuda:1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 复用 train_large6_counterfactual 的协议函数，保证与训练完全一致
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_large6_counterfactual import (
    load_counterfactual_data,
    preprocess_data,
    pad_action,
    load_caq_model,
    CKPT,
)

CKPT_DIR = Path("experiments/exp5_CAQ/checkpoints/train6_counterfactual/antv4_ctx1024_v4noise_flat")
N_CONTEXT_EVAL = 20000
SEED = 42


def eval_ablate(model, env_data, device, criterion, n_ctx, action_mode, shuffle_seed=0):
    """与 train_large6 的 evaluate_env 协议一致，仅对 action 做消融变换。"""
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

    n_ctx = min(n_ctx, len(X_state_tr))
    eval_action = pad_action(np.concatenate([X_action_tr[:n_ctx], X_action_te]))

    if action_mode == "zero":
        eval_action = np.zeros_like(eval_action)
    elif action_mode == "shuffle_query":
        rng = np.random.RandomState(shuffle_seed)
        q = eval_action[n_ctx:].copy()
        eval_action[n_ctx:] = q[rng.permutation(len(q))]

    model.eval()
    mse_dims = {}
    with torch.no_grad():
        for d in range(output_dims):
            eval_state = np.concatenate([X_state_tr[:n_ctx], X_state_te])
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


def build_env_data(env_name="Ant-v4", cfg_dir="Ant-v4"):
    """复刻 train_large6 main 的数据准备：全行打乱 + 80/20 切分 + 预处理。"""
    print(f"📦 加载反事实数据（{env_name}，1-3 分钟）...")
    X_state, X_action, y = load_counterfactual_data(cfg_dir)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_state))
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    n_train = int(len(X_state) * 0.8)
    n_test = min(10000, len(X_state) - n_train)
    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:n_train + n_test]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:n_train + n_test]
    y_tr, y_te = y[:n_train], y[n_train:n_train + n_test]

    (X_state_tr_pp, X_state_te_pp, X_action_tr_pp, X_action_te_pp,
     y_tr_pp, y_te_pp, y_stats) = preprocess_data(
        X_state_tr, X_state_te, X_action_tr, X_action_te,
        y_tr, y_te, norm_method="none", seed=SEED,
    )

    return {
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
    }


def main():
    parser = argparse.ArgumentParser(description="causal 通道 action 消融")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--env", type=str, default="Ant-v4")
    parser.add_argument("--env_dir", type=str, default="Ant-v4")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    env_data = build_env_data(args.env, args.env_dir)
    print(f"  训练 {len(env_data['X_state_tr_pp']):,} 行 测试 {len(env_data['X_state_te_pp']):,} 行")

    # 模型：与训练脚本相同的构造参数
    caq = load_caq_model(
        CKPT, action_dim=8, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False,
        freeze_icl=True, target_aware=True,
    )

    # ── 冻结初始（对照）──
    print("\n❄️ 冻结初始模型评估（对照）...")
    frozen = {}
    for mode in ("normal", "zero", "shuffle_query"):
        mse = eval_ablate(caq, env_data, device, criterion, N_CONTEXT_EVAL, mode)
        frozen[mode] = np.mean(list(mse.values()))
        print(f"  [{mode:<13}] avg={frozen[mode]:.6f}")

    # ── 训练后（best.pt）──
    best_path = CKPT_DIR / "best.pt"
    ckpt = torch.load(best_path, map_location=device)
    caq.load_state_dict(ckpt["model_state"], strict=False)
    print(f"\n🔥 训练后模型评估（{best_path.name}，epoch={ckpt.get('epoch')}）...")
    trained = {}
    for mode in ("normal", "zero", "shuffle_query"):
        mse = eval_ablate(caq, env_data, device, criterion, N_CONTEXT_EVAL, mode)
        trained[mode] = np.mean(list(mse.values()))
        print(f"  [{mode:<13}] avg={trained[mode]:.6f}")

    # ── 对比表 ──
    print("\n" + "=" * 66)
    print("  causal 通道消融对比（avg MSE，越低越好）")
    print("=" * 66)
    print(f"  {'模式':<14} {'冻结':>10} {'训练后':>10} {'训练后-冻结':>10}")
    for mode in ("normal", "zero", "shuffle_query"):
        print(f"  {mode:<14} {frozen[mode]:>10.6f} {trained[mode]:>10.6f} "
              f"{trained[mode]-frozen[mode]:>+10.6f}")
    print("-" * 66)
    print(f"  zero 相对 normal 上升:    冻结 {frozen['zero']/frozen['normal']:>5.3f}x  "
          f"训练后 {trained['zero']/trained['normal']:>5.3f}x")
    print(f"  shuffle 相对 normal 上升: 冻结 {frozen['shuffle_query']/frozen['normal']:>5.3f}x  "
          f"训练后 {trained['shuffle_query']/trained['normal']:>5.3f}x")
    print(f"  baseline（训练日志口径）= 0.047680；训练后 normal = {trained['normal']:.6f} "
          f"({trained['normal']/0.047680:.3f}x)")

    # 判据解读
    t_zero = trained["zero"] / trained["normal"]
    f_zero = frozen["zero"] / frozen["normal"]
    t_shuf = trained["shuffle_query"] / trained["normal"]
    f_shuf = frozen["shuffle_query"] / frozen["normal"]
    if t_zero > 1.02 or t_shuf > 1.02:
        if (t_zero - f_zero > 0.02) or (t_shuf - f_shuf > 0.02):
            print("\n  ✅ 结论: 训练后模型对 action 消融更敏感 → causal 通道学到了 action→δ")
        else:
            print("\n  ⚠️ 结论: 消融有影响但训练前后差异小 → 因果调制可能主要来自预训练结构")
    else:
        print("\n  ❌ 结论: action 消融几乎无影响 → causal 通道未学到 action→δ，"
              "0.87x 来自其他因素")


if __name__ == "__main__":
    main()
