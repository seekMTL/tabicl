"""CAQ-TabICL Hopper-v5 实验脚本。

对照实验设计：
  - Baseline: 12 个独立 TabICLRegressor（每个预测 1 维输出）
  - CAQ-TabICL: 1 个 CAQTabICL 模型（状态+动作分离输入，新增 N+1 层因果检索）

实验阶段：
  Phase I (Sanity Check):    4000 样本快速验证 + 注意力权重检查
  Phase II (Full Comparison): 全量数据对照实验

用法:
  # Sanity check
  python train_caq.py --mode sanity --epoch_idx 280 --train_samples 4000

  # 全量对照实验
  python train_caq.py --mode full --epoch_idx 280

  # 仅 baseline
  python train_caq.py --mode baseline_only --epoch_idx 280

  # 注意力分析
  python train_caq.py --mode analyze_attn --epoch_idx 280
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加项目根目录到 path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._model.caq_model import load_caq_model

# ── 路径配置 ──
CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
DATA_DIR = "/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble"

STATE_DIM = 11
ACTION_DIM = 3
OUTPUT_DIMS = 12  # 1 reward + 11 delta_state
DIM_LABELS = ["reward"] + [f"delta[{i}]" for i in range(11)]


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════


def load_hopper_data(epoch_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 Hopper-v5 ensemble 数据。

    Returns:
        X_state:  (n, 11) - 状态特征
        X_action: (n, 3)  - 动作特征
        y:        (n, 12) - 目标 (1 reward + 11 delta_state)
    """
    data = np.load(f"{DATA_DIR}/env_pool_epoch_{epoch_idx:04d}.npz")
    X_state = data["states"].astype(np.float32)
    X_action = data["actions"].astype(np.float32)
    delta = (data["next_states"] - data["states"]).astype(np.float32)
    y = np.concatenate([data["rewards"][:, None], delta], axis=1).astype(np.float32)
    return X_state, X_action, y


def concat_state_action(X_state: np.ndarray, X_action: np.ndarray) -> np.ndarray:
    """拼接状态和动作作为 Baseline 的输入（state_dim + action_dim 维特征）。"""
    return np.concatenate([X_state, X_action], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Baseline 评估 (12 个 TabICLRegressor)
# ═══════════════════════════════════════════════════════════════════


def evaluate_baseline(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    device: str = "cuda",
    n_estimators: int = 4,
) -> Tuple[np.ndarray, float]:
    """Baseline: 对每个输出维度独立运行 TabICLRegressor。

    Args:
        X_tr: (n_train, state_dim + action_dim) - 训练特征
        X_te: (n_test, state_dim + action_dim) - 测试特征
        y_tr: (n_train, 12) - 训练目标
        y_te: (n_test, 12) - 测试目标

    Returns:
        mse: (12,) per-dimension MSE
        elapsed: 总耗时 (秒)
    """
    mse = np.zeros(OUTPUT_DIMS)
    t0 = time.time()

    for d in range(OUTPUT_DIMS):
        reg = TabICLRegressor(
            n_estimators=n_estimators,
            model_path=CKPT,
            device=device,
            random_state=42,
        )
        reg.fit(X_tr, y_tr[:, d])
        pred = reg.predict(X_te)
        mse[d] = np.mean((pred - y_te[:, d]) ** 2)

    elapsed = time.time() - t0
    return mse, elapsed


# ═══════════════════════════════════════════════════════════════════
# CAQ-TabICL 训练与评估
# ═══════════════════════════════════════════════════════════════════


def train_caq_single_dim(
    model: nn.Module,
    X_state_tr: np.ndarray,
    X_action_tr: np.ndarray,
    y_tr: np.ndarray,
    X_state_te: np.ndarray,
    X_action_te: np.ndarray,
    y_te: np.ndarray,
    device: torch.device,
    lr: float = 1e-4,
    n_epochs: int = 100,
    weight_decay: float = 1e-5,
    context_ratio: float = 0.5,
    verbose: bool = True,
) -> float:
    """针对单个输出维度训练 CAQ 新增模块。

    每次迭代随机采样上下文样本，模型用上下文做 ICL 来预测测试样本。
    只训练 CAQ 新增模块（action_encoder + causal_block），
    原生 Block、ColEmbedding、ICLearning 保持冻结。

    Args:
        model: CAQTabICL 模型
        X_state_tr, X_action_tr: (n_train, dim) 训练集状态/动作
        y_tr: (n_train,) 训练集目标
        X_state_te, X_action_te: (n_test, dim) 测试集状态/动作
        y_te: (n_test,) 测试集目标
        device: 训练设备
        lr: 学习率
        n_epochs: 训练 epoch 数
        weight_decay: 权重衰减
        context_ratio: 上下文样本比例

    Returns:
        test_mse: 最优测试集 MSE
    """
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_context = int(len(X_state_tr) * context_ratio)
    n_total = len(X_state_tr)

    criterion = nn.MSELoss()
    best_mse = float("inf")

    for epoch in range(n_epochs):
        model.train()

        # 随机采样上下文和测试样本
        idx = np.random.permutation(n_total)
        ctx_idx = idx[:n_context]
        n_test = min(n_context, n_total - n_context)
        te_idx = idx[n_context : n_context + n_test]

        # 拼接 context + test → (B=1, T=n_context+n_test, dim)
        batch_state = np.concatenate(
            [X_state_tr[ctx_idx], X_state_tr[te_idx]], axis=0
        )
        batch_action = np.concatenate(
            [X_action_tr[ctx_idx], X_action_tr[te_idx]], axis=0
        )
        batch_y = np.concatenate([y_tr[ctx_idx], y_tr[te_idx]], axis=0)

        Xs = torch.from_numpy(batch_state).float().unsqueeze(0).to(device)
        Xa = torch.from_numpy(batch_action).float().unsqueeze(0).to(device)
        yb = torch.from_numpy(batch_y).float().unsqueeze(0).to(device)
        y_train = yb[:, :n_context]

        optimizer.zero_grad()
        pred = model(Xs, Xa, y_train)  # (1, n_test, num_quantiles)
        pred_mean = pred.mean(dim=-1).squeeze(0)  # 取分位数均值作为点预测
        target = yb[0, n_context : n_context + n_test]
        loss = criterion(pred_mean, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 验证（保持 train 模式避免触发 InferenceManager 的 AMP autocast）
        # 使用全部训练数据作为上下文，与 Baseline 对齐
        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                n_ctx_eval = len(X_state_tr)  # 全部训练数据做上下文
                eval_state = np.concatenate(
                    [X_state_tr[:n_ctx_eval], X_state_te], axis=0
                )
                eval_action = np.concatenate(
                    [X_action_tr[:n_ctx_eval], X_action_te], axis=0
                )
                eval_y = np.concatenate([y_tr[:n_ctx_eval], y_te], axis=0)

                Xs_e = torch.from_numpy(eval_state).float().unsqueeze(0).to(device)
                Xa_e = torch.from_numpy(eval_action).float().unsqueeze(0).to(device)
                yb_e = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
                y_train_e = yb_e[:, :n_ctx_eval]

                pred_e = model(Xs_e, Xa_e, y_train_e)
                pred_mean_e = pred_e.mean(dim=-1).squeeze(0)
                target_e = yb_e[0, n_ctx_eval:]
                test_mse = criterion(pred_mean_e, target_e).item()

            if test_mse < best_mse:
                best_mse = test_mse

            if verbose:
                print(
                    f"    Epoch {epoch+1:4d}/{n_epochs} | "
                    f"train_loss={loss.item():.6f} | test_mse={test_mse:.6f}"
                )

    return best_mse


def evaluate_caq(
    X_state_all: np.ndarray,
    X_action_all: np.ndarray,
    y_all: np.ndarray,
    train_ratio: float = 0.67,
    device: torch.device = None,
    lr: float = 1e-4,
    n_epochs: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """CAQ-TabICL 完整评估：对 12 个输出维度分别训练 CAQ 模块并评估。

    每个维度重新加载模型，避免跨维度污染。

    Returns:
        mse: (12,) per-dimension MSE
        elapsed: 总耗时 (秒)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train = int(len(X_state_all) * train_ratio)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_state_all))

    X_state_all_s = X_state_all[idx]
    X_action_all_s = X_action_all[idx]
    y_all_s = y_all[idx]

    X_state_tr = X_state_all_s[:n_train]
    X_action_tr = X_action_all_s[:n_train]
    y_tr = y_all_s[:n_train]

    X_state_te = X_state_all_s[n_train:]
    X_action_te = X_action_all_s[n_train:]
    y_te = y_all_s[n_train:]

    mse = np.zeros(OUTPUT_DIMS)
    t0 = time.time()

    for d in range(OUTPUT_DIMS):
        print(f"\n  [{d}] {DIM_LABELS[d]}:")

        caq = load_caq_model(
            CKPT,
            action_dim=ACTION_DIM,
            device=device,
            freeze_col_embedder=True,
            freeze_icl=True,
        )
        print(f"    {caq.summary()}")

        mse[d] = train_caq_single_dim(
            caq,
            X_state_tr,
            X_action_tr,
            y_tr[:, d],
            X_state_te,
            X_action_te,
            y_te[:, d],
            device=device,
            lr=lr,
            n_epochs=n_epochs,
        )

    elapsed = time.time() - t0
    return mse, elapsed


# ═══════════════════════════════════════════════════════════════════
# 注意力权重分析（Sanity Check 核心验证）
# ═══════════════════════════════════════════════════════════════════


def analyze_attention_weights(
    model: nn.Module,
    X_state: np.ndarray,
    X_action: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_context: int = 200,
    n_actions: int = 5,
):
    """分析 CausalInterventionBlock 的交叉注意力权重。

    对不同的动作输入，检查注意力权重矩阵 (4, G) 是否展现出动态变化。
    成功标志：不同动作输入下，前 4 个 Query Token 对特征组的注意力分布应有明显差异。

    Args:
        model: 训练后的 CAQTabICL
        X_state: (n, 11) 状态数据
        X_action: (n, 3) 动作数据
        y: (n,) 目标值
        device: 设备
        n_context: 上下文样本数
        n_actions: 采样的对比动作数量
    """
    model.train()  # 保持 train 模式避免 InferenceManager AMP 问题

    # 准备一批数据
    n_total = n_context + 50
    batch_state = X_state[:n_total]
    batch_action = X_action[:n_total]
    batch_y = y[:n_total]

    Xs = torch.from_numpy(batch_state).float().unsqueeze(0).to(device)
    Xa = torch.from_numpy(batch_action).float().unsqueeze(0).to(device)
    yb = torch.from_numpy(batch_y).float().unsqueeze(0).to(device)

    # 通过 ColEmbedding 获取特征嵌入
    row = model.row_interactor
    cross_attn = row.causal_block.cross_attn

    with torch.no_grad():
        col_emb = model.col_embedder(Xs, y_train=yb.squeeze(0))
        B, T, HC, E = col_emb.shape

        # 注入 CLS token
        cls_tokens = row.cls_tokens.expand(B, T, row.num_cls, row.embed_dim)
        col_emb[:, :, : row.num_cls] = cls_tokens.to(col_emb.device)

        # 前 N-1 个原生 Block
        emb = col_emb
        for block in row.blocks_except_last:
            emb = block(emb, rope=row.rope)

        # 切片 KV 特征组
        kv_features = emb[:, :, row.num_cls :, :]  # (1, T, G, E)

        # 第 N 个原生 Block → cls_1
        cls_outputs = row.last_native_block(
            q=emb[..., : row.num_cls, :], k=emb, v=emb, rope=row.rope
        )
        cls_1 = cls_outputs  # (1, T, 4, E)

        G = kv_features.shape[2]

        # 对多个不同动作，观察注意力权重变化
        print(f"\n  🔍 注意力权重分析 (CausalInterventionBlock 交叉注意力)")
        print(f"  特征组数 G={G}, CLS 数={row.num_cls}, 动作维度={ACTION_DIM}")
        print(f"\n  {'样本':<6} {'动作向量':<50} {'attn均值':<12} {'attn标准差':<12}")
        print(f"  {'-'*82}")

        all_attn = []
        for i in range(min(n_actions, T)):
            a_single = Xa[:, i : i + 1, :]  # (1, 1, action_dim)
            a_emb = row.action_encoder(a_single)  # (1, 1, 4, E)
            Q = cls_1[:, i : i + 1, :, :] + a_emb  # (1, 1, 4, E)
            KV = kv_features[:, i : i + 1, :, :]  # (1, 1, G, E)

            q_flat = Q.view(1, row.num_cls, E)
            kv_flat = KV.view(1, G, E)

            _, attn_weights = cross_attn(query=q_flat, key=kv_flat, value=kv_flat)
            attn_w = attn_weights.squeeze(0).cpu().numpy()  # (4, G)
            all_attn.append(attn_w)

            action_vec = Xa[0, i, :].cpu().numpy()
            print(
                f"  {i:<6} "
                f"{np.array2string(action_vec, precision=3, suppress_small=True):<50} "
                f"{attn_w.mean():<12.6f} "
                f"{attn_w.std():<12.6f}"
            )

        # 计算不同动作间注意力的差异
        if len(all_attn) >= 2:
            all_attn = np.stack(all_attn, axis=0)  # (n_actions, 4, G)
            pairwise_diff = []
            for i in range(len(all_attn)):
                for j in range(i + 1, len(all_attn)):
                    diff = np.abs(all_attn[i] - all_attn[j]).mean()
                    pairwise_diff.append(diff)

            avg_inter_action_diff = np.mean(pairwise_diff)
            print(f"\n  📊 动作间平均注意力差异: {avg_inter_action_diff:.6f}")
            if avg_inter_action_diff > 0.01:
                print(f"  ✅ 通过！不同动作下的注意力分布有明显差异，")
                print(f"     证明模型学到了动作对状态特征的因果检索能力。")
            else:
                print(f"  ⚠️  注意力差异较小，可能需要更多训练或调整架构。")


# ═══════════════════════════════════════════════════════════════════
# 结果输出
# ═══════════════════════════════════════════════════════════════════


def print_comparison(
    mse_base: np.ndarray, mse_caq: np.ndarray, t_base: float, t_caq: float
):
    """打印 Baseline vs CAQ-TabICL 逐维度对比表。"""
    print(f"\n{'='*80}")
    print(f"  Baseline vs CAQ-TabICL 逐维度 MSE 对比")
    print(f"  {'Dim':>4} {'Label':<12} {'Baseline':>12} {'CAQ':>12} {'Ratio(C/B)':>10} {'Winner':>8}")
    print(f"  {'-'*66}")

    for d in range(OUTPUT_DIMS):
        label = DIM_LABELS[d]
        ratio = mse_caq[d] / mse_base[d] if mse_base[d] > 0 else float("inf")
        winner = "CAQ ✓" if ratio < 1 else "Baseline"
        print(
            f"  {d:>4} {label:<12} {mse_base[d]:>12.6f} {mse_caq[d]:>12.6f} "
            f"{ratio:>10.2f}x {winner:>8}"
        )

    print(f"  {'-'*66}")

    reward_ratio = mse_caq[0] / mse_base[0] if mse_base[0] > 0 else float("inf")
    delta_ratio = (
        np.sum(mse_caq[1:]) / np.sum(mse_base[1:])
        if np.sum(mse_base[1:]) > 0
        else float("inf")
    )
    overall_ratio = (
        np.mean(mse_caq) / np.mean(mse_base)
        if np.mean(mse_base) > 0
        else float("inf")
    )

    print(f"  {'REWARD':>17} {mse_base[0]:>12.6f} {mse_caq[0]:>12.6f} {reward_ratio:>10.2f}x")
    print(f"  {'DELTA(sum)':>17} {np.sum(mse_base[1:]):>12.6f} {np.sum(mse_caq[1:]):>12.6f} {delta_ratio:>10.2f}x")
    print(f"  {'OVERALL':>17} {np.mean(mse_base):>12.6f} {np.mean(mse_caq):>12.6f} {overall_ratio:>10.2f}x")
    print(f"\n  耗时: Baseline={t_base:.1f}s | CAQ={t_caq:.1f}s")

    return reward_ratio, delta_ratio, overall_ratio


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="CAQ-TabICL Hopper-v5 实验")
    parser.add_argument(
        "--mode", type=str, default="sanity",
        choices=["sanity", "full", "baseline_only", "caq_only", "analyze_attn"],
        help="实验模式: sanity=快速验证 | full=全量对照 | baseline_only/caq_only=单项 | analyze_attn=注意力分析",
    )
    parser.add_argument("--epoch_idx", type=int, default=280, help="数据快照编号")
    parser.add_argument(
        "--train_samples", type=int, default=4000,
        help="Sanity check 模式的训练上下文样本数"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--n_epochs", type=int, default=100, help="CAQ 模块训练 epoch 数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 CAQ-TabICL Hopper-v5 实验")
    print(f"  模式: {args.mode} | 设备: {device} | 数据: epoch {args.epoch_idx}")

    # ── 加载数据 ──
    X_state, X_action, y = load_hopper_data(args.epoch_idx)
    print(f"  数据: {len(X_state)} 样本 | "
          f"state={X_state.shape[1]}d | action={X_action.shape[1]}d | "
          f"target={y.shape[1]}d")

    # 限制样本数（所有模式都适用，避免 OOM）
    if args.train_samples > 0:
        n_total = min(args.train_samples * 2, len(X_state))
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(X_state), n_total, replace=False)
        X_state, X_action, y = X_state[idx], X_action[idx], y[idx]
        print(f"  Sanity Check: 限制 {n_total} 样本")

    # 分割训练/测试
    n_train = int(len(X_state) * 0.67)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X_state))
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    # Baseline 需要拼接 state+action (14 维特征)
    X_tr = concat_state_action(X_state_tr, X_action_tr)
    X_te = concat_state_action(X_state_te, X_action_te)

    print(f"  训练: {len(X_tr)} | 测试: {len(X_te)}")
    print(f"  y 范围: reward [{y[:,0].min():.3f}, {y[:,0].max():.3f}]")

    mse_base, mse_caq = None, None
    t_base, t_caq = 0.0, 0.0

    # ── Baseline 评估 ──
    if args.mode in ("baseline_only", "sanity", "full"):
        print(f"\n{'='*60}")
        print(f"  📊 Baseline: 12 x TabICLRegressor(num_outputs=1)")
        print(f"     状态+动作拼接为 14 维特征输入")
        print(f"{'='*60}")
        mse_base, t_base = evaluate_baseline(X_tr, X_te, y_tr, y_te, device=args.device)
        print(f"\n  ✅ Baseline 完成 | 耗时: {t_base:.1f}s")
        for d in range(OUTPUT_DIMS):
            print(f"    [{d}] {DIM_LABELS[d]:<12} MSE={mse_base[d]:.6f}")

    # ── CAQ-TabICL 评估 ──
    if args.mode in ("caq_only", "sanity", "full"):
        print(f"\n{'='*60}")
        print(f"  🧬 CAQ-TabICL: 状态(11d) + 动作(3d) 分离输入")
        print(f"     新增第 N+1 层因果交叉检索")
        print(f"     冻结 ColEmbedding + ICL，仅训练 CAQ 模块")
        print(f"{'='*60}")

        n_epochs = min(50, args.n_epochs) if args.mode == "sanity" else args.n_epochs

        mse_caq, t_caq = evaluate_caq(
            np.concatenate([X_state_tr, X_state_te]),
            np.concatenate([X_action_tr, X_action_te]),
            np.concatenate([y_tr, y_te]),
            train_ratio=n_train / len(X_state),
            device=device,
            lr=args.lr,
            n_epochs=n_epochs,
            seed=args.seed,
        )
        print(f"\n  ✅ CAQ-TabICL 完成 | 耗时: {t_caq:.1f}s")
        for d in range(OUTPUT_DIMS):
            print(f"    [{d}] {DIM_LABELS[d]:<12} MSE={mse_caq[d]:.6f}")

    # ── 打印对比结果 ──
    if mse_base is not None and mse_caq is not None:
        print_comparison(mse_base, mse_caq, t_base, t_caq)

    # ── 注意力分析 ──
    if args.mode == "analyze_attn":
        print(f"\n{'='*60}")
        print(f"  🔍 注意力权重分析（Sanity Check 核心）")
        print(f"{'='*60}")
        caq = load_caq_model(CKPT, action_dim=ACTION_DIM, device=device)
        analyze_attention_weights(
            caq, X_state, X_action, y[:, 0], device, n_context=200
        )


if __name__ == "__main__":
    main()
