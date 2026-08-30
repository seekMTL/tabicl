"""CAQ-TabICL 通用 RL 环境训练脚本（并行 ICL 推理 + 完整模型微调）。

在 train_large3_multenv.py 基础上，新增 col_embedder 和 row_interactor 原生 blocks 的训练。
与 train_large3 仅训练 action_encoder + causal_block 不同，本脚本默认解冻 col_embedder
和 native_blocks，可通过 --freeze_* 参数控制。

通过 --env 指定 RL 环境，自动推断 state/action/output 维度，无需手动设置。
支持环境：Hopper, HalfCheetah, Walker2d, Ant

用法:
  python train_large3_multenv_colrow.py --env Hopper --epoch_idx 280 --lr 1e-4 --device cuda
  python train_large3_multenv_colrow.py --env Walker2d --epoch_idx 300 --device cuda
  python train_large3_multenv_colrow.py --env Hopper --freeze_col_embedder --freeze_native_blocks  # 等同 train_large3 行为
  python train_large3_multenv_colrow.py --resume checkpoints/xxx
"""

from __future__ import annotations

import sys, time, argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._sklearn.preprocessing import PreprocessingPipeline
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
DATA_BASE = "/home/lizitao/project/other/sample_collect"

# 环境名 → 数据目录映射
ENV_DIR_MAP = {
    "Hopper": "Hopper-v5-ensemble",
    "Walker2d": "walker2d",
    "HalfCheetah": "halfcheetah-1st",
    "Ant": "ant-v5",
}

# 以下维度在加载数据后自动推断，此处仅为默认占位
STATE_DIM = None
ACTION_DIM = None
OUTPUT_DIMS = None
DIM_LABELS = None


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def format_mse(val: float) -> str:
    if val == 0 or val >= 1e-5:
        return f"{val:.6f}"
    return f"{val:.6e}"


def load_data(env_dir: str, epoch_idx: int):
    data = np.load(f"{DATA_BASE}/{env_dir}/env_pool_epoch_{epoch_idx:04d}.npz")
    X_state = data["states"].astype(np.float32)
    X_action = data["actions"].astype(np.float32)
    delta = (data["next_states"] - data["states"]).astype(np.float32)
    y = np.concatenate([data["rewards"][:, None], delta], axis=1).astype(np.float32)
    return X_state, X_action, y


def preprocess_data(X_state_tr, X_state_te, y_tr, y_te,
                    norm_method="none", seed=42):
    """预处理数据，对齐 TabICL 预训练的 standard_scaling。

    X_state: PreprocessingPipeline（默认 none = 仅 Z-score + OutlierClip）
    y:      逐维 Z-score 标准化
    X_action 在外层保持原样（值域 [-1,1]，action_encoder 可训练自适应）

    Returns:
        X_state_tr_pp, X_state_te_pp: 预处理后的状态特征
        y_tr_pp, y_te_pp:             标准化后的目标值
        y_stats:                      {dim: (mean, std)}
    """
    preprocessor = PreprocessingPipeline(
        normalization_method=norm_method, random_state=seed
    )
    X_state_tr_pp = preprocessor.fit_transform(X_state_tr).astype(np.float32)
    X_state_te_pp = preprocessor.transform(X_state_te).astype(np.float32)

    y_tr_pp = np.zeros_like(y_tr)
    y_te_pp = np.zeros_like(y_te)
    y_stats = {}
    n_dims = y_tr.shape[1]
    for d in range(n_dims):
        mean = float(np.mean(y_tr[:, d]))
        std = float(np.std(y_tr[:, d]))
        if std < 1e-8:
            std = 1.0
        y_tr_pp[:, d] = (y_tr[:, d] - mean) / std
        y_te_pp[:, d] = (y_te[:, d] - mean) / std
        y_stats[d] = (mean, std)
    return X_state_tr_pp, X_state_te_pp, y_tr_pp, y_te_pp, y_stats


def create_icl_batch(X_state, X_action, y, n_context, n_query, device):
    """创建 ICL mini-batch，分离返回 state 和 action。

    CAQ 架构要求 state 和 action 分别输入：state 走 col_embedder，
    action 走 row_interactor 的 action_encoder → causal_block 路径。
    """
    n_total = len(X_state)
    all_idx = np.random.permutation(n_total)
    ctx_idx = all_idx[:n_context] # 取前 n_context 个样本作为 CAQ-ICL 训练时的上下文
    q_idx = all_idx[n_context : n_context + n_query] # 取后 n_query 个样本作为 CAQ-ICL 训练时的 query
    Xs_state = np.concatenate([X_state[ctx_idx], X_state[q_idx]])
    Xs_action = np.concatenate([X_action[ctx_idx], X_action[q_idx]])
    yb = np.concatenate([y[ctx_idx], y[q_idx]])
    return (
        torch.from_numpy(Xs_state).float().unsqueeze(0).to(device),
        torch.from_numpy(Xs_action).float().unsqueeze(0).to(device),
        torch.from_numpy(yb).float().unsqueeze(0).to(device),
    )


def create_icl_batch_shared(X_state, X_action, y, n_context, n_query, device):
    """创建共享 state/action 的 ICL batch，y 为所有输出维度批量返回。

    与 create_icl_batch 的区别：只采样一次索引，state/action 各维度共享，
    y 返回 (OUTPUT_DIMS, T) 形状的批量张量，用于 CAQ 模型的并行 ICL 训练。
    """
    n_total = len(X_state)
    all_idx = np.random.permutation(n_total)
    ctx_idx = all_idx[:n_context]
    q_idx = all_idx[n_context : n_context + n_query]
    Xs_state = np.concatenate([X_state[ctx_idx], X_state[q_idx]])
    Xs_action = np.concatenate([X_action[ctx_idx], X_action[q_idx]])
    # y 是 (n_total, OUTPUT_DIMS)，取所有维度
    yb = np.stack([np.concatenate([y[ctx_idx, d], y[q_idx, d]]) for d in range(y.shape[1])], axis=0)
    return (
        torch.from_numpy(Xs_state).float().unsqueeze(0).to(device),
        torch.from_numpy(Xs_action).float().unsqueeze(0).to(device),
        torch.from_numpy(yb).float().to(device),
    )


def evaluate_dim(model, X_state_tr, X_action_tr, y_tr,
                 X_state_te, X_action_te, y_te,
                 device, criterion, n_context_eval,
                 y_stat=None, y_tr_raw=None, y_te_raw=None):
    """CAQ 评估：从训练池取前 n_context_eval 样本做 ICL 上下文，预测测试集。

    y_tr, y_te 是标准化后的目标值（供 ICL context 使用）。
    若提供 y_stat + y_tr_raw + y_te_raw，则预测值逆变换到原始尺度后与原始目标计算 MSE（与 Baseline 公平比较）。
    """
    n_ctx = min(n_context_eval, len(X_state_tr)) # 限制评估时上下文的上限，默认 100K 样本
    model.eval()
    with torch.no_grad():
        eval_state = np.concatenate([X_state_tr[:n_ctx], X_state_te])
        eval_action = np.concatenate([X_action_tr[:n_ctx], X_action_te])
        eval_y = np.concatenate([y_tr[:n_ctx], y_te])  # 标准化目标，用于 ICL context
        X_state_t = torch.from_numpy(eval_state).float().unsqueeze(0).to(device)
        X_action_t = torch.from_numpy(eval_action).float().unsqueeze(0).to(device)
        yb = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
        y_train = yb[:, :n_ctx]
        # 模型前向，只对query位置（即后 n_test 个样本）做预测，输出 shape = (1, n_test, num_quantiles)
        out = model(X_state_t, X_action_t, y_train=y_train)
        # 标准化空间预测值，对 quantile 维度取均值 → (n_test,)
        pred = out.mean(dim=-1).squeeze(0)

        # 若提供了 y_stat，逆变换预测值后与原始目标计算 MSE
        if y_stat is not None:
            mean, std = y_stat
            pred = pred * std + mean
            raw = np.concatenate([y_tr_raw[:n_ctx], y_te_raw])
            target = torch.from_numpy(raw).float().unsqueeze(0).to(device)[0, n_ctx:] # 只取测试部分
        else:
            target = yb[0, n_ctx:]
        return criterion(pred, target).item()


# ═══════════════════════════════════════════════════════════════════
# 检查点（共享模型，不区分维度）
# ═══════════════════════════════════════════════════════════════════

def get_trainable_state_dict(model):
    """只提取可训练参数的 state_dict，排除冻结的 icl_predictor。

    与 train_large3 不同：本脚本默认解冻 col_embedder + native_blocks，
    检查点体积会增大（~2MB → ~tens of MB），取决于解冻的模块数量。
    """
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("icl_predictor")}


def save_checkpoint(save_dir, model, optimizer, scheduler, epoch, mse_caq, train_args, is_best=False):
    """保存训练检查点（仅保存可训练参数，~2MB/个）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "mse_caq": {str(k): float(v) for k, v in mse_caq.items()},
        "model_state": get_trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_args": train_args, # 恢复时校验超参数一致性
    }
    # epoch 补零到 4 位，确保字典序 = 数值序，否则如 epoch100 字符串排序在 epoch20 前面，会被清理
    path = save_dir / f"ckpt_epoch{epoch:04d}.pt"
    torch.save(state, path)
    if is_best:
        torch.save(state, save_dir / "best.pt")

    # 清理旧检查点（保留最新 3 个）
    ckpts = sorted(save_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-3]:
        old.unlink()


def save_progress(save_dir, mse_caq, mse_base, epoch, elapsed):
    """保存训练进度。每轮评估后调用，断电等异常中断后可通过 progress.json 查看训练状态。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "epoch": epoch,
        "mse_caq": {str(k): float(v) for k, v in mse_caq.items()},
        "mse_base": {str(k): float(v) for k, v in enumerate(mse_base)},
        "elapsed": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def load_progress(save_dir):
    """加载进度文件。"""
    path = save_dir / "progress.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# 共享模型训练
# ═══════════════════════════════════════════════════════════════════

def train_shared(
    model, optimizer, scheduler, criterion,
    X_state_tr, X_action_tr, y_tr,
    X_state_te, X_action_te, y_te,
    device, save_dir, train_args,
    n_context=1024, n_query=512,
    n_steps_per_dim=25, n_epochs=300, n_context_eval=100000,
    start_epoch=0, best_mse=None,
    early_stopping=False, patience_evals=4, min_epochs=100,
    eval_interval=20,
    mse_base=None, t_start=None, prev_elapsed=0,
    y_stats=None, y_tr_raw=None, y_te_raw=None,
):
    """共享 CAQ 模块训练。

    每步：采样一次共享的 state/action，ColEmbedding+RowInteraction 一次，
    ICL 阶段批量并行推理全部维度，累积 loss 后一次反向传播。
    """
    if best_mse is None:
        best_mse = {d: float("inf") for d in range(OUTPUT_DIMS)}
    mse_caq = {d: 0.0 for d in range(OUTPUT_DIMS)}
    evals_no_improve = 0

    for epoch in range(start_epoch, n_epochs):
        model.train() # 训练模式
        epoch_losses = []

        for _ in range(n_steps_per_dim):
            optimizer.zero_grad()
            # 1. 一次采样：state/action 共享，y 批量返回 (OUTPUT_DIMS, T)
            X_state, X_action, yb_all = create_icl_batch_shared(
                X_state_tr, X_action_tr, y_tr, n_context, n_query, device
            )
            y_train_all = yb_all[:, :n_context]  # (OUTPUT_DIMS, n_context)

            # 2. ColEmbedding + RowInteraction 只做一次（target_aware=False，y 仅用于提取 train_size）
            dummy_y = y_train_all[0:1]  # (1, n_context)
            representations = model._forward_embeddings(X_state, X_action, y_train=dummy_y)

            # 3. 扩展 CLS tokens + 批量 ICL（OUTPUT_DIMS 维并行推理）
            representations_batch = representations.repeat(OUTPUT_DIMS, 1, 1)  # (OUTPUT_DIMS, T, D)
            out = model.icl_predictor(representations_batch, y_train=y_train_all)

            # 4. 逐维计算 loss，累积后一次反向传播
            total_loss = torch.tensor(0.0, device=device)
            for d in range(OUTPUT_DIMS):
                loss = criterion(out[d].mean(dim=-1), yb_all[d, n_context:])
                total_loss = total_loss + loss
                epoch_losses.append(loss.item())
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            torch.cuda.empty_cache()  # 释放训练累积的缓存，为 100K 评估腾出显存
            improved_any = False
            improved_dims = set()  # 记录本次评估中打破历史最优的维度
            for d in range(OUTPUT_DIMS):
                test_mse = evaluate_dim(
                    model, X_state_tr, X_action_tr, y_tr[:, d],
                    X_state_te, X_action_te, y_te[:, d],
                    device, criterion, n_context_eval,
                    y_stat=y_stats[d], y_tr_raw=y_tr_raw[:, d], y_te_raw=y_te_raw[:, d],
                )
                if test_mse < best_mse[d]:
                    best_mse[d] = test_mse
                    improved_dims.add(d)
                    improved_any = True
                mse_caq[d] = test_mse

            if improved_any:
                evals_no_improve = 0
            else:
                evals_no_improve += 1

            # 打印全部维度的评估结果（表头 + MSE）
            header = "        " + "   ".join(f"{DIM_LABELS[d]:>12} " for d in range(OUTPUT_DIMS))
            values = "   ".join(
                f"{format_mse(mse_caq[d]):>12}{'▲' if d in improved_dims else ' '}"
                for d in range(OUTPUT_DIMS)
            )
            print(
                f"  epoch {epoch+1:4d}/{n_epochs} | "
                f"loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}\n"
                f"{header}\n"
                f"    MSE{values}"
            )

            if save_dir:
                save_checkpoint(save_dir, model, optimizer, scheduler,
                               epoch + 1, mse_caq, train_args, is_best=improved_any)
                # 每次评估后保存进度，防止异常中断丢失训练记录
                save_progress(save_dir, mse_caq, mse_base, epoch + 1,
                             prev_elapsed + (time.time() - t_start if t_start else 0))

            if early_stopping and epoch + 1 >= min_epochs and evals_no_improve >= patience_evals:
                print(f"  🛑 早停 (连续 {evals_no_improve} 轮评估无改善)")
                break

    return mse_caq


def evaluate_baseline_full(X_tr_pool, X_te, y_tr_pool, y_te, device="cuda", n_context_eval=100000):
    """Baseline: 输出维度 x TabICLRegressor，使用前 n_context_eval 个训练样本作为 ICL 上下文。"""
    n_ctx = min(n_context_eval, len(X_tr_pool))
    X_ctx, y_ctx = X_tr_pool[:n_ctx], y_tr_pool[:n_ctx]
    mse = np.zeros(OUTPUT_DIMS)
    t_start = time.time()
    for d in range(OUTPUT_DIMS):
        reg = TabICLRegressor(n_estimators=4, model_path=CKPT, device=device, random_state=42)
        reg.fit(X_ctx, y_ctx[:, d])
        pred = reg.predict(X_te)
        mse[d] = np.mean((pred - y_te[:, d]) ** 2)
    return mse, time.time() - t_start


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CAQ-TabICL 通用 RL 环境训练脚本")
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_DIR_MAP.keys()),
                        help=f"RL 环境名: {list(ENV_DIR_MAP.keys())}")
    parser.add_argument("--epoch_idx", type=int, default=280)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_context", type=int, default=1024)
    parser.add_argument("--n_query", type=int, default=512)
    parser.add_argument("--n_context_eval", type=int, default=100000, help="Baseline和CAQ评估的ICL上下文样本数")
    parser.add_argument("--n_test", type=int, default=10000, help="Baseline和CAQ的统一测试集样本数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_steps_per_dim", type=int, default=50,
                        help="每 epoch 的优化步数，每步累积全部维度 loss 后一次 backward")
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--early_stopping", action="store_true", default=False, help="启用早停")
    parser.add_argument("--freeze_col_embedder", action="store_true", default=False,
                        help="冻结 col_embedder（默认 False，即训练 col_embedder）")
    parser.add_argument("--freeze_native_blocks", action="store_true", default=False,
                        help="冻结 row_interactor 原生 blocks（默认 False，即训练原生 blocks）")
    parser.add_argument("--norm_method", type=str, default="none",
                        choices=["none", "power", "quantile", "quantile_rtdl", "robust"],
                        help="X_state 归一化方法，默认 none（与预训练分布一致）")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="从指定目录恢复训练")
    args = parser.parse_args()

    # ── 环境配置 ──
    env_dir = ENV_DIR_MAP[args.env]
    print(f"🎮 环境: {args.env} ({env_dir})")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) if args.save_dir else None

    # ── 数据加载 ──
    print(f"📦 加载数据...")
    X_state, X_action, y = load_data(env_dir, args.epoch_idx)

    # 从数据自动推断维度
    global STATE_DIM, ACTION_DIM, OUTPUT_DIMS, DIM_LABELS
    STATE_DIM = X_state.shape[1]
    ACTION_DIM = X_action.shape[1]
    OUTPUT_DIMS = y.shape[1]
    DIM_LABELS = ["reward"] + [f"delta[{i}]" for i in range(OUTPUT_DIMS - 1)]
    print(f"  样本: {len(X_state):,}, state={STATE_DIM}d, action={ACTION_DIM}d, target={OUTPUT_DIMS}d")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X_state)) # 随机打乱样本索引
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    # 划分训练集和测试集
    n_train = int(len(X_state) * 0.8) # 原始样本8成作为训练集
    n_test = min(args.n_test, len(X_state) - n_train) # 限制测试集上限10k
    # 训练集
    X_state_tr = X_state[:n_train]
    X_action_tr = X_action[:n_train]
    y_tr = y[:n_train]
    # 测试集
    X_state_te = X_state[n_train:n_train+n_test]
    X_action_te = X_action[n_train:n_train+n_test]
    y_te = y[n_train:n_train+n_test]
    print(f"  训练: {len(X_state_tr):,} | 测试: {len(X_state_te):,}")

    # ── 数据预处理 ──
    print(f"\n🔧 数据预处理 (norm_method={args.norm_method})...")
    X_state_tr_pp, X_state_te_pp, y_tr_pp, y_te_pp, y_stats = preprocess_data(
        X_state_tr, X_state_te, y_tr, y_te, norm_method=args.norm_method, seed=args.seed,
    )
    print(f"  X_state 预处理完成，y 逐维标准化完成")

    # ── Baseline ──
    print(f"\n{'='*60}\n  📊 Baseline 评估中...")
    X_tr = np.concatenate([X_state_tr, X_action_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([X_state_te, X_action_te], axis=1).astype(np.float32)
    mse_base, t_base = evaluate_baseline_full(
        # baseline评估时是state和action拼接一起作为特征输入
        X_tr, X_te, y_tr, y_te, device=str(device), n_context_eval=args.n_context_eval
    )
    print(f"  ✅ Baseline (context={min(args.n_context_eval, len(y_tr)):,}, test={len(y_te):,}) 用时 {t_base:.1f}s")
    for d in range(OUTPUT_DIMS):
        print(f"    [{d}] {DIM_LABELS[d]:<12} MSE={format_mse(mse_base[d])}")

    # ── 创建共享模型（仅一次）──
    caq = load_caq_model(
        CKPT, action_dim=ACTION_DIM, device=device,
        freeze_col_embedder=args.freeze_col_embedder,
        freeze_native_blocks=args.freeze_native_blocks,
        freeze_icl=True,
    )
    optimizer = AdamW(caq.trainable_parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    start_epoch, best_mse, prev_elapsed = 0, None, 0

    if args.resume and save_dir:
        best_path = save_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            caq.load_state_dict(ckpt["model_state"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"]
            best_mse = {int(k): float(v) for k, v in ckpt.get("mse_caq", {}).items()}
            progress = load_progress(save_dir)
            if progress:
                prev_elapsed = progress.get("elapsed", 0)
                prev_caq = np.mean([float(v) for v in progress.get("mse_caq", {}).values()])
                prev_base = np.mean([float(v) for v in progress.get("mse_base", {}).values()])
                print(f"  📂 恢复: epoch={start_epoch}, 此前训练 {prev_elapsed/3600:.1f}h, "
                      f"CAQ/Baseline={prev_caq/prev_base:.2f}x")
            else:
                print(f"  📂 从 best.pt 恢复: epoch={start_epoch} (progress.json 缺失)")

    train_args = {
        "lr": args.lr, "n_epochs": args.n_epochs, "n_context": args.n_context, "n_query": args.n_query,
        "n_steps_per_dim": args.n_steps_per_dim, "weight_decay": 1e-5,
        "norm_method": args.norm_method,
        "freeze_col_embedder": args.freeze_col_embedder, "freeze_native_blocks": args.freeze_native_blocks,
    }

    total_steps = args.n_steps_per_dim * args.n_epochs
    total_forwards = total_steps * OUTPUT_DIMS
    print(f"\n{'='*60}")
    print(f"  🧬 CAQ 模型训练（col_embedder={'❄️冻结' if args.freeze_col_embedder else '🔥训练'}，"
          f"native_blocks={'❄️冻结' if args.freeze_native_blocks else '🔥训练'}）")
    print(f"  {OUTPUT_DIMS} 维 loss 累积后一次反向传播")
    print(f"  {args.n_steps_per_dim} steps/epoch × {args.n_epochs} epochs = {total_steps:,} 步优化")
    print(f"  每步前向 {OUTPUT_DIMS} 维，共 {total_forwards:,} 次前向传播")
    if save_dir:
        print(f"  检查点: {save_dir}/best.pt")

    # ── 训练 ──
    t0 = time.time()
    mse_caq = None
    try:
        mse_caq = train_shared(
            caq, optimizer, scheduler, criterion,
            X_state_tr_pp, X_action_tr, y_tr_pp,
            X_state_te_pp, X_action_te, y_te_pp,
            device, save_dir=save_dir, train_args=train_args,
            n_context=args.n_context, n_query=args.n_query,
            n_steps_per_dim=args.n_steps_per_dim, n_epochs=args.n_epochs,
            n_context_eval=args.n_context_eval,
            start_epoch=start_epoch, best_mse=best_mse,
            early_stopping=args.early_stopping,
            eval_interval=args.eval_interval,
            mse_base=mse_base, t_start=t0, prev_elapsed=prev_elapsed,
            y_stats=y_stats, y_tr_raw=y_tr, y_te_raw=y_te,
        )
    except Exception as e:
        print(f"  ❌ 训练异常: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t0

    # ── 训练完成后统一重新评估全部维度（不依赖训练中 eval_interval 的快照）──
    mse_caq_final = None
    if mse_caq is not None:
        print(f"\n  🔍 最终评估（全 {OUTPUT_DIMS} 维，context={min(args.n_context_eval, len(X_state_tr)):,}）...")
        mse_caq_final = {}
        for d in range(OUTPUT_DIMS):
            mse_caq_final[d] = evaluate_dim(
                caq, X_state_tr_pp, X_action_tr, y_tr_pp[:, d],
                X_state_te_pp, X_action_te, y_te_pp[:, d],
                device, criterion, args.n_context_eval,
                y_stat=y_stats[d], y_tr_raw=y_tr[:, d], y_te_raw=y_te[:, d],
            )
        caq_overall = np.mean(list(mse_caq_final.values()))
        print(f"  ✅ 最终评估完成, OVERALL={format_mse(caq_overall)}")

    if save_dir and mse_caq_final is not None:
        save_progress(save_dir, mse_caq_final, mse_base, args.n_epochs, prev_elapsed + elapsed)

    print(f"\n  ✅ 训练完成，耗时 {elapsed/3600:.1f}h")

    # ── 最终结果 ──
    if mse_caq_final is None:
        print(f"\n  ❌ 无有效 CAQ 评估结果，跳过最终对比。")
        return

    print(f"\n{'='*70}")
    print(f"  📊 最终对比（共享 CAQ 模型）")
    print(f"  {'Dim':>4} {'Label':<12} {'Baseline':>12} {'CAQ':>12} {'Ratio':>10} {'Winner':>8}")
    print(f"  {'-'*60}")

    for d in range(OUTPUT_DIMS):
        b = mse_base[d]
        c = mse_caq_final[d]
        ratio = c / b if b > 0 else float("inf")
        winner = "✅ CAQ" if ratio < 1 else ("≈平" if ratio < 1.05 else "—")
        note = " (≈0)" if b < 1e-5 else ""
        print(f"  {d:>4} {DIM_LABELS[d]:<12} {format_mse(b):>12} {format_mse(c):>12} {ratio:>10.2f}x{note:>5} {winner:>8}")

    reward_b, reward_c = mse_base[0], mse_caq_final[0]
    delta_b_avg = np.mean(mse_base[1:])
    delta_c_avg = np.mean([mse_caq_final[d] for d in range(1, OUTPUT_DIMS)])
    overall_b = np.mean(mse_base)
    overall_c = np.mean(list(mse_caq_final.values()))

    print(f"  {'-'*60}")
    print(f"  {'REWARD':>17} {format_mse(reward_b):>12} {format_mse(reward_c):>12} {reward_c/reward_b:>10.2f}x")
    print(f"  {'DELTA(avg)':>17} {format_mse(delta_b_avg):>12} {format_mse(delta_c_avg):>12} {delta_c_avg/delta_b_avg:>10.2f}x")
    print(f"  {'OVERALL':>17} {format_mse(overall_b):>12} {format_mse(overall_c):>12} {overall_c/overall_b:>10.2f}x")

    print(f"\n  检查点保存在: {save_dir}/")


if __name__ == "__main__":
    main()