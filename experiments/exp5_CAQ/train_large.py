"""CAQ-TabICL 大规模训练脚本：全量 Hopper-v5 数据 + mini-batch 训练 + 检查点

用法:
  # 从头训练
  python train_large.py --epoch_idx 280 --lr 1e-4 --n_epochs 300 --device cuda

  # 从检查点恢复
  python train_large.py --resume checkpoints

  # 仅训练 reward 维度测试
  python train_large.py --dims reward_only --save_dir checkpoints
"""

from __future__ import annotations

import sys, time, argparse, json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
DATA_DIR = "/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble"

STATE_DIM = 11
ACTION_DIM = 3
OUTPUT_DIMS = 12
DIM_LABELS = ["reward"] + [f"delta[{i}]" for i in range(11)]


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════

def load_all_hopper(epoch_idx: int):
    data = np.load(f"{DATA_DIR}/env_pool_epoch_{epoch_idx:04d}.npz")
    X_state = data["states"].astype(np.float32)
    X_action = data["actions"].astype(np.float32)
    delta = (data["next_states"] - data["states"]).astype(np.float32)
    y = np.concatenate([data["rewards"][:, None], delta], axis=1).astype(np.float32)
    return X_state, X_action, y


def create_icl_batch(
    X_state: np.ndarray, X_action: np.ndarray, y: np.ndarray,
    n_context: int, n_query: int, device: torch.device
):
    n_total = len(X_state)
    all_idx = np.random.permutation(n_total)
    ctx_idx = all_idx[:n_context]
    q_idx = all_idx[n_context : n_context + n_query]
    Xs = np.concatenate([
        np.concatenate([X_state[ctx_idx], X_action[ctx_idx]], axis=1),
        np.concatenate([X_state[q_idx], X_action[q_idx]], axis=1),
    ])
    Xa = np.concatenate([X_action[ctx_idx], X_action[q_idx]])
    yb = np.concatenate([y[ctx_idx], y[q_idx]])
    return (
        torch.from_numpy(Xs).float().unsqueeze(0).to(device),
        torch.from_numpy(Xa).float().unsqueeze(0).to(device),
        torch.from_numpy(yb).float().unsqueeze(0).to(device),
    )


def evaluate_dim(
    model, X_state_tr, X_action_tr, y_tr,
    X_state_te, X_action_te, y_te,
    device, criterion, n_context_eval: int
):
    """CAQ 评估：从训练池取前 n_context_eval 样本做 ICL 上下文，预测测试集。"""
    n_ctx = min(n_context_eval, len(X_state_tr))
    # eval() 禁用 Dropout/BatchNorm 的训练行为，确保评估结果确定且可复现
    model.eval()
    with torch.no_grad():
        eval_all = np.concatenate([
            np.concatenate([X_state_tr[:n_ctx], X_action_tr[:n_ctx]], axis=1),
            np.concatenate([X_state_te, X_action_te], axis=1),
        ])
        eval_a = np.concatenate([X_action_tr[:n_ctx], X_action_te])
        eval_y = np.concatenate([y_tr[:n_ctx], y_te])
        Xs = torch.from_numpy(eval_all).float().unsqueeze(0).to(device)
        Xa = torch.from_numpy(eval_a).float().unsqueeze(0).to(device)
        yb = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
        # col_embedder 在 eval 模式下可能输出 float16，显式转 float32 确保 dtype 一致
        col_emb = model.col_embedder(Xs, y_train=yb[:, :n_ctx]).float()
        reps = model.row_interactor(col_emb, Xa)
        out = model.icl_predictor(reps, y_train=yb[:, :n_ctx])
        return criterion(out.mean(dim=-1).squeeze(0), yb[0, n_ctx:]).item()


def format_mse(val: float) -> str:
    """≥1e-5 用浮点，<1e-5 用科学计数法，避免 near-zero 值显示为 0.000000。"""
    if val == 0 or val >= 1e-5:
        return f"{val:.6f}"
    return f"{val:.6e}"


# ═══════════════════════════════════════════════════════════════════
# 检查点
# ═══════════════════════════════════════════════════════════════════

def get_trainable_state_dict(model: nn.Module) -> dict:
    """只提取可训练参数的 state_dict，大幅减小检查点体积（110MB → ~2MB）。"""
    return {k: v for k, v in model.state_dict().items()
            if any(k.startswith(prefix) for prefix in
                   ["row_interactor.action_encoder", "row_interactor.causal_block"])}


def save_checkpoint(
    save_dir: Path, dim: int, model: nn.Module,
    optimizer, scheduler, epoch: int, best_mse: float,
    train_args: dict, is_best: bool = False,
):
    """保存训练检查点（仅保存可训练参数，~2MB/个）。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "best_mse": best_mse,
        "dim": dim,
        "model_state": get_trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_args": train_args,  # 恢复时校验超参数一致性
    }

    # epoch 补零到 4 位，确保字典序 = 数值序，否则如 epoch100 字符串排序在 epoch20 前面，会被清理
    ckpt_path = save_dir / f"ckpt_dim{dim}_epoch{epoch:04d}.pt"
    torch.save(state, ckpt_path)

    if is_best:
        best_path = save_dir / f"best_dim{dim}.pt"
        torch.save(state, best_path)

    # 清理旧检查点（保留最新 3 个）
    ckpts = sorted(save_dir.glob(f"ckpt_dim{dim}_epoch*.pt"))
    for old in ckpts[:-3]:
        old.unlink()


def save_progress(save_dir: Path, completed_dims: list,
                  mse_caq: dict, mse_base, elapsed: float = 0.0,
                  current_dim: int = None, current_epoch: int = None, current_best_mse: float = None):
    """保存训练进度。mse_base 每次传入，内部统一转为 string-keyed dict。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    progress_path = save_dir / "progress.json"

    # mse_base: ndarray → {str: float}
    _mse_base = {str(i): float(v) for i, v in enumerate(mse_base)}

    progress = {
        "completed_dims": completed_dims,
        "mse_caq": {str(k): float(v) for k, v in mse_caq.items()},
        "mse_base": _mse_base,
        "elapsed": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if current_dim is not None:
        progress["current_dim"] = current_dim
        progress["current_epoch"] = current_epoch
        progress["current_best_mse"] = current_best_mse

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress(save_dir: Path) -> Optional[dict]:
    """加载进度文件。"""
    path = save_dir / "progress.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════

def train_single_dim(
    model, optimizer, scheduler, criterion,
    X_state_tr, X_action_tr, y_tr,
    X_state_te, X_action_te, y_te,
    device, dim: int,
    save_dir: Optional[Path],
    train_args: dict,
    n_context: int = 1024, n_query: int = 512,
    n_steps_per_epoch: int = 50, n_epochs: int = 300,
    n_context_eval: int = 100000,
    start_epoch: int = 0, best_mse: float = float("inf"),
    early_stopping: bool = False, patience_evals: int = 4, min_epochs: int = 100,
    completed_dims: list = None, mse_caq: dict = None,
    mse_base = None, accumulated_elapsed: float = 0.0, t0: float = None,
):
    """针对单个输出维度训练 CAQ 模块。

    检查点：每 20 epoch 评估时保存 rolling 快照（保留最新 3 个）+ 创新低时另存 best。
    早停（可选）：开启后连续 patience_evals 轮评估无改善且已过 min_epochs 则提前停止。
    进度文件：每 20 epoch 同步更新 progress.json（含累计耗时），断电后最多丢失 20 epoch 计时。
    """
    evals_no_improve = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_losses = []

        for _ in range(n_steps_per_epoch):
            Xs, Xa, yb = create_icl_batch(X_state_tr, X_action_tr, y_tr, n_context, n_query, device)
            y_train = yb[:, :n_context]

            optimizer.zero_grad()
            col_emb = model.col_embedder(Xs, y_train=y_train)
            reps = model.row_interactor(col_emb, Xa)
            out = model.icl_predictor(reps, y_train=y_train)
            loss = criterion(out.mean(dim=-1).squeeze(0), yb[0, n_context:])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        # 每 20 epoch 评估一次
        if (epoch + 1) % 20 == 0 or epoch == 0:
            test_mse = evaluate_dim(
                model, X_state_tr, X_action_tr, y_tr, X_state_te, X_action_te, y_te,
                device, criterion, n_context_eval,
            )
            is_best = test_mse < best_mse
            if is_best:
                best_mse = test_mse
                evals_no_improve = 0
            else:
                evals_no_improve += 1

            print(
                f"  epoch {epoch+1:4d}/{n_epochs} | "
                f"loss={avg_loss:.6f} | mse={format_mse(test_mse)} | "
                f"best={format_mse(best_mse)} | lr={scheduler.get_last_lr()[0]:.2e}"
                f"{' 🔥' if is_best else ''}"
            )

            # 保存检查点：每次评估都存 rolling（保留最新 3 个），创新低时另存 best
            if save_dir:
                save_checkpoint(
                    save_dir, dim, model, optimizer, scheduler,
                    epoch + 1, best_mse, train_args, is_best,
                )

            # 每 20 epoch 更新进度文件（含累计耗时），断电等异常中断时最多丢失 20 epoch 的计时
            if save_dir and t0 is not None:
                save_progress(save_dir, completed_dims, mse_caq, mse_base,
                             elapsed=accumulated_elapsed + (time.time() - t0),
                             current_dim=dim, current_epoch=epoch + 1,
                             current_best_mse=best_mse)

            # 早停（可选）
            if early_stopping and epoch + 1 >= min_epochs and evals_no_improve >= patience_evals:
                print(f"  🛑 早停 (连续 {evals_no_improve} 轮评估无改善)")
                break

    return best_mse


def evaluate_baseline_full(X_tr_pool, X_te, y_tr_pool, y_te, device="cuda", n_context_eval=100000):
    """Baseline: 12 x TabICLRegressor，使用前 n_context_eval 个训练样本作为 ICL 上下文。"""
    n_ctx = min(n_context_eval, len(X_tr_pool))
    X_ctx = X_tr_pool[:n_ctx]
    y_ctx = y_tr_pool[:n_ctx]
    mse = np.zeros(OUTPUT_DIMS)
    t_start = time.time()
    for d in range(OUTPUT_DIMS):
        reg = TabICLRegressor(n_estimators=4, model_path=CKPT, device=device, random_state=42)
        reg.fit(X_ctx, y_ctx[:, d])
        pred = reg.predict(X_te)
        mse[d] = np.mean((pred - y_te[:, d]) ** 2)
    elapsed = time.time() - t_start
    return mse, elapsed


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CAQ-TabICL 大规模训练")
    parser.add_argument("--epoch_idx", type=int, default=280)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_context", type=int, default=1024)
    parser.add_argument("--n_query", type=int, default=512)
    parser.add_argument("--n_context_eval", type=int, default=100000, help="Baseline和CAQ评估的ICL上下文样本数")
    parser.add_argument("--n_test", type=int, default=10000, help="Baseline和CAQ的统一测试集样本数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dims", type=str, default="all")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--early_stopping", action="store_true", default=False, help="启用早停")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="从指定目录恢复训练")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) if args.save_dir else None

    # 解析维度
    if args.dims == "all":
        dims = list(range(OUTPUT_DIMS))
    elif args.dims == "reward_only":
        dims = [0]
    else:
        dims = [int(x.strip()) for x in args.dims.split(",")]

    # ── 数据加载 ──
    print("📦 加载 Hopper-v5 数据...")
    X_state, X_action, y = load_all_hopper(args.epoch_idx)
    print(f"  样本: {len(X_state)}, state={X_state.shape[1]}d, action={X_action.shape[1]}d, target={y.shape[1]}d")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X_state)) # 随机打乱样本索引
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    # 划分训练集和测试集
    n_train = int(len(X_state) * 0.8)
    n_test = min(args.n_test, len(X_state) - n_train)
    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:n_train+n_test]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:n_train+n_test]
    y_tr, y_te = y[:n_train], y[n_train:n_train+n_test]
    print(f"  训练: {len(X_state_tr):,} | 测试: {len(X_state_te):,}")
    print(f"  y 范围: reward [{y[:,0].min():.3f}, {y[:,0].max():.3f}]")

    # ── Baseline ──
    print(f"\n{'='*60}")
    print(f"  📊 Baseline 评估中...")
    X_tr = np.concatenate([X_state_tr, X_action_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([X_state_te, X_action_te], axis=1).astype(np.float32)
    mse_base, t_base = evaluate_baseline_full(
        X_tr, X_te, y_tr, y_te, device=str(device), n_context_eval=args.n_context_eval
    )
    print(f"  ✅ Baseline (context={min(args.n_context_eval, len(y_tr)):,}, test={len(y_te):,}) 用时 {t_base:.1f}s")
    for d in range(OUTPUT_DIMS):
        print(f"    [{d}] {DIM_LABELS[d]:<12} MSE={format_mse(mse_base[d])}")

    # ── 恢复/初始化进度 ──
    mse_caq = {d: 0.0 for d in dims}
    completed_dims = []
    accumulated_elapsed = 0.0  # 之前的会话已累计的训练耗时（秒）

    if args.resume:
        resume_dir = Path(args.resume)
        progress = load_progress(resume_dir)
        if progress:
            completed_dims = progress.get("completed_dims", [])
            for k, v in progress.get("mse_caq", {}).items():
                mse_caq[int(k)] = v
            accumulated_elapsed = progress.get("elapsed", 0.0)
            print(f"\n📂 从 {resume_dir} 恢复: 已完成 {len(completed_dims)} 维度: {completed_dims}")
            if accumulated_elapsed > 0:
                print(f"   累计耗时: {accumulated_elapsed/3600:.1f}h")
        else:
            print(f"\n⚠️  未找到进度文件，从头开始训练")

    # ── 训练 ──
    t0 = time.time()

    # 当前训练的超参数（保存到检查点，恢复时校验）
    train_args = {
        "lr": args.lr, "n_epochs": args.n_epochs, "n_context": args.n_context, "n_query": args.n_query,
        "n_steps": args.n_steps, "weight_decay": 1e-5,
    }

    for d in dims:
        if d in completed_dims:
            print(f"\n  ⏭️  [{d}] {DIM_LABELS[d]}: 已完成 (MSE={format_mse(mse_caq[d])})，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"  🧬 [{d}] {DIM_LABELS[d]}: 训练 CAQ 模块")
        print(f"    配置: {args.n_steps} steps/epoch × {args.n_epochs} epochs")
        if save_dir:
            print(f"    检查点: {save_dir}/best_dim{d}.pt")

        try:
            # 加载模型
            caq = load_caq_model(
                CKPT, action_dim=ACTION_DIM, device=device,
                freeze_col_embedder=True, freeze_native_blocks=True, freeze_icl=True,
            )
            optimizer = AdamW(caq.trainable_parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr*0.01)
            criterion = nn.MSELoss()

            start_epoch, best_mse = 0, float("inf")

            # 从检查点恢复
            best_path = save_dir / f"best_dim{d}.pt" if save_dir else None
            if best_path and best_path.exists():
                ckpt = torch.load(best_path, map_location=device)
                # 校验超参数
                saved_args = ckpt.get("train_args", {})
                for k in ["lr", "n_epochs"]:
                    if saved_args.get(k) != train_args[k]:
                        print(f"    ⚠️  {k} 不匹配: 保存={saved_args.get(k)}, 当前={train_args[k]}")
                # 加载权重（strict=False 因为只保存了可训练参数）
                caq.load_state_dict(ckpt["model_state"], strict=False)
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = ckpt["epoch"]
                best_mse = ckpt.get("best_mse", float("inf"))
                print(f"    📂 恢复: epoch={start_epoch}, best_mse={format_mse(best_mse)}")

            best_mse = train_single_dim(
                caq, optimizer, scheduler, criterion,
                X_state_tr, X_action_tr, y_tr[:, d],
                X_state_te, X_action_te, y_te[:, d],
                device, dim=d, save_dir=save_dir, train_args=train_args,
                n_context=args.n_context, n_query=args.n_query,
                n_steps_per_epoch=args.n_steps, n_epochs=args.n_epochs,
                n_context_eval=args.n_context_eval,
                start_epoch=start_epoch, best_mse=best_mse,
                early_stopping=args.early_stopping,
                completed_dims=completed_dims, mse_caq=mse_caq,
                mse_base=mse_base, accumulated_elapsed=accumulated_elapsed, t0=t0,
            )

            mse_caq[d] = best_mse
            completed_dims.append(d)

        except Exception as e:
            print(f"  ❌ [{d}] {DIM_LABELS[d]} 训练异常: {e}")
            import traceback; traceback.print_exc()
        finally:
            # dim 完成（或异常）后保存进度：更新 completed_dims 和 mse_caq
            if save_dir:
                save_progress(save_dir, completed_dims, mse_caq, mse_base=mse_base,
                             elapsed=accumulated_elapsed + (time.time() - t0))

        print(f"  ✅ [{d}] {DIM_LABELS[d]}: best_mse={format_mse(mse_caq[d])}")

    # ── 最终结果 ──
    elapsed = accumulated_elapsed + (time.time() - t0)
    print(f"\n{'='*70}")
    print(f"  📊 最终对比 (训练耗时: {elapsed/3600:.1f}h)")
    print(f"  {'Dim':>4} {'Label':<12} {'Baseline':>12} {'CAQ':>12} {'Ratio':>10} {'Winner':>8}")
    print(f"  {'-'*60}")

    for d in range(OUTPUT_DIMS):
        label = DIM_LABELS[d]
        b = mse_base[d]
        c = mse_caq.get(d, 0)
        if d in dims:
            ratio = c / b if b > 0 else float("inf")
            winner = "✅ CAQ" if ratio < 1 else ("≈平" if ratio < 1.05 else "—")
            note = " (≈0)" if b < 1e-5 else ""
            print(f"  {d:>4} {label:<12} {format_mse(b):>12} {format_mse(c):>12} {ratio:>10.2f}x{note:>5} {winner:>8}")
        else:
            print(f"  {d:>4} {label:<12} {format_mse(b):>12} {'—':>12} {'—':>10}")

    trained_dims = [d for d in dims if d in completed_dims]
    if trained_dims:
        overall_base = np.mean([mse_base[d] for d in trained_dims])
        overall_caq = np.mean([mse_caq[d] for d in trained_dims])
        reward_base, reward_caq = mse_base[0], mse_caq.get(0, 0)
        delta_base_avg = np.mean(mse_base[1:])
        delta_caq_avg = np.mean([mse_caq.get(d, 0) for d in range(1, OUTPUT_DIMS)])

        print(f"  {'-'*60}")
        print(f"  {'REWARD':>17} {format_mse(reward_base):>12} {format_mse(reward_caq):>12} {reward_caq/reward_base:>10.2f}x")
        print(f"  {'DELTA(avg)':>17} {format_mse(delta_base_avg):>12} {format_mse(delta_caq_avg):>12} {delta_caq_avg/delta_base_avg:>10.2f}x")
        print(f"  {'OVERALL':>17} {format_mse(overall_base):>12} {format_mse(overall_caq):>12} {overall_caq/overall_base:>10.2f}x")

    print(f"\n  检查点保存在: {save_dir}/")


if __name__ == "__main__":
    main()
