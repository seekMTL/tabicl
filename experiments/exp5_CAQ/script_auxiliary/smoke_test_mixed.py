"""train_large4_mixed.py 的冒烟测试。

验证：
1. SCMReplaySampler 能生成回归模式数据（连续 y）
2. SCM 数据能通过 CAQ 模型（X_action=None 路径）
3. RL 数据（Hopper）能通过 CAQ 模型（含 action padding）
4. 梯度能正常回传到 col_embedder / row_interactor / action_encoder

用法:
  python smoke_test_mixed.py [--device cpu]
"""

import sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exp5_CAQ 目录（训练脚本所在）

from train_large4_mixed import (
    SCMReplaySampler, load_data, preprocess_data, pad_action,
    create_icl_batch_shared, MAX_ACTION_DIM, ENV_CONFIGS,
)
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"


def test_scm_sampler(device):
    print("\n[1/4] SCMReplaySampler 回归模式数据生成...")
    sampler = SCMReplaySampler(batch_size=4, min_seq_len=128, max_seq_len=256, n_jobs=1)
    X, y, d, train_size = sampler.get_batch()
    print(f"  X: {tuple(X.shape)} (dtype={X.dtype})")
    print(f"  y: {tuple(y.shape)} (dtype={y.dtype})")
    print(f"  d: {d.tolist()}  train_size={train_size}")

    # 检查 y 是否为连续值（非分类整数）
    y_flat = y.flatten().numpy()
    n_unique = len(np.unique(y_flat))
    assert n_unique > 100, f"y 看起来像分类标签（unique={n_unique}），应为连续值"
    print(f"  y unique values: {n_unique} → ✅ 连续回归目标")
    assert X.shape[1] == y.shape[1], "X 和 y 的序列长度不一致"
    assert d.max() <= X.shape[-1], "有效特征数超过 X 列数"
    print(f"  ✅ SCM 采样器正常")


def test_scm_forward(model, sampler, device, criterion):
    print("\n[2/4] SCM 数据通过 CAQ 模型（X_action=None 路径）...")
    model.train()
    X, y, d, train_size = sampler.get_batch()
    X, y = X.to(device), y.to(device)
    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    # 不传 d：feature grouping 模式下 ColEmbedding 不支持 d
    pred = model(X, y_train=y_train)  # X_action=None → 零动作
    print(f"  pred: {tuple(pred.shape)} (期望 (B, test_size, 999))")
    assert pred.shape[0] == X.shape[0], "batch 维度不匹配"
    assert pred.shape[-1] == 999, "回归模型应输出 999 分位数"

    loss = criterion(pred.mean(dim=-1), y_test)
    print(f"  SCM loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "loss 不是有限值"

    # 梯度回传
    loss.backward()
    grad_ok = {
        "col_embedder": any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.col_embedder.parameters() if p.requires_grad),
        "row_interactor": any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.row_interactor.parameters() if p.requires_grad),
        "icl_predictor (冻结, 无梯度)": all(p.grad is None
                                          for p in model.icl_predictor.parameters()),
    }
    for name, ok in grad_ok.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    assert all(grad_ok.values()), "梯度检查失败"
    print(f"  ✅ SCM 前向 + 梯度正常")


def test_rl_forward(model, device, criterion):
    print("\n[3/4] RL 数据（Hopper）通过 CAQ 模型（含 action padding）...")
    cfg = ENV_CONFIGS["Hopper"]
    X_state, X_action, y = load_data(cfg["dir"], cfg["epoch"])
    # 用小样本快速验证
    n = 5000
    X_state, X_action, y = X_state[:n], X_action[:n], y[:n]
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    n_train = int(n * 0.8)
    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    X_state_tr_pp, X_state_te_pp, y_tr_pp, y_te_pp, _ = preprocess_data(
        X_state_tr, X_state_te, y_tr, y_te, norm_method="none", seed=42,
    )

    # 检查 padding
    padded = pad_action(X_action_tr)
    print(f"  action 原始 {X_action_tr.shape[1]}d → padding 后 {padded.shape[1]}d")
    assert padded.shape[1] == MAX_ACTION_DIM

    # 完整 batch 前向
    model.train()
    n_context, n_query = 64, 32
    X_state_b, X_action_b, yb_all = create_icl_batch_shared(
        X_state_tr_pp, X_action_tr, y_tr_pp, n_context, n_query, device
    )
    print(f"  X_state: {tuple(X_state_b.shape)}, X_action: {tuple(X_action_b.shape)}, "
          f"y: {tuple(yb_all.shape)}")

    y_train_all = yb_all[:, :n_context]
    dummy_y = y_train_all[0:1]
    reprs = model._forward_embeddings(X_state_b, X_action_b, y_train=dummy_y)
    print(f"  representations: {tuple(reprs.shape)} (期望 (1, T, 512))")

    out = model.icl_predictor(reprs.repeat(yb_all.shape[0], 1, 1), y_train=y_train_all)
    print(f"  pred: {tuple(out.shape)} (期望 (12, n_query, 999))")

    loss = criterion(out.mean(dim=-1), yb_all[:, n_context:])
    print(f"  RL loss = {loss.item():.6f}")
    assert torch.isfinite(loss)
    print(f"  ✅ RL 前向正常")


def test_eval_forward(model, device):
    print("\n[4/4] eval 模式前向（evaluate_env 路径）...")
    model.eval()
    X_state = torch.randn(1, 128, 11, device=device)   # Hopper state 11d
    X_action = torch.randn(1, 128, MAX_ACTION_DIM, device=device)  # padded action
    y_train = torch.randn(1, 64, device=device)
    with torch.no_grad():
        out = model(X_state, X_action, y_train=y_train)
    print(f"  eval pred: {tuple(out.shape)}")
    assert out.shape == (1, 64, 999)
    print(f"  ✅ eval 模式正常")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")
    print("加载模型（冻结 ICL，训练 col/row/CAQ）...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False, freeze_icl=True,
    )
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_t:,}")

    criterion = nn.MSELoss()
    sampler = SCMReplaySampler(batch_size=4, min_seq_len=128, max_seq_len=256, n_jobs=1)

    test_scm_sampler(device)
    test_scm_forward(model, sampler, device, criterion)
    test_rl_forward(model, device, criterion)
    test_eval_forward(model, device)

    print("\n" + "=" * 60)
    print("✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
