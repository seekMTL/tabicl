"""train_large4_actionfeat.py 的冒烟测试。

验证 action 作为特征的新管线：
1. preprocess_data: state+action 拼接后一起 z-score（action 均值≈0 方差≈1）
2. create_icl_batch_shared: Xs_state 为 H+8 列表格（Hopper 11+8=19）
3. RL 数据全链路前向 + 梯度回传（特征表含 action 列）
4. 评估路径组装（evaluate_env 的信息流）与 eval 模式前向
5. SCM 路径（X_action=None）保持可用

用法:
  python smoke_test_actionfeat.py [--device cpu]
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
    load_data, preprocess_data, pad_action,
    create_icl_batch_shared, MAX_ACTION_DIM, ENV_CONFIGS,
)
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"


def test_preprocess():
    print("\n[1/5] preprocess_data: state+action 一起 z-score...")
    cfg = ENV_CONFIGS["Hopper"]
    X_state, X_action, y = load_data(cfg["dir"], cfg["epoch"])
    n = 5000
    X_state, X_action, y = X_state[:n], X_action[:n], y[:n]
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    n_train = int(n * 0.8)
    (Xs_tr_pp, Xs_te_pp, Xa_tr_pp, Xa_te_pp,
     y_tr_pp, y_te_pp, y_stats) = preprocess_data(
        X_state[:n_train], X_state[n_train:], X_action[:n_train], X_action[n_train:],
        y[:n_train], y[n_train:], norm_method="none", seed=42,
    )
    print(f"  state_pp: {Xs_tr_pp.shape}（应 (4000, 11)）")
    print(f"  action_pp: {Xa_tr_pp.shape}（应 (4000, 3)，已 z-score）")
    assert Xs_tr_pp.shape[1] == 11 and Xa_tr_pp.shape[1] == 3
    am = Xa_tr_pp.mean(axis=0)
    ast = Xa_tr_pp.std(axis=0)
    print(f"  action z-score 检查: mean={am}（应≈0）, std={ast}（应≈1）")
    assert np.allclose(am, 0, atol=0.05), f"action z-score mean 异常: {am}"
    assert np.allclose(ast, 1, atol=0.15), f"action z-score std 异常: {ast}"
    print(f"  ✅ 预处理正常")
    return Xs_tr_pp, Xa_tr_pp, y_tr_pp


def test_create_batch(Xs_tr_pp, Xa_tr_pp, y_tr_pp, device):
    print("\n[2/5] create_icl_batch_shared: 特征表含 action 列...")
    Xs_b, Xa_b, yb = create_icl_batch_shared(Xs_tr_pp, Xa_tr_pp, y_tr_pp, 64, 32, device)
    H = 11
    print(f"  Xs_state: {tuple(Xs_b.shape)}（应 (1, 96, {H + MAX_ACTION_DIM})）")
    print(f"  Xs_action: {tuple(Xa_b.shape)}（应 (1, 96, 8)）")
    print(f"  y: {tuple(yb.shape)}（应 (12, 96)）")
    assert Xs_b.shape == (1, 96, H + MAX_ACTION_DIM), f"特征表形状错误: {Xs_b.shape}"
    assert Xa_b.shape == (1, 96, MAX_ACTION_DIM)
    assert yb.shape == (1 * y_tr_pp.shape[1], 96)
    print(f"  ✅ batch 组装正常")
    return Xs_b, Xa_b, yb


def test_rl_forward(model, Xs_b, Xa_b, yb, device, criterion):
    print("\n[3/5] RL 前向 + 梯度回传（action 作为特征）...")
    model.train()
    n_context = 64
    y_train_all = yb[:, :n_context]
    dummy_y = y_train_all[0:1]
    reprs = model._forward_embeddings(Xs_b, Xa_b, y_train=dummy_y)
    print(f"  representations: {tuple(reprs.shape)}（应 (1, 96, 512)）")

    out = model.icl_predictor(reprs.repeat(yb.shape[0], 1, 1), y_train=y_train_all)
    print(f"  pred: {tuple(out.shape)}（应 (12, 32, 999)）")

    loss = criterion(out.mean(dim=-1), yb[:, n_context:])
    print(f"  RL loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "loss 非有限"

    loss.backward()
    grad_ok = {
        "col_embedder": any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.col_embedder.parameters() if p.requires_grad),
        "row_interactor": any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.row_interactor.parameters() if p.requires_grad),
        "icl_predictor (冻结)": all(p.grad is None
                                    for p in model.icl_predictor.parameters()),
    }
    for name, ok in grad_ok.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    assert all(grad_ok.values()), "梯度检查失败"
    print(f"  ✅ RL 前向 + 梯度正常")


def test_eval_path(model, Xs_tr_pp, Xa_tr_pp, y_tr_pp, device):
    print("\n[4/5] 评估路径组装 + eval 前向...")
    model.eval()
    n_ctx, n_te = 256, 128
    Xa_ctx = pad_action(Xa_tr_pp[:n_ctx])
    Xa_te = pad_action(Xa_tr_pp[n_ctx:n_ctx + n_te])
    eval_state = np.concatenate([
        np.concatenate([Xs_tr_pp[:n_ctx], Xa_ctx], axis=1),
        np.concatenate([Xs_tr_pp[n_ctx:n_ctx + n_te], Xa_te], axis=1),
    ])
    eval_action = np.concatenate([Xa_ctx, Xa_te])
    eval_y = y_tr_pp[:n_ctx + n_te, 0]
    X_state_t = torch.from_numpy(eval_state).float().unsqueeze(0).to(device)
    X_action_t = torch.from_numpy(eval_action).float().unsqueeze(0).to(device)
    yb = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(X_state_t, X_action_t, y_train=yb[:, :n_ctx])
    print(f"  eval pred: {tuple(out.shape)}（应 (1, 128, 999)）")
    assert out.shape == (1, n_te, 999)
    assert torch.isfinite(out).all(), "eval 输出非有限"
    print(f"  ✅ 评估路径正常")


def test_scm_path(model, device, criterion):
    print("\n[5/5] SCM 路径（X_action=None）...")
    from train_large4_actionfeat import SCMReplaySampler
    model.train()
    sampler = SCMReplaySampler(batch_size=2, min_seq_len=64, max_seq_len=128, n_jobs=1)
    X, y, d, train_size = sampler.get_batch()
    X, y = X.to(device), y.to(device)
    pred = model(X, y_train=y[:, :train_size])
    print(f"  SCM pred: {tuple(pred.shape)}（应 (2, 128-train_size, 999)）")
    assert torch.isfinite(pred).all(), "SCM 输出非有限"
    loss = criterion(pred.mean(dim=-1), y[:, train_size:])
    assert torch.isfinite(loss)
    print(f"  SCM loss = {loss.item():.6f} → ✅ SCM 路径正常")


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
    Xs_tr_pp, Xa_tr_pp, y_tr_pp = test_preprocess()
    Xs_b, Xa_b, yb = test_create_batch(Xs_tr_pp, Xa_tr_pp, y_tr_pp, device)
    test_rl_forward(model, Xs_b, Xa_b, yb, device, criterion)
    test_eval_path(model, Xs_tr_pp, Xa_tr_pp, y_tr_pp, device)
    test_scm_path(model, device, criterion)

    print("\n" + "=" * 60)
    print("✅ 全部冒烟测试通过（action 作为特征管线）")


if __name__ == "__main__":
    main()
