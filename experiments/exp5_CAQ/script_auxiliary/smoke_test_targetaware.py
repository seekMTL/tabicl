"""train_large5_targetaware.py 的冒烟测试（target-aware 路径）。

验证：
1. target_aware=True 模型能正常构造（y_encoder 路径存在）
2. rl_train_step(TA-on)：逐维 y 条件的批处理前向 + 梯度回传
3. SCM 步（TA-on，X_action=None）与预训练同构
4. 评估路径（evaluate_env，TA-on）正常

用法:
  python smoke_test_targetaware.py [--device cpu]
"""

import sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exp5_CAQ 目录

from train_large5_targetaware import (
    SCMReplaySampler, load_data, preprocess_data, ENV_CONFIGS,
    rl_train_step, scm_train_step, evaluate_env, MAX_ACTION_DIM,
)
from tabicl._model.caq_model import load_caq_model

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"


def build_mini_env_data():
    """构造小样本 env_data（Hopper）。"""
    cfg = ENV_CONFIGS["Hopper"]
    X_state, X_action, y = load_data(cfg["dir"], cfg["epoch"])
    n = 3000
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_state))[:n]
    X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

    n_train = int(n * 0.8)
    X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:]
    X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    (Xs_tr_pp, Xs_te_pp, Xa_tr_pp, Xa_te_pp,
     y_tr_pp, y_te_pp, y_stats) = preprocess_data(
        X_state_tr, X_state_te, X_action_tr, X_action_te,
        y_tr, y_te, norm_method="none", seed=42,
    )
    return {
        "X_state_tr_pp": Xs_tr_pp,
        "X_state_te_pp": Xs_te_pp,
        "X_action_tr": Xa_tr_pp,
        "X_action_te": Xa_te_pp,
        "y_tr_pp": y_tr_pp,
        "y_te_pp": y_te_pp,
        "y_tr_raw": y_tr,
        "y_te_raw": y_te,
        "y_stats": y_stats,
        "output_dims": y.shape[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")
    print("[0/4] 加载 target_aware=True 模型...")
    model = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=False, freeze_native_blocks=False, freeze_icl=True,
        target_aware=True,
    )
    assert model.col_embedder.target_aware, "target_aware 未生效"
    assert model.col_embedder.y_encoder is not None, "y_encoder 不存在（checkpoint 配置无 target-aware？）"
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  target_aware=True ✓, y_encoder 存在 ✓, 可训练参数 {n_t:,}")

    criterion = nn.MSELoss()
    env_data = build_mini_env_data()
    print(f"  迷你数据集: {env_data['X_state_tr_pp'].shape[0]} 训练样本, "
          f"{env_data['output_dims']} 输出维")

    print("\n[1/4] rl_train_step(TA-on) 前向 + 梯度...")
    model.train()  # 与训练循环一致；col_embedder 继承 pretrained.eval()，需显式恢复
    loss = rl_train_step(model, env_data, criterion, device,
                         n_context=64, n_query=32, target_aware=True)
    print(f"  loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "loss 非有限"

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-5)
    optimizer.zero_grad()
    loss.backward()
    grad_ok = {
        "col_embedder": any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.col_embedder.parameters() if p.requires_grad),
        "row_interactor": any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.row_interactor.parameters() if p.requires_grad),
        "y_encoder (训练)": model.col_embedder.y_encoder.weight.grad is not None
                            and model.col_embedder.y_encoder.weight.grad.abs().sum() > 0,
        "icl_predictor (冻结)": all(p.grad is None
                                    for p in model.icl_predictor.parameters()),
    }
    for name, ok in grad_ok.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    assert all(grad_ok.values()), "梯度检查失败"
    print("  ✅ TA-on 训练步正常")

    print("\n[1.5/4] rl_train_step(TA-on, 分块模式 chunk=3) 梯度累积...")
    optimizer.zero_grad()
    loss = rl_train_step(model, env_data, criterion, device,
                         n_context=64, n_query=32, target_aware=True,
                         ta_dim_chunk=3, do_backward=True)
    print(f"  loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "分块 loss 非有限"
    grad_ok = {
        "col_embedder": any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.col_embedder.parameters() if p.requires_grad),
        "row_interactor": any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.row_interactor.parameters() if p.requires_grad),
    }
    for name, ok in grad_ok.items():
        print(f"  {name}（分块累积）: {'✅' if ok else '❌'}")
    assert all(grad_ok.values()), "分块梯度累积失败"
    print("  ✅ 分块模式正常（图随块释放，梯度累积等效全维）")

    print("\n[2/4] SCM 步（TA-on, X_action=None）...")
    model.train()
    sampler = SCMReplaySampler(batch_size=2, min_seq_len=64, max_seq_len=128, n_jobs=1)
    loss = scm_train_step(model, sampler, criterion, device)
    print(f"  SCM loss = {loss.item():.6f}")
    assert torch.isfinite(loss), "SCM loss 非有限"
    print("  ✅ SCM 步正常（TA-on 下与预训练同构）")

    print("\n[3/4] 评估路径（evaluate_env, TA-on）...")
    mse_dims = evaluate_env(model, env_data, device, criterion, n_context_eval=200)
    print(f"  avg MSE = {np.mean(list(mse_dims.values())):.6f}")
    assert all(np.isfinite(v) for v in mse_dims.values())
    print("  ✅ 评估路径正常")

    print("\n[4/4] 与 TA-off 输出差异检查（y 条件嵌入确实改变了表示）...")
    model_off = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=True, freeze_native_blocks=True, freeze_icl=True,
        target_aware=False,
    )
    model_off.eval()  # eval 路径（row_interactor 内部有 fp16→fp32 转换）
    Xs = torch.randn(1, 32, 11, device=device)
    Xa = torch.randn(1, 32, MAX_ACTION_DIM, device=device)
    yt = torch.randn(1, 16, device=device)
    with torch.no_grad():
        r_taoff = model_off._forward_embeddings(Xs, Xa, y_train=yt)
        model_off.col_embedder.target_aware = True
        r_taon = model_off._forward_embeddings(Xs, Xa, y_train=yt)
    diff = (r_taoff - r_taon).abs().mean().item()
    print(f"  TA-off vs TA-on 表示平均差异 = {diff:.6f}（应 > 0）")
    assert diff > 0, "target-aware 未改变表示，实现有误"
    print("  ✅ target-aware 确实改变了嵌入")

    print("\n" + "=" * 60)
    print("✅ 全部冒烟测试通过（target-aware 训练管线）")


if __name__ == "__main__":
    main()
