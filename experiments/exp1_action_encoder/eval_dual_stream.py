#!/usr/bin/env python3
"""
实验 1：双流分离 + ActionEncoder 评估脚本

与实验 0 对比，验证 ActionEncoder 的价值。
- 模式: use_action_encoder=True, state_dim=11, action_dim=3
- ActionEncoder + fusion_proj 随机初始化，需要 fine-tune

用法:
  python experiments/exp1_action_encoder/eval_dual_stream.py
"""

from __future__ import annotations

import csv, logging, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

BASELINE_CSV = (
    "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3"
    "/distribution_tabicl2_mse/mse_comparison_results.csv"
)
SNAPSHOT_DIR = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
CKPT_PATH = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp1_action_encoder" / "outputs"


# ---- 数据加载 ----
def load_snapshot(npz_path: Path):
    data = np.load(npz_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    delta = np.asarray(data["next_states"], dtype=np.float32) - states
    inputs = np.concatenate([states, actions], axis=-1)
    labels = np.concatenate([rewards, delta], axis=-1)
    return inputs, labels, {
        "epoch": int(data["epoch"][0]),
        "num_samples": int(data["num_samples"][0]),
        "reward_mean": float(data["reward_mean"][0]),
        "reward_std": float(data["reward_std"][0]),
    }


def compute_mse(pred, target, reward_size=1):
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return (float(np.mean(diff ** 2)),
            float(np.mean(diff[:, :reward_size] ** 2)),
            float(np.mean(diff[:, reward_size:] ** 2)))


def load_baseline_results(csv_path: str) -> dict:
    baseline = {}
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            baseline[ep] = {k: float(row[k]) for k in row
                            if k not in ('epoch', 'num_samples', 'tabicl_train_size', 'tabicl_val_size')}
            baseline[ep]['num_samples'] = int(row['num_samples'])
    return baseline


# ---- 模型加载 ----
def load_dual_stream_model(ckpt_path: str, num_outputs: int, device: torch.device):
    """加载双流多输出模型。"""
    from tabicl._model.tabicl import TabICL

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    config.update({
        "num_outputs": num_outputs,
        "use_action_encoder": True,
        "state_dim": 11,
        "action_dim": 3,
        "action_encoder_mode": "mlp",
    })
    model = TabICL(**config)
    model.load_state_dict(state_dict, strict=False)

    # 复制预训练权重到所有 head
    y_w = state_dict["icl_predictor.y_encoder.weight"]
    y_b = state_dict["icl_predictor.y_encoder.bias"]
    for j in range(num_outputs):
        model.icl_predictor.y_encoders[j].weight.data.copy_(y_w)
        model.icl_predictor.y_encoders[j].bias.data.copy_(y_b)
    for sfx in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        src = state_dict[f"icl_predictor.decoder.{sfx}"]
        for j in range(num_outputs):
            dict(model.named_parameters())[f"icl_predictor.decoders.{j}.{sfx}"].data.copy_(src)

    # ActionEncoder + fusion_proj 保持随机初始化（需要 fine-tune）
    log.info(f"ActionEncoder params: {sum(p.numel() for p in model.action_encoder.parameters()):,}")
    log.info(f"FusionProj params: {sum(p.numel() for p in model.fusion_proj.parameters()):,}")

    model.to(device)
    return model


# ---- 评估 ----
def evaluate_snapshot(snapshot_path: Path, model: torch.nn.Module, device: torch.device,
                      rng: np.random.Generator, holdout_ratio: float = 0.2,
                      max_val_samples: int = 10000, fine_tune_epochs: int = 0, lr: float = 1e-4):
    """在单个快照上评估双流模型。可选 fine-tune。"""
    inputs, labels, meta = load_snapshot(snapshot_path)
    num_outputs = labels.shape[1]

    n = inputs.shape[0]
    n_val = int(round(n * holdout_ratio))
    n_val = min(max(n_val, 1), n - 1)
    perm = rng.permutation(n)
    x_tr_raw, y_tr_raw = inputs[perm[n_val:]], labels[perm[n_val:]]
    x_va_raw, y_va_raw = inputs[perm[:n_val]], labels[perm[:n_val]]

    if max_val_samples and x_va_raw.shape[0] > max_val_samples:
        sub = rng.choice(x_va_raw.shape[0], size=max_val_samples, replace=False)
        x_va_raw = x_va_raw[sub]
        y_va_raw = y_va_raw[sub]

    # 数据标准化
    x_scaler = StandardScaler()
    x_tr = x_scaler.fit_transform(x_tr_raw).astype(np.float32)
    x_va = x_scaler.transform(x_va_raw).astype(np.float32)

    y_scalers = []
    y_tr = np.zeros_like(y_tr_raw, dtype=np.float32)
    y_va = np.zeros_like(y_va_raw, dtype=np.float32)
    for d in range(num_outputs):
        s = StandardScaler()
        y_tr[:, d] = s.fit_transform(y_tr_raw[:, d].reshape(-1, 1)).ravel()
        y_va[:, d] = s.transform(y_va_raw[:, d].reshape(-1, 1)).ravel()
        y_scalers.append(s)

    # Fine-tune 时限制训练集大小（避免 OOM）
    max_ft_train = 4000
    if fine_tune_epochs > 0 and x_tr.shape[0] > max_ft_train:
        ft_idx = rng.choice(x_tr.shape[0], size=max_ft_train, replace=False)
        x_tr_ft = x_tr[ft_idx]
        y_tr_ft = y_tr[ft_idx]
    else:
        x_tr_ft = x_tr
        y_tr_ft = y_tr

    if fine_tune_epochs > 0:
        log.info(f"  Fine-tuning ({fine_tune_epochs} epochs, train_size={x_tr_ft.shape[0]})...")
        model.train()
        # 冻结 backbone，只训练 ActionEncoder + fusion_proj + heads
        for name, param in model.named_parameters():
            if not (name.startswith("action_encoder") or
                    name.startswith("fusion_proj") or
                    name.startswith("icl_predictor.y_encoders") or
                    name.startswith("icl_predictor.decoders")):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Trainable params: {trainable:,}")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
        )

        test_size = min(128, x_va.shape[0])  # 128 to reduce OOM risk
        for epoch in range(fine_tune_epochs):
            test_idx = rng.choice(x_va.shape[0], size=test_size, replace=False)
            X_batch = np.concatenate([x_tr_ft, x_va[test_idx]], axis=0)
            X_t = torch.from_numpy(X_batch).unsqueeze(0).to(device)
            y_batch = np.concatenate([y_tr_ft, y_va[test_idx]], axis=0)
            y_full = torch.from_numpy(y_batch).unsqueeze(0).to(device)
            y_train_t = y_full[:, :x_tr_ft.shape[0], :]

            # Quantile loss
            from experiments.exp0_multi_output.train_multi_output import quantile_loss
            pred = model(X_t, y_train=y_train_t)
            y_test_t = y_full[:, x_tr_ft.shape[0]:, :]
            loss = sum(quantile_loss(pred[:, :, j, :], y_test_t[:, :, j])
                       for j in range(num_outputs)) / num_outputs

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % max(1, fine_tune_epochs // 5) == 0:
                log.info(f"    Epoch {epoch+1}/{fine_tune_epochs}, loss={loss.item():.6f}")

        model.eval()

    # 推理
    all_preds_scaled = []
    eval_bs = 256
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, x_va.shape[0], eval_bs):
            end = min(start + eval_bs, x_va.shape[0])
            X_np = np.concatenate([x_tr, x_va[start:end]], axis=0)
            X_t = torch.from_numpy(X_np).unsqueeze(0).to(device)
            y_tr_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)
            pred = model(X_t, y_train=y_tr_t)
            all_preds_scaled.append(pred.squeeze(0).mean(dim=-1).cpu().numpy())

    elapsed = time.time() - t0
    preds_scaled = np.concatenate(all_preds_scaled, axis=0)[:x_va.shape[0]]
    preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
    for d in range(num_outputs):
        preds_raw[:, d] = y_scalers[d].inverse_transform(preds_scaled[:, d].reshape(-1, 1)).ravel()

    mse_all, mse_r, mse_d = compute_mse(preds_raw, y_va_raw)

    return {
        "epoch": meta["epoch"],
        "num_samples": meta["num_samples"],
        "reward_mean": meta["reward_mean"],
        "reward_std": meta["reward_std"],
        "mse_all": mse_all, "mse_reward": mse_r, "mse_delta": mse_d,
        "train_size": x_tr.shape[0], "val_size": x_va.shape[0],
        "infer_time": elapsed, "fine_tune_epochs": fine_tune_epochs,
    }


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "eval_dual_stream.log", mode='w'),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    baseline = load_baseline_results(BASELINE_CSV)
    log.info(f"Loaded {len(baseline)} baseline points")

    model = load_dual_stream_model(CKPT_PATH, num_outputs=12, device=device)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 只测试前几个小快照
    snapshot_files = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))
    test_snapshots = [sp for sp in snapshot_files
                      if int(sp.stem.split("_")[-1]) in [0, 5, 10, 20]]

    results = []
    rng = np.random.default_rng(123456)

    import gc
    for sp in test_snapshots:
        log.info(f"\n{'='*60}")
        log.info(f"处理: {sp.name}")

        # 为每个快照重新加载模型（避免 fine-tune 状态污染 + OOM）
        torch.cuda.empty_cache()
        gc.collect()
        model = load_dual_stream_model(CKPT_PATH, num_outputs=12, device=device)

        # 先 zero-shot
        result_zs = evaluate_snapshot(sp, model, device, rng, fine_tune_epochs=0)
        ep = result_zs["epoch"]
        bl = baseline.get(ep, {})
        vs_orig = result_zs["mse_all"] / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else float("inf")

        log.info(f"  [Zero-shot] MSE all={result_zs['mse_all']:.6g} | vs orig: {vs_orig:.2f}x")
        log.info(f"    reward={result_zs['mse_reward']:.6g}, delta={result_zs['mse_delta']:.6g}")

        # 再 fine-tune
        result_ft = evaluate_snapshot(sp, model, device, rng, fine_tune_epochs=50, lr=1e-4)
        vs_orig_ft = result_ft["mse_all"] / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else float("inf")
        vs_ens = result_ft["mse_all"] / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else float("inf")

        log.info(f"  [Fine-tuned] MSE all={result_ft['mse_all']:.6g} | vs orig: {vs_orig_ft:.2f}x | vs ens: {vs_ens:.2f}x")
        log.info(f"    reward={result_ft['mse_reward']:.6g}, delta={result_ft['mse_delta']:.6g}")

        results.append({"type": "zero_shot", **result_zs, "ratio_vs_orig": vs_orig})
        results.append({"type": "fine_tuned", **result_ft, "ratio_vs_orig": vs_orig_ft})

    # 保存结果
    csv_path = output_dir / "dual_stream_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log.info(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
