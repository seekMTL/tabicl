#!/usr/bin/env python3
"""
实验 0 完整对比测试：多输出 TabICL vs 原始 12×TabICL vs Ensemble MLP

在所有 15 个快照上评估多输出模型，对比 baseline 结果。

用法:
  python experiments/exp0_multi_output/full_comparison.py
"""

from __future__ import annotations

import csv, logging, sys, time
from pathlib import Path
from collections import OrderedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Baseline 结果（来自 distribution_tabicl2_mse 实验的 mse_comparison_results.csv）
BASELINE_CSV = (
    "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3"
    "/distribution_tabicl2_mse/mse_comparison_results.csv"
)

SNAPSHOT_DIR = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
CKPT_PATH = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp0_multi_output" / "outputs"


def load_snapshot(npz_path: Path):
    data = np.load(npz_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    delta = next_states - states
    inputs = np.concatenate([states, actions], axis=-1)       # (N, 14)
    labels = np.concatenate([rewards, delta], axis=-1)        # (N, 12)
    metadata = {
        "epoch": int(data["epoch"][0]),
        "num_samples": int(data["num_samples"][0]),
        "reward_mean": float(data["reward_mean"][0]),
        "reward_std": float(data["reward_std"][0]),
    }
    return inputs, labels, metadata


def compute_mse(pred: np.ndarray, target: np.ndarray, reward_size: int = 1):
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse_all = float(np.mean(diff ** 2))
    mse_r = float(np.mean(diff[:, :reward_size] ** 2))
    mse_d = float(np.mean(diff[:, reward_size:] ** 2))
    return mse_all, mse_r, mse_d


def load_baseline_results(csv_path: str) -> dict:
    """从 CSV 加载 baseline 结果。"""
    baseline = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            baseline[epoch] = {
                "num_samples": int(row["num_samples"]),
                "ensemble_mse_all": float(row["ensemble_mse_all"]),
                "ensemble_mse_reward": float(row["ensemble_mse_reward"]),
                "ensemble_mse_delta": float(row["ensemble_mse_delta"]),
                "tabicl_mse_all": float(row["tabicl_mse_all"]),
                "tabicl_mse_reward": float(row["tabicl_mse_reward"]),
                "tabicl_mse_delta": float(row["tabicl_mse_delta"]),
                "tabicl_fit_time": float(row["tabicl_fit_time"]),
                "tabicl_predict_time": float(row["tabicl_predict_time"]),
            }
    return baseline


def load_multi_output_model(ckpt_path: str, num_outputs: int, device: torch.device):
    """加载多输出模型并复制预训练权重。"""
    from tabicl._model.tabicl import TabICL

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    config["num_outputs"] = num_outputs
    model = TabICL(**config)
    model.load_state_dict(state_dict, strict=False)

    # 复制单输出权重到所有 head
    y_enc_weight = state_dict["icl_predictor.y_encoder.weight"]
    y_enc_bias = state_dict["icl_predictor.y_encoder.bias"]
    for j in range(num_outputs):
        model.icl_predictor.y_encoders[j].weight.data.copy_(y_enc_weight)
        model.icl_predictor.y_encoders[j].bias.data.copy_(y_enc_bias)

    for suffix in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        src_key = f"icl_predictor.decoder.{suffix}"
        src_weight = state_dict[src_key]
        for j in range(num_outputs):
            target = dict(model.named_parameters())[f"icl_predictor.decoders.{j}.{suffix}"]
            target.data.copy_(src_weight)

    model.to(device)
    model.eval()
    return model


def evaluate_snapshot(snapshot_path: Path, model: torch.nn.Module, device: torch.device,
                      rng: np.random.Generator, holdout_ratio: float = 0.2,
                      max_val_samples: int = 10000):
    """在单个快照上评估多输出模型。"""
    inputs, labels, metadata = load_snapshot(snapshot_path)
    num_outputs = labels.shape[1]

    # 划分训练/验证
    n = inputs.shape[0]
    n_val = int(round(n * holdout_ratio))
    n_val = min(max(n_val, 1), n - 1)
    perm = rng.permutation(n)
    x_tr_raw, y_tr_raw = inputs[perm[n_val:]], labels[perm[n_val:]]
    x_va_raw, y_va_raw = inputs[perm[:n_val]], labels[perm[:n_val]]

    # 限制验证集大小
    if max_val_samples and x_va_raw.shape[0] > max_val_samples:
        sub = rng.choice(x_va_raw.shape[0], size=max_val_samples, replace=False)
        x_va_raw = x_va_raw[sub]
        y_va_raw = y_va_raw[sub]

    # 预处理：StandardScaler（对应 baseline 的单视图标准化）
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

    # 推理
    all_preds_scaled = []
    eval_bs = 256  # 批大小，与 baseline 的 tabicl_infer_batch_size=1024 类似但更保守
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, x_va.shape[0], eval_bs):
            end = min(start + eval_bs, x_va.shape[0])
            X_np = np.concatenate([x_tr, x_va[start:end]], axis=0)
            X_t = torch.from_numpy(X_np).unsqueeze(0).to(device)
            y_tr_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)

            pred = model(X_t, y_train=y_tr_t)
            # pred: (1, chunk_size, 12, 999)
            pred_mean = pred.squeeze(0).mean(dim=-1)  # (chunk_size, 12)，取分位数均值
            all_preds_scaled.append(pred_mean.cpu().numpy())

    elapsed = time.time() - t0
    preds_scaled = np.concatenate(all_preds_scaled, axis=0)[:x_va.shape[0]]

    # 逆归一化
    preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
    for d in range(num_outputs):
        preds_raw[:, d] = y_scalers[d].inverse_transform(
            preds_scaled[:, d].reshape(-1, 1)
        ).ravel()

    mse_all, mse_r, mse_d = compute_mse(preds_raw, y_va_raw)

    return {
        "epoch": metadata["epoch"],
        "num_samples": metadata["num_samples"],
        "reward_mean": metadata["reward_mean"],
        "reward_std": metadata["reward_std"],
        "mse_all": mse_all,
        "mse_reward": mse_r,
        "mse_delta": mse_d,
        "train_size": x_tr.shape[0],
        "val_size": x_va.shape[0],
        "infer_time": elapsed,
    }, metadata, y_va_raw


def print_summary(results: list, baseline: dict):
    """打印汇总表格。"""
    sep = "=" * 120
    log.info(f"\n{sep}")
    header = (f"{'Epoch':>6} | {'Samples':>8} | "
              f"{'Ensemble MSE':>14} | {'OrigTabICL MSE':>14} | {'MultiOut MSE':>14} | "
              f"{'vs Ens':>8} | {'vs Orig':>8} | {'Time':>8}")
    log.info(header)
    log.info(f"{'':>6} | {'':>8} | {'(all)':>14} | {'(all)':>14} | {'(all)':>14} | "
             f"{'(ratio)':>8} | {'(ratio)':>8} | {'(s)':>8}")
    log.info("-" * 120)

    for r in results:
        ep = r["epoch"]
        bl = baseline.get(ep)
        if bl is None:
            continue
        vs_ens = r["mse_all"] / bl["ensemble_mse_all"] if bl["ensemble_mse_all"] > 0 else float("inf")
        vs_orig = r["mse_all"] / bl["tabicl_mse_all"] if bl["tabicl_mse_all"] > 0 else float("inf")
        log.info(f"{ep:>6} | {r['num_samples']:>8} | "
                 f"{bl['ensemble_mse_all']:>14.6g} | {bl['tabicl_mse_all']:>14.6g} | "
                 f"{r['mse_all']:>14.6g} | {vs_ens:>8.4f} | {vs_orig:>8.4f} | "
                 f"{r['infer_time']:>8.1f}")
    log.info(sep)


def save_results(results: list, baseline: dict, output_dir: Path):
    """保存结果为 CSV。"""
    csv_path = output_dir / "full_comparison_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [
            "epoch", "num_samples", "reward_mean", "reward_std",
            "ensemble_mse_all", "ensemble_mse_reward", "ensemble_mse_delta",
            "orig_tabicl_mse_all", "orig_tabicl_mse_reward", "orig_tabicl_mse_delta",
            "multiout_mse_all", "multiout_mse_reward", "multiout_mse_delta",
            "ratio_vs_ensemble_all", "ratio_vs_ensemble_reward", "ratio_vs_ensemble_delta",
            "ratio_vs_orig_all", "ratio_vs_orig_reward", "ratio_vs_orig_delta",
            "train_size", "val_size", "infer_time",
        ]
        writer.writerow(headers)
        for r in results:
            ep = r["epoch"]
            bl = baseline.get(ep, {})
            writer.writerow([
                ep, r["num_samples"],
                f"{r['reward_mean']:.4f}", f"{r['reward_std']:.4f}",
                f"{bl.get('ensemble_mse_all', 'N/A'):.8g}" if isinstance(bl.get('ensemble_mse_all'), (int, float)) else "N/A",
                f"{bl.get('ensemble_mse_reward', 'N/A'):.8g}" if isinstance(bl.get('ensemble_mse_reward'), (int, float)) else "N/A",
                f"{bl.get('ensemble_mse_delta', 'N/A'):.8g}" if isinstance(bl.get('ensemble_mse_delta'), (int, float)) else "N/A",
                f"{bl.get('tabicl_mse_all', 'N/A'):.8g}" if isinstance(bl.get('tabicl_mse_all'), (int, float)) else "N/A",
                f"{bl.get('tabicl_mse_reward', 'N/A'):.8g}" if isinstance(bl.get('tabicl_mse_reward'), (int, float)) else "N/A",
                f"{bl.get('tabicl_mse_delta', 'N/A'):.8g}" if isinstance(bl.get('tabicl_mse_delta'), (int, float)) else "N/A",
                f"{r['mse_all']:.8g}", f"{r['mse_reward']:.8g}", f"{r['mse_delta']:.8g}",
                f"{r['mse_all'] / bl['ensemble_mse_all']:.4f}" if bl.get('ensemble_mse_all', 0) > 0 else "N/A",
                f"{r['mse_reward'] / bl['ensemble_mse_reward']:.4f}" if bl.get('ensemble_mse_reward', 0) > 0 else "N/A",
                f"{r['mse_delta'] / bl['ensemble_mse_delta']:.4f}" if bl.get('ensemble_mse_delta', 0) > 0 else "N/A",
                f"{r['mse_all'] / bl['tabicl_mse_all']:.4f}" if bl.get('tabicl_mse_all', 0) > 0 else "N/A",
                f"{r['mse_reward'] / bl['tabicl_mse_reward']:.4f}" if bl.get('tabicl_mse_reward', 0) > 0 else "N/A",
                f"{r['mse_delta'] / bl['tabicl_mse_delta']:.4f}" if bl.get('tabicl_mse_delta', 0) > 0 else "N/A",
                r["train_size"], r["val_size"], f"{r['infer_time']:.1f}",
            ])
    log.info(f"CSV 已保存至: {csv_path}")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "full_comparison.log", mode='w'),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 加载 baseline 结果
    log.info(f"加载 baseline 结果: {BASELINE_CSV}")
    baseline = load_baseline_results(BASELINE_CSV)
    log.info(f"已加载 {len(baseline)} 个 baseline 数据点")

    # 加载多输出模型（所有快照共享同一模型）
    log.info(f"加载多输出模型: {CKPT_PATH}")
    model = load_multi_output_model(CKPT_PATH, num_outputs=12, device=device)
    log.info(f"模型加载完成")

    # 评估所有快照
    snapshot_files = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))
    log.info(f"找到 {len(snapshot_files)} 个快照文件")

    results = []
    rng = np.random.default_rng(123456)  # 与 baseline 脚本相同的 seed

    for sp in snapshot_files:
        log.info(f"\n{'='*60}")
        log.info(f"处理: {sp.name}")

        result, metadata, _ = evaluate_snapshot(sp, model, device, rng)

        ep = result["epoch"]
        bl = baseline.get(ep, {})
        log.info(f"  samples={result['num_samples']}, reward={result['reward_mean']:.2f}±{result['reward_std']:.2f}")
        log.info(f"  MultiOut MSE: all={result['mse_all']:.6g}, reward={result['mse_reward']:.6g}, "
                 f"delta={result['mse_delta']:.6g}")

        if bl:
            vs_ens = result["mse_all"] / bl["ensemble_mse_all"]
            vs_orig = result["mse_all"] / bl["tabicl_mse_all"]
            log.info(f"  vs Ensemble: {vs_ens:.4f}x  |  vs OrigTabICL: {vs_orig:.4f}x")
            log.info(f"  OrigTabICL MSE: {bl['tabicl_mse_all']:.6g}  |  "
                     f"Ensemble MSE: {bl['ensemble_mse_all']:.6g}")

        log.info(f"  Infer time: {result['infer_time']:.1f}s")
        results.append(result)

    # 打印汇总
    print_summary(results, baseline)

    # 保存结果
    save_results(results, baseline, output_dir)

    # 统计
    valid = [(r, baseline[r["epoch"]]) for r in results if r["epoch"] in baseline]
    ratios_vs_orig = [r["mse_all"] / bl["tabicl_mse_all"] for r, bl in valid]
    ratios_vs_ens = [r["mse_all"] / bl["ensemble_mse_all"] for r, bl in valid]
    log.info(f"\n=== 统计汇总 ===")
    log.info(f"MultiOut vs OrigTabICL: {np.mean(ratios_vs_orig):.2f}x ± {np.std(ratios_vs_orig):.2f}")
    log.info(f"MultiOut vs Ensemble:   {np.mean(ratios_vs_ens):.2f}x ± {np.std(ratios_vs_ens):.2f}")
    log.info(f"解释: ratio < 1 表示 MultiOut 更好, ratio > 1 表示 MultiOut 更差")


if __name__ == "__main__":
    main()
