#!/usr/bin/env python3
"""
实验 0 完整对比（含 Ensemble）：多输出 TabICL + 简单 ensemble  vs  原始 12×TabICL  vs  Ensemble MLP

与 full_comparison.py 的区别：加入 n_estimators 个 ensemble 视图（特征随机排列），取平均。
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
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp0_multi_output" / "outputs"


def load_snapshot(npz_path: Path):
    data = np.load(npz_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    delta = next_states - states
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
            baseline[ep] = {k: float(row[k]) if 'mse' in k or 'time' in k or 'ratio' in k
                            else int(row[k]) if k in ('epoch', 'num_samples', 'tabicl_train_size', 'tabicl_val_size')
                            else row[k] for k in row}
    return baseline


def load_multi_output_model(ckpt_path: str, num_outputs: int, device: torch.device):
    from tabicl._model.tabicl import TabICL

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]
    config["num_outputs"] = num_outputs
    model = TabICL(**config)
    model.load_state_dict(state_dict, strict=False)

    y_w = state_dict["icl_predictor.y_encoder.weight"]
    y_b = state_dict["icl_predictor.y_encoder.bias"]
    for j in range(num_outputs):
        model.icl_predictor.y_encoders[j].weight.data.copy_(y_w)
        model.icl_predictor.y_encoders[j].bias.data.copy_(y_b)
    for suffix in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        src = state_dict[f"icl_predictor.decoder.{suffix}"]
        for j in range(num_outputs):
            dict(model.named_parameters())[f"icl_predictor.decoders.{j}.{suffix}"].data.copy_(src)

    model.to(device)
    model.eval()
    return model


def generate_shuffle_patterns(n_features: int, n_estimators: int, rng: np.random.Generator) -> list:
    """生成 n_estimators 个特征排列模式（仿照 Latin square 思想，这里用随机排列）。"""
    patterns = []
    for _ in range(n_estimators):
        patterns.append(rng.permutation(n_features).tolist())
    return patterns


def evaluate_snapshot_with_ensemble(snapshot_path: Path, model: torch.nn.Module,
                                    device: torch.device, rng: np.random.Generator,
                                    n_estimators: int = 4, holdout_ratio: float = 0.2,
                                    max_val_samples: int = 10000):
    """在单个快照上评估多输出模型（含 ensemble 多视图平均）。"""
    inputs, labels, metadata = load_snapshot(snapshot_path)
    num_outputs = labels.shape[1]
    n_features = inputs.shape[1]

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

    # 生成 ensemble 特征排列
    shuffle_patterns = generate_shuffle_patterns(n_features, n_estimators, rng)

    all_ensemble_preds = []  # 收集各 ensemble 视图的预测
    t0 = time.time()

    for est_idx, shuffle_pat in enumerate(shuffle_patterns):
        # 应用特征排列
        x_tr_shuffled = x_tr_raw[:, shuffle_pat]
        x_va_shuffled = x_va_raw[:, shuffle_pat]

        # StandardScaler 预处理
        x_scaler = StandardScaler()
        x_tr = x_scaler.fit_transform(x_tr_shuffled).astype(np.float32)
        x_va = x_scaler.transform(x_va_shuffled).astype(np.float32)

        # y 预处理（scaling per dimension）
        y_scalers, y_tr, y_va = [], np.zeros_like(y_tr_raw, dtype=np.float32), np.zeros_like(y_va_raw, dtype=np.float32)
        for d in range(num_outputs):
            s = StandardScaler()
            y_tr[:, d] = s.fit_transform(y_tr_raw[:, d].reshape(-1, 1)).ravel()
            y_va[:, d] = s.transform(y_va_raw[:, d].reshape(-1, 1)).ravel()
            y_scalers.append(s)

        # 推理
        preds_scaled = []
        eval_bs = 256
        with torch.no_grad():
            for start in range(0, x_va.shape[0], eval_bs):
                end = min(start + eval_bs, x_va.shape[0])
                X_t = torch.from_numpy(np.concatenate([x_tr, x_va[start:end]], axis=0)).unsqueeze(0).to(device)
                y_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)
                pred = model(X_t, y_train=y_t)
                preds_scaled.append(pred.squeeze(0).mean(dim=-1).cpu().numpy())

        preds_scaled = np.concatenate(preds_scaled, axis=0)[:x_va.shape[0]]

        # 逆归一化
        preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
        for d in range(num_outputs):
            preds_raw[:, d] = y_scalers[d].inverse_transform(preds_scaled[:, d].reshape(-1, 1)).ravel()

        all_ensemble_preds.append(preds_raw)

    elapsed = time.time() - t0

    # Ensemble 平均
    ensemble_preds = np.mean(np.stack(all_ensemble_preds, axis=0), axis=0)  # (val_size, 12)

    mse_all, mse_r, mse_d = compute_mse(ensemble_preds, y_va_raw)

    return {
        "epoch": metadata["epoch"],
        "num_samples": metadata["num_samples"],
        "reward_mean": metadata["reward_mean"],
        "reward_std": metadata["reward_std"],
        "mse_all": mse_all, "mse_reward": mse_r, "mse_delta": mse_d,
        "train_size": x_tr.shape[0], "val_size": x_va.shape[0],
        "infer_time": elapsed, "n_estimators": n_estimators,
    }, metadata


def print_summary(results: list, baseline: dict):
    sep = "=" * 130
    log.info(f"\n{sep}")
    header = (f"{'Epoch':>6} | {'Samples':>8} | "
              f"{'Ensemble MLP':>14} | {'OrigTabICL':>14} | {'MultiOut+Ens':>14} | "
              f"{'vs Ens':>8} | {'vs Orig':>8} | {'Time':>8}")
    log.info(header)
    log.info(f"{'':>6} | {'':>8} | {'(MSE all)':>14} | {'(MSE all)':>14} | {'(MSE all)':>14} | "
             f"{'(ratio)':>8} | {'(ratio)':>8} | {'(s)':>8}")
    log.info("-" * 130)
    for r in results:
        ep = r["epoch"]
        bl = baseline.get(ep, {})
        vs_ens = r["mse_all"] / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else float("inf")
        vs_orig = r["mse_all"] / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else float("inf")
        log.info(f"{ep:>6} | {r['num_samples']:>8} | "
                 f"{bl.get('ensemble_mse_all', 0):>14.6g} | {bl.get('tabicl_mse_all', 0):>14.6g} | "
                 f"{r['mse_all']:>14.6g} | {vs_ens:>8.4f} | {vs_orig:>8.4f} | "
                 f"{r['infer_time']:>8.1f}")
    log.info(sep)


def save_results(results: list, baseline: dict, output_dir: Path):
    csv_path = output_dir / "ensemble_comparison_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "num_samples", "reward_mean", "reward_std",
            "ensemble_mlp_mse_all", "orig_tabicl_mse_all",
            "multiout_ensemble_mse_all", "multiout_ensemble_mse_reward", "multiout_ensemble_mse_delta",
            "ratio_vs_ensemble_all", "ratio_vs_orig_all",
            "n_estimators", "infer_time",
        ])
        for r in results:
            ep = r["epoch"]
            bl = baseline.get(ep, {})
            writer.writerow([
                ep, r["num_samples"],
                f"{r['reward_mean']:.4f}", f"{r['reward_std']:.4f}",
                f"{bl.get('ensemble_mse_all', 0):.8g}",
                f"{bl.get('tabicl_mse_all', 0):.8g}",
                f"{r['mse_all']:.8g}", f"{r['mse_reward']:.8g}", f"{r['mse_delta']:.8g}",
                f"{r['mse_all'] / bl['ensemble_mse_all']:.4f}" if bl.get('ensemble_mse_all', 0) > 0 else "N/A",
                f"{r['mse_all'] / bl['tabicl_mse_all']:.4f}" if bl.get('tabicl_mse_all', 0) > 0 else "N/A",
                r["n_estimators"], f"{r['infer_time']:.1f}",
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
            logging.FileHandler(output_dir / "ensemble_comparison.log", mode='w'),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 加载 baseline
    baseline = load_baseline_results(BASELINE_CSV)

    # 加载模型
    log.info("加载多输出模型...")
    model = load_multi_output_model(CKPT_PATH, num_outputs=12, device=device)

    # 评估
    snapshot_files = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))
    log.info(f"找到 {len(snapshot_files)} 个快照")

    results = []
    rng = np.random.default_rng(123456)

    for sp in snapshot_files:
        log.info(f"\n{'='*60}")
        log.info(f"处理: {sp.name}")

        result, meta = evaluate_snapshot_with_ensemble(sp, model, device, rng, n_estimators=4)

        ep = result["epoch"]
        bl = baseline.get(ep, {})
        vs_ens = result["mse_all"] / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else float("inf")
        vs_orig = result["mse_all"] / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else float("inf")

        log.info(f"  samples={result['num_samples']}, reward={result['reward_mean']:.2f}")
        log.info(f"  MultiOut+Ens MSE: all={result['mse_all']:.6g}, reward={result['mse_reward']:.6g}, "
                 f"delta={result['mse_delta']:.6g}")
        log.info(f"  vs Ensemble: {vs_ens:.4f}x  |  vs OrigTabICL: {vs_orig:.4f}x")
        log.info(f"  OrigTabICL: {bl.get('tabicl_mse_all', 0):.6g}  |  "
                 f"Ensemble MLP: {bl.get('ensemble_mse_all', 0):.6g}")
        log.info(f"  Time: {result['infer_time']:.1f}s (n_estimators={result['n_estimators']})")

        results.append(result)

    print_summary(results, baseline)
    save_results(results, baseline, output_dir)

    valid = [(r, baseline[r["epoch"]]) for r in results if r["epoch"] in baseline]
    ratios_vs_orig = [r["mse_all"] / bl["tabicl_mse_all"] for r, bl in valid]
    ratios_vs_ens = [r["mse_all"] / bl["ensemble_mse_all"] for r, bl in valid]
    log.info(f"\n=== 统计汇总 ===")
    log.info(f"MultiOut+Ensemble vs OrigTabICL: {np.mean(ratios_vs_orig):.2f}x ± {np.std(ratios_vs_orig):.2f}")
    log.info(f"MultiOut+Ensemble vs Ensemble MLP: {np.mean(ratios_vs_ens):.2f}x ± {np.std(ratios_vs_ens):.2f}")


if __name__ == "__main__":
    main()
