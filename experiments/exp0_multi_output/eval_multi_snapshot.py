#!/usr/bin/env python3
"""评估多输出模型在多个快照上的表现，对比 baseline。"""

import csv, logging, sys, time
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval_zero_shot import load_snapshot, load_multi_output_model, compute_mse

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    snapshot_dir = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
    ckpt_path = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
    output_dir = _PROJECT_ROOT / "experiments" / "exp0_multi_output" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    # Baseline results from distribution_tabicl2_mse experiment
    baseline_results = {
        0:   (0.00265815, 0.00550285, 0.00239954),
        5:   (0.00249776, 0.00413317, 0.00234909),
        10:  (0.00208060, 0.00514020, 0.00180245),
        20:  (0.00158494, 0.00276447, 0.00147771),
        50:  (0.00161951, 0.00145373, 0.00163458),
    }

    snapshot_files = sorted(snapshot_dir.glob("env_pool_epoch_*.npz"))
    # 只评估部分快照（从 epoch 0 到 epoch 50）
    test_epochs = [0, 5, 10, 20, 50]

    results = []
    for sp in snapshot_files:
        epoch_num = int(sp.stem.split("_")[-1])
        if epoch_num not in test_epochs:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"评估: {sp.name}")

        inputs, labels, _ = load_snapshot(sp)
        num_outputs = labels.shape[1]
        n = inputs.shape[0]
        n_val = int(round(n * 0.2))
        n_val = min(max(n_val, 1), n - 1)
        perm = rng.permutation(n)
        x_tr_raw, y_tr_raw = inputs[perm[n_val:]], labels[perm[n_val:]]
        x_va_raw, y_va_raw = inputs[perm[:n_val]], labels[perm[:n_val]]

        # 预处理
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

        # 加载模型
        model = load_multi_output_model(ckpt_path, num_outputs, device)

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
                pred_mean = pred.squeeze(0).mean(dim=-1)
                all_preds_scaled.append(pred_mean.cpu().numpy())

        elapsed = time.time() - t0
        preds_scaled = np.concatenate(all_preds_scaled, axis=0)[:x_va.shape[0]]

        # 逆归一化
        preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
        for d in range(num_outputs):
            preds_raw[:, d] = y_scalers[d].inverse_transform(
            preds_scaled[:, d].reshape(-1, 1)  # 注意：是 preds_scaled，不是 preds_raw
        ).ravel()

        mse_all, mse_r, mse_d = compute_mse(preds_raw, y_va_raw)

        # Baseline
        b_all, b_r, b_d = baseline_results.get(epoch_num, (float('nan'),)*3)

        results.append({
            "epoch": epoch_num, "samples": n,
            "multi_mse_all": mse_all, "multi_mse_r": mse_r, "multi_mse_d": mse_d,
            "base_mse_all": b_all, "base_mse_r": b_r, "base_mse_d": b_d,
            "ratio_all": mse_all / b_all, "ratio_r": mse_r / b_r, "ratio_d": mse_d / b_d,
            "infer_time": elapsed,
        })

        log.info(f"Epoch {epoch_num}: overall={mse_all:.6f} (baseline={b_all:.6f}, ratio={mse_all/b_all:.2f})")
        log.info(f"  Reward={mse_r:.6f} (ratio={mse_r/b_r:.2f}), Delta={mse_d:.6f} (ratio={mse_d/b_d:.2f})")
        log.info(f"  Time: {elapsed:.1f}s")

    # 保存结果
    csv_path = output_dir / "multi_output_vs_baseline.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    log.info(f"\n结果已保存至: {csv_path}")
    log.info(f"\n{'='*60}")
    log.info(f"Summary:")
    ratios_all = [r['ratio_all'] for r in results]
    ratios_r = [r['ratio_r'] for r in results]
    ratios_d = [r['ratio_d'] for r in results]
    log.info(f"  Overall ratio: {np.mean(ratios_all):.2f}x ± {np.std(ratios_all):.2f}")
    log.info(f"  Reward ratio:  {np.mean(ratios_r):.2f}x ± {np.std(ratios_r):.2f}")
    log.info(f"  Delta ratio:   {np.mean(ratios_d):.2f}x ± {np.std(ratios_d):.2f}")


if __name__ == "__main__":
    main()
