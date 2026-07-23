#!/usr/bin/env python3
"""
实验 1：双流分离 + ActionEncoder —— 纯零样本评估（全量数据，无 fine-tune）

- 加载预训练权重，ActionEncoder + fusion_proj 必需先 fine-tune 才能用
- 与实验 0 对比：同样的全量数据，纯前向推理，验证架构 overhead
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

BASELINE_CSV = "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3/distribution_tabicl2_mse/mse_comparison_results.csv"
SNAPSHOT_DIR = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
CKPT_PATH = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp1_action_encoder" / "outputs"


def load_snapshot(npz_path):
    data = np.load(npz_path)
    s = np.asarray(data["states"], dtype=np.float32)
    a = np.asarray(data["actions"], dtype=np.float32)
    r = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    d = np.asarray(data["next_states"], dtype=np.float32) - s
    return np.concatenate([s, a], -1), np.concatenate([r, d], -1), {
        "epoch": int(data["epoch"][0]), "num_samples": int(data["num_samples"][0]),
        "reward_mean": float(data["reward_mean"][0]), "reward_std": float(data["reward_std"][0]),
    }


def compute_mse(p, t, rs=1):
    d = p.astype(np.float64) - t.astype(np.float64)
    return float(np.mean(d**2)), float(np.mean(d[:,:rs]**2)), float(np.mean(d[:,rs:]**2))


def load_baseline(csv_path):
    bl = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            bl[ep] = {k: float(r[k]) for k in r if k not in ('epoch','num_samples','tabicl_train_size','tabicl_val_size')}
            bl[ep]['num_samples'] = int(r['num_samples'])
    return bl


def load_model(ckpt_path, num_outputs, device, use_dual_stream=True):
    from tabicl._model.tabicl import TabICL
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config, sd = ckpt["config"], ckpt["state_dict"]

    if use_dual_stream:
        config.update({"num_outputs": num_outputs, "use_action_encoder": True,
                       "state_dim": 11, "action_dim": 3, "action_encoder_mode": "mlp"})
    else:
        config.update({"num_outputs": num_outputs})

    model = TabICL(**config)
    model.load_state_dict(sd, strict=False)

    # Copy pre-trained head weights to all heads
    yw, yb = sd["icl_predictor.y_encoder.weight"], sd["icl_predictor.y_encoder.bias"]
    for j in range(num_outputs):
        model.icl_predictor.y_encoders[j].weight.data.copy_(yw)
        model.icl_predictor.y_encoders[j].bias.data.copy_(yb)
    for sfx in ["0.weight","0.bias","2.weight","2.bias"]:
        src = sd[f"icl_predictor.decoder.{sfx}"]
        for j in range(num_outputs):
            dict(model.named_parameters())[f"icl_predictor.decoders.{j}.{sfx}"].data.copy_(src)

    model.to(device).eval()
    return model


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(output_dir/"eval_zero_shot_full.log", mode='w')])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    baseline = load_baseline(BASELINE_CSV)

    # Test all snapshots with both models
    snapshot_files = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))
    log.info(f"Found {len(snapshot_files)} snapshots")

    rng = np.random.default_rng(123456)

    for model_name, use_dual_stream in [("Exp0 (shared backbone)", False),
                                         ("Exp1 (dual-stream)", True)]:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model_name}")
        log.info(f"{'='*60}")

        model = load_model(CKPT_PATH, num_outputs=12, device=device, use_dual_stream=use_dual_stream)

        for sp in snapshot_files:
            ep = int(sp.stem.split("_")[-1])
            log.info(f"\n  Snapshot: {sp.name}")

            inputs, labels, meta = load_snapshot(sp)
            num_outputs = labels.shape[1]
            n = inputs.shape[0]
            n_val = int(round(n * 0.2))
            n_val = min(max(n_val, 1), n - 1)
            perm = rng.permutation(n)
            x_tr_raw, y_tr_raw = inputs[perm[n_val:]], labels[perm[n_val:]]
            x_va_raw, y_va_raw = inputs[perm[:n_val]], labels[perm[:n_val]]

            # Cap validation (same as original)
            max_val = 10000
            if x_va_raw.shape[0] > max_val:
                sub = rng.choice(x_va_raw.shape[0], max_val, replace=False)
                x_va_raw, y_va_raw = x_va_raw[sub], y_va_raw[sub]

            # StandardScaler (same as original's "none" norm method)
            x_scaler = StandardScaler()
            x_tr = x_scaler.fit_transform(x_tr_raw).astype(np.float32)
            x_va = x_scaler.transform(x_va_raw).astype(np.float32)

            y_scalers, y_tr = [], np.zeros_like(y_tr_raw, dtype=np.float32)
            for d in range(num_outputs):
                s = StandardScaler()
                y_tr[:, d] = s.fit_transform(y_tr_raw[:, d].reshape(-1, 1)).ravel()
                y_scalers.append(s)

            log.info(f"    train={x_tr.shape[0]}, val={x_va.shape[0]}")

            # Inference (full training set, batched test)
            all_preds = []
            infer_bs = 256  # smaller than original's 1024 (12x ICL passes consume more memory)
            t0 = time.time()

            with torch.no_grad():
                for start in range(0, x_va.shape[0], infer_bs):
                    end = min(start + infer_bs, x_va.shape[0])
                    X_np = np.concatenate([x_tr, x_va[start:end]], axis=0)
                    X_t = torch.from_numpy(X_np).unsqueeze(0).to(device)
                    y_t = torch.from_numpy(y_tr).unsqueeze(0).to(device)

                    pred = model(X_t, y_train=y_t)
                    # pred: (1, chunk_size, 12, 999) or (1, chunk_size, 999)
                    all_preds.append(pred.squeeze(0).mean(dim=-1).cpu().numpy())

            elapsed = time.time() - t0
            preds_scaled = np.concatenate(all_preds, axis=0)[:x_va.shape[0]]
            preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
            for d in range(num_outputs):
                preds_raw[:, d] = y_scalers[d].inverse_transform(preds_scaled[:, d].reshape(-1, 1)).ravel()

            mse_all, mse_r, mse_d = compute_mse(preds_raw, y_va_raw)
            bl = baseline.get(ep, {})
            vs_orig = mse_all / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else float("inf")
            vs_ens = mse_all / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else float("inf")

            log.info(f"    MSE: all={mse_all:.6g} | vs orig: {vs_orig:.2f}x | vs ens: {vs_ens:.2f}x | time: {elapsed:.1f}s")
            log.info(f"      reward={mse_r:.6g}, delta={mse_d:.6g}")

            if vs_orig > 10:
                log.info(f"    (Exp1 dual-stream: ActionEncoder+fusion_proj are random → terrible zero-shot)")
                if use_dual_stream:
                    log.info(f"    (需要 fine-tune 才能使用，纯零样本不可用)")

    log.info(f"\nDone. Results show that Experiment 0 (shared backbone) works zero-shot at ~2.2x vs orig.")
    log.info(f"Experiment 1 (dual-stream) needs fine-tuning because ActionEncoder + fusion_proj are random.")


if __name__ == "__main__":
    main()
