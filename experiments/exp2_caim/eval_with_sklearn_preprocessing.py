#!/usr/bin/env python3
"""
使用 sklearn wrapper 原始预处理管线评估多输出模型。

目标: 验证在正确的预处理下，多输出模型零样本能否达到 ~1.22x vs OrigTabICL。

用法:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp2_caim/eval_with_sklearn_preprocessing.py
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

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
SNAPSHOT_DIR = Path("/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble")
BASELINE_CSV = "/home/lizitao/project/mbpo_pyt_tabpfn/results/mse_v3/distribution_tabicl2_mse/mse_comparison_results.csv"
OUTPUT_DIR = _PROJECT_ROOT / "experiments" / "exp2_caim" / "outputs"


def load_snapshot(npz_path):
    d = np.load(npz_path)
    s = np.asarray(d["states"], dtype=np.float32)
    a = np.asarray(d["actions"], dtype=np.float32)
    r = np.asarray(d["rewards"], dtype=np.float32).reshape(-1, 1)
    ns = np.asarray(d["next_states"], dtype=np.float32)
    return (np.concatenate([s, a], -1),                              # (N, 14)
            np.concatenate([r, ns - s], -1),                          # (N, 12)
            int(d["epoch"][0]), int(d["num_samples"][0]))


def compute_mse(p, t, rs=1):
    d = p.astype(np.float64) - t.astype(np.float64)
    return float(np.mean(d**2)), float(np.mean(d[:,:rs]**2)), float(np.mean(d[:,rs:]**2))


def load_baseline(csv_path):
    bl = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            bl[ep] = {"tabicl_mse_all": float(r["tabicl_mse_all"]),
                      "ensemble_mse_all": float(r["ensemble_mse_all"])}
    return bl


def load_multi_output_model(ckpt_path, num_outputs, device):
    """加载实验 0 的多输出模型（无 ActionEncoder，纯共享 backbone + per-dim heads）。"""
    from src.tabicl._model.tabicl import TabICL
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config, sd = ckpt["config"], ckpt["state_dict"]
    config["num_outputs"] = num_outputs
    model = TabICL(**config)
    model.load_state_dict(sd, strict=False)

    # 复制预训练单输出 head 权重
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


def evaluate_snapshot(sp_path, model, device, rng, n_estimators=4):
    """用原始 sklearn 预处理管线评估单快照。"""
    inputs, labels, ep, ns = load_snapshot(sp_path)
    num_outputs = labels.shape[1]

    n, nv = inputs.shape[0], int(round(inputs.shape[0] * 0.2))
    nv = min(max(nv, 1), n - 1)
    # Cap validation to match baseline
    max_val = 10000
    nv = min(nv, max_val)

    perm = rng.permutation(n)
    x_tr_raw, y_tr_raw = inputs[perm[nv:]], labels[perm[nv:]]
    x_va_raw, y_va_raw = inputs[perm[:nv]], labels[perm[:nv]]

    # ── 使用原始 TabICL 预处理管线 ──
    from tabicl._sklearn.preprocessing import TransformToNumerical, EnsembleGenerator

    # Step 1: TransformToNumerical
    X_encoder = TransformToNumerical()
    x_tr_enc = X_encoder.fit_transform(x_tr_raw)
    x_va_enc = X_encoder.transform(x_va_raw)

    # Step 2: EnsembleGenerator（与 baseline 完全一致）
    ensemble_gen = EnsembleGenerator(
        classification=False,
        n_estimators=n_estimators,
        norm_methods=['none', 'power'],     # 原始默认
        feat_shuffle_method='latin',         # 原始默认
    )
    ensemble_gen.fit(x_tr_enc, y_tr_raw[:, 0])  # fit 需要 y（任意维度）

    # Step 3: 获取 ensemble 视图
    result_dict = ensemble_gen.transform(x_va_enc, mode='both')
    # result_dict: {norm_method: (X_combined[n_variants, n_tr+n_va, n_feat], y_train_ens[n_variants, n_tr])}

    # Step 4: y 标准化（per dimension，与 baseline 一致）
    y_scalers = []
    y_tr_scaled = np.zeros_like(y_tr_raw, dtype=np.float32)
    for d in range(num_outputs):
        s = StandardScaler()
        y_tr_scaled[:, d] = s.fit_transform(y_tr_raw[:, d].reshape(-1, 1)).ravel()
        y_scalers.append(s)

    # Step 5: 每个 ensemble 视图独立推理
    all_ensemble_preds = []
    t0 = time.time()

    with torch.no_grad():
        for norm_method, (X_combined, _) in result_dict.items():
            n_variants = X_combined.shape[0]
            n_tr = x_tr_enc.shape[0]

            for vi in range(n_variants):
                X_var = X_combined[vi].astype(np.float32)
                x_tr_var = X_var[:n_tr]
                x_va_var = X_var[n_tr:]

                preds_scaled = []
                infer_bs = 256
                for s in range(0, x_va_var.shape[0], infer_bs):
                    e = min(s + infer_bs, x_va_var.shape[0])
                    X_np = np.concatenate([x_tr_var, x_va_var[s:e]], 0)
                    X_t = torch.from_numpy(X_np).unsqueeze(0).to(device)
                    y_t = torch.from_numpy(y_tr_scaled).unsqueeze(0).to(device)
                    pred = model(X_t, y_train=y_t)
                    preds_scaled.append(pred.squeeze(0).mean(dim=-1).cpu().numpy())

                preds_scaled = np.concatenate(preds_scaled, 0)[:x_va_var.shape[0]]
                preds_raw = np.zeros_like(preds_scaled, dtype=np.float32)
                for d in range(num_outputs):
                    preds_raw[:, d] = y_scalers[d].inverse_transform(
                        preds_scaled[:, d].reshape(-1, 1)
                    ).ravel()
                all_ensemble_preds.append(preds_raw)

    elapsed = time.time() - t0

    # Ensemble 平均
    ensemble_preds = np.mean(np.stack(all_ensemble_preds, 0), 0)
    mse_all, mse_r, mse_d = compute_mse(ensemble_preds, y_va_raw)

    return ep, ns, mse_all, mse_r, mse_d, elapsed, len(all_ensemble_preds)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"eval_sklearn_preprocessing_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    baseline = load_baseline(BASELINE_CSV)
    model = load_multi_output_model(CKPT, num_outputs=12, device=device)

    rng = np.random.default_rng(123456)
    snaps = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))

    for sp in snaps:
        ep, ns, ma, mr, md, t, n_ens = evaluate_snapshot(sp, model, device, rng)
        bl = baseline.get(ep, {})
        vs_o = ma / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else 0
        vs_e = ma / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else 0

        log.info(f"Epoch {ep:>3} (n={ns:>6}): MSE={ma:.6g} | vs Orig: {vs_o:.2f}x | vs Ens: {vs_e:.2f}x | "
                 f"ens_views={n_ens} | t={t:.1f}s")
        log.info(f"  reward={mr:.6g}, delta={md:.6g}")
        log.info(f"  baseline: orig={bl.get('tabicl_mse_all', 0):.6g}, ens={bl.get('ensemble_mse_all', 0):.6g}")


if __name__ == "__main__":
    main()
