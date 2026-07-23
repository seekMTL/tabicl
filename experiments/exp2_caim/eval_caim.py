#!/usr/bin/env python3
"""CAIM 评估脚本：零样本 + fine-tune 后评估。

用法:
  # 零样本评估（不加载 fine-tune 权重）
  CUDA_VISIBLE_DEVICES=1 python experiments/exp2_caim/eval_caim.py --mode zero_shot

  # 加载 fine-tune 权重后评估
  CUDA_VISIBLE_DEVICES=1 python experiments/exp2_caim/eval_caim.py --mode fine_tuned --weights outputs/caim_phase2_epoch50.pt

  # 全快照对比
  CUDA_VISIBLE_DEVICES=1 python experiments/exp2_caim/eval_caim.py --mode zero_shot --all_snapshots
"""

from __future__ import annotations
import argparse, csv, logging, sys, time
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="zero_shot", choices=["zero_shot", "fine_tuned"])
    p.add_argument("--weights", type=str, default=None, help="fine-tune 权重路径")
    p.add_argument("--all_snapshots", action="store_true")
    p.add_argument("--snapshot_name", default="env_pool_epoch_0000.npz")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_snapshot(npz_path):
    d = np.load(npz_path)
    return (np.asarray(d["states"], dtype=np.float32),
            np.asarray(d["actions"], dtype=np.float32),
            np.concatenate([np.asarray(d["rewards"], dtype=np.float32).reshape(-1,1),
                           np.asarray(d["next_states"], dtype=np.float32) - np.asarray(d["states"], dtype=np.float32)], -1),
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


def evaluate_snapshot(sp_path, model, device, rng, max_val=10000):
    S, A, Y, ep, ns = load_snapshot(sp_path)
    n, nv = S.shape[0], int(round(S.shape[0] * 0.2))
    nv = min(max(nv, 1), n-1)
    p = rng.permutation(n)
    S_tr, A_tr, Y_tr = S[p[nv:]], A[p[nv:]], Y[p[nv:]]
    S_va, A_va, Y_va = S[p[:nv]], A[p[:nv]], Y[p[:nv]]
    if S_va.shape[0] > max_val:
        sub = rng.choice(S_va.shape[0], max_val, replace=False)
        S_va, A_va, Y_va = S_va[sub], A_va[sub], Y_va[sub]

    s_s, a_s = StandardScaler(), StandardScaler()
    S_tr = s_s.fit_transform(S_tr).astype(np.float32); S_va = s_s.transform(S_va).astype(np.float32)
    A_tr = a_s.fit_transform(A_tr).astype(np.float32); A_va = a_s.transform(A_va).astype(np.float32)

    ys, Y_tr_s = [], np.zeros_like(Y_tr, dtype=np.float32)
    for d in range(Y.shape[1]):
        s = StandardScaler()
        Y_tr_s[:, d] = s.fit_transform(Y_tr[:, d].reshape(-1,1)).ravel()
        ys.append(s)

    all_p = []
    t0 = time.time()
    with torch.no_grad():
        for st in range(0, S_va.shape[0], 256):
            en = min(st+256, S_va.shape[0])
            X_s = torch.from_numpy(np.concatenate([S_tr, S_va[st:en]], 0)).unsqueeze(0).to(device)
            X_a = torch.from_numpy(np.concatenate([A_tr, A_va[st:en]], 0)).unsqueeze(0).to(device)
            y_t = torch.from_numpy(Y_tr_s).unsqueeze(0).to(device)
            all_p.append(model(X_s, X_a, y_t).squeeze(0).mean(-1).cpu().numpy())

    ps = np.concatenate(all_p, 0)[:S_va.shape[0]]
    pr = np.zeros_like(ps, dtype=np.float32)
    for d in range(ps.shape[1]):
        pr[:, d] = ys[d].inverse_transform(ps[:, d].reshape(-1,1)).ravel()

    mse = compute_mse(pr, Y_va)
    return ep, ns, mse[0], mse[1], mse[2], time.time()-t0


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"eval_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ],
    )
    log.info(f"Log file: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    baseline = load_baseline(BASELINE_CSV)

    from src.tabicl._model.caim import CAIM
    model = CAIM(CKPT, num_outputs=12, state_dim=11, action_dim=3)
    model.to(device).eval()

    if args.mode == "fine_tuned" and args.weights:
        log.info(f"Loading fine-tuned weights: {args.weights}")
        w = torch.load(args.weights, map_location=device)
        model.action_encoder.load_state_dict(w["action_encoder"])
        model.causal_fusion.load_state_dict(w["causal_fusion"])
        model.y_encoders.load_state_dict(w["y_encoders"])
        model.decoders.load_state_dict(w["decoders"])
        log.info("Weights loaded")

    snaps = sorted(SNAPSHOT_DIR.glob("env_pool_epoch_*.npz"))
    if not args.all_snapshots:
        snaps = [SNAPSHOT_DIR / args.snapshot_name]

    log.info(f"Mode: {args.mode}, snapshots: {len(snaps)}")
    for sp in snaps:
        ep, ns, ma, mr, md, t = evaluate_snapshot(sp, model, device, rng)
        bl = baseline.get(ep, {})
        vs_o = ma / bl.get("tabicl_mse_all", 1) if bl.get("tabicl_mse_all", 0) > 0 else 0
        vs_e = ma / bl.get("ensemble_mse_all", 1) if bl.get("ensemble_mse_all", 0) > 0 else 0
        log.info(f"Epoch {ep} ({ns}): MSE={ma:.6g} | vs Orig: {vs_o:.2f}x | vs Ens: {vs_e:.2f}x | t={t:.1f}s")
        if not args.all_snapshots:
            log.info(f"  reward={mr:.6g}, delta={md:.6g}")


if __name__ == "__main__":
    main()
