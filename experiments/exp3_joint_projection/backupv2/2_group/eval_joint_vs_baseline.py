"""方案1：全向量多维目标投影 (Joint Vector Target Projection)

对比 Baseline (12x 单输出 TabICLRegressor) vs Joint (1x 多输出 TabICLRegressor)。

用法:
  python eval_joint_vs_baseline.py [epoch_idx]
  epoch_idx: 数据快照编号，默认 280
"""

import sys, time, argparse
import numpy as np
import torch
from tabicl._sklearn.regressor import TabICLRegressor

CKPT = '/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt'
DATA_DIR = '/home/lizitao/project/other/sample_collect/Hopper-v5-ensemble'
OUTPUT_DIMS = 12  # reward + 11 delta_state

# ── 加载 Hopper 数据: X=[state|action](14列), y=[reward|delta](12列) ──

def load_hopper_data(epoch_idx):
    data = np.load(f'{DATA_DIR}/env_pool_epoch_{epoch_idx:04d}.npz')
    X = np.concatenate([data['states'], data['actions']], axis=1).astype(np.float32)
    delta = (data['next_states'] - data['states']).astype(np.float32)
    y = np.concatenate([data['rewards'][:, None], delta], axis=1).astype(np.float32)
    return X, y

# ── 评估一个模型：fit + predict → MSE ──

def evaluate(reg, X_tr, X_te, y_tr, y_te):
    """fit + predict，返回 per-dim MSE 向量和耗时。"""
    t0 = time.time()
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    t = time.time() - t0
    if pred.ndim == 1:
        # 单输出：(n,) → 与 y_te 形状一致，直接算标量 MSE
        return np.array([np.mean((pred - y_te) ** 2)]), t
    # 多输出：(n, d) → 逐维度 MSE
    return np.mean((pred - y_te) ** 2, axis=0), t

# ── 主函数 ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch_idx', nargs='?', type=int, default=280)
    parser.add_argument('--samples', type=int, default=10000,
                        help='限制测试集样本数 (default: 10000), 0=不限制')
    args = parser.parse_args()

    epoch_idx = args.epoch_idx
    max_test = args.samples

    X_all, y_all = load_hopper_data(epoch_idx)
    n_train = int(len(X_all) * 0.67)
    # n_train = int(len(X_all) * 0.8)
    np.random.seed(42)
    idx = np.random.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]
    X_tr, X_te = X_all[:n_train], X_all[n_train:]
    y_tr, y_te = y_all[:n_train], y_all[n_train:]

    if max_test > 0 and max_test < len(X_te):
        X_te, y_te = X_te[:max_test], y_te[:max_test]

    print(f'Epoch {epoch_idx}: {n_train} train / {len(X_te)} test'
          f'{" (limited to " + str(max_test) + ")" if max_test > 0 else ""}')
    # print(f'Epoch {epoch_idx}: {n_train} train / {len(X_all) - n_train} test')
    print(f'y 范围: reward [{y_all[:,0].min():.3f},{y_all[:,0].max():.3f}]  '
          f'delta [{y_all[:,1:].min():.3f},{y_all[:,1:].max():.3f}]')

    # ── Baseline: 12 个单输出 TabICLRegressor ──
    print('\n=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===')
    mse_base, t_base = np.zeros(OUTPUT_DIMS), 0.0
    for d in range(OUTPUT_DIMS):
        reg = TabICLRegressor(n_estimators=4, model_path=CKPT, device='cuda', random_state=42)
        mse_d, t_d = evaluate(reg, X_tr, X_te, y_tr[:, d], y_te[:, d])
        mse_base[d] = mse_d[0]  # evaluate 返回 per-dim 向量，单输出取下标 [0]
        t_base += t_d
    print(f'耗时: {t_base:.1f}s')

    # ── Joint: 1 个多输出 TabICLRegressor ──
    print('\n=== Joint: 1 x TabICLRegressor(num_outputs=12) ===')
    torch.cuda.empty_cache()  # 清理 baseline 残留显存
    reg12 = TabICLRegressor(n_estimators=4, num_outputs=12, model_path=CKPT, device='cuda', random_state=42)
    mse_vec, t_joint = evaluate(reg12, X_tr, X_te, y_tr, y_te)
    mse_joint = mse_vec  # (12,) per-dim MSE 向量
    print(f'耗时: {t_joint:.1f}s')

    # ── 结果：12 维度分离为 reward(1维) + delta(11维) 分组汇总 ──
    # reward 和 delta 量级差 ~1000x，混合算整体 MSE 会被 reward 主导
    reward_mse_b = mse_base[0]
    reward_mse_j = mse_joint[0]
    delta_mse_b = np.sum(mse_base[1:])   # 11 个 delta 维度的 MSE 之和
    delta_mse_j = np.sum(mse_joint[1:])

    print(f'\n{"="*70}')
    print(f'  {"Dim":>4} {"Label":<12} {"Baseline MSE":>14} {"Joint MSE":>14} {"Ratio(J/B)":>10}')
    print(f'  {"-"*66}')
    for d in range(OUTPUT_DIMS):
        label = 'reward' if d == 0 else f'delta[{d-1}]'
        print(f'  {d:>4} {label:<12} {mse_base[d]:>14.6f} {mse_joint[d]:>14.6f} {mse_joint[d]/mse_base[d]:>10.2f}x')
    print(f'  {"-"*66}')
    print(f'  {"REWARD MSE":>17} {reward_mse_b:>14.6f} {reward_mse_j:>14.6f} {reward_mse_j/reward_mse_b:>10.2f}x')
    print(f'  {"DELTA(avg) MSE":>17} {delta_mse_b/11:>14.6f} {delta_mse_j/11:>14.6f} {delta_mse_j/delta_mse_b:>10.2f}x')
    print(f'  {"OVERALL MSE":>17} {np.mean(mse_base):>14.6f} {np.mean(mse_joint):>14.6f} {np.mean(mse_joint)/np.mean(mse_base):>10.2f}x')
    print(f'\n  加速比: {t_base/t_joint:.1f}x ({t_base:.1f}s → {t_joint:.1f}s)')

if __name__ == '__main__':
    main()
