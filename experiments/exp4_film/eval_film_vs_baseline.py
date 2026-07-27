"""FiLM 条件化调制 vs 原版 TabICL 对比实验

对比 A. 原版 TabICL (12x 单输出) vs B. FiLM TabICL (1x 多输出)。

关键：A 使用 TabICLRegressor(num_outputs=1)，内部走原始 y_encoder + 加性注入路径；
B 使用 TabICLRegressor(num_outputs=12)，内部走 MultiDimTargetFiLM 特征级线性调制路径。
为确保对比公正，脚本会自检 A 的 ColEmbedding 是否使用 y_encoder（非 film）。

用法:
  python eval_film_vs_baseline.py [epoch_idx] [--samples N]
  epoch_idx: 数据快照编号，默认 280
  --samples: 限制测试集样本数（默认 10000）
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
    """加载指定 epoch 的 Hopper-v5 ensemble 数据。

    X: (n, 11 state + 3 action) = (n, 14)
    y: (n, 1 reward + 11 delta_state) = (n, 12)
    """
    data = np.load(f'{DATA_DIR}/env_pool_epoch_{epoch_idx:04d}.npz')
    X = np.concatenate([data['states'], data['actions']], axis=1).astype(np.float32)
    delta = (data['next_states'] - data['states']).astype(np.float32)
    y = np.concatenate([data['rewards'][:, None], delta], axis=1).astype(np.float32)
    return X, y


# ── 评估 ──


def evaluate(reg, X_tr, X_te, y_tr, y_te):
    """fit + predict，返回 per-dim MSE 向量和耗时。

    单输出 y_tr/y_te 为 1D → pred 为 (n,)；多输出为 2D → pred 为 (n, d)。
    """
    t0 = time.time()
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    t = time.time() - t0
    if pred.ndim == 1:
        return np.array([np.mean((pred - y_te) ** 2)]), t
    return np.mean((pred - y_te) ** 2, axis=0), t


# ── 结果输出 ──


def print_results(mse_base, mse_film, t_base, t_film):
    """按维度打印 Baseline vs FiLM 对比表。"""
    print(f'\n{"="*70}')
    print(f'  {"Dim":>4} {"Label":<12} {"Baseline MSE":>14} {"FiLM MSE":>14} {"Ratio(F/B)":>10}')
    print(f'  {"-"*66}')

    for d in range(OUTPUT_DIMS):
        label = 'reward' if d == 0 else f'delta[{d-1}]'
        print(f'  {d:>4} {label:<12} {mse_base[d]:>14.6f} {mse_film[d]:>14.6f} {mse_film[d]/mse_base[d]:>10.2f}x')

    print(f'  {"-"*66}')

    reward_r = mse_film[0] / mse_base[0]
    delta_base_avg = np.sum(mse_base[1:]) / 11
    delta_film_avg = np.sum(mse_film[1:]) / 11
    delta_r = np.sum(mse_film[1:]) / np.sum(mse_base[1:])
    overall_r = np.mean(mse_film) / np.mean(mse_base)

    print(f'  {"REWARD MSE":>17} {mse_base[0]:>14.6f} {mse_film[0]:>14.6f} {reward_r:>10.2f}x')
    print(f'  {"DELTA(avg) MSE":>17} {delta_base_avg:>14.6f} {delta_film_avg:>14.6f} {delta_r:>10.2f}x')
    print(f'  {"OVERALL MSE":>17} {np.mean(mse_base):>14.6f} {np.mean(mse_film):>14.6f} {overall_r:>10.2f}x')
    print(f'\n  加速比: {t_base/t_film:.1f}x ({t_base:.1f}s → {t_film:.1f}s)')

    return reward_r, delta_r, overall_r


# ── 自检：确认 Baseline 用的是原版 y_encoder 路径 ──


def verify_baseline_is_original(reg):
    """验证 num_outputs=1 时走的是原始 y_encoder 路径（非 FiLM）。"""
    col = reg.model_.col_embedder
    icl = reg.model_.icl_predictor

    checks = []
    # ColEmbedding: 应有 y_encoder（原始），不应有 film（FiLM 专用）
    checks.append(('ColEmbedding.y_encoder', hasattr(col, 'y_encoder')))
    checks.append(('ColEmbedding.film (应为 False)', not hasattr(col, 'film')))
    # ICLearning: num_outputs 应为 1
    checks.append(('ICLearning.num_outputs=1', icl.num_outputs == 1))

    all_ok = True
    for name, ok in checks:
        status = '✓' if ok else '✗ FAIL'
        if not ok:
            all_ok = False
        print(f'    {status}  {name}')

    return all_ok


# ── 主函数 ──


def main():
    parser = argparse.ArgumentParser(
        description='FiLM 条件化调制 vs 原版 TabICL 对比实验')
    parser.add_argument('epoch_idx', nargs='?', type=int, default=280,
                        help='数据快照编号 (default: 280)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='限制测试集样本数 (default: 10000), 0=不限制')
    args = parser.parse_args()

    epoch_idx = args.epoch_idx
    max_test = args.samples

    # ── 加载并分割数据 ──
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
    print(f'y 范围: reward [{y_all[:,0].min():.3f},{y_all[:,0].max():.3f}]  '
          f'delta [{y_all[:,1:].min():.3f},{y_all[:,1:].max():.3f}]')

    # ── A. 原版 TabICL: 12 个单输出 TabICLRegressor ──
    # num_outputs=1 时 _load_model 走 else 分支：
    #   self.model_ = TabICL(**config)
    #   self.model_.load_state_dict(state_dict)
    # ColEmbedding 创建 y_encoder=Linear(1,128)，不创建 film。
    # ICLearning 创建 y_encoder=Linear(1,512)，_icl_predictions 走单输出路径。
    print('\n=== A. 原版 TabICL: 12 x TabICLRegressor(num_outputs=1) ===')

    # 先用第 0 维 fit 一个实例来做完整性自检
    reg0 = TabICLRegressor(n_estimators=4, model_path=CKPT, device='cuda', random_state=42)
    reg0.fit(X_tr, y_tr[:, 0])
    print('  [自检] 确认 Baseline 使用原始 y_encoder 路径（非 FiLM）:')
    if not verify_baseline_is_original(reg0):
        print('\n  ❌ 自检失败！Baseline 模型结构与预期不符，对比无效。')
        sys.exit(1)

    mse_base, t_base = np.zeros(OUTPUT_DIMS), 0.0
    t_base += time.time() - time.time()  # reset; reg0 already fitted above
    t0 = time.time()
    pred0 = reg0.predict(X_te)
    t_base += time.time() - t0
    mse_base[0] = np.mean((pred0 - y_te[:, 0]) ** 2)

    for d in range(1, OUTPUT_DIMS):
        reg = TabICLRegressor(n_estimators=4, model_path=CKPT, device='cuda', random_state=42)
        t0 = time.time()
        reg.fit(X_tr, y_tr[:, d])
        pred = reg.predict(X_te)
        t_base += time.time() - t0
        mse_base[d] = np.mean((pred - y_te[:, d]) ** 2)
    print(f'  耗时: {t_base:.1f}s')

    # ── B. FiLM TabICL: 1 个多输出 TabICLRegressor ──
    # num_outputs=12 时 _load_model 走 if 分支：
    #   self.model_ = TabICL(**config, num_outputs=12)
    #   film = MultiDimTargetFiLM(12, 128) 创建并初始化权重
    # ColEmbedding 创建 film（fc_gamma + fc_beta），不创建 y_encoder。
    print('\n=== B. FiLM TabICL: 1 x TabICLRegressor(num_outputs=12) ===')
    torch.cuda.empty_cache()
    reg_film = TabICLRegressor(n_estimators=4, num_outputs=12, model_path=CKPT, device='cuda', random_state=42)
    mse_vec, t_film = evaluate(reg_film, X_tr, X_te, y_tr, y_te)
    mse_film = mse_vec

    # 自检 B 确实用了 FiLM
    col_film = reg_film.model_.col_embedder
    print(f'  [自检] FiLM 模型: has film={hasattr(col_film, "film")}, '
          f'has y_encoder={hasattr(col_film, "y_encoder")}')
    print(f'  耗时: {t_film:.1f}s')

    # ── 结果 ──
    print_results(mse_base, mse_film, t_base, t_film)


if __name__ == '__main__':
    main()
