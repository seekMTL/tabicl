"""检查 19 个扰动环境的反事实数据质量（2026-09-06 采集版）。

对每个环境输出: shape、行数、NaN/inf 计数、reward 与 delta 的分布统计。
"""
import numpy as np

BASE = "/home/lizitao/project/other/sample_collect/Counterfactual/mujoco_v4_260906"
ENVS = [
    "HopperGravity-g-1", "HopperGravity-g-0.5", "HopperGravity-g0",
    "HopperGravity-g0.5", "HopperGravity-g1",
    "Walker2dFriction-f0.3", "Walker2dFriction-f1", "Walker2dFriction-f3",
    "HalfCheetahSlope-s-10", "HalfCheetahSlope-s0", "HalfCheetahSlope-s10",
    "AntJoints-af0", "AntJoints-af1", "AntJoints-af2", "AntJoints-af3",
    "AntJoints-af4", "AntJoints-af5", "AntJoints-af6", "AntJoints-af7",
]

print(f"{'env':<24} {'rows':>9} {'state':>6} {'act':>4} "
      f"{'r_mean':>8} {'r_std':>8} {'d_mean':>8} {'d_std':>8} {'nan':>6}")
print("-" * 96)
for env in ENVS:
    try:
        d = np.load(f"{BASE}/{env}/fork_samples.npz", mmap_mode="r")
        keys = list(d.keys())
        if not {"states", "actions", "next_states", "rewards"} <= set(keys):
            print(f"{env:<24} 键不完整: {keys}")
            continue
        S, A, NS, R = d["states"], d["actions"], d["next_states"], d["rewards"]
        n = S.shape[0]
        delta = NS - S
        # 按块统计 NaN（大文件全量 float64 计算太慢）
        def nan_cnt(arr, block=200_000):
            c = 0
            for i in range(0, arr.shape[0], block):
                c += int(np.isnan(arr[i:i+block]).sum())
            return c
        n_nan = nan_cnt(R) + nan_cnt(delta)
        print(f"{env:<24} {n:>9,} {S.shape[1]:>6} {A.shape[1]:>4} "
              f"{float(R.mean()):>8.3f} {float(R.std()):>8.3f} "
              f"{float(delta.mean()):>8.4f} {float(delta.std()):>8.4f} {n_nan:>6}")
        del d, S, A, NS, R
    except Exception as e:
        print(f"{env:<24} 加载失败: {e}")
