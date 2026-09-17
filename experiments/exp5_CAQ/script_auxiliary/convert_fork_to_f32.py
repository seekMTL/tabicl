"""将 fork_samples.npz（float64 + zip 压缩）转换为 fork_samples_f32.npz（float32 + 无压缩）。

背景：npz 的 zip 解压是单线程 CPU 密集，Humanoid 两环境 18GB 压缩数据
解压+转换要 ~35 分钟。转为无压缩 float32 后，训练脚本加载零解压，
加载时间降到 ~1 分钟。

用法（新服务器上执行）:
  python convert_fork_to_f32.py --envs "Humanoid-v4,HumanoidStandup-v4"
输出: 同目录下 fork_samples_f32.npz（保留原 npz 不动）
内存峰值 ~50GB（float64 全量 + float32 副本），需大内存机器。
"""
import argparse
import time
from pathlib import Path

import numpy as np

DEFAULT_BASE = "/opt/data/private/xdcode/envData/Counterfactual/mujoco_v4_260906"


def convert_one(base: str, env: str):
    src = Path(base) / env / "fork_samples.npz"
    dst = Path(base) / env / "fork_samples_f32.npz"
    if dst.exists():
        print(f"  [{env}] {dst.name} 已存在，跳过（如需重转请先删除）", flush=True)
        return
    print(f"  [{env}] 解压 {src.name}（{src.stat().st_size / 1e9:.1f}GB）...", flush=True)
    t0 = time.time()
    d = np.load(src)
    print(f"    解压完成 {time.time() - t0:.0f}s，转换 float32 并写入无压缩 npz...", flush=True)
    np.savez(
        dst,
        states=d["states"].astype(np.float32),
        actions=d["actions"].astype(np.float32),
        next_states=d["next_states"].astype(np.float32),
        rewards=d["rewards"].astype(np.float32),
    )
    print(f"    ✓ 完成 {time.time() - t0:.0f}s → {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="fork_samples.npz → float32 无压缩 npz")
    parser.add_argument("--envs", type=str, default="Humanoid-v4,HumanoidStandup-v4",
                        help="环境列表（逗号分隔）")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE,
                        help="Counterfactual 数据根目录")
    args = parser.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    for env in envs:
        convert_one(args.base, env)


if __name__ == "__main__":
    main()
