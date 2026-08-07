#!/bin/bash
# 训练进程存活监控 — 定期记录时间戳，用于断电后计算训练耗时。
#
# 用法:
#   nohup bash monitor_training.sh 521882 checkpoints/bkup4/heartbeat &
#
# 断电后恢复:
#   1. cat checkpoints/bkup4/heartbeat        # 得到最后的 Unix 时间戳
#   2. cat checkpoints/bkup4/session_t0.json  # 得到 t0_epoch
#   3. 前次会话耗时 = heartbeat - t0_epoch
#   4. 写入 progress.json 的 elapsed 字段，然后 --resume 恢复

PID=${1:?Usage: $0 <pid> <output> [interval]}
OUTPUT=${2:?Usage: $0 <pid> <output> [interval]}
INTERVAL=${3:-60}

mkdir -p "$(dirname "$OUTPUT")"

echo "[$(date '+%F %T')] 开始监听 PID=$PID  间隔=${INTERVAL}s  输出=$OUTPUT"

N=0
while [ -d "/proc/$PID" ]; do
    N=$((N + 1))
    echo "timestamp_epoch $(date +%s)" > "$OUTPUT"
    echo "[$(date '+%F %T')] 第 ${N} 次记录  PID=$PID 运行中，继续监听..."
    sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] PID=$PID 已退出，累计记录 ${N} 次，监听结束"
