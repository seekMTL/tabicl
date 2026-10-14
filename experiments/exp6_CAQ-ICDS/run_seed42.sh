#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS
ICDS_DIR=/home/lizitao/project/Industrial_synthetic_data
PYTHON=/home/lizitao/software/miniconda3/envs/tabicl/bin/python
MANIFEST="$ICDS_DIR/data/pilot-m3/manifest.json"

mkdir -p "$EXP_DIR/Logs/paper-main" "$EXP_DIR/checkpoints/paper-main"

run_one() {
  local model_kind=$1
  local label=$2
  local gpu=$3
  "$PYTHON" "$EXP_DIR/train_industrial.py" \
    --manifest "$MANIFEST" \
    --out "$EXP_DIR/checkpoints/paper-main/${label}-A4-seed42-hr0" \
    --device "cuda:${gpu}" \
    --model-kind "$model_kind" \
    --prior A4 \
    --steps 30000 \
    --context-size 960 \
    --query-size 256 \
    --targets-per-task 1 \
    --replay-ratio 0.2 \
    --lr 1e-4 \
    --lr-schedule cosine \
    --eval-interval 1000 \
    --seed 42 \
    --paired-query \
    --target-selector \
    --target-anchor \
    --absolute-state \
    --persistence-residual \
    --persistence-residual-scope next_observation \
    --external-context-mode observed \
    --target-floor-profile event_v2 \
    --history-ratio 0 \
    --history-length 8 \
    --history-families hybrid_line \
    --sensor-noise-multiplier 10 \
    --skip-paired-eval \
    --threads 4 >"$EXP_DIR/Logs/paper-main/${label}-A4-seed42-hr0.log" 2>&1
}

# hr0: 只移除 hybrid_line history 混合，其余训练协议保持不变。
# CAQ 用 state-only（action 只走 causal 通道），不再是 caq-concat。
# 两模型各占一张卡，互不抢显存（train_industrial.py 已 set_device 修复
# InferenceManager 默认设备坑，评估不会 device mismatch）
run_one caq caq 1 &
caq_pid=$!
run_one tabicl tabicl 0 &
tabicl_pid=$!
wait "$caq_pid"
wait "$tabicl_pid"
