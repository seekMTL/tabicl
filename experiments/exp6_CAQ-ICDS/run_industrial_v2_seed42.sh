#!/usr/bin/env bash
set -euo pipefail
cd /home/lizitao/project/tabicl
experiment_dir=/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS
python_bin=/home/lizitao/software/miniconda3/envs/tabicl/bin/python
manifest=/home/lizitao/project/Industrial_synthetic_data/data/industrial-v2/manifest.json
checkpoint=/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt
phase=${1:?Use smoke or formal}
model_kind=${2:?Use caq or tabicl}
[[ "$phase" == smoke || "$phase" == formal ]]
[[ "$model_kind" == caq || "$model_kind" == tabicl ]]
steps=30000
eval_interval=1000
run_name="${model_kind}-A4-seed42-hr0"
if [[ "$phase" == smoke ]]; then
    steps=200
    eval_interval=100
    run_name="smoke-${model_kind}-seed42"
fi
out="$experiment_dir/checkpoints/industrial-v2/$run_name"
log="$experiment_dir/Logs/industrial-v2/$run_name.log"
if [[ -e "$out" || -e "$log" ]]; then
    echo "Refusing to overwrite run: $out or $log" >&2
    exit 1
fi
"$python_bin" "$experiment_dir/check_industrial_v2_ready.py" --phase "$phase"
mkdir -p "$experiment_dir/Logs/industrial-v2" "$experiment_dir/checkpoints/industrial-v2"
command=("$python_bin" -u "$experiment_dir/train_industrial.py"
    --manifest "$manifest" --checkpoint "$checkpoint" --out "$out"
    --device cuda:1 --cuda-memory-fraction 0.45 --model-kind "$model_kind" --prior A4
    --steps "$steps" --schedule-steps 30000 --context-size 960 --query-size 256
    --targets-per-task 1 --replay-ratio 0.2 --lr 1e-4 --lr-schedule cosine
    --eval-interval "$eval_interval" --seed 42 --paired-query
    --industrial-sampling-protocol factual --information-track full_state
    --target-selector --target-anchor --absolute-state --input-scale-profile industrial_v2
    --persistence-residual --persistence-residual-scope next_observation
    --external-context-mode observed --target-floor-profile event_v2
    --history-ratio 0 --skip-paired-eval --threads 4)
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "$log"
if [[ "$phase" == formal ]]; then
    echo "Training finished successfully: $model_kind"
fi
