#!/usr/bin/env bash
# Exp5-style within-world row split; two 10k pilots on GPU 1.
set -euo pipefail
cd /home/lizitao/project/tabicl
experiment_dir=/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS
python_bin=/home/lizitao/software/miniconda3/envs/tabicl/bin/python
for model in caq tabicl; do
    run_name="${model}-A4-seed42-hr0-row-flat-10k"
    if [[ -e "$experiment_dir/checkpoints/paper-main/$run_name" || -e "$experiment_dir/Logs/$run_name.log" ]]; then
        echo "Refusing to overwrite $run_name" >&2
        exit 1
    fi
done
for model in caq tabicl; do
    run_name="${model}-A4-seed42-hr0-row-flat-10k"
    "$python_bin" -u "$experiment_dir/train_industrial.py" \
        --manifest /home/lizitao/project/Industrial_synthetic_data/data/pilot-m3/manifest.json \
        --out "$experiment_dir/checkpoints/paper-main/$run_name" \
        --device cuda:1 --model-kind "$model" --prior A4 \
        --steps 10000 --schedule-steps 30000 --context-size 960 --query-size 256 \
        --targets-per-task 1 --replay-ratio 0.2 --lr 1e-4 --lr-schedule cosine \
        --eval-interval 1000 --seed 42 --industrial-sampling-protocol row_flat \
        --target-selector --target-anchor --absolute-state \
        --persistence-residual --persistence-residual-scope next_observation \
        --external-context-mode observed --target-floor-profile event_v2 \
        --history-ratio 0 --skip-paired-eval --threads 4 \
        > "$experiment_dir/Logs/$run_name.log" 2>&1
done
