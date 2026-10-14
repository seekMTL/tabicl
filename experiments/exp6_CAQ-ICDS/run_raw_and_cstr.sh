#!/usr/bin/env bash
set -euo pipefail
cd /home/lizitao/project/tabicl
experiment_dir=/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS
python_bin=/home/lizitao/software/miniconda3/envs/tabicl/bin/python
for suite in icds cstr; do
    if [[ -e "$experiment_dir/checkpoints/raw-input-cstr-audit/$suite" || -e "$experiment_dir/Logs/raw-input-cstr-$suite.log" ]]; then
        echo "Refusing to overwrite existing $suite evaluation" >&2
        exit 1
    fi
done
for suite in icds cstr; do
    "$python_bin" -u "$experiment_dir/eval_raw_and_cstr.py" \
        --suite "$suite" --device cuda:1 --contexts 128 512 960 \
        --support-seeds 42 43 44 --query-size 256 --estimators 8 \
        --out "$experiment_dir/checkpoints/raw-input-cstr-audit/$suite" \
        > "$experiment_dir/Logs/raw-input-cstr-$suite.log" 2>&1
done
