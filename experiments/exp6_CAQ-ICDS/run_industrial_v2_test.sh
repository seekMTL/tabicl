#!/usr/bin/env bash
set -euo pipefail
cd /home/lizitao/project/tabicl
experiment_dir=/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS
python_bin=/home/lizitao/software/miniconda3/envs/tabicl/bin/python
phase=${1:-formal}
[[ "$phase" == formal || "$phase" == preflight ]]
run_name=meta-test-seed42-context960-query256-official8
extra=()
if [[ "$phase" == preflight ]]; then
    run_name=meta-test-official-preflight
    extra=(--preflight-only)
fi
out="$experiment_dir/checkpoints/industrial-v2/$run_name"
log="$experiment_dir/Logs/industrial-v2/$run_name.log"
if [[ -e "$out" || -e "$log" ]]; then
    echo "Refusing to overwrite $out or $log" >&2
    exit 1
fi
command=("$python_bin" -u "$experiment_dir/test_industrial_v2.py"
    --manifest /home/lizitao/project/Industrial_synthetic_data/data/industrial-v2/manifest.json
    --caq-checkpoint "$experiment_dir/checkpoints/industrial-v2/caq-A4-seed42-hr0/best.pt"
    --ft-checkpoint "$experiment_dir/checkpoints/industrial-v2/tabicl-A4-seed42-hr0/best.pt"
    --context-size 960 --query-size 256 --official-estimators 8
    --device cuda:1 --cuda-memory-fraction 0.55 --out "$out" "${extra[@]}")
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "$log"
