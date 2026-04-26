#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

batch_sizes=(2048 16834)
dropouts=(0 0.2)
weight_decays=(2e-2 2e-3)
lrs=(1e-3)

format_float() {
    local value="$1"
    echo "${value//./p}"
}

launch_run() {
    local batch_size="$1"
    local dropout="$2"
    local weight_decay="$3"
    local lr="$4"

    local prefix="bs${batch_size}_do$(format_float "$dropout")_wd$(format_float "$weight_decay")_lr$(format_float "$lr")"
    local log_dir="experiments_scripting_adam_transformer/logs_v2/${prefix}"
    mkdir -p "$log_dir"

    nohup python3 grokker_trainer.py \
        --batch_size "$batch_size" \
        --dropout "$dropout" \
        --weight_decay "$weight_decay" \
        --lr "$lr" \
        --prefix "$prefix" \
        --model transformer \
        --optimizer adam \
        >"${log_dir}/console.log" 2>&1 &

    echo "Started ${prefix}"
}

for batch_size in "${batch_sizes[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for lr in "${lrs[@]}"; do
                launch_run "$batch_size" "$dropout" "$weight_decay" "$lr"
            done
        done
    done
done