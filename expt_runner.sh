# #!/usr/bin/env bash

# set -euo pipefail

# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$script_dir"

# batch_sizes=(2048 16834 128)
# lrs=(1e-3)
# dropouts=(0 0.2)
# weight_decays=(5e-2 5e-3)
# optimizers=(adam second_order_adam)
# weight_norm_ratios=(-1.0 0.5 0.75)

# format_float() {
#     local value="$1"
#     echo "${value//./p}"
# }

# launch_run() {
#     local batch_size="$1"
#     local dropout="$2"
#     local weight_decay="$3"
#     local lr="$4"
#     local optimizer="$5"
#     local weight_norm_ratio="$6"

#     local prefix="bs${batch_size}_do$(format_float "$dropout")_wd$(format_float "$weight_decay")_lr$(format_float "$lr")_opt${optimizer}_wnr$(format_float "$weight_norm_ratio")"
#     local base_log_dir="experiments_adam_second_order_v2"
#     mkdir -p "${base_log_dir}/${prefix}"

#     local cmd="python3 grokker_trainer.py \
#         --batch_size $batch_size \
#         --dropout $dropout \
#         --weight_decay $weight_decay \
#         --lr $lr \
#         --prefix $prefix \
#         --model transformer \
#         --optimizer $optimizer \
#         --log_dir $base_log_dir"

#     # Add weight norm params if weight_norm_ratio != -1
#     if [[ "$weight_norm_ratio" != "-1.0" && "$weight_norm_ratio" != "-1" ]]; then
#         cmd="$cmd --do_weight_norm --weight_norm_ratio $weight_norm_ratio"
#     fi


#     cmd="$cmd >${base_log_dir}/${prefix}/console.log 2>&1"

#     # print command
#     echo "$cmd"
#     eval "$cmd"

#     echo "Completed ${prefix}"
# }

# for batch_size in "${batch_sizes[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for dropout in "${dropouts[@]}"; do
#             for weight_decay in "${weight_decays[@]}"; do
#                 for optimizer in "${optimizers[@]}"; do
#                     for weight_norm_ratio in "${weight_norm_ratios[@]}"; do
#                         launch_run "$batch_size" "$dropout" "$weight_decay" "$lr" "$optimizer" "$weight_norm_ratio"
#                     done
#                 done
#             done
#         done
#     done
# done


#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

batch_sizes=(512 64 128 2048)
lrs=(1e-3)
dropouts=(0)
weight_decays=(5e-2)
optimizers=(adam second_order_adam)
weight_norm_ratios=(-1.0 0.75)

format_float() {
    local value="$1"
    echo "${value//./p}"
}

launch_run() {
    local batch_size="$1"
    local dropout="$2"
    local weight_decay="$3"
    local lr="$4"
    local optimizer="$5"
    local weight_norm_ratio="$6"

    local prefix="bs${batch_size}_do$(format_float "$dropout")_wd$(format_float "$weight_decay")_lr$(format_float "$lr")_opt${optimizer}_wnr$(format_float "$weight_norm_ratio")"
    local base_log_dir="experiments_transformer_final_results_batch_size_comparison"
    mkdir -p "${base_log_dir}/${prefix}"

    local cmd="python3 grokker_trainer.py \
        --batch_size $batch_size \
        --dropout $dropout \
        --weight_decay $weight_decay \
        --lr $lr \
        --prefix $prefix \
        --model transformer \
        --optimizer $optimizer \
        --log_dir $base_log_dir"

    # Add weight norm params if weight_norm_ratio != -1
    if [[ "$weight_norm_ratio" != "-1.0" && "$weight_norm_ratio" != "-1" ]]; then
        cmd="$cmd --do_weight_norm --weight_norm_ratio $weight_norm_ratio"
    fi


    cmd="$cmd >${base_log_dir}/${prefix}/console.log 2>&1"

    # print command
    echo "$cmd"
    eval "$cmd"

    echo "Completed ${prefix}"
}

for lr in "${lrs[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for optimizer in "${optimizers[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for weight_norm_ratio in "${weight_norm_ratios[@]}"; do
                        launch_run "$batch_size" "$dropout" "$weight_decay" "$lr" "$optimizer" "$weight_norm_ratio"
                    done
                done
            done
        done
    done
done