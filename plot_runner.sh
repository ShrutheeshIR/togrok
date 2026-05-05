#!/bin/bash

plot_all_experiments() {
    local base_dir="experiments_transformer_final_results_batch_size_comparison"
    
    # Iterate through all config subdirectories
    for config_dir in "$base_dir"/*/ ; do
        config_name=$(basename "$config_dir")
        
        # Iterate through all timestamp subdirectories within each config
        for timestamp_dir in "$config_dir"*/ ; do
            # Skip if not a directory (e.g., console.log file)
            if [ ! -d "$timestamp_dir" ]; then
                continue
            fi
            
            timestamp=$(basename "$timestamp_dir")
            exp_path="$base_dir/$config_name/$timestamp"
            
            echo "Running plot_from_tf_events.py for $config_name/$timestamp..."
            python3 plot_from_tf_events.py --output_dir "$exp_path" "$exp_path"
            
            if [ $? -ne 0 ]; then
                echo "Error processing $exp_path"
            fi
        done
    done
}

# Call the function if script is executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    plot_all_experiments
fi
