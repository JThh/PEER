#!/bin/bash

declare -a models=( "ESM-1b") 
declare -a tasks=("subloc" "solubility" "contact" "yeast" "aav") 
# output_dir="~/code/scratch/torchprotein_output/"
# task="fluorescence"
log_dir="/home/yz979/code/scratch/torchprotein_logs/"

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        yaml_file="config/single_task/ESM/${task}_ESM.yaml"
        batch_size=1

        log_file="${log_dir}task_${task}_run_${model}_bs${batch_size}.log"

        echo "Running with task: $task, model: $model, batch size: $batch_size"

        # Create the log file
        touch "$log_file"

        # Modify the YAML file with the desired model and batch size
        awk -v model="$model" -v batch_size="$batch_size" '
        /model:/ { sub(/ESM-.*/, model) }
        /batch_size:/ { sub(/[0-9]+/, batch_size) }
        /path:/ {sub(/~\/scratch/, "~\/code\/scratch")}
        1' "$yaml_file" > temp_lora.yaml

        # /gpus:/ { sub(/\[0, 1, 2, 3\]/, "[0, 1, 2, 3, 4, 5, 6]") }

        # Execute the Python comman
        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master-port=29500 script/run_single.py -c temp.yaml --seed 0 > "$log_file" 2>&1
        
        rm temp_lora.yaml

        echo "Finished running. Log saved to $log_file"
        echo ""
    done
done
