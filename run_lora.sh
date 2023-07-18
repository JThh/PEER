#!/bin/bash

declare -a models=("ESM-2-8M" "ESM-2-35M" "ESM-2-150M" "ESM-2-650M" "ESM-2-3B" "ESM-2-15B") # "ESM-2-650M" "ESM-2-3B"
# output_dir="~/code/scratch/torchprotein_output/"
task="subloc"
log_dir="/home/yz979/code/scratch/torchprotein_logs/"
yaml_file="config/single_task/ESM/${task}_ESM.yaml"

for model in "${models[@]}"
do
    if [[ "$model" == "ESM-2-8M" || "$model" == "ESM-2-35M" || "$model" == "ESM-2-150M" || "$model" == "ESM-2-650M" ]]; then
        batch_size=8
    else
        batch_size=1
    fi

    log_file="${log_dir}lora_task_${task}_run_${model}_bs${batch_size}.log"

    echo "Running with model: $model, batch size: $batch_size"

    # Create the log file
    touch "$log_file"

    # Modify the YAML file with the desired model and batch size
    awk -v model="$model" -v batch_size="$batch_size" '/model:/ {sub(/ESM-.*/, model)} /batch_size:/ {sub(/[0-9]+/, batch_size)} /path:/ {sub(/~\/scratch/, "~\/code\/scratch")} 1' "$yaml_file" > temp_lora.yaml # && mv temp.yaml "$yaml_file"

    # Execute the Python command
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master-port=29500 script/run_single.py -c temp_lora.yaml --seed 0 > "$log_file" 2>&1

    rm temp_lora.yaml

    echo "Finished running. Log saved to $log_file"
    echo ""
done
