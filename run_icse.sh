#!/bin/bash

PYTHON_SCRIPT_PATH="icse/run.py"

# Define an array of tasks
TASKS=('root_cause' 'impact' 'vulnerability_type' 'attack_vector')

# Loop over tasks
for TASK in "${TASKS[@]}"
do

    echo "Running task $TASK"

    if [[ $TASK == "root_cause" ]]; then
        TARGET_LEN=153
    elif [[ $TASK == "impact" ]]; then
        TARGET_LEN=53
    elif [[ $TASK == "vulnerability_type" ]]; then
        TARGET_LEN=53
    elif [[ $TASK == "attack_vector" ]]; then
        TARGET_LEN=146
    fi

    python $PYTHON_SCRIPT_PATH \
        --do_train \
        --do_eval \
        --do_test \
        --do_lower_case \
        --train_filename "tmp_data_new_v1/$TASK/train.jsonl" \
        --dev_filename "tmp_data_new_v1/$TASK/validation.jsonl" \
        --test_filename "tmp_data_new_v1/$TASK/test.jsonl" \
        --output_dir "results/$TASK/icse_linevul10" \
        --max_source_length 512 \
        --max_target_length $TARGET_LEN \
        --beam_size 5 \
        --train_batch_size  40 \
        --eval_batch_size 10 \
        --learning_rate "5e-5" \
        --num_train_epochs 100 \
        --linevul_kline 10

done
