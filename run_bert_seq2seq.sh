#!/bin/bash

PYTHON_SCRIPT_PATH="code2nl/run.py"

# Define an array of tasks
TASKS=('root_cause' 'impact' 'vulnerability_type' 'attack_vector')

# Loop over tasks
for TASK in "${TASKS[@]}"
do
    echo "Running task $TASK"
    python $PYTHON_SCRIPT_PATH --task root_cause \
        --do_train \
        --do_eval \
        --do_test \
        --train_filename "tmp_data/$TASK/train.jsonl" \
        --dev_filename "tmp_data/$TASK/valid.jsonl" \
        --test_filename "tmp_data/$TASK/test.jsonl" \
        --do_lower_case \
        --model_type roberta \
        --model_name_or_path "neulab/codebert-cpp" \
        --output_dir "results/$TASK/neulab-12layers" \
        --beam_size 5 \
        --train_batch_size  40 \
        --eval_batch_size 10 \
        --learning_rate "5e-5" \
        --num_train_epochs 100 \
        --num_decoder_layers 12

done
