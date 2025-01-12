#!/bin/bash

PYTHON_SCRIPT_PATH="code2nl/run.py"

# Define an array of tasks
TASKS=('root_cause' 'impact' 'vulnerability_type' 'attack_vector')

# Loop over tasks
for TASK in "${TASKS[@]}"
do
    echo "Running task $TASK"
    python $PYTHON_SCRIPT_PATH --task $TASK \
        --do_train \
        --do_eval \
        --do_test \
        --train_filename "tmp_data_new_v1/$TASK/train.jsonl" \
        --dev_filename "tmp_data_new_v1/$TASK/validation.jsonl" \
        --test_filename "tmp_data_new_v1/$TASK/test.jsonl" \
        --do_lower_case \
        --model_type roberta \
        --model_name_or_path "neulab/codebert-cpp" \
        --output_dir "results/$TASK/neulab-12layers_linevul10" \
        --beam_size 5 \
        --train_batch_size  40 \
        --eval_batch_size 10 \
        --learning_rate "5e-5" \
        --num_train_epochs 100 \
        --num_decoder_layers 12 \
        --linevul_kline 10

    python $PYTHON_SCRIPT_PATH --task $TASK \
        --do_train \
        --do_eval \
        --do_test \
        --train_filename "tmp_data_new_v1/$TASK/train.jsonl" \
        --dev_filename "tmp_data_new_v1/$TASK/validation.jsonl" \
        --test_filename "tmp_data_new_v1/$TASK/test.jsonl" \
        --do_lower_case \
        --model_type roberta \
        --model_name_or_path "neulab/codebert-cpp" \
        --output_dir "results/$TASK/neulab-6layers_linevul10" \
        --beam_size 5 \
        --train_batch_size  40 \
        --eval_batch_size 10 \
        --learning_rate "5e-5" \
        --num_train_epochs 100 \
        --num_decoder_layers 6 \
        --linevul_kline 10
    
    python $PYTHON_SCRIPT_PATH --task $TASK \
        --do_train \
        --do_eval \
        --do_test \
        --train_filename "tmp_data_new_v1/$TASK/train.jsonl" \
        --dev_filename "tmp_data_new_v1/$TASK/validation.jsonl" \
        --test_filename "tmp_data_new_v1/$TASK/test.jsonl" \
        --do_lower_case \
        --model_type roberta \
        --model_name_or_path "microsoft/codebert-base" \
        --output_dir "results/$TASK/base-linevul10" \
        --beam_size 5 \
        --train_batch_size  40 \
        --eval_batch_size 10 \
        --learning_rate "5e-5" \
        --num_train_epochs 100 \
        --num_decoder_layers 6 \
        --linevul_kline 10

done
