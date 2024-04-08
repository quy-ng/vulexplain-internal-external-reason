#!/bin/bash

PYTHON_SCRIPT_PATH="t5p_seq2seq.py"

# Define an array of tasks
TASKS=('root_cause' 'impact' 'vulnerability_type' 'attack_vector')

# Loop over tasks
for TASK in "${TASKS[@]}"
do
  echo "Running task $TASK"
  python $PYTHON_SCRIPT_PATH -t $TASK -b 50 -w 220 --train_fp16 --prexfix "new_v1"
done
