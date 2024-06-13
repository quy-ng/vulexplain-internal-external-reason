#!/bin/bash

PYTHON_SCRIPT_PATH="t5p_seq2seq.py"

# Define an array of tasks
TASKS=('root_cause' 'impact' 'vulnerability_type' 'attack_vector')
# Create an array of percentages
PERCENTS=(5 10 15 20 25 30 40 50 60 70 80 90)

# Loop over tasks
for TASK in "${TASKS[@]}"
do
  # echo "Running task $TASK"
  for PERCENT in "${PERCENTS[@]}"
  do
    echo "Running task $TASK with $PERCENT% of data"
    prefix="new_${TASK}_${PERCENT}_linevul"
    python $PYTHON_SCRIPT_PATH -t $TASK -b 50 -w 220 --train_fp16 --prexfix $prefix  --percent --linevul_kline $PERCENT
  done
done
