#!/bin/bash

# COnfiguration Parameters
LR="1e-4"
STEPS="100"
BATCH_SIZE="8"
RESULT_DIR="./results/run_1"

# POINT TO THE CORRECT SCRIPT LOCATION
TRAIN_SCRIPT="/workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py"

echo "Starting Training with LR=${LR}, Steps=${STEPS}..."

# Command to run the training
python $TRAIN_SCRIPT \
    --train-data-path /workspace/project/data/splits/human/train.csv \
    --valid-data-path /workspace/project/data/splits/human/val.csv \
    --restore-from-checkpoint-path /workspace/project/esm2_650m.nemo \
    --task-type regression \
    --mlp-target-size 1 \
    --label-column target \
    --lr $LR \
    --micro-batch-size $BATCH_SIZE \
    --num-steps $STEPS \
    --num-gpus 1 \
    --result-dir $RESULT_DIR \
    --save-last-checkpoint

# Before running this script, ensure that it is an executable: "chmod +x run_centralized_training.sh"