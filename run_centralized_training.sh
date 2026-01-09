#!/bin/bash

# Configuration
LR="1e-4"
STEPS="1000"           
BATCH_SIZE="8"

# 1. Pointing to local results (since the mount failed)
RESULT_DIR="./results/run_centralized_human"

# The Bionemo Script Path (Inside Docker)
TRAIN_SCRIPT="/workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/finetune_esm2.py"

# Data Paths (Inside Docker)
TRAIN_DATA="/workspace/project/data/splits/human/train.csv"
VAL_DATA="/workspace/project/data/splits/human/val.csv"
CHECKPOINT="/workspace/project/esm2_650m.nemo"

echo "🚀 Starting Centralized Training..."
echo "   - Steps: $STEPS"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Output: $RESULT_DIR"

# Run command
python $TRAIN_SCRIPT \
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VAL_DATA \
    --restore-from-checkpoint-path $CHECKPOINT \
    --task-type regression \
    --mlp-target-size 1 \
    --label-column DMS_score \
    --lr $LR \
    --micro-batch-size $BATCH_SIZE \
    --num-steps $STEPS \
    --num-gpus 1 \
    --result-dir $RESULT_DIR \
    --val-check-interval 50 \
    --log-every-n-steps 10 \
    --save-top-k 1 \
    --metric-to-monitor-for-checkpoints val_loss \
    --save-last-checkpoint \
    --avoid-ckpt-async-save \
    --limit-val-batches 1.0 \
    --encoder-frozen # Freeze the ESM2 Encoder or comment if you want to fine-tune the whole model

echo "✅ Run Complete! Logs are in $RESULT_DIR"