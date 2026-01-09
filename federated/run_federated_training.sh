#!/bin/bash

# FEDERATED LEARNING LAUNCHER (NVFlare Sim)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# --- 1. FEDERATED CONFIGURATION ---
ROUNDS=5            # Number of times server aggregates weights
LOCAL_STEPS=50      # Steps per client per round
EXP_NAME="fed_dms_experiment_1"
SIM_GPUS="0"

# --- 2. MODEL HYPERPARAMETERS (Must match Centralized) ---
LR="1e-3"
BATCH_SIZE="8"
TASK_TYPE="regression"
TARGET_SIZE="1"
LABEL_COL="DMS_score"

# Set to "--encoder-frozen" to freeze, or empty string "" to unfreeze
FROZEN_FLAG="--encoder-frozen" 

echo "Starting Federated Simulation..."
echo "   - Rounds: $ROUNDS"
echo "   - Local Steps: $LOCAL_STEPS"
echo "   - Task: $TASK_TYPE"
echo "   - Frozen: $FROZEN_FLAG"

# --- 3. EXECUTION ---
# We pass these new args to run_sim_scl.py
python3 run_sim_scl.py \
    --num_rounds $ROUNDS \
    --local_steps $LOCAL_STEPS \
    --exp_name $EXP_NAME \
    --sim_gpus $SIM_GPUS \
    --model "650m" \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --task_type $TASK_TYPE \
    --label_column $LABEL_COL \
    --target_size $TARGET_SIZE \
    $FROZEN_FLAG

echo "Simulation Finished."