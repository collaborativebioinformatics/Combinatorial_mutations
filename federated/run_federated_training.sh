#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------
# CENTRALIZED-ALIGNED CONFIG
# ---------------------------
TOTAL_STEPS="8000"
ROUNDS="160"
LOCAL_STEPS="50"

EXP_NAME="fed_dms_scl_esm2_650m_centralized_aligned"
SIM_GPUS="0"

# ---------------------------
# MODEL HYPERPARAMETERS
# ---------------------------
LR="1e-4"
BATCH_SIZE="8"
TASK_TYPE="regression"
TARGET_SIZE="1"
LABEL_COL="DMS_score"

# Correct flag name
FROZEN_FLAG="--encoder_frozen"

# Validation
LIMIT_VAL_BATCHES="100"

echo "🚀 Starting Federated Simulation..."
echo "   - Rounds:                ${ROUNDS}"
echo "   - Local steps:           ${LOCAL_STEPS}"
echo "   - Total steps:           $((ROUNDS * LOCAL_STEPS))"
echo "   - LR:                    ${LR}"
echo "   - Batch size:            ${BATCH_SIZE}"
echo "   - Encoder frozen:        yes"
echo "   - limit_val_batches:     ${LIMIT_VAL_BATCHES}"
echo "   - GPUs:                  ${SIM_GPUS}"
echo "   - Experiment:            ${EXP_NAME}"
echo ""

python3 run_sim_scl.py \
  --num_rounds "${ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --exp_name "${EXP_NAME}" \
  --sim_gpus "${SIM_GPUS}" \
  --model "650m" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --task_type "${TASK_TYPE}" \
  --label_column "${LABEL_COL}" \
  --target_size "${TARGET_SIZE}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  ${FROZEN_FLAG}

echo "✅ Federated simulation finished."
