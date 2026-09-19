#!/bin/bash
# =============================================================================
# CBD-DFB Pipeline for Deepseek Dataset
# =============================================================================
# This script runs the full pipeline:
#   Stage 1: Extract CBD-DFB basis (generalized eigen subspace from gradients)
#   Stage 2: Train A1 model (unlearning via gradient descent + KL)
#   Stage 3: Compute sym-KL threshold (200 samples from each test set)
#   Stage 4: Score remaining samples & count above/below threshold
# =============================================================================

set -e  # Exit on error

# Ensure Python can find the local 'uld' module and disable output buffering
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# --------------- Configuration ---------------
# Model
ASSIST_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Data paths (relative to project root)
DEEPSEEK_DATA="../Data-Collection/deepseek"
FORGET_DATA="${DEEPSEEK_DATA}/D_forget.json"

# Basis extraction params (use all D_forget.json samples)
MAX_FORGET=99999
MAX_RETAIN=99999
MAX_LEN=512
TOP_K=192
SEED=42

# Output directories
BASIS_DIR="artifacts/basis_cbd_dfb/deepseek"
TRAIN_OUTPUT_DIR="artifacts/outputs_trained_models/cbd_dfb_tinyllama_deepseek"

# Inference params
THRESHOLD_SAMPLES=200  # Calibration samples per test set
SCORE_SAMPLES=500      # Cap on negative test set
BATCH_SIZE=8

# --------------- Derived paths ---------------
BASIS_FILE="${BASIS_DIR}/cbd_dfb_basis_deepseek_forget_vs_deepseek_retain.pkl"

# =============================================================================
# Stage 1: Extract CBD-DFB Basis
# =============================================================================
echo ""
echo "============================================================"
echo "  Stage 1: Extract CBD-DFB Basis"
echo "============================================================"
echo ""

python scripts/extract_cbd_dfb_basis.py \
    --base_model_name "${ASSIST_MODEL}" \
    --data_path "${FORGET_DATA}" \
    --train_ratio 1.0 \
    --max_forget ${MAX_FORGET} \
    --max_retain ${MAX_RETAIN} \
    --max_len ${MAX_LEN} \
    --top_k ${TOP_K} \
    --seed ${SEED} \
    --batch_size 8 \
    --output_dir "${BASIS_DIR}"

echo ""
echo "[Stage 1] Basis saved to: ${BASIS_FILE}"
echo ""

# =============================================================================
# Stage 2: Train A1 Model
# =============================================================================
echo ""
echo "============================================================"
echo "  Stage 2: Train A1 (Unlearning)"
echo "============================================================"
echo ""

DISABLE_INTERNAL_EVAL=1 python scripts/hf_forget_train.py \
    --config-name cbd_dfb_tinyllama_deepseek \
    enable_cbd_dfb=true \
    cbd_dfb_basis_path="${BASIS_FILE}" \
    seed=${SEED} \
    lora_seed=${SEED} \
    trainer.batch_size=16 \
    trainer.gradient_accumulation_steps=1 \
    trainer.max_epochs=3 \
    OUTPUTMODELDIR="${TRAIN_OUTPUT_DIR}"

# Find the latest checkpoint (may be nested in subdirectories)
CHECKPOINT=$(find "${TRAIN_OUTPUT_DIR}" -type d -name "checkpoint-*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "${CHECKPOINT}" ]; then
    echo "[ERROR] No checkpoint found in ${TRAIN_OUTPUT_DIR}"
    exit 1
fi
echo ""
echo "[Stage 2] Checkpoint: ${CHECKPOINT}"
echo ""

# =============================================================================
# Stage 3: Infer + Score (sym-KL routing)
# =============================================================================

# Find the latest checkpoint (works whether Stage 2 just ran or was skipped)
CHECKPOINT=$(find "${TRAIN_OUTPUT_DIR}" -type d -name "checkpoint-*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "${CHECKPOINT}" ]; then
    echo "[ERROR] No checkpoint found in ${TRAIN_OUTPUT_DIR}"
    exit 1
fi
echo ""
echo "  Using checkpoint: ${CHECKPOINT}"

echo ""
echo "============================================================"
echo "  Stage 3: Sym-KL Routing Evaluation"
echo "============================================================"
echo ""

EVAL_DIR="artifacts/eval_outputs/deepseek"
mkdir -p "${EVAL_DIR}"

python scripts/infer_deepseek.py \
    --original_model_path "${ASSIST_MODEL}" \
    --finetuned_model_path "${CHECKPOINT}" \
    --test_dep_path "${DEEPSEEK_DATA}/D_test_U_dep.json" \
    --test_nondep_path "${DEEPSEEK_DATA}/D_test_U_nondep.json" \
    --output_dir "${EVAL_DIR}" \
    --calib_dep_n ${THRESHOLD_SAMPLES} \
    --calib_nondep_n ${THRESHOLD_SAMPLES} \
    --test_nondep_n ${SCORE_SAMPLES} \
    --max_len ${MAX_LEN} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --optimize accuracy

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Basis:     ${BASIS_FILE}"
echo "  Model:     ${CHECKPOINT}"
echo "  Eval:      ${EVAL_DIR}/routing_statistics.json"
echo ""

