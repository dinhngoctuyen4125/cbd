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

# Ensure Python can find the local 'uld' module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# --------------- Configuration ---------------
# Model
ASSIST_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Data paths (relative to project root)
DEEPSEEK_DATA="../Data-Collection/deepseek"
FORGET_DATA="${DEEPSEEK_DATA}/D_forget.json"

# Basis extraction params
MAX_FORGET=400
MAX_RETAIN=400
MAX_LEN=512
TOP_K=192
SEED=42

# Output directories
BASIS_DIR="artifacts/basis_cbd_dfb/deepseek"
TRAIN_OUTPUT_DIR="artifacts/outputs_trained_models/cbd_dfb_tinyllama_deepseek"
THRESHOLD_DIR="artifacts/ce_deepseek/threshold"
SCORE_DIR="artifacts/ce_deepseek/scoring"

# Threshold/Scoring params
THRESHOLD_SAMPLES=200  # First 200 samples for threshold
SCORE_SAMPLES=500      # Up to 500 remaining samples for scoring
MAX_NEW_TOKENS=20
BATCH_SIZE=4

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
    --deepseek_data_path "${FORGET_DATA}" \
    --max_forget ${MAX_FORGET} \
    --max_retain ${MAX_RETAIN} \
    --max_len ${MAX_LEN} \
    --top_k ${TOP_K} \
    --seed ${SEED} \
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

python scripts/hf_forget_train.py \
    --config-name cbd_dfb_tinyllama_deepseek \
    enable_cbd_dfb=true \
    cbd_dfb_basis_path="${BASIS_FILE}" \
    seed=${SEED} \
    lora_seed=${SEED} \
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
# Stage 3: Compute Threshold
# =============================================================================
echo ""
echo "============================================================"
echo "  Stage 3: Compute Sym-KL Threshold"
echo "============================================================"
echo ""

mkdir -p "${THRESHOLD_DIR}"

# 3a: Sym-KL on first 200 samples of D_test_U_dep (forget-like group)
echo "[Stage 3a] Scoring D_test_U_dep (first ${THRESHOLD_SAMPLES} samples)..."
python scripts/assis_tinyllama_test_path.py \
    --model_path "${CHECKPOINT}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${DEEPSEEK_DATA}" \
    --dataset_split "D_test_U_dep" \
    --question_key "probing input" \
    --answer_key "y_neg" \
    --max_samples ${THRESHOLD_SAMPLES} \
    --raw_prompt \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --output_file "tinyllama_comparison_results.json" \
    --output_dir "${THRESHOLD_DIR}"

# 3b: Sym-KL on first 200 samples of D_test_U_nondep (retain-like group)
echo "[Stage 3b] Scoring D_test_U_nondep (first ${THRESHOLD_SAMPLES} samples)..."
python scripts/assis_tinyllama_test_path.py \
    --model_path "${CHECKPOINT}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${DEEPSEEK_DATA}" \
    --dataset_split "D_test_U_nondep" \
    --question_key "probing input" \
    --max_samples ${THRESHOLD_SAMPLES} \
    --raw_prompt \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --output_file "tinyllama_comparison_results.json" \
    --output_dir "${THRESHOLD_DIR}"

# 3c: Find optimal threshold
echo "[Stage 3c] Finding optimal threshold..."
python scripts/analyze_cross_entropy.py \
    --data-dir "${THRESHOLD_DIR}" \
    --forget-split "D_test_U_dep" \
    --retain-split "D_test_U_nondep" \
    --optimize accuracy

# =============================================================================
# Stage 4: Score Remaining Samples
# =============================================================================
echo ""
echo "============================================================"
echo "  Stage 4: Score Remaining Samples"
echo "============================================================"
echo ""

mkdir -p "${SCORE_DIR}"

# 4a: Score remaining D_test_U_dep (skip first 200)
echo "[Stage 4a] Scoring D_test_U_dep (skip ${THRESHOLD_SAMPLES}, max ${SCORE_SAMPLES})..."
python scripts/assis_tinyllama_test_path.py \
    --model_path "${CHECKPOINT}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${DEEPSEEK_DATA}" \
    --dataset_split "D_test_U_dep" \
    --question_key "probing input" \
    --answer_key "y_neg" \
    --skip_samples ${THRESHOLD_SAMPLES} \
    --max_samples ${SCORE_SAMPLES} \
    --raw_prompt \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --output_file "tinyllama_comparison_results.json" \
    --output_dir "${SCORE_DIR}"

# 4b: Score remaining D_test_U_nondep (skip first 200)
echo "[Stage 4b] Scoring D_test_U_nondep (skip ${THRESHOLD_SAMPLES}, max ${SCORE_SAMPLES})..."
python scripts/assis_tinyllama_test_path.py \
    --model_path "${CHECKPOINT}" \
    --pretrained_model_name "${ASSIST_MODEL}" \
    --dataset_name "${DEEPSEEK_DATA}" \
    --dataset_split "D_test_U_nondep" \
    --question_key "probing input" \
    --skip_samples ${THRESHOLD_SAMPLES} \
    --max_samples ${SCORE_SAMPLES} \
    --raw_prompt \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --output_file "tinyllama_comparison_results.json" \
    --output_dir "${SCORE_DIR}"

# 4c: Analyze scores with same threshold tool
echo "[Stage 4c] Analyzing scores..."
python scripts/analyze_cross_entropy.py \
    --data-dir "${SCORE_DIR}" \
    --forget-split "D_test_U_dep" \
    --retain-split "D_test_U_nondep" \
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
echo "  Threshold: ${THRESHOLD_DIR}/"
echo "  Scores:    ${SCORE_DIR}/"
echo ""
