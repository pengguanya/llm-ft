#!/usr/bin/env bash
set -euo pipefail

# Profile LLM training with NVIDIA Nsight Systems.
# Produces a .nsys-rep file in profiler_logs/nsys/ for timeline analysis.
#
# Usage:
#   bash profile_nsys.sh                              # HuggingFace Trainer (default)
#   bash profile_nsys.sh --lightning                   # Lightning Trainer
#   bash profile_nsys.sh -- --max_samples 500          # pass extra args to training script
#   bash profile_nsys.sh --lightning -- --epochs 2     # Lightning + extra args

SCRIPT="train_lora.py"
EXTRA_ARGS=()
OUTPUT_DIR="profiler_logs/nsys"

# Parse our flags (before --)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lightning)
            SCRIPT="train_lora_lightning.py"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/nsys_${SCRIPT%.py}_${TIMESTAMP}"

echo "============================================================"
echo "  Nsight Systems GPU Profiling"
echo "============================================================"
echo "Script:  ${SCRIPT}"
echo "Output:  ${OUTPUT_FILE}.nsys-rep"
echo ""

# Both scripts support --profile_nsys for NVTX markers
TRAIN_CMD="python ${SCRIPT} --profile_nsys --max_samples 200 --epochs 1 ${EXTRA_ARGS[*]:-}"

echo "Command: ${TRAIN_CMD}"
echo "============================================================"
echo ""

nsys profile \
    -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s none \
    -o "${OUTPUT_FILE}" \
    -x true \
    ${TRAIN_CMD}

echo ""
echo "============================================================"
echo "  Profiling complete!"
echo ""
echo "  View summary:    nsys stats ${OUTPUT_FILE}.nsys-rep"
echo "  View timeline:   nsys-ui ${OUTPUT_FILE}.nsys-rep"
echo "============================================================"
