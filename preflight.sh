#!/usr/bin/env bash
set -euo pipefail

# Pre-flight test: validates the full training + profiling pipeline locally
# before running on DGX Spark. Uses minimal data (50 samples, 1 epoch)
# to fit any GPU (8GB VRAM is enough).
#
# Usage:
#   bash preflight.sh          # run all checks
#   bash preflight.sh --quick  # skip training, just verify imports

cd "$(dirname "$0")"

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

PASS=0
FAIL=0
SKIP=0

run_check() {
    local name="$1"
    shift
    echo ""
    echo "--- CHECK: ${name} ---"
    if "$@"; then
        echo "  ✓ ${name}"
        ((PASS++))
    else
        echo "  ✗ ${name}"
        ((FAIL++))
    fi
}

skip_check() {
    local name="$1"
    local reason="$2"
    echo ""
    echo "--- SKIP: ${name} (${reason}) ---"
    ((SKIP++))
}

echo "============================================================"
echo "  Pre-flight Check — LLM LoRA Fine-Tuning Pipeline"
echo "============================================================"
echo "Mode: $(if $QUICK; then echo 'quick (imports only)'; else echo 'full (train + profile)'; fi)"

# ---------- 1. Python environment ----------
run_check "Python imports" python3 -c "
import torch, transformers, peft, datasets, accelerate, pytorch_lightning, tensorboard
print(f'  PyTorch:    {torch.__version__}')
print(f'  Lightning:  {pytorch_lightning.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  VRAM:       {mem:.1f} GB')
    print(f'  Arch:       sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
"

# ---------- 2. Script syntax ----------
run_check "train_lora.py syntax" python3 -m py_compile train_lora.py
run_check "train_lora_lightning.py syntax" python3 -m py_compile train_lora_lightning.py
run_check "generate.py syntax" python3 -m py_compile generate.py

# ---------- 3. CLI help (catches import errors) ----------
run_check "train_lora.py --help" python3 train_lora.py --help > /dev/null
run_check "train_lora_lightning.py --help" python3 train_lora_lightning.py --help > /dev/null

if $QUICK; then
    echo ""
    echo "============================================================"
    echo "  Quick check done: ${PASS} passed, ${FAIL} failed"
    echo "  Run without --quick for full training + profiling tests"
    echo "============================================================"
    exit $FAIL
fi

# ---------- 4. Dry run (verify model + data loading) ----------
run_check "train_lora.py --verify" \
    python3 train_lora.py --verify --max_samples 50

# ---------- 5. Training with torch.profiler ----------
TRAIN_ARGS="--max_samples 50 --epochs 1 --batch_size 2 --max_length 256"

run_check "train_lora.py --profile (torch.profiler)" \
    python3 train_lora.py --profile --profile_steps 2 \
    --output_dir results/preflight-hf \
    ${TRAIN_ARGS}

# Verify profiler output exists
run_check "torch.profiler trace files exist" \
    test -n "$(find profiler_logs/torch -name '*.json' -o -name '*.pt.trace*' 2>/dev/null | head -1)"

# ---------- 6. Training with NVTX markers ----------
run_check "train_lora.py --profile_nsys (NVTX markers)" \
    python3 train_lora.py --profile_nsys \
    --output_dir results/preflight-nvtx \
    ${TRAIN_ARGS}

# ---------- 7. Lightning training with profiler ----------
run_check "train_lora_lightning.py --profile (Lightning profiler)" \
    python3 train_lora_lightning.py --profile --profile_steps 2 \
    --output_dir results/preflight-lightning \
    ${TRAIN_ARGS}

# Verify Lightning profiler output exists
run_check "Lightning profiler trace files exist" \
    test -n "$(find profiler_logs/lightning -name '*.json' -o -name '*.pt.trace*' 2>/dev/null | head -1)"

# ---------- 8. Generation test ----------
run_check "generate.py (inference from HF-trained model)" \
    python3 generate.py --model_dir results/preflight-hf \
    --prompt "What is machine learning?"

run_check "generate.py (inference from Lightning-trained model)" \
    python3 generate.py --model_dir results/preflight-lightning \
    --prompt "What is machine learning?"

# ---------- 9. Nsight Systems availability ----------
echo ""
if command -v nsys &>/dev/null; then
    run_check "nsys available" nsys --version
else
    skip_check "nsys CLI" "not installed — will be available in NGC container on DGX Spark"
fi

# ---------- 10. TensorBoard can load traces ----------
run_check "TensorBoard import" python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
print('  TensorBoard can parse event files')
"

# ---------- Cleanup ----------
echo ""
echo "Cleaning up preflight results..."
rm -rf results/preflight-hf results/preflight-nvtx results/preflight-lightning

# ---------- Summary ----------
echo ""
echo "============================================================"
echo "  Pre-flight Results"
echo "============================================================"
echo "  Passed:  ${PASS}"
echo "  Failed:  ${FAIL}"
echo "  Skipped: ${SKIP}"
echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "  ✓ All checks passed — ready for DGX Spark!"
    echo ""
    echo "  On DGX Spark, run:"
    echo "    python train_lora.py --profile --max_samples 200 --epochs 1"
    echo "    python train_lora_lightning.py --profile --max_samples 200 --epochs 1"
    echo "    bash profile_nsys.sh"
else
    echo "  ✗ ${FAIL} check(s) failed — fix before running on DGX Spark"
fi
echo "============================================================"

exit $FAIL
