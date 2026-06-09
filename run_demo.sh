#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "  LLM LoRA Fine-Tuning Demo — DGX Spark"
echo "============================================================"
echo ""

# ---------- Step 1: Verify environment ----------
echo ">>> Step 1/5: Checking environment..."
python3 -c "import torch, transformers, peft, datasets; print('  All packages OK')"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    p = torch.cuda.get_device_properties(0)
    mem = getattr(p, 'total_memory', getattr(p, 'total_mem', 0)) / 1e9
    print(f'  Memory: {mem:.1f} GB')
else:
    print('  WARNING: No GPU detected — training will be slow')
"

# ---------- Step 2: Quick test (200 samples, 1 epoch) ----------
echo ""
echo ">>> Step 2/5: Quick test (200 samples, 1 epoch)..."
python3 train_lora.py --max_samples 200 --epochs 1 --output_dir results/quick-test

# ---------- Step 3: Quick generation test ----------
echo ""
echo ">>> Step 3/5: Testing generation..."
python3 generate.py --model_dir results/quick-test --prompt "Explain what DNA is in one sentence."

# ---------- Step 4: Full training ----------
echo ""
echo ">>> Step 4/5: Full training (15K examples, 3 epochs)..."
python3 train_lora.py

# ---------- Step 5: Generate with full model ----------
echo ""
echo ">>> Step 5/5: Testing full model..."
python3 generate.py --model_dir results/qwen2.5-1.5b-instruct-lora --prompt "What is machine learning and why is it useful?"

echo ""
echo "============================================================"
echo "  Demo complete!"
echo ""
echo "  Try interactive chat:"
echo "    python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora"
echo "============================================================"
