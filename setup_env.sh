#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== LLM Fine-Tuning Environment Setup ==="
echo "Architecture: $(uname -m)"

# ---------- Install uv if not present ----------
if ! command -v uv &>/dev/null; then
    echo ""
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---------- Install Python (if needed) ----------
echo ""
echo ">>> Ensuring Python is available..."
uv python install

# ---------- Install dependencies ----------
echo ""
echo ">>> Installing dependencies (uv sync)..."
uv sync

# ---------- Optional: bitsandbytes for 4-bit quantization ----------
echo ""
echo ">>> Installing bitsandbytes (optional, for --load_in_4bit)..."
if uv sync --extra qlora 2>/dev/null; then
    echo "  bitsandbytes installed — 4-bit quantization available"
else
    echo "  WARNING: bitsandbytes install failed (common on aarch64)."
    echo "  4-bit quantization (--load_in_4bit) won't be available."
    echo "  This is fine — bf16 training works without it."
fi

# ---------- Verify ----------
echo ""
echo "=== Verification ==="
uv run python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU memory:    {mem:.1f} GB')

import transformers, peft, datasets, accelerate, pytorch_lightning
print(f'Transformers:  {transformers.__version__}')
print(f'PEFT:          {peft.__version__}')
print(f'Datasets:      {datasets.__version__}')
print(f'Accelerate:    {accelerate.__version__}')
print(f'Lightning:     {pytorch_lightning.__version__}')

try:
    import bitsandbytes
    print(f'BnB:           {bitsandbytes.__version__}')
except ImportError:
    print('BnB:           not installed')

print()
print('Setup complete!')
"

echo ""
echo "=== Done ==="
echo "Run with:  uv run python train_lora.py --verify"
echo "Or activate the venv:  source .venv/bin/activate"
