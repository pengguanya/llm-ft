#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
ARCH=$(uname -m)

echo "=== LLM Fine-Tuning Environment Setup ==="
echo "Architecture: ${ARCH}"

if [ "${ARCH}" = "aarch64" ]; then
    echo ""
    echo "NOTE: On DGX Spark, Docker is the recommended approach."
    echo "  See README.md for Docker instructions."
    echo "  Continuing with bare-metal setup..."
    echo ""
fi

# ---------- Virtual environment ----------
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------- Install PyTorch ----------
echo "=== Installing PyTorch ==="
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    if [ "${ARCH}" = "aarch64" ]; then
        pip install torch
    else
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# ---------- Install HuggingFace stack ----------
echo "=== Installing HuggingFace stack ==="
pip install transformers peft datasets accelerate

# ---------- Install bitsandbytes (for 4-bit quantization) ----------
echo "=== Installing bitsandbytes ==="
if pip install bitsandbytes; then
    echo "bitsandbytes installed — 4-bit quantization available (--load_in_4bit)"
else
    echo "WARNING: bitsandbytes install failed."
    echo "  4-bit quantization (--load_in_4bit) won't be available."
    echo "  On aarch64, try the Docker workflow instead."
fi

# ---------- Verify ----------
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU memory:    {mem:.1f} GB')

import transformers, peft, datasets, accelerate
print(f'Transformers:  {transformers.__version__}')
print(f'PEFT:          {peft.__version__}')
print(f'Datasets:      {datasets.__version__}')
print(f'Accelerate:    {accelerate.__version__}')

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
echo "Activate with: source .venv/bin/activate"
echo "Then run:      python train_lora.py --verify"
