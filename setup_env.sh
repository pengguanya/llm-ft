#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH=$(uname -m)

echo "=== LLM Fine-Tuning Environment Setup ==="
echo "Architecture: ${ARCH}"

# ---------- SSL certs for corporate environments ----------
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi

# ---------- Install uv if not present ----------
if ! command -v uv &>/dev/null; then
    echo ""
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---------- Install Python (if needed) ----------
echo ""
echo ">>> Ensuring Python is available..."
uv python install

# ---------- Install dependencies ----------
echo ""
if [ "${ARCH}" = "aarch64" ]; then
    # DGX Spark: PyTorch is pre-installed in the NGC container or system
    # with CUDA 13.0 and sm_121 support. No cu130 aarch64 wheels exist
    # on PyPI, so we skip torch and use the system-installed version.
    echo ">>> aarch64 detected — using system PyTorch, installing other deps..."
    uv sync --no-install-package torch
else
    # x86_64: Install everything including PyTorch from cu130 index.
    echo ">>> Installing dependencies (uv sync)..."
    uv sync
fi

# ---------- Optional: bitsandbytes for 4-bit quantization ----------
echo ""
echo ">>> Installing bitsandbytes (optional, for --load_in_4bit)..."
if [ "${ARCH}" = "aarch64" ]; then
    UV_EXTRA="uv sync --extra qlora --no-install-package torch"
else
    UV_EXTRA="uv sync --extra qlora"
fi
if ${UV_EXTRA} 2>/dev/null; then
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
    p = torch.cuda.get_device_properties(0)
    mem = getattr(p, 'total_memory', getattr(p, 'total_mem', 0)) / 1e9
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
