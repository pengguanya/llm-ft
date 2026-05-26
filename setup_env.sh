#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-ft"

echo "=== Setting up ${ENV_NAME} environment ==="

ARCH=$(uname -m)
echo "Architecture: ${ARCH}"

# ---------- Create isolated environment ----------
if command -v conda &>/dev/null; then
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo "Creating conda environment: ${ENV_NAME}"
        conda create -n "${ENV_NAME}" python=3.11 -y
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
        echo ""
        echo "Activate the environment first, then re-run:"
        echo "  conda activate ${ENV_NAME} && bash setup_env.sh"
        exit 0
    fi
else
    echo "conda not found — using python venv"
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated venv at .venv/"
fi

# ---------- Install PyTorch ----------
echo "=== Installing PyTorch ==="
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed: $(python3 -c 'import torch; print(torch.__version__)')"
elif python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch ${TORCH_VER} found (no CUDA). Keeping as-is — install CUDA build manually if needed."
else
    echo "Installing PyTorch..."
    if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
        echo "  ARM64 detected — trying pip install (NGC container is the safest option)"
        pip install torch || {
            echo ""
            echo "ERROR: PyTorch install failed on ARM64."
            echo "On DGX Spark, use the NGC PyTorch container instead:"
            echo "  docker run --gpus all -it -v \$PWD:/workspace nvcr.io/nvidia/pytorch:24.12-py3"
            echo "  cd /workspace && pip install peft datasets accelerate transformers"
            exit 1
        }
    else
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# ---------- Install HuggingFace stack ----------
echo "=== Installing HuggingFace stack ==="
# Pick transformers version based on PyTorch version
TORCH_MAJOR_MINOR=$(python3 -c "import torch; v=torch.__version__.split('+')[0].split('.')[:2]; print('.'.join(v))" 2>/dev/null || echo "0.0")
if python3 -c "
v = '${TORCH_MAJOR_MINOR}'.split('.')
exit(0 if int(v[0]) >= 2 and int(v[1]) >= 4 else 1)
" 2>/dev/null; then
    echo "  PyTorch >= 2.4 detected — installing latest transformers"
    pip install --upgrade transformers peft datasets accelerate
else
    echo "  PyTorch < 2.4 detected — pinning transformers < 5"
    pip install --upgrade "transformers>=4.42,<5" peft datasets accelerate
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
else:
    print('GPU:           not available (CPU only)')

import transformers, peft, datasets, accelerate
print(f'Transformers:  {transformers.__version__}')
print(f'PEFT:          {peft.__version__}')
print(f'Datasets:      {datasets.__version__}')
print(f'Accelerate:    {accelerate.__version__}')
print()
print('All dependencies installed!')
"

echo ""
echo "=== Setup Complete ==="
echo "Next: python train_lora.py --verify"
