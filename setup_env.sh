#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-ft"

echo "=== Setting up ${ENV_NAME} environment ==="

ARCH=$(uname -m)
echo "Architecture: ${ARCH}"

# ---------- Create conda environment ----------
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
    echo "conda not found — installing into current Python environment"
fi

# ---------- Install PyTorch ----------
echo "=== Installing PyTorch ==="
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

# ---------- Install HuggingFace stack ----------
echo "=== Installing HuggingFace stack ==="
pip install --upgrade "transformers>=4.42,<5" peft datasets accelerate

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
print()
print('All dependencies installed!')
"

echo ""
echo "=== Setup Complete ==="
echo "Next: python train_lora.py --verify"
