# DGX Spark (GB10, aarch64, sm_121) — recommended path for fine-tuning.
# 25.12+ includes Blackwell (sm_120/sm_121) support.
# Check for newer tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Install deps via uv, skipping torch (NGC image has it with sm_121 support).
# No cu130 aarch64 wheels exist on PyPI — must use NGC's pre-built PyTorch.
COPY pyproject.toml .python-version ./
RUN uv sync --no-dev --no-install-package torch && \
    uv sync --extra qlora --no-dev --no-install-package torch || true

# Copy scripts
COPY train_lora.py train_lora_lightning.py generate.py \
     profile_nsys.sh preflight.sh ./

CMD ["bash"]
