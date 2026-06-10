# DGX Spark (GB10, aarch64, sm_121) — recommended path for fine-tuning.
# 25.12+ includes Blackwell (sm_120/sm_121) support.
# Check for newer tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

# Install deps into the system Python (alongside NGC's pre-built torch).
# pip avoids uv's cert/venv issues inside NGC containers.
COPY pyproject.toml ./
RUN pip install --no-cache-dir transformers peft datasets accelerate pytorch-lightning tensorboard && \
    pip install --no-cache-dir bitsandbytes || true && \
    pip uninstall -y torchao || true

# Copy scripts
COPY train_lora.py train_lora_lightning.py generate.py \
     profile_nsys.sh preflight.sh ./

CMD ["bash"]
