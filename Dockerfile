# DGX Spark (GB10, aarch64, sm_121) — recommended path for fine-tuning.
# 25.12+ includes Blackwell (sm_120/sm_121) support.
# Check for newer tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN pip install --no-cache-dir \
    transformers \
    peft \
    datasets \
    accelerate \
    bitsandbytes

COPY train_lora.py generate.py ./

CMD ["bash"]
