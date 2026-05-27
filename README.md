# LLM LoRA Fine-Tuning

Fine-tune Qwen2.5 (or any HuggingFace model) with LoRA for instruction following.
Everything downloads automatically — no manual data setup needed.

## Quick Start (x86 / standard GPU)

```bash
bash setup_env.sh
source .venv/bin/activate
python train_lora.py --verify
python train_lora.py --max_samples 200 --epochs 1    # quick test
python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora
```

## DGX Spark (recommended: Docker)

```bash
# Build
docker build -t llm-ft .

# Run (interactive)
docker run -it --gpus all --ipc=host \
  -v $(pwd):/workspace -w /workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  llm-ft

# Inside container:
python train_lora.py --verify
python train_lora.py --max_samples 200 --epochs 1
```

### DGX Spark bare-metal (alternative)

```bash
bash setup_env.sh
source .venv/bin/activate
python train_lora.py --verify
```

## Scaling Up

```bash
# 7B model (bf16, fits in 128GB unified memory)
python train_lora.py --model Qwen/Qwen2.5-7B-Instruct

# 72B model (4-bit quantization, ~40GB memory)
python train_lora.py --model Qwen/Qwen2.5-72B-Instruct \
  --load_in_4bit --gradient_checkpointing --batch_size 2

# Generate
python generate.py --model_dir results/qwen2.5-72b-instruct-lora
```

## Memory Tips for DGX Spark

The 128GB unified memory is shared between CPU and GPU. If training crashes:

```bash
# Reduce memory usage (cumulative — combine as needed)
--batch_size 2              # smaller batches
--max_length 256            # shorter sequences
--gradient_checkpointing    # trade compute for memory
--load_in_4bit              # 4-bit quantization (QLoRA)
```

## Files

| Script | Purpose |
|--------|---------|
| `setup_env.sh` | Create environment, install dependencies |
| `train_lora.py` | Fine-tune with LoRA (auto-downloads model + data) |
| `generate.py` | Interactive chat with fine-tuned model |
| `Dockerfile` | Container build for DGX Spark |
