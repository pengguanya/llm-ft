# LLM LoRA Fine-Tuning on DGX Spark

Fine-tune Qwen2.5 (or any HuggingFace model) with LoRA for instruction following.
Everything downloads automatically — no manual data setup needed.

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
conda activate llm-ft

# 2. Verify everything loads (~1 min, downloads model + data)
python train_lora.py --verify

# 3. Quick test (200 samples, 1 epoch)
python train_lora.py --max_samples 200 --epochs 1

# 4. Full training (15K examples, 3 epochs)
python train_lora.py

# 5. Try the model
python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora
```

## Scaling Up

```bash
# Use 7B model (fits easily in 128GB unified memory)
python train_lora.py --model Qwen/Qwen2.5-7B-Instruct

python generate.py --model_dir results/qwen2.5-7b-instruct-lora
```

## Files

| Script | Purpose |
|--------|---------|
| `setup_env.sh` | Create environment, install dependencies |
| `train_lora.py` | Fine-tune with LoRA (auto-downloads model + data) |
| `generate.py` | Interactive chat with fine-tuned model |

## Hardware

Designed for NVIDIA DGX Spark (128GB unified memory).
Also works on any CUDA GPU or CPU (slower).
