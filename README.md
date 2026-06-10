# LLM LoRA Fine-Tuning with GPU Profiling

Fine-tune Qwen2.5 (or any HuggingFace model) with LoRA for instruction following,
and compare GPU profiling across three tools: **torch.profiler**, **PyTorch Lightning Profiler**,
and **NVIDIA Nsight Systems**.

Everything downloads automatically from HuggingFace Hub — no manual data or model setup needed.

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Local PC Setup (Windows 11 + WSL2)](#2-local-pc-setup-windows-11--wsl2)
3. [DGX Spark Setup](#3-dgx-spark-setup)
4. [Pre-flight Testing](#4-pre-flight-testing)
5. [Training — Quick Test](#5-training--quick-test)
6. [Training — Full Run](#6-training--full-run)
7. [GPU Profiling Overview](#7-gpu-profiling-overview)
8. [Profiling on Local PC](#8-profiling-on-local-pc)
9. [Profiling on DGX Spark](#9-profiling-on-dgx-spark)
10. [Comparing Profiler Results](#10-comparing-profiler-results)
11. [Viewing and Analyzing Traces](#11-viewing-and-analyzing-traces)
12. [Scaling Up on DGX Spark](#12-scaling-up-on-dgx-spark)
13. [Troubleshooting](#13-troubleshooting)
14. [File Reference](#14-file-reference)

---

## 1. Platform Overview

| | **Local PC** | **DGX Spark** |
|---|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) | Debian 12 container on Ubuntu host |
| Architecture | x86_64 | aarch64 (ARM64) |
| GPU | RTX 5060 (8 GB VRAM) | Grace Blackwell GB10 (128 GB unified memory) |
| CUDA arch | sm_120 | sm_121 |
| Dependency manager | uv (auto-installs Python + packages) | uv (same) |
| PyTorch install | CUDA 13.0 wheels via uv (cu130 index) | NGC container (pre-built with sm_121) |
| Nsight Systems | Install separately | Pre-installed in NGC container |
| Max model size | 1.5B (bf16) or 7B (4-bit) | 72B+ (bf16 with unified memory) |

Both platforms run the **same Python scripts** with the **same `pyproject.toml`**.
The only prerequisite is [`uv`](https://docs.astral.sh/uv/) — it handles Python
installation, virtual environment creation, and dependency resolution automatically.

**Note on Blackwell + PyTorch:** Both RTX 5060 (sm_120) and DGX Spark (sm_121) are
Blackwell GPUs requiring CUDA 13.0+. On x86_64, `uv sync` installs PyTorch from
the cu130 index. On aarch64 (DGX Spark), no cu130 pip wheels exist — PyTorch must
come from the NGC container or be pre-installed on the system. The setup script
handles this automatically.

---

## 2. Local PC Setup (Windows 11 + WSL2)

### 2.1 Prerequisites

- Windows 11 with WSL2 installed (`wsl --install` in PowerShell)
- NVIDIA GPU driver for Windows (enables CUDA in WSL2 automatically)
- No Python installation needed — `uv` handles it

### 2.2 Install the environment

Open a WSL2 Ubuntu terminal:

```bash
# Clone the repo
git clone https://github.com/pengguanya/llm-ft.git
cd llm-ft

# Option A: Automated setup (installs uv if needed, then syncs dependencies)
bash setup_env.sh

# Option B: Manual (if you already have uv)
uv sync                    # creates .venv, installs Python 3.11 + all deps
uv sync --extra qlora      # optional: adds bitsandbytes for 4-bit quantization
```

`uv sync` reads `pyproject.toml` and `.python-version`, installs Python 3.11 if
missing, creates a `.venv`, and resolves all dependencies including PyTorch with
CUDA 13.0 wheels (automatic for x86_64, required for Blackwell GPUs).

### 2.3 Verify the environment

```bash
# Using uv run (no need to activate .venv)
uv run python train_lora.py --verify

# Or activate and use python directly
source .venv/bin/activate
python train_lora.py --verify
```

Expected output:

```
============================================================
LLM LoRA Fine-Tuning
============================================================
Model:       Qwen/Qwen2.5-1.5B-Instruct
...
GPU:         NVIDIA GeForce RTX 5060
GPU memory:  8.0 GB
============================================================
...
VERIFY: Everything loaded successfully!
```

### 2.4 Install Nsight Systems (optional)

Nsight Systems is not required for torch.profiler or Lightning profiling,
but if you want to test the nsys workflow locally:

1. Download "Nsight Systems" from [NVIDIA Developer Tools](https://developer.nvidia.com/nsight-systems)
2. Install the Linux CLI package inside WSL2:
   ```bash
   sudo dpkg -i NsightSystems-linux-cli-*.deb
   nsys --version   # verify
   ```

---

## 3. DGX Spark Setup

SSH into the DGX Spark and clone the repo directly from GitHub:

```bash
ssh username@<dgx-spark-ip>
git clone https://github.com/pengguanya/llm-ft.git
cd llm-ft
```

### 3.1 Option A: Docker (recommended)

```bash
# Build the container (uses NGC base image + uv for dependency install)
docker build -t llm-ft .

# Run interactive container with GPU access
docker run -it --gpus all --ipc=host \
  -v $(pwd):/workspace -w /workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  llm-ft

# Inside the container — verify and start training
python train_lora.py --verify
```

The Dockerfile uses the NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`)
which includes PyTorch pre-built with CUDA 13.0 and Blackwell (sm_121) support.
`pip install` adds the non-torch dependencies (transformers, peft, etc.)
directly into the system Python alongside NGC's pre-built torch.
Nsight Systems is pre-installed in the NGC image.

> **Installing extra packages inside the container:** Use
> `pip install <pkg>` directly. Avoid `uv` inside NGC containers — it
> has SSL cert issues with NVIDIA's package indexes on aarch64.

### 3.2 Option B: Bare-metal (if Docker is not available)

Requires PyTorch to be pre-installed on the system (DGX Spark ships with it).

```bash
# Detects aarch64, skips torch install, installs other deps
bash setup_env.sh

# Run with uv (or activate .venv first)
uv run python train_lora.py --verify
```

On aarch64, the setup script runs `uv sync --no-install-package torch` so it
uses the system-installed PyTorch (which has CUDA 13.0 + sm_121 support) and
only installs the other dependencies into the venv.

### 3.3 Verify

```bash
python train_lora.py --verify      # inside Docker
uv run python train_lora.py --verify  # bare-metal
```

Expected output:

```
GPU:         NVIDIA Graphics Device    # or specific Blackwell name
GPU memory:  128.0 GB                  # unified memory
```

---

## 4. Pre-flight Testing

Run this on your **local PC** before going to DGX Spark. It validates the entire
pipeline — training, profiling, and generation — with minimal data (50 samples)
that fits any GPU.

### 4.1 Quick check (no training, ~30 seconds)

```bash
bash preflight.sh --quick
```

Verifies: Python imports, script syntax, CLI flags.

### 4.2 Full check (trains + profiles + generates, ~5 minutes)

```bash
bash preflight.sh
```

Verifies:
- Model + data download from HuggingFace
- Training with torch.profiler (HuggingFace Trainer)
- Training with Lightning PyTorchProfiler
- NVTX marker emission
- Profiler trace file generation
- Text generation from both trained models
- TensorBoard can parse the traces

The script uses `--batch_size 2 --max_length 256 --max_samples 50` to fit
in 8 GB VRAM. It cleans up all results after running.

### 4.3 What to expect

```
============================================================
  Pre-flight Results
============================================================
  Passed:  13
  Failed:  0
  Skipped: 1          # nsys — expected if not installed locally
  ✓ All checks passed — ready for DGX Spark!
============================================================
```

If any check fails, fix the issue before moving to DGX Spark.

---

## 5. Training — Quick Test

Use these settings on **both** local PC and DGX Spark for a fast validation run:

```bash
# HuggingFace Trainer (200 samples, 1 epoch, ~2 minutes)
python train_lora.py --max_samples 200 --epochs 1

# Lightning Trainer (same data, for comparison)
python train_lora_lightning.py --max_samples 200 --epochs 1
```

### Test the fine-tuned model

```bash
# Single prompt
python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora \
  --prompt "Explain what DNA is in one sentence."

# Interactive chat
python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora
```

---

## 6. Training — Full Run

### 6.1 Local PC (RTX 5060, 8 GB)

The 1.5B model fits comfortably. Use reduced sequence length for safety:

```bash
python train_lora.py --max_length 256
```

### 6.2 DGX Spark (128 GB unified memory)

Scale up to larger models:

```bash
# 1.5B (default, fast)
python train_lora.py

# 7B model (bf16, fits in 128 GB)
python train_lora.py --model Qwen/Qwen2.5-7B-Instruct

# 72B model (4-bit quantization)
python train_lora.py --model Qwen/Qwen2.5-72B-Instruct \
  --load_in_4bit --gradient_checkpointing --batch_size 2
```

### Memory tips (DGX Spark)

The 128 GB is shared between CPU and GPU. If training crashes with OOM:

```bash
--batch_size 2              # smaller batches
--max_length 256            # shorter sequences
--gradient_checkpointing    # trade compute for memory
--load_in_4bit              # 4-bit quantization (QLoRA)
```

---

## 7. GPU Profiling Overview

Three profiling tools are available. Each captures different levels of detail:

| Profiler | What it captures | Overhead | Output format | Viewer |
|----------|-----------------|----------|--------------|--------|
| **torch.profiler** | CUDA kernel timing, CPU ops, memory allocations, operator shapes | Low–Medium | TensorBoard traces + Chrome JSON | TensorBoard, Perfetto |
| **Lightning PyTorchProfiler** | Everything torch.profiler captures + Lightning hook annotations + automatic NVTX markers | Low–Medium | TensorBoard traces + Chrome JSON | TensorBoard, Perfetto |
| **NVIDIA Nsight Systems** | Full system GPU timeline: CUDA API calls, kernel launches, memory transfers, cuDNN/cuBLAS, NVTX ranges | Very Low | `.nsys-rep` binary | nsys-ui, nsys stats |

### How they relate

```
torch.profiler ──────────────┐
  (PyTorch built-in)          │  Both output TensorBoard traces
                              ├──► directly comparable in same TensorBoard UI
Lightning PyTorchProfiler ───┘
  (wraps torch.profiler,
   adds hook annotations)
         │
         │ emit_nvtx=True
         ▼
  NVTX markers ──────────────► appear in Nsight Systems timeline
                                (bridges Lightning ↔ nsys)

Nsight Systems ──────────────► captures everything from outside the process
  (system-level profiler)       no code changes needed (just wrap with nsys)
```

### Profiling flags reference

| Flag | Script | What it does |
|------|--------|-------------|
| `--profile` | `train_lora.py` | Enables torch.profiler (kernel timing + memory) |
| `--profile` | `train_lora_lightning.py` | Enables Lightning PyTorchProfiler (kernel timing + memory + NVTX) |
| `--profile_steps N` | Both | Number of training steps to actively profile (default: 3) |
| `--profile_nsys` | `train_lora.py` | Emits NVTX range markers for each training step |
| `--profile_nsys` | `train_lora_lightning.py` | Enables NVTX-only mode (lightweight, no torch.profiler overhead) |

---

## 8. Profiling on Local PC

All three profilers work on your RTX 5060. Use small data to keep runs fast.

### 8.1 torch.profiler (HuggingFace Trainer)

```bash
python train_lora.py --profile --max_samples 200 --epochs 1
```

Output:
- Traces in `profiler_logs/torch/`
- Summary table of top 20 CUDA operations printed to terminal

### 8.2 Lightning PyTorchProfiler

```bash
python train_lora_lightning.py --profile --max_samples 200 --epochs 1
```

Output:
- Traces in `profiler_logs/lightning/`
- Includes Lightning hook annotations (training_step, backward, optimizer_step)

### 8.3 Nsight Systems (if installed)

```bash
# Profile HuggingFace training
bash profile_nsys.sh

# Profile Lightning training
bash profile_nsys.sh --lightning
```

Output:
- `.nsys-rep` files in `profiler_logs/nsys/`

If nsys is not installed locally, skip this — you'll run it on DGX Spark where
it's pre-installed in the NGC container.

### 8.4 View results

```bash
# Open TensorBoard (shows both torch and Lightning traces)
tensorboard --logdir=profiler_logs
# → Open http://localhost:6006 in your browser
# → Click "PyTorch Profiler" tab in the top nav
# → Use the "Runs" dropdown to switch between torch/ and lightning/
```

---

## 9. Profiling on DGX Spark

On DGX Spark, you can run all three profilers with larger workloads and have
Nsight Systems available natively.

### 9.1 torch.profiler

```bash
# Profile the 1.5B model
python train_lora.py --profile --max_samples 200 --epochs 1

# Profile the 7B model
python train_lora.py --profile --model Qwen/Qwen2.5-7B-Instruct \
  --max_samples 200 --epochs 1

# Profile more steps for better coverage
python train_lora.py --profile --profile_steps 10 --max_samples 500 --epochs 1
```

### 9.2 Lightning PyTorchProfiler

```bash
python train_lora_lightning.py --profile --max_samples 200 --epochs 1
```

### 9.3 Nsight Systems

```bash
# Profile HuggingFace training
bash profile_nsys.sh

# Profile Lightning training (NVTX markers from Lightning hooks)
bash profile_nsys.sh --lightning

# Profile a larger run with custom args
bash profile_nsys.sh -- --max_samples 500 --epochs 2

# Profile the 7B model
bash profile_nsys.sh -- --model Qwen/Qwen2.5-7B-Instruct --max_samples 200
```

### 9.4 Combined: Lightning + TensorBoard + Nsight in one run

Get both TensorBoard traces AND NVTX markers, then wrap with nsys:

```bash
nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s none \
  -o profiler_logs/nsys/combined \
  -x true \
  python train_lora_lightning.py --profile --profile_nsys \
    --max_samples 200 --epochs 1
```

This produces:
- TensorBoard traces in `profiler_logs/lightning/` (view in TensorBoard)
- `.nsys-rep` file in `profiler_logs/nsys/` with full NVTX annotations (view in nsys-ui)

### 9.5 Retrieve results for local viewing

If DGX Spark doesn't have a browser, copy traces back to your local PC:

```bash
# On your local PC:
scp -r dgx-spark:~/llm-ft/profiler_logs ./profiler_logs

# View TensorBoard traces locally
tensorboard --logdir=profiler_logs

# View nsys traces (requires Nsight Systems installed locally)
nsys-ui profiler_logs/nsys/*.nsys-rep
```

---

## 10. Comparing Profiler Results

### 10.1 Step-by-step comparison workflow

Run all profilers on the same workload for a fair comparison:

```bash
# Use identical settings for all runs
ARGS="--max_samples 200 --epochs 1"

# --- Run 1: HuggingFace + torch.profiler ---
python train_lora.py --profile ${ARGS}
# → profiler_logs/torch/

# --- Run 2: Lightning + PyTorchProfiler ---
python train_lora_lightning.py --profile ${ARGS}
# → profiler_logs/lightning/

# --- Run 3: Nsight Systems + HuggingFace ---
bash profile_nsys.sh -- ${ARGS}
# → profiler_logs/nsys/nsys_train_lora_*.nsys-rep

# --- Run 4: Nsight Systems + Lightning ---
bash profile_nsys.sh --lightning -- ${ARGS}
# → profiler_logs/nsys/nsys_train_lora_lightning_*.nsys-rep
```

### 10.2 What to compare

**torch.profiler vs Lightning (TensorBoard):**

```bash
tensorboard --logdir=profiler_logs
```

In the TensorBoard "PyTorch Profiler" tab:
- **Runs dropdown** — switch between `torch/` and `lightning/` traces
- **Overview** — total training time, GPU utilization percentage
- **Operator view** — which CUDA kernels take the most time (should be similar across both trainers)
- **Memory view** — peak GPU memory, allocation timeline
- **Trace view** — timeline of individual kernel launches

Look for:
- Does Lightning add measurable overhead from its hook system?
- Are the same CUDA kernels dominant in both?
- Any difference in memory usage patterns?

**Nsight Systems (HF vs Lightning):**

```bash
# CLI summary — quick comparison of kernel time distribution
nsys stats profiler_logs/nsys/nsys_train_lora_*.nsys-rep
nsys stats profiler_logs/nsys/nsys_train_lora_lightning_*.nsys-rep

# GUI timeline — visual comparison
nsys-ui profiler_logs/nsys/*.nsys-rep
```

In the nsys-ui timeline view:
- **NVTX Ranges row** — shows step_0, step_1, etc. (HF) or training_step, backward (Lightning)
- **CUDA API row** — kernel launches, memory copies
- **GPU Kernels row** — actual GPU execution
- Look for gaps between kernels (idle GPU time) and compare between HF and Lightning

**Cross-tool: Lightning TensorBoard ↔ Lightning nsys:**

Lightning's `emit_nvtx=True` creates a bridge — the same training run produces:
- Fine-grained operator breakdown in TensorBoard
- System-level GPU timeline with named NVTX ranges in nsys
- Compare: does TensorBoard's "GPU utilization %" match what you see in the nsys kernel timeline?

### 10.3 Expected differences

| Aspect | torch.profiler (HF) | Lightning PyTorchProfiler | Nsight Systems |
|--------|---------------------|--------------------------|----------------|
| Kernel timing accuracy | Operator-level | Same + hook-level | Kernel-level (most precise) |
| Memory tracking | Python allocator | Python allocator | CUDA driver-level |
| Overhead | ~5-15% | ~5-15% | ~1-2% |
| Annotations | CUDA ops only | CUDA ops + Lightning hooks | NVTX markers + CUDA API |
| Ease of use | Easy (built-in flag) | Easy (built-in flag) | Moderate (need nsys CLI) |
| Best for | Operator hotspot analysis | Same + understanding training loop structure | Diagnosing GPU idle time, kernel launch overhead, memory transfer bottlenecks |

---

## 11. Viewing and Analyzing Traces

### 11.1 TensorBoard (torch.profiler + Lightning traces)

```bash
tensorboard --logdir=profiler_logs
```

Open `http://localhost:6006` and navigate to the **PyTorch Profiler** tab (top nav).

Key views:
- **Overview** — GPU utilization, step time breakdown
- **Operator** — Top CUDA operators sorted by total/self time
- **Memory** — Allocation timeline, peak memory
- **Trace** — Zoomable timeline (similar to chrome://tracing)

### 11.2 Chrome Tracing / Perfetto (alternative to TensorBoard)

For a lighter-weight viewer, load the JSON trace files directly:

1. Open [Perfetto UI](https://ui.perfetto.dev/) in your browser
2. Drag and drop a `.json` file from `profiler_logs/torch/` or `profiler_logs/lightning/`
3. Use Ctrl+scroll to zoom, click on spans for details

### 11.3 Nsight Systems

**CLI summary (headless / SSH):**

```bash
# Overview statistics
nsys stats profiler_logs/nsys/*.nsys-rep

# Specific analysis
nsys stats --report cuda_gpu_kern_sum profiler_logs/nsys/*.nsys-rep   # kernel summary
nsys stats --report cuda_api_sum profiler_logs/nsys/*.nsys-rep        # CUDA API summary
nsys stats --report nvtx_sum profiler_logs/nsys/*.nsys-rep            # NVTX range summary
```

**GUI timeline (desktop or remote):**

```bash
nsys-ui profiler_logs/nsys/*.nsys-rep
```

In the timeline:
- **Top rows**: CPU threads, CUDA API calls
- **Middle rows**: NVTX annotations (step_0, training_step, etc.)
- **Bottom rows**: GPU kernel execution, memory operations
- Zoom in on individual steps to see kernel launch patterns

---

## 12. Scaling Up on DGX Spark

After validating the pipeline with small runs, scale up:

```bash
# 7B model with profiling (bf16, ~15 GB)
python train_lora.py --profile \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max_samples 1000 --epochs 1

# 72B model (4-bit quantization, ~40 GB)
python train_lora.py --profile \
  --model Qwen/Qwen2.5-72B-Instruct \
  --load_in_4bit --gradient_checkpointing --batch_size 2 \
  --max_samples 500 --epochs 1

# Full dataset (15K examples, 3 epochs)
python train_lora.py --profile --profile_steps 20

# Generate from the trained model
python generate.py --model_dir results/qwen2.5-7b-instruct-lora
```

---

## 13. Troubleshooting

### CUDA out of memory (local PC)

The RTX 5060 has 8 GB VRAM. If training crashes:

```bash
python train_lora.py --batch_size 2 --max_length 256 --max_samples 200
```

### CUDA out of memory (DGX Spark)

The 128 GB is shared. For large models:

```bash
python train_lora.py --load_in_4bit --gradient_checkpointing --batch_size 2
```

### nsys not found

- **Local PC**: Nsight Systems is optional. Install from [NVIDIA Developer Tools](https://developer.nvidia.com/nsight-systems), or skip and use it on DGX Spark.
- **DGX Spark Docker**: Should be pre-installed. If not: `apt-get update && apt-get install -y nsight-systems`
- **DGX Spark bare-metal**: Install the ARM64 `.deb` package from NVIDIA.

### TensorBoard shows no data

Ensure you ran with `--profile` and traces exist:

```bash
ls profiler_logs/torch/    # should have .json files
ls profiler_logs/lightning/ # should have .json files
```

### Lightning import error

```bash
uv sync    # re-installs all dependencies from pyproject.toml
```

### Model download fails

HuggingFace Hub is accessed via HTTPS. If behind a corporate proxy:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

---

## 14. File Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project dependencies and uv configuration |
| `.python-version` | Pins Python 3.11 for uv |
| `setup_env.sh` | Installs uv (if needed) and runs `uv sync` |
| `train_lora.py` | Fine-tune with LoRA using HuggingFace Trainer (supports `--profile`, `--profile_nsys`) |
| `train_lora_lightning.py` | Fine-tune with LoRA using Lightning Trainer (supports `--profile`, `--profile_nsys`) |
| `generate.py` | Interactive chat / single-prompt inference with fine-tuned model |
| `profile_nsys.sh` | Nsight Systems wrapper — runs nsys with proper flags |
| `preflight.sh` | Pre-flight validation — tests full pipeline with minimal data |
| `run_demo.sh` | End-to-end demo script (quick test → full training → generation) |
| `Dockerfile` | NGC-based container for DGX Spark (aarch64, Blackwell support) |
