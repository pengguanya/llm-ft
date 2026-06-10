"""Pre-download HuggingFace models and datasets before running training.

Run this on the host (outside the container) to cache assets into a local
directory that gets bind-mounted into the container. Avoids slow/stuck
downloads inside NGC containers.

Usage:
    python prefetch.py                          # download defaults
    python prefetch.py --model meta-llama/Llama-3.2-1B-Instruct
    python prefetch.py --dataset myorg/mydataset
    python prefetch.py --model Qwen/Qwen2.5-7B-Instruct --cache-dir ./my_cache
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Pre-download HF models and datasets")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model to download (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        default="databricks/databricks-dolly-15k",
        help="HuggingFace dataset to download (default: databricks/databricks-dolly-15k)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Local cache directory (default: .hf_cache)",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.environ["HF_HOME"] = cache_dir

    print(f"Cache directory: {cache_dir}")
    print()

    # --- Dataset ---
    print(f">>> Downloading dataset: {args.dataset}")
    try:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split="train")
        print(f"    OK — {len(ds)} examples")
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Tokenizer ---
    print(f"\n>>> Downloading tokenizer: {args.model}")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"    OK — vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Model weights ---
    print(f"\n>>> Downloading model weights: {args.model}")
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(args.model)
        print(f"    OK — cached at: {path}")
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("All assets cached. Run the container with:")
    print(f"  docker run -it --gpus all --ipc=host \\")
    print(f"    -v $(pwd):/workspace -w /workspace \\")
    print(f"    -v $(pwd)/{args.cache_dir}:/root/.cache/huggingface \\")
    print(f"    llm-ft")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
