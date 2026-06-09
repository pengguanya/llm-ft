"""
Fine-tune a language model with LoRA using PyTorch Lightning.

This is a Lightning equivalent of train_lora.py, created to compare
Lightning's PyTorchProfiler against raw torch.profiler and Nsight Systems.

Usage:
    python train_lora_lightning.py                                  # defaults
    python train_lora_lightning.py --max_samples 200 --epochs 1     # quick test
    python train_lora_lightning.py --profile                        # with Lightning profiler
    python train_lora_lightning.py --profile --max_samples 200 --epochs 1
"""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from train_lora import format_example


class LoRALightningModule(pl.LightningModule):
    def __init__(self, model, lr=2e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM with LoRA using PyTorch Lightning"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        default="databricks/databricks-dolly-15k",
        help="HuggingFace dataset (default: databricks/databricks-dolly-15k)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable Lightning PyTorchProfiler (GPU kernel timing + NVTX)",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=3,
        help="Number of steps to actively profile (default: 3)",
    )
    parser.add_argument(
        "--profile_nsys",
        action="store_true",
        help="Enable NVTX-only profiler for Nsight Systems (no torch.profiler overhead)",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    output_dir = args.output_dir or str(
        Path("results") / f"{args.model.split('/')[-1].lower()}-lora-lightning"
    )

    print("=" * 60)
    print("LLM LoRA Fine-Tuning (Lightning)")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"LoRA rank:   {args.lora_r}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Profiling:   {args.profile}")
    print(f"Output:      {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU memory:  {mem:.1f} GB")
    print("=" * 60)

    # -- Tokenizer --
    print("\n>>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Dataset --
    print(f"\n>>> Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"  {len(dataset)} examples")

    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))
        print(f"  Subsampled to {args.max_samples}")

    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_data, val_data = split["train"], split["test"]
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    # -- Tokenize --
    print("\n>>> Tokenizing...")

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            texts.append(
                format_example(
                    examples["instruction"][i],
                    examples["context"][i] or "",
                    examples["response"][i],
                    tokenizer,
                )
            )
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    train_data = train_data.map(
        tokenize_fn, batched=True, remove_columns=train_data.column_names
    )
    val_data = val_data.map(
        tokenize_fn, batched=True, remove_columns=val_data.column_names
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=collator
    )

    # -- Model --
    # Lightning manages device placement — load to CPU, let Trainer move to GPU.
    print(f"\n>>> Loading model: {args.model}")
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # -- Lightning module --
    lit_model = LoRALightningModule(model, lr=args.lr)

    # -- Profiler --
    # --profile:      full PyTorchProfiler (kernel timing + memory + NVTX)
    # --profile_nsys: lightweight NVTX-only mode for clean Nsight Systems traces
    # Both together:  full profiler with NVTX, run under nsys for cross-referencing
    profiler = None
    if args.profile or args.profile_nsys:
        from pytorch_lightning.profilers import PyTorchProfiler

        profile_dir = "profiler_logs/lightning"
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        use_nvtx = args.profile_nsys or torch.cuda.is_available()

        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = PyTorchProfiler(
                dirpath=profile_dir,
                filename="lightning_trace",
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=args.profile_steps, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                emit_nvtx=use_nvtx,
            )
            mode = "full + NVTX" if use_nvtx else "full"
            print(f"\n>>> Lightning PyTorchProfiler ({mode}) — traces to {profile_dir}/")
        else:
            profiler = PyTorchProfiler(
                dirpath=profile_dir,
                filename="lightning_nvtx",
                emit_nvtx=True,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            print("\n>>> Lightning NVTX-only profiler — run with nsys for GPU timeline")

    # -- Trainer --
    print("\n>>> Training...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        default_root_dir=output_dir,
        profiler=profiler,
        enable_checkpointing=True,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # -- Save --
    print(f"\n>>> Saving to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    info = {
        "base_model": args.model,
        "dataset": args.dataset,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "framework": "pytorch-lightning",
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    if args.profile:
        print("\n" + "=" * 60)
        print("Lightning PyTorchProfiler — profiling complete")
        print("=" * 60)
        print(f"Traces saved to: profiler_logs/lightning/")
        print("View with: tensorboard --logdir=profiler_logs")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Next: python generate.py --model_dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
