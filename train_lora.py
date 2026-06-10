"""
Fine-tune a language model with LoRA for instruction following.

Model and dataset download automatically from HuggingFace Hub.
No login required — uses ungated models and datasets.

Usage:
    python train_lora.py                                     # defaults
    python train_lora.py --max_samples 200 --epochs 1        # quick test (~2 min)
    python train_lora.py --model Qwen/Qwen2.5-7B-Instruct    # larger model
    python train_lora.py --load_in_4bit --model Qwen/Qwen2.5-72B-Instruct  # QLoRA
    python train_lora.py --verify                             # dry run
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class ProfilerCallback(TrainerCallback):
    """Wraps torch.profiler to capture GPU kernel timing and memory during training."""

    def __init__(self, output_dir="profiler_logs/torch", active_steps=3):
        self.output_dir = output_dir
        self.active_steps = active_steps
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=self.active_steps, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self.prof.__enter__()
        print(f"\n>>> torch.profiler enabled — traces will be saved to {self.output_dir}/")

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.__exit__(None, None, None)
            sort_key = (
                "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            )
            print("\n" + "=" * 60)
            print(f"torch.profiler — Top 20 operations by {sort_key}")
            print("=" * 60)
            print(self.prof.key_averages().table(sort_by=sort_key, row_limit=20))
            print(f"\nTraces saved to: {self.output_dir}/")
            print("View with: tensorboard --logdir=profiler_logs")
            print("=" * 60)


class NsysCallback(TrainerCallback):
    """Emits NVTX markers for Nsight Systems profiling."""

    def __init__(self):
        self.enabled = torch.cuda.is_available()

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            print("\n>>> NVTX markers skipped — no CUDA device available")
            return
        torch.cuda.nvtx.range_push("training")
        print("\n>>> NVTX markers enabled — run with nsys profile for GPU timeline")

    def on_step_begin(self, args, state, control, **kwargs):
        if self.enabled:
            torch.cuda.nvtx.range_push(f"step_{state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.enabled:
            torch.cuda.nvtx.range_pop()  # step

    def on_train_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        torch.cuda.nvtx.range_pop()  # training
        print("\n>>> NVTX markers complete — view in Nsight Systems UI")


def format_example(instruction, context, response, tokenizer):
    user_content = instruction
    if context:
        user_content += f"\n\n{context}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    text = f"### Instruction:\n{instruction}"
    if context:
        text += f"\n\n### Input:\n{context}"
    text += f"\n\n### Response:\n{response}{tokenizer.eos_token}"
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM with LoRA for instruction following"
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
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length (default: 512)"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (QLoRA, needs bitsandbytes)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load data + model, print config, exit without training",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler (GPU kernel timing + memory)",
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
        help="Emit NVTX markers for Nsight Systems profiling",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = args.output_dir or str(
        Path("results") / f"{args.model.split('/')[-1].lower()}-lora"
    )

    # -- Config --
    print("=" * 60)
    print("LLM LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"LoRA rank:   {args.lora_r}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"Max length:  {args.max_length}")
    print(f"4-bit:       {args.load_in_4bit}")
    print(f"Grad ckpt:   {args.gradient_checkpointing}")
    print(f"Output:      {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"GPU memory:  {mem:.1f} GB")
    else:
        print("GPU:         NOT AVAILABLE (will be slow on CPU)")
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

    # -- Model --
    print(f"\n>>> Loading model: {args.model}")
    load_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
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

    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFY: Everything loaded successfully!")
        print(f"  Data:  {len(train_data)} train / {len(val_data)} val")
        print(f"  Model: {args.model}")
        print(f"  LoRA:  {trainable:,} params ({100 * trainable / total:.2f}%)")
        print("  Remove --verify to start training.")
        print("=" * 60)
        return

    # -- Train --
    print("\n>>> Training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )

    callbacks = []
    if args.profile:
        callbacks.append(
            ProfilerCallback(
                output_dir="profiler_logs/torch", active_steps=args.profile_steps
            )
        )
    if args.profile_nsys:
        callbacks.append(NsysCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nOUT OF MEMORY! Try:")
            print("  --batch_size 2")
            print("  --max_length 256")
            print("  --gradient_checkpointing")
            print("  --load_in_4bit")
            print("  --model Qwen/Qwen2.5-1.5B-Instruct  (smaller model)")
            sys.exit(1)
        raise

    # -- Save --
    print(f"\n>>> Saving to {output_dir}")
    trainer.save_model(output_dir)
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
        "load_in_4bit": args.load_in_4bit,
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Next: python generate.py --model_dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
