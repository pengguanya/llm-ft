"""
Generate text with a LoRA fine-tuned model.

Usage:
    python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora
    python generate.py --model_dir results/qwen2.5-1.5b-instruct-lora --prompt "What is DNA?"
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_dir):
    model_dir = Path(model_dir)

    base_model = None
    load_in_4bit = False

    for config_file, key in [
        ("training_info.json", "base_model"),
        ("adapter_config.json", "base_model_name_or_path"),
    ]:
        path = model_dir / config_file
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                base_model = base_model or data.get(key)
                if config_file == "training_info.json":
                    load_in_4bit = data.get("load_in_4bit", False)

    if not base_model:
        raise ValueError(
            f"Cannot determine base model from {model_dir}. "
            "Expected training_info.json or adapter_config.json."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Base model: {base_model}")
    print(f"Adapter:    {model_dir}")
    print(f"4-bit:      {load_in_4bit}")
    print("Loading...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(model, str(model_dir))

    if device == "cpu":
        model = model.to(device)

    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, instruction, device, max_tokens=256, temperature=0.7):
    messages = [{"role": "user", "content": instruction}]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate with fine-tuned model")
    parser.add_argument("--model_dir", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--prompt", default=None, help="Single prompt (skip interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)
    print("Ready!\n")

    if args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt, device, args.max_tokens, args.temperature
        )
        print(f"Prompt: {args.prompt}\n")
        print(f"Response: {response}")
        return

    print("Interactive mode — type your instruction, 'quit' to exit")
    print("-" * 40)

    while True:
        try:
            instruction = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not instruction or instruction.lower() in ("quit", "exit", "q"):
            break

        response = generate_response(
            model, tokenizer, instruction, device, args.max_tokens, args.temperature
        )
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
